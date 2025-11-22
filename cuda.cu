// cuda.cu
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem;

using Clock = chrono::high_resolution_clock;

// ------------------------------------------------------------
// Fast whitespace tokenizer using string_view (no string copy)
// same semantics as stringstream >> w
// ------------------------------------------------------------
static inline void tokenize_ws_sv(const string& text, vector<string_view>& out_tokens) {
    out_tokens.clear();
    const char* s = text.c_str();
    size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        while (i < n && isspace((unsigned char)s[i])) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !isspace((unsigned char)s[j])) ++j;
        out_tokens.emplace_back(s + i, j - i);
        i = j;
    }
}

// ------------------------------------------------------------
// Load 20-Newsgroups files
// ------------------------------------------------------------
void load_20newsgroups(const string& root, vector<string>& documents) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path());
            if (!fin.is_open()) continue;

            string line;
            string content;
            while (getline(fin, line)) {
                content += line;
                content.push_back(' ');
            }
            documents.push_back(std::move(content));
        }
    }
}

// ------------------------------------------------------------
// CUDA kernels for sparse pipeline
// ------------------------------------------------------------
__global__
void scatter_df_kernel(const int* terms_df,
                       const int* df_vals,
                       int* df_dense,
                       int n_df)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_df) return;
    int t = terms_df[i];
    df_dense[t] = df_vals[i];
}

__global__
void compute_idf_kernel(const int* df_dense,
                        double* idf_dense,
                        int num_docs,
                        int vocab_size)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= vocab_size) return;
    int df_t = df_dense[t];
    if (df_t > 0) {
        idf_dense[t] = log((double)num_docs / (double)df_t);
    } else {
        idf_dense[t] = 0.0;
    }
}

__global__
void tfidf_sparse_kernel(const int* doc_ids,
                         const int* term_ids,
                         const int* counts_nnz,
                         const int* doc_len,
                         const double* idf_dense,
                         double* tfidf_vals,
                         int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int d = doc_ids[i];
    int t = term_ids[i];
    int c = counts_nnz[i];
    int len = doc_len[d];

    if (len <= 0 || c <= 0) {
        tfidf_vals[i] = 0.0;
        return;
    }

    double tf = (double)c / (double)len;
    tfidf_vals[i] = tf * idf_dense[t];
}

// ------------------------------------------------------------
// Thrust functors
// ------------------------------------------------------------
struct MakeKeyFunctor {
    int V;
    __host__ __device__
    uint64_t operator()(const thrust::tuple<int,int>& x) const {
        uint64_t d = (uint64_t)thrust::get<0>(x);
        uint64_t t = (uint64_t)thrust::get<1>(x);
        return d * (uint64_t)V + t;
    }
};

struct KeyToDocFunctor {
    int V;
    __host__ __device__
    int operator()(const uint64_t& k) const {
        return (int)(k / (uint64_t)V);
    }
};

struct KeyToTermFunctor {
    int V;
    __host__ __device__
    int operator()(const uint64_t& k) const {
        return (int)(k % (uint64_t)V);
    }
};

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Phase 1: Load documents
    auto t0 = Clock::now();
    vector<string> documents;
    string dataset_path = "dataset";
    load_20newsgroups(dataset_path, documents);
    auto t1 = Clock::now();

    int N = (int)documents.size();
    cout << "--- CUDA TF-IDF Timing Report (Sparse TF DF IDF TF-IDF) ---" << endl;
    cout << "Total Documents Loaded: " << N << endl;
    cout << "Document Loading Time: "
         << chrono::duration<double>(t1 - t0).count() << " seconds" << endl;

    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"" << endl;
        return 1;
    }

    // Phase 2: Tokenization on CPU (string_view fast scan)
    auto t2 = Clock::now();
    vector<vector<string_view>> tokenized_docs;
    tokenized_docs.reserve(N);

    vector<string_view> tmp_tokens_sv;
    for (const auto& doc : documents) {
        tokenize_ws_sv(doc, tmp_tokens_sv);
        tokenized_docs.push_back(tmp_tokens_sv);
    }
    auto t3 = Clock::now();
    cout << "Tokenization Time (CPU string_view scan): "
         << chrono::duration<double>(t3 - t2).count() << " seconds" << endl;

    // Phase 3: Build vocab and flatten tokens (reserve + std::unordered_map)
    auto t4 = Clock::now();

    size_t total_tokens_est = 0;
    for (int d = 0; d < N; ++d) total_tokens_est += tokenized_docs[d].size();
    size_t est_vocab = total_tokens_est / 6 + 1024;

    unordered_map<string, int> word2id;
    word2id.reserve(est_vocab);
    word2id.max_load_factor(0.7f);

    vector<string> vocab_unsorted;
    vocab_unsorted.reserve(est_vocab);

    vector<int> doc_len(N, 0);
    vector<int> terms_flat;
    terms_flat.reserve(total_tokens_est);
    vector<int> doc_ids_flat;
    doc_ids_flat.reserve(total_tokens_est);

    size_t total_tokens = 0;

    for (int d = 0; d < N; ++d) {
        for (const auto& w_sv : tokenized_docs[d]) {
            string w_str(w_sv);

            auto it = word2id.find(w_str);
            int term_id;
            if (it == word2id.end()) {
                term_id = (int)vocab_unsorted.size();
                vocab_unsorted.push_back(w_str);
                word2id.emplace(vocab_unsorted.back(), term_id);
            } else {
                term_id = it->second;
            }

            terms_flat.push_back(term_id);
            doc_ids_flat.push_back(d);
            ++total_tokens;
            ++doc_len[d];
        }
    }

    auto t5 = Clock::now();
    cout << "Build vocab (reserve + unordered_map + string_view tokenize) + term indices Time (CPU): "
         << chrono::duration<double>(t5 - t4).count() << " seconds" << endl;
    cout << "Total mapped tokens: " << total_tokens << endl;

    if (total_tokens == 0) {
        cout << "No tokens after mapping to vocabulary. Exit." << endl;
        return 0;
    }

    // Phase 4: Sort vocab lexicographically and remap ids
    auto t6 = Clock::now();

    int oldV = (int)vocab_unsorted.size();
    vector<int> perm(oldV);
    iota(perm.begin(), perm.end(), 0);

    sort(perm.begin(), perm.end(),
         [&](int a, int b) { return vocab_unsorted[a] < vocab_unsorted[b]; });

    int V = oldV;
    vector<string> vocab(V);
    vector<int> old2new(oldV);

    for (int new_id = 0; new_id < V; ++new_id) {
        int old_id = perm[new_id];
        vocab[new_id] = vocab_unsorted[old_id];
        old2new[old_id] = new_id;
    }

    for (size_t i = 0; i < terms_flat.size(); ++i) {
        terms_flat[i] = old2new[terms_flat[i]];
    }

    auto t7 = Clock::now();
    cout << "Sort vocab + remap term ids Time (CPU): "
         << chrono::duration<double>(t7 - t6).count() << " seconds" << endl;
    cout << "Final Vocabulary Size: " << V << endl;

    // Phase 5: Sparse GPU pipeline
    auto t8 = Clock::now();

    thrust::device_vector<int> d_terms(terms_flat.begin(), terms_flat.end());
    thrust::device_vector<int> d_doc_ids(doc_ids_flat.begin(), doc_ids_flat.end());
    thrust::device_vector<int> d_doc_len(doc_len.begin(), doc_len.end());

    // Build keys = doc * V + term
    thrust::device_vector<uint64_t> d_keys(total_tokens);
    MakeKeyFunctor make_key{V};
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_doc_ids.begin(), d_terms.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_doc_ids.end(), d_terms.end())),
        d_keys.begin(),
        make_key
    );

    // Sort keys and reduce to sparse counts
    auto t_tf_start = Clock::now();

    thrust::sort(d_keys.begin(), d_keys.end());

    thrust::device_vector<uint64_t> d_ukeys(total_tokens);
    thrust::device_vector<int> d_ones(total_tokens, 1);
    thrust::device_vector<int> d_counts_nnz(total_tokens);

    auto new_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_ones.begin(),
        d_ukeys.begin(),
        d_counts_nnz.begin()
    );

    int nnz = (int)(new_end.first - d_ukeys.begin());
    d_ukeys.resize(nnz);
    d_counts_nnz.resize(nnz);

    cudaDeviceSynchronize();
    auto t_tf_end = Clock::now();

    cout << "TF counting (sparse sort reduce on GPU) Time: "
         << chrono::duration<double>(t_tf_end - t_tf_start).count()
         << " seconds (nnz=" << nnz << ")" << endl;

    // Extract doc_ids_nnz and term_ids_nnz
    thrust::device_vector<int> d_doc_ids_nnz(nnz);
    thrust::device_vector<int> d_term_ids_nnz(nnz);

    KeyToDocFunctor key_to_doc{V};
    KeyToTermFunctor key_to_term{V};
    thrust::transform(d_ukeys.begin(), d_ukeys.end(), d_doc_ids_nnz.begin(), key_to_doc);
    thrust::transform(d_ukeys.begin(), d_ukeys.end(), d_term_ids_nnz.begin(), key_to_term);

    // DF by reducing term ids (each unique (doc,term) contributes 1)
    auto t_df_start = Clock::now();

    thrust::device_vector<int> d_term_for_df = d_term_ids_nnz;
    thrust::sort(d_term_for_df.begin(), d_term_for_df.end());

    thrust::device_vector<int> d_terms_df(nnz);
    thrust::device_vector<int> d_df_vals(nnz);
    thrust::device_vector<int> d_ones_df(nnz, 1);

    auto df_end = thrust::reduce_by_key(
        d_term_for_df.begin(), d_term_for_df.end(),
        d_ones_df.begin(),
        d_terms_df.begin(),
        d_df_vals.begin()
    );

    int n_df = (int)(df_end.first - d_terms_df.begin());
    d_terms_df.resize(n_df);
    d_df_vals.resize(n_df);

    cudaDeviceSynchronize();
    auto t_df_end = Clock::now();

    cout << "DF reduce on GPU Time: "
         << chrono::duration<double>(t_df_end - t_df_start).count()
         << " seconds (unique_terms=" << n_df << ")" << endl;

    // Scatter DF to dense and compute dense IDF
    thrust::device_vector<int> d_df_dense(V, 0);
    thrust::device_vector<double> d_idf_dense(V, 0.0);

    {
        int block = 256;

        int grid_scatter = (n_df + block - 1) / block;
        scatter_df_kernel<<<grid_scatter, block>>>(
            thrust::raw_pointer_cast(d_terms_df.data()),
            thrust::raw_pointer_cast(d_df_vals.data()),
            thrust::raw_pointer_cast(d_df_dense.data()),
            n_df
        );
        cudaDeviceSynchronize();

        int grid_idf = (V + block - 1) / block;
        compute_idf_kernel<<<grid_idf, block>>>(
            thrust::raw_pointer_cast(d_df_dense.data()),
            thrust::raw_pointer_cast(d_idf_dense.data()),
            N, V
        );
        cudaDeviceSynchronize();
    }

    // Sparse TF-IDF
    thrust::device_vector<double> d_tfidf_vals(nnz);

    auto t_tfidf_start = Clock::now();
    {
        int block = 256;
        int grid = (nnz + block - 1) / block;
        tfidf_sparse_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_doc_ids_nnz.data()),
            thrust::raw_pointer_cast(d_term_ids_nnz.data()),
            thrust::raw_pointer_cast(d_counts_nnz.data()),
            thrust::raw_pointer_cast(d_doc_len.data()),
            thrust::raw_pointer_cast(d_idf_dense.data()),
            thrust::raw_pointer_cast(d_tfidf_vals.data()),
            nnz
        );
        cudaDeviceSynchronize();
    }
    auto t_tfidf_end = Clock::now();

    cout << "TF-IDF (sparse GPU) Time: "
         << chrono::duration<double>(t_tfidf_end - t_tfidf_start).count()
         << " seconds" << endl;

    auto t9 = Clock::now();
    cout << "Total sparse GPU pipeline Time: "
         << chrono::duration<double>(t9 - t8).count()
         << " seconds" << endl;

    // Phase 6: Copy nnz results and save CSV
    vector<int> doc_ids_nnz(nnz);
    vector<int> term_ids_nnz(nnz);
    vector<int> counts_nnz(nnz);
    vector<double> tfidf_vals(nnz);

    thrust::copy(d_doc_ids_nnz.begin(), d_doc_ids_nnz.end(), doc_ids_nnz.begin());
    thrust::copy(d_term_ids_nnz.begin(), d_term_ids_nnz.end(), term_ids_nnz.begin());
    thrust::copy(d_counts_nnz.begin(), d_counts_nnz.end(), counts_nnz.begin());
    thrust::copy(d_tfidf_vals.begin(), d_tfidf_vals.end(), tfidf_vals.begin());

    ofstream fout("cuda.csv");
    fout << "document_id,word,tfidf_value\n";
    for (int i = 0; i < nnz; ++i) {
        int d = doc_ids_nnz[i];
        int t = term_ids_nnz[i];
        double val = tfidf_vals[i];
        fout << d << "," << vocab[t] << "," << val << "\n";
    }
    fout.close();
    cout << "TF-IDF saved to cuda.csv" << endl;

    // Summary
    double total_time = chrono::duration<double>(t9 - t0).count();
    cout << "------------------------------------------" << endl;
    cout << "Total Execution Time (including load): "
         << total_time << " seconds" << endl;
    cout << "------------------------------------------" << endl;

    return 0;
}
