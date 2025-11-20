// cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

// ---------------------------
// Tokenize text into words
// ---------------------------
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string w;
    while (ss >> w) {
        tokens.push_back(w);
    }
    return tokens;
}

// ---------------------------
// Load 20-Newsgroups files
// ---------------------------
void load_20newsgroups(const string& root, vector<string>& documents) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path());
            if (!fin.is_open()) continue;

            string line;
            string content;
            while (getline(fin, line)) {
                content += line + " ";
            }
            documents.push_back(content);
        }
    }
}

// ---------------------------
// CUDA kernel 1: TF counting on GPU (dense N x V)
// 每個 block 處理一篇 document，threads 在該 doc 的 tokens 上並行，對 counts[d, term] 作 atomicAdd
// counts 使用 int，避免 atomicAdd(double*) 的相容性問題
// ---------------------------
__global__
void tf_count_kernel(const int* terms_flat,
                     const int* doc_offsets,
                     const int* doc_len,
                     int* counts,
                     int num_docs,
                     int vocab_size)
{
    int d = blockIdx.x; // one block per doc
    if (d >= num_docs) return;

    int start = doc_offsets[d];
    int len   = doc_len[d];

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        int term = terms_flat[start + i];
        if (term >= 0 && term < vocab_size) {
            atomicAdd(&counts[(size_t)d * vocab_size + term], 1);
        }
    }
}

// ---------------------------
// CUDA kernel 2 (Level 3): 從 counts 計算每個 term 的 df[t]
// df[t] = 有幾篇 doc 的 counts[d, t] > 0
// ---------------------------
__global__
void df_from_counts_kernel(const int* counts,
                           int* df,
                           int num_docs,
                           int vocab_size)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= vocab_size) return;

    int df_count = 0;
    for (int d = 0; d < num_docs; ++d) {
        int c = counts[d * vocab_size + t];
        if (c > 0) {
            ++df_count;
        }
    }
    df[t] = df_count;
}

// ---------------------------
// CUDA kernel 3: TF-IDF = (count / doc_len) * IDF
// counts: [N * V] (int), idf: [V] (double), doc_len: [N], tfidf: [N * V] (double)
// ---------------------------
__global__
void tfidf_dense_kernel(const int* counts,
                        const double* idf,
                        const int* doc_len,
                        double* tfidf,
                        int num_docs,
                        int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total_ll = (long long)num_docs * (long long)vocab_size;
    if ((long long)idx >= total_ll) return;

    int doc  = idx / vocab_size;
    int term = idx % vocab_size;
    int len  = doc_len[doc];

    if (len <= 0) {
        tfidf[idx] = 0.0;
        return;
    }

    int c = counts[idx];
    if (c == 0) {
        tfidf[idx] = 0.0;
        return;
    }

    double tf = static_cast<double>(c) / static_cast<double>(len);
    tfidf[idx] = tf * idf[term];
}

// ---------------------------
// Main
// ---------------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    using Clock = chrono::high_resolution_clock;

    // 這四個 vector 提前宣告，避免 goto 跳過初始化
    vector<int>    df_host;
    vector<double> idf_vec;
    vector<int>    counts_host;
    vector<double> tfidf_host;

    // ---------------------------
    // Phase 1: Load documents
    // ---------------------------
    auto t0 = Clock::now();
    vector<string> documents;
    string dataset_path = "dataset";
    load_20newsgroups(dataset_path, documents);
    auto t1 = Clock::now();

    int N = static_cast<int>(documents.size());
    cout << "--- CUDA TF-IDF Timing Report (TF+IDF+TF-IDF with DF on GPU) ---" << endl;
    cout << "Total Documents Loaded: " << N << endl;
    cout << "Document Loading Time: "
         << chrono::duration<double>(t1 - t0).count() << " seconds" << endl;

    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"" << endl;
        return 1;
    }

    // ---------------------------
    // Phase 2: Tokenization (CPU)
    // ---------------------------
    auto t2 = Clock::now();
    vector<vector<string>> tokenized_docs;
    tokenized_docs.reserve(N);
    for (const auto& doc : documents) {
        tokenized_docs.push_back(tokenize(doc));
    }
    auto t3 = Clock::now();
    cout << "Tokenization Time (CPU): "
         << chrono::duration<double>(t3 - t2).count() << " seconds" << endl;

    // ---------------------------
    // Phase 3: Build vocab (unordered) + term indices (old id)，同時建立 doc_offsets / doc_len
    // ---------------------------
    auto t4 = Clock::now();

    unordered_map<string, int> word2id;
    word2id.reserve(400000); // 粗估，避免 rehash
    vector<string> vocab_unsorted;
    vocab_unsorted.reserve(400000);

    vector<int> doc_offsets(N + 1, 0);
    vector<int> doc_len(N, 0);
    vector<int> terms_flat;  // 使用 old_id
    terms_flat.reserve(10000000); // 粗估

    size_t total_tokens = 0;
    for (int d = 0; d < N; ++d) {
        doc_offsets[d] = static_cast<int>(total_tokens);
        for (const auto& w : tokenized_docs[d]) {
            int term_id;
            auto it = word2id.find(w);
            if (it == word2id.end()) {
                term_id = static_cast<int>(vocab_unsorted.size());
                vocab_unsorted.push_back(w);
                word2id.emplace(w, term_id);
            } else {
                term_id = it->second;
            }
            terms_flat.push_back(term_id);
            ++total_tokens;
            ++doc_len[d];
        }
    }
    doc_offsets[N] = static_cast<int>(total_tokens);

    auto t5 = Clock::now();
    cout << "Build vocab (unordered) + term indices Time (CPU): "
         << chrono::duration<double>(t5 - t4).count() << " seconds" << endl;
    cout << "Total mapped tokens: " << total_tokens << endl;

    if (total_tokens == 0) {
        cout << "No tokens after mapping to vocabulary. Exit." << endl;
        return 0;
    }

    // ---------------------------
    // Phase 4: 將 vocab 依字典序排序，並把 terms_flat 的 term id remap 成「字典序順序」
    // 這樣輸出的 vocabulary 順序會與 serial.cpp 的 map<string,double> 一致
    // ---------------------------
    auto t6 = Clock::now();

    int oldV = static_cast<int>(vocab_unsorted.size());
    vector<int> perm(oldV);
    std::iota(perm.begin(), perm.end(), 0);

    std::sort(perm.begin(), perm.end(),
              [&](int a, int b) {
                  return vocab_unsorted[a] < vocab_unsorted[b];
              });

    int V = oldV;
    vector<string> vocab(V);
    vector<int> old2new(oldV);

    for (int new_id = 0; new_id < V; ++new_id) {
        int old_id = perm[new_id];
        vocab[new_id] = vocab_unsorted[old_id];
        old2new[old_id] = new_id;
    }

    // remap terms_flat: old_id -> new_id (字典序)
    for (size_t i = 0; i < terms_flat.size(); ++i) {
        int old_id = terms_flat[i];
        terms_flat[i] = old2new[old_id];
    }

    auto t7 = Clock::now();
    cout << "Sort vocab + remap term ids Time (CPU): "
         << chrono::duration<double>(t7 - t6).count() << " seconds" << endl;
    cout << "Final Vocabulary Size: " << V << endl;

    // host buffer 在知道 N / V 之後再 resize
    df_host.assign(V, 0);
    idf_vec.assign(V, 0.0);
    counts_host.assign(static_cast<size_t>(N) * V, 0);
    tfidf_host.assign(static_cast<size_t>(N) * V, 0.0);

    // ---------------------------
    // Phase 5: TF / DF (GPU) + IDF (CPU log) + TF-IDF (GPU)
    // 1) tf_count_kernel: counts[d,t]
    // 2) df_from_counts_kernel: df[t] = #docs with counts>0
    // 3) host: idf[t] = log(N / df[t])
    // 4) tfidf_dense_kernel: tfidf = (counts / doc_len) * idf
    // ---------------------------
    auto t8 = Clock::now();

    int*    d_counts      = nullptr; // [N * V] int
    double* d_tfidf       = nullptr; // [N * V] double
    int*    d_terms       = nullptr; // [total_tokens] int
    int*    d_doc_offsets = nullptr; // [N+1] int
    int*    d_doc_len     = nullptr; // [N] int
    int*    d_df          = nullptr; // [V] int
    double* d_idf         = nullptr; // [V] double

    size_t counts_bytes      = static_cast<size_t>(N) * V * sizeof(int);
    size_t tfidf_bytes       = static_cast<size_t>(N) * V * sizeof(double);
    size_t terms_bytes       = static_cast<size_t>(total_tokens) * sizeof(int);
    size_t doc_offsets_bytes = static_cast<size_t>(N + 1) * sizeof(int);
    size_t doc_len_bytes     = static_cast<size_t>(N) * sizeof(int);
    size_t df_bytes          = static_cast<size_t>(V) * sizeof(int);
    size_t idf_bytes         = static_cast<size_t>(V) * sizeof(double);

    cudaError_t err;
    // --- allocate device memory ---
    err = cudaMalloc(&d_counts, counts_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_counts failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    err = cudaMalloc(&d_tfidf, tfidf_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_tfidf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        return 1;
    }
    err = cudaMalloc(&d_terms, terms_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_terms failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        cudaFree(d_tfidf);
        return 1;
    }
    err = cudaMalloc(&d_doc_offsets, doc_offsets_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_doc_offsets failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        cudaFree(d_tfidf);
        cudaFree(d_terms);
        return 1;
    }
    err = cudaMalloc(&d_doc_len, doc_len_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_doc_len failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        cudaFree(d_tfidf);
        cudaFree(d_terms);
        cudaFree(d_doc_offsets);
        return 1;
    }
    err = cudaMalloc(&d_df, df_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_df failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        cudaFree(d_tfidf);
        cudaFree(d_terms);
        cudaFree(d_doc_offsets);
        cudaFree(d_doc_len);
        return 1;
    }
    err = cudaMalloc(&d_idf, idf_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_idf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_counts);
        cudaFree(d_tfidf);
        cudaFree(d_terms);
        cudaFree(d_doc_offsets);
        cudaFree(d_doc_len);
        cudaFree(d_df);
        return 1;
    }

    // 初始化 counts / df
    err = cudaMemset(d_counts, 0, counts_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMemset d_counts failed: " << cudaGetErrorString(err) << endl;
        goto GPU_CLEANUP;
    }
    err = cudaMemset(d_df, 0, df_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMemset d_df failed: " << cudaGetErrorString(err) << endl;
        goto GPU_CLEANUP;
    }

    // Host -> Device
    err = cudaMemcpy(d_terms, terms_flat.data(), terms_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy terms_flat -> d_terms failed: " << cudaGetErrorString(err) << endl;
        goto GPU_CLEANUP;
    }
    err = cudaMemcpy(d_doc_offsets, doc_offsets.data(), doc_offsets_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy doc_offsets -> d_doc_offsets failed: " << cudaGetErrorString(err) << endl;
        goto GPU_CLEANUP;
    }
    err = cudaMemcpy(d_doc_len, doc_len.data(), doc_len_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy doc_len -> d_doc_len failed: " << cudaGetErrorString(err) << endl;
        goto GPU_CLEANUP;
    }

    // ---- Kernel 1: TF counting ----
    {
        int blockSizeTF = 256;
        int gridSizeTF  = N; // one block per doc

        auto t_tf_start = Clock::now();
        tf_count_kernel<<<gridSizeTF, blockSizeTF>>>(d_terms,
                                                     d_doc_offsets,
                                                     d_doc_len,
                                                     d_counts,
                                                     N,
                                                     V);
        err = cudaDeviceSynchronize();
        auto t_tf_end = Clock::now();

        if (err != cudaSuccess) {
            cerr << "cudaDeviceSynchronize failed after tf_count_kernel: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }
        cout << "TF counting (GPU) Time: "
             << chrono::duration<double>(t_tf_end - t_tf_start).count() << " seconds" << endl;
    }

    // ---- Kernel 2: DF from counts (GPU) + host 上算 IDF ----
    {
        int blockSizeDF = 256;
        int gridSizeDF  = (V + blockSizeDF - 1) / blockSizeDF;

        auto t_df_start = Clock::now();
        df_from_counts_kernel<<<gridSizeDF, blockSizeDF>>>(d_counts,
                                                           d_df,
                                                           N,
                                                           V);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "cudaDeviceSynchronize failed after df_from_counts_kernel: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }

        // 取回 df[t]
        err = cudaMemcpy(df_host.data(), d_df, df_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cerr << "cudaMemcpy d_df -> df_host failed: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }

        // 在 CPU 上算 idf[t] = log(N / df[t])，保持與 serial.cpp 相同的 log 實作
        for (int t = 0; t < V; ++t) {
            int df_t = df_host[t];
            if (df_t > 0) {
                idf_vec[t] = std::log(static_cast<double>(N) / static_cast<double>(df_t));
            } else {
                idf_vec[t] = 0.0;
            }
        }

        // idf_vec -> d_idf
        err = cudaMemcpy(d_idf, idf_vec.data(), idf_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cerr << "cudaMemcpy idf_vec -> d_idf failed: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }

        auto t_df_end = Clock::now();
        cout << "IDF: DF on GPU + log on CPU Time: "
             << chrono::duration<double>(t_df_end - t_df_start).count() << " seconds" << endl;
    }

    // ---- Kernel 3: TF-IDF (GPU) ----
    {
        int blockSizeTFIDF = 256;
        long long totalElems_ll = static_cast<long long>(N) * static_cast<long long>(V);
        int totalElems = static_cast<int>(totalElems_ll);
        int gridSizeTFIDF = (totalElems + blockSizeTFIDF - 1) / blockSizeTFIDF;

        auto t_tfidf_start = Clock::now();
        tfidf_dense_kernel<<<gridSizeTFIDF, blockSizeTFIDF>>>(d_counts,
                                                              d_idf,
                                                              d_doc_len,
                                                              d_tfidf,
                                                              N,
                                                              V);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "cudaDeviceSynchronize failed after tfidf_dense_kernel: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }

        // 把 counts / tfidf 拿回 host
        err = cudaMemcpy(counts_host.data(), d_counts, counts_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cerr << "cudaMemcpy d_counts -> counts_host failed: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }
        err = cudaMemcpy(tfidf_host.data(), d_tfidf, tfidf_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cerr << "cudaMemcpy d_tfidf -> tfidf_host failed: "
                 << cudaGetErrorString(err) << endl;
            goto GPU_CLEANUP;
        }

        auto t_tfidf_end = Clock::now();
        cout << "TF-IDF (GPU) Time: "
             << chrono::duration<double>(t_tfidf_end - t_tfidf_start).count() << " seconds" << endl;
    }

GPU_CLEANUP:
    cudaFree(d_counts);
    cudaFree(d_tfidf);
    cudaFree(d_terms);
    cudaFree(d_doc_offsets);
    cudaFree(d_doc_len);
    cudaFree(d_df);
    cudaFree(d_idf);

    auto t9 = Clock::now();
    cout << "Total TF/IDF/TF-IDF GPU pipeline Time: "
         << chrono::duration<double>(t9 - t8).count() << " seconds" << endl;

    // 如果前面有錯（err != cudaSuccess），counts_host / tfidf_host 內容可能無效，
    // 這裡簡單判斷一下，錯了就直接結束，不寫 CSV。
    if (err != cudaSuccess) {
        cerr << "Error occurred in GPU pipeline, abort writing cuda.csv" << endl;
        double total_time_err = chrono::duration<double>(t9 - t0).count();
        cout << "------------------------------------------" << endl;
        cout << "Total Execution Time (including load, with error): "
             << total_time_err << " seconds" << endl;
        cout << "------------------------------------------" << endl;
        return 1;
    }

    // ---------------------------
    // Phase 6: Save to CSV
    // 只要該詞在該 doc 出現過（count > 0），就輸出一列。
    // 即使 idf = 0 導致 tfidf = 0 也會輸出，與 serial.cpp 行為一致。
    // 順序：doc 0..N-1，vocab 依字典序。
// ---------------------------
    ofstream fout("cuda.csv");
    fout << "document_id,word,tfidf_value\n";

    for (int d = 0; d < N; ++d) {
        size_t base = static_cast<size_t>(d) * V;
        for (int j = 0; j < V; ++j) {
            int cnt = counts_host[base + j];
            if (cnt > 0) {
                double val = tfidf_host[base + j];
                fout << d << "," << vocab[j] << "," << val << "\n";
            }
        }
    }
    fout.close();
    cout << "TF-IDF saved to cuda.csv" << endl;

    // ---------------------------
    // Summary
    // ---------------------------
    double total_time = chrono::duration<double>(t9 - t0).count();
    cout << "------------------------------------------" << endl;
    cout << "Total Execution Time (including load): "
         << total_time << " seconds" << endl;
    cout << "------------------------------------------" << endl;

    return 0;
}
