// cuda.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>
#include <fstream>
#include <filesystem>
#include <chrono>

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
// Compute TF for a single document
// ---------------------------
map<string, double> computeTF(const vector<string>& words) {
    map<string, double> tf;
    if (words.empty()) return tf;

    for (const string& w : words) {
        tf[w] += 1.0;
    }
    double total_words = static_cast<double>(words.size());
    for (auto& kv : tf) {
        kv.second /= total_words;
    }
    return tf;
}

// ---------------------------
// Compute IDF across all documents
// ---------------------------
map<string, double> computeIDF(const vector<vector<string>>& docs) {
    map<string, double> idf;
    set<string> vocab;
    int N = static_cast<int>(docs.size());

    for (const auto& doc : docs) {
        set<string> seen;
        for (const string& w : doc) {
            seen.insert(w);
            vocab.insert(w);
        }
        for (const string& w : seen) {
            idf[w] += 1.0;
        }
    }

    for (const string& w : vocab) {
        idf[w] = log(static_cast<double>(N) / idf.at(w));
    }

    return idf;
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
// CUDA kernel: TF-IDF = TF * IDF
// tf, tfidf: [num_docs * vocab_size]
// idf: [vocab_size]
// ---------------------------
__global__
void tfidf_kernel(const double* tf,
                  const double* idf,
                  double* tfidf,
                  int num_docs,
                  int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_docs * vocab_size;
    if (idx >= total) return;

    int term = idx % vocab_size;
    tfidf[idx] = tf[idx] * idf[term];
}

// ---------------------------
// Main
// ---------------------------
int main() {
    using Clock = chrono::high_resolution_clock;

    // ---------------------------
    // Phase 1: Load documents
    // ---------------------------
    auto t0 = Clock::now();
    vector<string> documents;
    string dataset_path = "dataset";
    load_20newsgroups(dataset_path, documents);
    auto t1 = Clock::now();

    int N = static_cast<int>(documents.size());
    cout << "--- CUDA TF-IDF Timing Report ---" << endl;
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
    // Phase 3: Compute IDF (CPU)
    // ---------------------------
    auto t4 = Clock::now();
    map<string, double> idf_map = computeIDF(tokenized_docs);
    auto t5 = Clock::now();
    cout << "Compute IDF Time (CPU): "
         << chrono::duration<double>(t5 - t4).count()
         << " seconds (Vocabulary Size: " << idf_map.size() << ")" << endl;

    // ---------------------------
    // Build vocabulary and IDF vector
    // ---------------------------
    vector<string> vocab;
    vocab.reserve(idf_map.size());
    unordered_map<string, int> word2idx;
    vector<double> idf_vec;
    idf_vec.reserve(idf_map.size());

    int idx = 0;
    for (const auto& kv : idf_map) {
        const string& w = kv.first;
        double idf_val = kv.second;
        vocab.push_back(w);
        word2idx[w] = idx;
        idf_vec.push_back(idf_val);  // already double
        idx++;
    }
    int V = static_cast<int>(vocab.size());
    cout << "Final Vocabulary Size: " << V << endl;

    // ---------------------------
    // Phase 4: Compute TF for all docs (CPU), fill dense matrix
    // ---------------------------
    auto t6 = Clock::now();
    vector<double> tf_host(static_cast<size_t>(N) * V, 0.0);

    for (int d = 0; d < N; ++d) {
        map<string, double> tf_map = computeTF(tokenized_docs[d]);
        for (const auto& kv : tf_map) {
            auto it = word2idx.find(kv.first);
            if (it == word2idx.end()) continue;
            int j = it->second;
            tf_host[static_cast<size_t>(d) * V + j] = kv.second;  // double
        }
    }
    auto t7 = Clock::now();
    cout << "Compute All TFs Time (CPU, dense fill): "
         << chrono::duration<double>(t7 - t6).count() << " seconds" << endl;

    // ---------------------------
    // Phase 5: TF-IDF on GPU
    // ---------------------------
    auto t8 = Clock::now();

    double* d_tf = nullptr;
    double* d_idf = nullptr;
    double* d_tfidf = nullptr;

    size_t tf_bytes  = static_cast<size_t>(N) * V * sizeof(double);
    size_t idf_bytes = static_cast<size_t>(V) * sizeof(double);

    cudaError_t err;
    err = cudaMalloc(&d_tf, tf_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_tf failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    err = cudaMalloc(&d_idf, idf_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_idf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        return 1;
    }
    err = cudaMalloc(&d_tfidf, tf_bytes);
    if (err != cudaSuccess) {
        cerr << "cudaMalloc d_tfidf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        cudaFree(d_idf);
        return 1;
    }

    err = cudaMemcpy(d_tf, tf_host.data(), tf_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy tf_host -> d_tf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        cudaFree(d_idf);
        cudaFree(d_tfidf);
        return 1;
    }
    err = cudaMemcpy(d_idf, idf_vec.data(), idf_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy idf_vec -> d_idf failed: " << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        cudaFree(d_idf);
        cudaFree(d_tfidf);
        return 1;
    }

    int total = N * V;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    tfidf_kernel<<<gridSize, blockSize>>>(d_tf, d_idf, d_tfidf, N, V);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "cudaDeviceSynchronize failed after kernel: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        cudaFree(d_idf);
        cudaFree(d_tfidf);
        return 1;
    }

    vector<double> tfidf_host(static_cast<size_t>(N) * V);
    err = cudaMemcpy(tfidf_host.data(), d_tfidf, tf_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy d_tfidf -> tfidf_host failed: "
             << cudaGetErrorString(err) << endl;
        cudaFree(d_tf);
        cudaFree(d_idf);
        cudaFree(d_tfidf);
        return 1;
    }

    cudaFree(d_tf);
    cudaFree(d_idf);
    cudaFree(d_tfidf);

    auto t9 = Clock::now();
    cout << "Compute All TF-IDFs Time (GPU, kernel + copy): "
         << chrono::duration<double>(t9 - t8).count() << " seconds" << endl;

    // ---------------------------
    // Phase 6: Save to CSV
    // 注意：判斷輸出時改用 tf_host != 0.0，
    // 這樣就算 idf = 0 導致 tfidf = 0，也會輸出（和 serial.cpp 行為一致）
    // ---------------------------
    ofstream fout("cuda.csv");
    fout << "document_id,word,tfidf_value\n";
    for (int d = 0; d < N; ++d) {
        for (int j = 0; j < V; ++j) {
            double tf_val     = tf_host[static_cast<size_t>(d) * V + j];
            double tfidf_val  = tfidf_host[static_cast<size_t>(d) * V + j];
            if (tf_val != 0.0) {  // 只要該字在此 doc 的 TF 非 0，就輸出
                fout << d << "," << vocab[j] << "," << tfidf_val << "\n";
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
