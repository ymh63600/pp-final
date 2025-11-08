#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <cmath>

using namespace std;

// CUDA kernel: 計算 TF-IDF (每個 thread 處理一個非零元素)
__global__ void compute_tfidf_sparse(int *d_row, int *d_col, float *d_tf, float *d_idf, float *d_tfidf, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nnz){
        int col_idx = d_col[idx];
        d_tfidf[idx] = d_tf[idx] * d_idf[col_idx];
    }
}

// CPU helper: 分詞
vector<string> tokenize(const string &text) {
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while(ss >> word) tokens.push_back(word);
    return tokens;
}

int main() {
    // 範例文檔
    vector<string> documents = {
        "this is a sample",
        "this is another example example",
        "one more sample document"
    };

    int D = documents.size();

    // 建立詞彙表
    map<string,int> vocab_map;
    int vocab_id = 0;

    // COO sparse representation
    vector<int> row_idx;
    vector<int> col_idx;
    vector<float> tf_values;

    vector<int> df_count; // 文檔頻率計算
    vector<string> vocab_list;

    // CPU: 生成 sparse TF + DF
    for(int doc_id=0; doc_id<D; doc_id++){
        vector<string> words = tokenize(documents[doc_id]);
        map<int,int> word_count;
        for(auto &w: words){
            if(vocab_map.find(w) == vocab_map.end()){
                vocab_map[w] = vocab_id++;
                vocab_list.push_back(w);
                df_count.push_back(0);
            }
            int id = vocab_map[w];
            word_count[id]++;
        }
        for(auto &[id, cnt]: word_count){
            row_idx.push_back(doc_id);
            col_idx.push_back(id);
            tf_values.push_back((float)cnt / words.size());
            df_count[id] += 1;
        }
    }

    int V = vocab_id;
    int nnz = tf_values.size();

    // CPU: 計算 IDF
    vector<float> idf(V);
    for(int i=0;i<V;i++){
        idf[i] = log((float)D / df_count[i]);
    }

    // GPU memory
    int *d_row, *d_col;
    float *d_tf, *d_idf, *d_tfidf;
    cudaMalloc(&d_row, nnz*sizeof(int));
    cudaMalloc(&d_col, nnz*sizeof(int));
    cudaMalloc(&d_tf, nnz*sizeof(float));
    cudaMalloc(&d_idf, V*sizeof(float));
    cudaMalloc(&d_tfidf, nnz*sizeof(float));

    // 複製到 GPU
    cudaMemcpy(d_row, row_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tf, tf_values.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idf, idf.data(), V*sizeof(float), cudaMemcpyHostToDevice);

    // GPU 計算 TF-IDF
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    compute_tfidf_sparse<<<blocks, threads>>>(d_row, d_col, d_tf, d_idf, d_tfidf, nnz);
    cudaDeviceSynchronize();

    // 複製結果回 CPU
    vector<float> tfidf_values(nnz);
    cudaMemcpy(tfidf_values.data(), d_tfidf, nnz*sizeof(float), cudaMemcpyDeviceToHost);

    // 輸出結果
    cout << "TF-IDF (COO format):\n";
    for(int i=0;i<nnz;i++){
        cout << "Doc " << row_idx[i] << ", Word '" << vocab_list[col_idx[i]] << "' = " << tfidf_values[i] << endl;
    }

    // 釋放 GPU memory
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_tf);
    cudaFree(d_idf);
    cudaFree(d_tfidf);

    return 0;
}
