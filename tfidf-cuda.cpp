#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace std;

#define D 3  // 文檔數
#define V 6  // 詞彙量

// GPU kernel 計算 TF-IDF
__global__ void compute_tfidf(int *doc_matrix, double *tfidf_matrix, double *idf) {
    int doc_idx = blockIdx.x;
    int word_idx = threadIdx.x;

    int offset = doc_idx * V + word_idx;

    // 計算 TF
    int tf_count = doc_matrix[offset];
    int doc_sum = 0;
    for(int i=0; i<V; i++){
        doc_sum += doc_matrix[doc_idx * V + i];
    }
    double tf = (doc_sum == 0) ? 0 : (double)tf_count / doc_sum;

    // 計算 TF-IDF
    tfidf_matrix[offset] = tf * idf[word_idx];
}

int main() {
    // 範例文檔矩陣: 每行=文檔, 每列=詞頻
    int h_doc_matrix[D][V] = {
        {1,1,1,1,0,0}, // "this is a sample"
        {1,1,0,1,2,0}, // "this is another example example"
        {0,0,1,0,0,1}  // "one more sample document"
    };

    // 計算 IDF (CPU 版本)
    double idf[V];
    for(int j=0; j<V; j++){
        int df = 0;
        for(int i=0; i<D; i++){
            if(h_doc_matrix[i][j] > 0) df++;
        }
        idf[j] = log((double)D / (df == 0 ? 1 : df));
    }

    // 分配 GPU 記憶體
    int *d_doc_matrix;
    double *d_tfidf_matrix, *d_idf;
    cudaMalloc((void**)&d_doc_matrix, D*V*sizeof(int));
    cudaMalloc((void**)&d_tfidf_matrix, D*V*sizeof(double));
    cudaMalloc((void**)&d_idf, V*sizeof(double));

    // 複製資料到 GPU
    cudaMemcpy(d_doc_matrix, h_doc_matrix, D*V*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idf, idf, V*sizeof(double), cudaMemcpyHostToDevice);

    // 每個 block 一個文檔, 每個 thread 對應一個詞
    compute_tfidf<<<D, V>>>(d_doc_matrix, d_tfidf_matrix, d_idf);

    // 複製結果回 CPU
    double h_tfidf[D][V];
    cudaMemcpy(h_tfidf, d_tfidf_matrix, D*V*sizeof(double), cudaMemcpyDeviceToHost);

    // 輸出結果
    for(int i=0; i<D; i++){
        cout << "Document " << i+1 << " TF-IDF: ";
        for(int j=0; j<V; j++){
            cout << h_tfidf[i][j] << " ";
        }
        cout << endl;
    }

    // 釋放 GPU 記憶體
    cudaFree(d_doc_matrix);
    cudaFree(d_tfidf_matrix);
    cudaFree(d_idf);

    return 0;
}
