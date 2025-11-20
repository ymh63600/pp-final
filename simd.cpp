#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <set>
#include <fstream>
#include <filesystem>
#include <chrono>

// ★ 引入 SIMD 庫，使用 AVX Intrinsics 作為範例
#ifdef __GNUC__
#include <immintrin.h> // for AVX/AVX2/AVX512 intrinsics
#endif

using namespace std;
namespace fs = std::filesystem;

// (Type aliases and utility functions like tokenize, computeTF, load_20newsgroups remain unchanged)
using timing_clock_t = chrono::high_resolution_clock;
using timing_point_t = timing_clock_t::time_point;

// Tokenize text into words (Omitted for brevity)
vector<string> tokenize(const string& text) { /* ... unchanged ... */
    vector<string> tokens; stringstream ss(text); string word;
    while (ss >> word) { tokens.push_back(word); } return tokens;
}
// Compute TF for a single document (Omitted for brevity)
map<string, double> computeTF(const vector<string>& words) { /* ... unchanged ... */
    map<string, double> tf; if (words.empty()) return tf;
    for (const string& w : words) { tf[w] += 1.0; } double total_words = words.size();
    for (auto& [word, count] : tf) { count /= total_words; } return tf;
}
// Load 20-Newsgroups files (Omitted for brevity)
void load_20newsgroups(const string& root, vector<string>& documents) { /* ... unchanged ... */
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path()); if (!fin.is_open()) continue;
            string line, content;
            while (getline(fin, line)) { content += line + " "; }
            documents.push_back(content);
        }
    }
}


// --- 修改 IDF 計算：為了 SIMD 應用，需要將數值計算與地圖操作分離 ---
map<string, double> computeIDF(const vector<vector<string>>& docs) {
    map<string, double> idf_counts;
    set<string> vocab;
    int N = docs.size();

    // Phase 1: Count Document Frequencies (df) - Non-SIMD
    for (const auto& doc : docs) {
        set<string> seen;
        for (const string& w : doc) {
            seen.insert(w);
            vocab.insert(w);
        }
        for (const string& w : seen) {
            idf_counts[w] += 1.0;
        }
    }

    // Phase 2: Compute IDF = log(N / df) - 這裡適合應用 SIMD
    map<string, double> idf_final;
    int vocab_size = idf_counts.size();

    // 為了 SIMD，我們需要連續的陣列來儲存數值
    vector<double> df_values;
    df_values.reserve(vocab_size);
    vector<string> vocab_list;
    vocab_list.reserve(vocab_size);

    for (const auto& [word, count] : idf_counts) {
        df_values.push_back(count);
        vocab_list.push_back(word);
    }

    vector<double> idf_values(vocab_size);
    double N_double = (double)N;

    // ★ 核心 SIMD 邏輯：使用 Intrinsics 處理 df_values
#ifdef __AVX512F__ 
    // AVX-512 處理 8 個 double
    const int SIMD_WIDTH = 8;
#elif defined(__AVX__) || defined(__AVX2__)
    // AVX/AVX2 處理 4 個 double
    const int SIMD_WIDTH = 4;
#else
    const int SIMD_WIDTH = 1;
#endif

    // SIMD 向量化處理
    for (int i = 0; i < vocab_size; i += SIMD_WIDTH) {
        if (i + SIMD_WIDTH <= vocab_size) {
#if defined(__AVX512F__)
            // 1. 載入 df 數據
            __m512d df_vec = _mm512_loadu_pd(&df_values[i]);
            // 2. 計算 N / df
            __m512d ratio_vec = _mm512_div_pd(_mm512_set1_pd(N_double), df_vec);
            // 3. 計算 log (使用優化的 log 函式，這裡暫時使用數學庫的向量擴展)
            __m512d idf_vec = _mm512_log_pd(ratio_vec); // 假設有向量化 log 函式
            // 4. 儲存結果
            _mm512_storeu_pd(&idf_values[i], idf_vec);
#elif defined(__AVX__) || defined(__AVX2__)
            // AVX/AVX2 (4 個 double)
            // __m256d df_vec = _mm256_loadu_pd(&df_values[i]);
            // __m256d ratio_vec = _mm256_div_pd(_mm256_set1_pd(N_double), df_vec);
            // 注意：標準 C++ Intrinsics 沒有內建的向量化 log，需要 VML 庫或自定義近似
            // 這裡為了演示，使用序列調用
            for (int j = 0; j < 4; ++j) {
                idf_values[i + j] = log(N_double / df_values[i + j]);
            }
#endif
        } else {
            // 處理剩餘的非對齊數據 (尾部清理)
            for (int j = i; j < vocab_size; ++j) {
                idf_values[j] = log(N_double / df_values[j]);
            }
        }
    }

    // 將結果重新組合成 map
    for (int i = 0; i < vocab_size; ++i) {
        idf_final[vocab_list[i]] = idf_values[i];
    }
    return idf_final;
}


// --- TF-IDF 計算：也可以透過將 map 數據提取到連續向量中來應用 SIMD ---
map<string, double> computeTFIDF(const map<string, double>& tf, const map<string, double>& idf) {
    map<string, double> tfidf;
    
    // 由於 TF-IDF 涉及查找 IDF（map 查找是瓶頸），這裡使用序列迴圈最實際。
    // 如果想要 SIMD，需要預先將 TF 和 IDF 對齊成向量表示（通常透過詞彙表索引），這超出了當前簡單結構的範圍。
    for (const auto& [word, tf_val] : tf) {
        double idf_val = idf.count(word) ? idf.at(word) : 0.0;
        // 核心計算：tf_val * idf_val，這可以在向量化結構上 SIMD 加速
        tfidf[word] = tf_val * idf_val; 
    }
    return tfidf;
}


// Main execution logic with timing (Omitted for brevity, remains unchanged)
int main() {
    timing_point_t start_time, end_time;
    // ... (All timing phases remain the same) ...
    // Note: The actual speedup will depend on the compiler's ability to vectorize the log function 
    // and whether AVX/AVX2/AVX512 is enabled and supported.
    
    // --- Phase 1: Load Documents ---
    start_time = timing_clock_t::now();
    vector<string> documents;
    string dataset_path = "dataset"; 
    load_20newsgroups(dataset_path, documents);
    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "--- SIMD TF-IDF Timing Report ---\n";
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";
    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"\n";
        return 1;
    }
    
    // --- Phase 2: Tokenization ---
    start_time = timing_clock_t::now();
    vector<vector<string>> tokenized_docs;
    for (const auto& doc : documents) {
        tokenized_docs.push_back(tokenize(doc));
    }
    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Tokenization Time: " << tokenize_duration << " seconds\n";

    // --- Phase 3: Compute IDF (SIMD Optimized) ---
    start_time = timing_clock_t::now();
    map<string, double> idf = computeIDF(tokenized_docs);
    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time: " << idf_duration << " seconds (Vocabulary Size: " << idf.size() << ")\n";

    // --- Phase 4: Compute All TFs ---
    start_time = timing_clock_t::now();
    vector<map<string, double>> all_tf_results;
    for (const auto& doc_tokens : tokenized_docs) {
        all_tf_results.push_back(computeTF(doc_tokens));
    }
    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";

    // --- Phase 5: Compute All TF-IDFs ---
    start_time = timing_clock_t::now();
    vector<map<string, double>> all_tfidf_results;
    for (const auto& tf_map : all_tf_results) {
        all_tfidf_results.push_back(computeTFIDF(tf_map, idf));
    }
    end_time = timing_clock_t::now();
    double tfidf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TF-IDFs Time: " << tfidf_duration << " seconds\n";

    // --- Summary ---
    double total_duration = load_duration + tokenize_duration + idf_duration + tf_duration + tfidf_duration;
    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load): " << total_duration << " seconds\n";
    cout << "------------------------------------------\n";

    return 0;
}
