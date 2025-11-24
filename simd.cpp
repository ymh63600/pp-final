#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <utility>

// SIMD Header
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#define CTZ(x) _tzcnt_u32(x)
#else
#define CTZ(x) __builtin_ctz(x)
#endif

using namespace std;
namespace fs = std::filesystem;

using timing_clock_t = chrono::high_resolution_clock;
using timing_point_t = timing_clock_t::time_point;

// ------------------------------------------------------------
// Load .txt documents recursively from dataset folder
// ------------------------------------------------------------
static void load_20newsgroups(const string& dataset_path,
                              vector<string>& documents) {
    documents.clear();

    if (!fs::exists(dataset_path)) return;

    // Collect paths first to allow deterministic sorting
    vector<fs::path> files;
    for (auto& p : fs::recursive_directory_iterator(dataset_path)) {
        if (p.is_regular_file() && p.path().extension() == ".txt") {
            files.push_back(p.path());
        }
    }
    sort(files.begin(), files.end());

    for (const auto& path : files) {
        ifstream fin(path, ios::in | ios::binary);
        if (!fin) continue;

        fin.seekg(0, ios::end);
        size_t sz = (size_t)fin.tellg();
        fin.seekg(0, ios::beg);
        
        string content;
        content.resize(sz);
        fin.read(&content[0], sz);

        documents.push_back(std::move(content));
    }
}

// ------------------------------------------------------------
// AVX2 Fast Tokenizer
// ------------------------------------------------------------
static inline void tokenize_ws_to_ids_avx2(const string& text,
                                           vector<int>& out_ids,
                                           unordered_map<string, int>& vocab,
                                           vector<string>& id2word) {
    out_ids.clear();
    const char* s = text.data();
    const size_t n = text.size();
    
    // Constant vector for whitespace comparison (Space is ASCII 32)
    // We treat anything <= 32 as whitespace (Space, Tab, Newline, CR, etc.)
    const __m256i v_space = _mm256_set1_epi8(32);

    size_t i = 0;
    string word;
    word.reserve(64);

    while (i < n) {
        // -------------------------------------------------
        // 1. Skip Whitespace (Find start of word)
        // -------------------------------------------------
        
        // Scalar loop for boundaries or small chunks
        while (i < n && (unsigned char)s[i] <= 32) i++;
        
        // AVX2 Loop: Skip whitespace blocks
        // Only run if we have at least 32 bytes left
        while (i + 32 <= n) {
            __m256i v_data = _mm256_loadu_si256((const __m256i*)(s + i));
            
            // Logic: standard check is isspace(c). Fast check: c <= 32.
            // _mm256_max_epu8(a, b) returns max(a, b). 
            // If c <= 32, max(c, 32) == 32.
            // If c > 32, max(c, 32) == c (which is != 32).
            __m256i v_max = _mm256_max_epu8(v_data, v_space);
            __m256i v_cmp = _mm256_cmpeq_epi8(v_max, v_space); // 0xFF where c <= 32
            
            int mask = _mm256_movemask_epi8(v_cmp);
            
            // mask bits are 1 where char is whitespace.
            // If mask is all 1s (0xFFFFFFFF), everything is whitespace.
            if (mask == -1) { // -1 is all 1s in 2's complement int
                i += 32;
            } else {
                // Found a non-whitespace character!
                // We need the index of the first ZERO bit in the mask.
                // Invert mask, count trailing zeros.
                int valid_mask = ~mask;
                int offset = CTZ(valid_mask);
                i += offset;
                goto start_word_found;
            }
        }
        
        // Tail handling (scalar)
        while (i < n && (unsigned char)s[i] <= 32) i++;

    start_word_found:
        if (i >= n) break;

        size_t start = i;
        size_t j = i;

        // -------------------------------------------------
        // 2. Scan Word (Find end of word)
        // -------------------------------------------------

        // AVX2 Loop: Find next whitespace
        while (j + 32 <= n) {
            __m256i v_data = _mm256_loadu_si256((const __m256i*)(s + j));
            
            // Same logic: check for <= 32
            __m256i v_max = _mm256_max_epu8(v_data, v_space);
            __m256i v_cmp = _mm256_cmpeq_epi8(v_max, v_space); // 0xFF where c <= 32
            
            int mask = _mm256_movemask_epi8(v_cmp);
            
            // mask bits are 1 where char is whitespace.
            // If mask is 0, no whitespace found in this block.
            if (mask == 0) {
                j += 32;
            } else {
                // Found a whitespace!
                // Find index of first ONE bit.
                int offset = CTZ(mask);
                j += offset;
                goto end_word_found;
            }
        }

        // Tail handling (scalar)
        while (j < n && !((unsigned char)s[j] <= 32)) j++;

    end_word_found:
        // Extract word
        // Optimization: Assign directly to reduce allocation overhead
        word.assign(s + start, j - start);

        // Dictionary Lookup (Unordered Map is scalar bound)
        auto it = vocab.find(word);
        int id;
        if (it == vocab.end()) {
            id = (int)id2word.size();
            vocab.emplace(word, id);
            id2word.push_back(word);
        } else {
            id = it->second;
        }

        out_ids.push_back(id);
        i = j;
    }
}

int main(int argc, char** argv) {
    timing_point_t t_total0 = timing_clock_t::now();
    timing_point_t start_time, end_time;

    // Phase 1: Load documents
    start_time = timing_clock_t::now();

    vector<string> documents;
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];
    load_20newsgroups(dataset_path, documents);

    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "--- AVX2 TF-IDF Timing Report ---\n";
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";

    if (documents.empty()) {
        cout << "Error: No documents found at \"" << dataset_path << "\"\n";
        return 1;
    }

    const int N = (int)documents.size();

    // Phase 2: Tokenization (AVX2 Optimized)
    start_time = timing_clock_t::now();

    unordered_map<string, int> vocab;
    vocab.reserve(200000);
    vector<string> id2word;
    id2word.reserve(200000);

    vector<vector<int>> doc_term_ids(N);

    for (int d = 0; d < N; d++) {
        tokenize_ws_to_ids_avx2(documents[d], doc_term_ids[d], vocab, id2word);
    }

    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Tokenization Time (AVX2): " << tokenize_duration << " seconds\n";

    const int V = (int)id2word.size();
    cout << "Vocabulary Size: " << V << "\n";

    // Phase 3: Compute DF and IDF (AVX2 Optimized Division)
    start_time = timing_clock_t::now();

    vector<int> df(V, 0);
    // Temporary vector for unique check
    vector<int> tmp_unique; 
    tmp_unique.reserve(10000); // Pre-allocate reasonable size

    for (int d = 0; d < N; d++) {
        const auto& ids = doc_term_ids[d];
        if (ids.empty()) continue;
        
        tmp_unique.assign(ids.begin(), ids.end());
        sort(tmp_unique.begin(), tmp_unique.end());
        tmp_unique.erase(unique(tmp_unique.begin(), tmp_unique.end()), tmp_unique.end());

        for (int term_id : tmp_unique) {
            df[term_id]++;
        }
    }

    // AVX2 IDF Calculation
    // IDF[t] = log(N / df[t])
    // We vectorize the division. Log is computed serially on the vector results
    // because AVX2 standard headers don't have a log intrinsic.
    vector<double> idf(V);
    
    int t = 0;
    const __m256d v_N = _mm256_set1_pd((double)N); // Broadcast N

    for (; t <= V - 4; t += 4) {
        // Load 4 integers (df values)
        __m128i v_df_i = _mm_loadu_si128((const __m128i*)&df[t]);
        
        // Convert int32 to double (precision)
        __m256d v_df_d = _mm256_cvtepi32_pd(v_df_i);
        
        // Compute N / df[t] for 4 items
        // Note: df[t] could be 0? (Shouldn't be if logic is correct, but safe to handle implies mask)
        // Assuming df > 0 based on logic.
        __m256d v_ratio = _mm256_div_pd(v_N, v_df_d);

        // Store to temporary array to call scalar log
        alignas(32) double temp_ratios[4];
        _mm256_store_pd(temp_ratios, v_ratio);

        // Standard log is often auto-vectorized by O3, but explicit unroll here ensures logic flow
        idf[t]     = (df[t] > 0) ? log(temp_ratios[0]) : 0.0;
        idf[t + 1] = (df[t+1] > 0) ? log(temp_ratios[1]) : 0.0;
        idf[t + 2] = (df[t+2] > 0) ? log(temp_ratios[2]) : 0.0;
        idf[t + 3] = (df[t+3] > 0) ? log(temp_ratios[3]) : 0.0;
    }

    // Tail loop
    for (; t < V; t++) {
        if (df[t] > 0) idf[t] = log((double)N / (double)df[t]);
        else idf[t] = 0.0;
    }

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time (AVX2 Mixed): " << idf_duration << " seconds\n";

    // Phase 4: Compute TF (Partially AVX2 Optimized)
    start_time = timing_clock_t::now();

    vector<vector<pair<int, double>>> all_tf_results(N);
    // Reuse buffer
    vector<int> tmp_ids;
    
    // Helper to perform batch division for TF using AVX2
    auto batch_tf_normalize = [](vector<pair<int, double>>& sparse_vec, double total_terms) {
        size_t sz = sparse_vec.size();
        size_t k = 0;
        __m256d v_total_inv = _mm256_set1_pd(1.0 / total_terms);

        // We process counts (already in sparse_vec.second) and divide by total
        for (; k <= sz - 4; k += 4) {
            // Gather/Load is tricky because struct is pair<int, double> (12 bytes or 16 with padding).
            // It's AoS. Explicit scalar load, Vector op, Scalar store is safest without gather.
            // Or better: just load the doubles? 
            // They are at stride 16 bytes (assuming 64-bit align). 
            // _mm256_i32gather_pd requires indices.
            
            // For simple code, we manually load the doubles
            __m256d v_counts = _mm256_set_pd(
                sparse_vec[k+3].second, sparse_vec[k+2].second,
                sparse_vec[k+1].second, sparse_vec[k].second
            );
            
            __m256d v_tf = _mm256_mul_pd(v_counts, v_total_inv); // mul by inverse is faster than div
            
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, v_tf);
            
            sparse_vec[k].second = tmp[0];
            sparse_vec[k+1].second = tmp[1];
            sparse_vec[k+2].second = tmp[2];
            sparse_vec[k+3].second = tmp[3];
        }
        for (; k < sz; k++) {
            sparse_vec[k].second /= total_terms;
        }
    };

    for (int d = 0; d < N; d++) {
        const auto& ids = doc_term_ids[d];
        if (ids.empty()) continue;

        tmp_ids.assign(ids.begin(), ids.end());
        sort(tmp_ids.begin(), tmp_ids.end());

        double total = (double)tmp_ids.size();
        vector<pair<int, double>>& tf_sparse = all_tf_results[d];
        tf_sparse.reserve(tmp_ids.size() / 2); // Heuristic

        // Run length encoding equivalent
        size_t i = 0;
        while (i < tmp_ids.size()) {
            int term_id = tmp_ids[i];
            size_t j = i + 1;
            while (j < tmp_ids.size() && tmp_ids[j] == term_id) j++;
            
            // Temporarily store COUNT in the double field
            tf_sparse.emplace_back(term_id, (double)(j - i));
            i = j;
        }

        // Apply AVX2 Normalization (Count -> TF)
        batch_tf_normalize(tf_sparse, total);
    }

    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time (AVX2): " << tf_duration << " seconds\n";

    // Phase 5: Compute TF-IDF & Serialize
    // Serialization is I/O bound, SIMD logic for formatting is overkill/complex.
    start_time = timing_clock_t::now();

    vector<string> out_lines;
    out_lines.reserve(1000000);
    vector<pair<string, double>> tfidf_pairs;

    for (int d = 0; d < N; d++) {
        tfidf_pairs.clear();
        tfidf_pairs.reserve(all_tf_results[d].size());

        // We access IDF randomly (gather), so standard SIMD linear load doesn't work well
        // without gather instructions. AVX2 has gather, but latency is high.
        // Keeping this loop scalar for stability unless vocab is huge.
        for (const auto& kv : all_tf_results[d]) {
            int term_id = kv.first;
            double tf_val = kv.second;
            double tfidf_val = tf_val * idf[term_id]; // Random access to dense vector
            tfidf_pairs.emplace_back(id2word[term_id], tfidf_val);
        }

        sort(tfidf_pairs.begin(), tfidf_pairs.end(),
             [](const auto& a, const auto& b) {
                 return a.first < b.first;
             });

        for (const auto& p : tfidf_pairs) {
            out_lines.push_back(to_string(d) + "," + p.first + "," + to_string(p.second) + "\n");
        }
    }

    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    ofstream fout("simd.csv");
    fout << "document_id,word,tfidf_value\n";
    for (const auto& line : out_lines) {
        fout << line;
    }
    fout.close();

    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "TF-IDF saved to simd.csv\n";
    cout << "------------------------------------------\n";
    cout << "Total Execution Time (incl load, excl write): " << total_excl_write << " seconds\n";
    cout << "------------------------------------------\n";

    return 0;
}
