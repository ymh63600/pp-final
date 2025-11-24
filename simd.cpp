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

    cout << "--- SIMD TF-IDF Timing Report --- " << endl;
    // Phase 1: Load documents
    start_time = timing_clock_t::now();
    vector<string> documents;
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];
    load_20newsgroups(dataset_path, documents);
    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << fixed << setprecision(7);
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";

    if (documents.empty()) {
        cout << "Error: No documents found at \"" << dataset_path << "\"\n";
        return 1;
    }

    const int N = (int)documents.size();

    // Phase 2: Tokenization
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
    cout << "Tokenization Time: " << tokenize_duration << " seconds\n";
    cout << "Vocabulary Size: " << id2word.size() << "\n";

    // Phase 3: Compute IDF
    start_time = timing_clock_t::now();
    vector<int> df(id2word.size(), 0);
    vector<int> tmp_unique; tmp_unique.reserve(10000);
    for (int d = 0; d < N; d++) {
        tmp_unique.assign(doc_term_ids[d].begin(), doc_term_ids[d].end());
        sort(tmp_unique.begin(), tmp_unique.end());
        tmp_unique.erase(unique(tmp_unique.begin(), tmp_unique.end()), tmp_unique.end());
        for (int term_id : tmp_unique) df[term_id]++;
    }

    vector<double> idf(id2word.size());
    for (size_t t = 0; t < id2word.size(); t++) {
        idf[t] = df[t] > 0 ? log((double)N / (double)df[t]) : 0.0;
    }
    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time: " << idf_duration << " seconds\n";

    // Phase 4: Compute TF
    start_time = timing_clock_t::now();
    vector<vector<pair<int, double>>> all_tf_results(N);
    vector<int> tmp_ids;
    auto batch_tf_normalize = [](vector<pair<int,double>>& sparse_vec, double total_terms){
        for (auto &p : sparse_vec) p.second /= total_terms;
    };
    for (int d = 0; d < N; d++) {
        tmp_ids.assign(doc_term_ids[d].begin(), doc_term_ids[d].end());
        sort(tmp_ids.begin(), tmp_ids.end());
        double total = (double)tmp_ids.size();
        vector<pair<int,double>>& tf_sparse = all_tf_results[d];
        size_t i = 0;
        while (i < tmp_ids.size()) {
            int term_id = tmp_ids[i];
            size_t j = i + 1;
            while (j < tmp_ids.size() && tmp_ids[j] == term_id) j++;
            tf_sparse.emplace_back(term_id, (double)(j-i));
            i = j;
        }
        batch_tf_normalize(tf_sparse, total);
    }
    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";

    // Phase 5: Compute TF-IDF & Serialize
    start_time = timing_clock_t::now();
    vector<string> out_lines;
    out_lines.reserve(1000000);
    for (int d = 0; d < N; d++) {
        for (const auto &kv : all_tf_results[d]) {
            double tfidf_val = kv.second * idf[kv.first];
            out_lines.push_back(to_string(d) + "," + id2word[kv.first] + "," + to_string(tfidf_val) + "\n");
        }
    }
    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    ofstream fout("simd.csv");
    fout << "document_id,word,tfidf_value\n";
    for (const auto &line : out_lines) fout << line;
    fout.close();
    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TF-IDFs Time: " << (total_excl_write - load_duration - tokenize_duration - idf_duration - tf_duration) << " seconds\n";
    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "TF-IDF saved to simd.csv\n";
    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load, excluding write): " << total_excl_write << " seconds\n";
    cout << "------------------------------------------\n";
}
