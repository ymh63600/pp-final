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

struct AlignedText {
    char* data;
    size_t size;
};

static void load_20newsgroups_aligned(const string& dataset_path,
                                      vector<AlignedText>& documents) {
    documents.clear();
    if (!fs::exists(dataset_path)) return;

    vector<fs::path> files;
    for (auto& p : fs::recursive_directory_iterator(dataset_path)) {
        if (p.is_regular_file() && p.path().extension() == ".txt")
            files.push_back(p.path());
    }
    sort(files.begin(), files.end());

    for (const auto& path : files) {
        ifstream fin(path, ios::in | ios::binary);
        if (!fin) continue;

        fin.seekg(0, ios::end);
        size_t sz = (size_t)fin.tellg();
        fin.seekg(0, ios::beg);

        char* aligned_buf = (char*)aligned_alloc(32, sz + 32);
        fin.read(aligned_buf, sz);

        documents.push_back({aligned_buf, sz});
    }
}

static inline void tokenize_ws_to_ids_avx2_aligned(
    const AlignedText& text,
    vector<int>& out_ids,
    unordered_map<string,int>& vocab,
    vector<string>& id2word)
{
    out_ids.clear();

    const char* s = text.data;
    const size_t n = text.size;

    const __m256i v_space = _mm256_set1_epi8(32);

    size_t i = 0;
    string word;
    word.reserve(64);

    while (i < n) {

        while (i < n && (unsigned char)s[i] <= 32) i++;

        while (i + 32 <= n) {
            __m256i v = _mm256_load_si256((const __m256i*)(s + i));
            __m256i vmax = _mm256_max_epu8(v, v_space);
            __m256i vcmp = _mm256_cmpeq_epi8(vmax, v_space);
            int mask = _mm256_movemask_epi8(vcmp);
            if (mask == -1)
                i += 32;
            else {
                int valid_mask = ~mask;
                int offset = CTZ(valid_mask);
                i += offset;
                goto start_word_found;
            }
        }

        while (i < n && (unsigned char)s[i] <= 32) i++;

    start_word_found:
        if (i >= n) break;

        size_t start = i;
        size_t j = i;

        while (j + 32 <= n) {
            __m256i v = _mm256_load_si256((const __m256i*)(s + j));
            __m256i vmax = _mm256_max_epu8(v, v_space);
            __m256i vcmp = _mm256_cmpeq_epi8(vmax, v_space);
            int mask = _mm256_movemask_epi8(vcmp);
            if (mask == 0)
                j += 32;
            else {
                int offset = CTZ(mask);
                j += offset;
                goto end_word_found;
            }
        }

        while (j < n && !((unsigned char)s[j] <= 32)) j++;

    end_word_found:
        word.assign(s + start, j - start);

        auto it = vocab.find(word);
        int id;
        if (it == vocab.end()) {
            id = id2word.size();
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

    cout << "--- SIMD TF-IDF Timing Report ---\n";

    start_time = timing_clock_t::now();
    vector<AlignedText> documents;
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];
    load_20newsgroups_aligned(dataset_path, documents);
    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << fixed << setprecision(7);
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";

    if (documents.empty()) {
        cout << "Error: No documents found\n";
        return 1;
    }

    const int N = documents.size();

    start_time = timing_clock_t::now();
    unordered_map<string,int> vocab;
    vocab.reserve(200000);
    vector<string> id2word;
    id2word.reserve(200000);
    vector<vector<int>> doc_term_ids(N);

    for (int d = 0; d < N; d++)
        tokenize_ws_to_ids_avx2_aligned(documents[d], doc_term_ids[d], vocab, id2word);

    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "Tokenization Time: " << tokenize_duration << " seconds\n";
    cout << "Vocabulary Size: " << id2word.size() << "\n";

    start_time = timing_clock_t::now();
    vector<int> df(id2word.size(), 0);
    vector<int> tmp_unique;
    tmp_unique.reserve(10000);

    for (int d = 0; d < N; d++) {
        tmp_unique.assign(doc_term_ids[d].begin(), doc_term_ids[d].end());
        sort(tmp_unique.begin(), tmp_unique.end());
        tmp_unique.erase(unique(tmp_unique.begin(), tmp_unique.end()), tmp_unique.end());
        for (int t : tmp_unique) df[t]++;
    }

    vector<double> idf(id2word.size());
    for (size_t t = 0; t < id2word.size(); t++)
        idf[t] = df[t] ? log((double)N / df[t]) : 0.0;

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "Compute IDF Time: " << idf_duration << " seconds\n";

    start_time = timing_clock_t::now();
    vector<vector<pair<int,double>>> all_tf_results(N);

    for (int d = 0; d < N; d++) {
        vector<int>& v = doc_term_ids[d];
        sort(v.begin(), v.end());
        size_t i = 0;
        double total = v.size();
        auto& tf_res = all_tf_results[d];
        while (i < v.size()) {
            int term = v[i];
            size_t j = i + 1;
            while (j < v.size() && v[j] == term) j++;
            tf_res.emplace_back(term, (j - i) / total);
            i = j;
        }
    }

    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";

    start_time = timing_clock_t::now();
    vector<string> out_lines;
    out_lines.reserve(1000000);
    for (int d = 0; d < N; d++) {
        for (auto& kv : all_tf_results[d]) {
            double tfidf = kv.second * idf[kv.first];
            out_lines.push_back(to_string(d) + "," + id2word[kv.first] + "," + to_string(tfidf) + "\n");
        }
    }

    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    ofstream fout("simd.csv");
    fout << "document_id,word,tfidf_value\n";
    for (auto& s : out_lines) fout << s;
    fout.close();

    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "Compute All TF-IDFs Time: "
         << (total_excl_write - load_duration - tokenize_duration - idf_duration - tf_duration) << " seconds\n";
    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "Total Execution Time (excluding write): " << total_excl_write << " seconds\n";
}

