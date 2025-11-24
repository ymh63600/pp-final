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
#include <omp.h> // 引入 OpenMP 標頭檔

using namespace std;
namespace fs = std::filesystem;

using timing_clock_t = chrono::high_resolution_clock;
using timing_point_t = timing_clock_t::time_point;

// ------------------------------------------------------------
// Load .txt documents recursively from dataset folder (Unchanged)
// ------------------------------------------------------------
static void load_20newsgroups(const string& dataset_path,
                             vector<string>& documents) {
    documents.clear();

    vector<fs::path> files;
    if (!fs::exists(dataset_path)) return;

    for (auto& p : fs::recursive_directory_iterator(dataset_path)) {
        if (p.is_regular_file() && p.path().extension() == ".txt") {
            files.push_back(p.path());
        }
    }
    sort(files.begin(), files.end());

    for (const auto& path : files) {
        ifstream fin(path, ios::in | ios::binary);
        if (!fin) continue;

        string content;
        fin.seekg(0, ios::end);
        size_t sz = (size_t)fin.tellg();
        fin.seekg(0, ios::beg);
        content.resize(sz);
        fin.read(&content[0], sz);

        documents.push_back(std::move(content));
    }
}

// ------------------------------------------------------------
// Fast whitespace tokenizer -> term strings (Optimized for Parallel)
// 此函式只負責將文檔分割為單字字串，不進行ID轉換，因此是 Lock-Free 的。
// ------------------------------------------------------------
static inline void tokenize_ws_to_strings(const string& text,
                                          vector<string>& out_words) {
    out_words.clear();
    const char* s = text.data();
    const size_t n = text.size();

    size_t i = 0;
    string word;
    word.reserve(32);

    while (i < n) {
        while (i < n && std::isspace((unsigned char)s[i])) i++;
        if (i >= n) break;

        size_t j = i;
        while (j < n && !std::isspace((unsigned char)s[j])) j++;

        word.assign(s + i, j - i);
        out_words.push_back(std::move(word));

        i = j;
    }
}

int main(int argc, char** argv) {
    timing_point_t t_total0 = timing_clock_t::now();
    timing_point_t start_time, end_time;

    // Phase 1: Load documents (IO, included in total)
    start_time = timing_clock_t::now();

    vector<string> documents;
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];
    load_20newsgroups(dataset_path, documents);

    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "--- OpenMP TF-IDF Timing Report (Optimized Tokenization) ---\n";
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";

    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"\n";
        return 1;
    }

    const int N = (int)documents.size();

    // --------------------------------------------------------------------------------
    // Phase 2: Tokenization + build vocab + store term ids per doc (Optimized Parallel)
    // --------------------------------------------------------------------------------
    start_time = timing_clock_t::now();

    // 儲存每個文件分割後的單字字串 (臨時需要)
    vector<vector<string>> doc_term_strings(N);

    // 2a. 並行 Tokenization: 將文件分割為單字字串 (Lock-Free)
    #pragma omp parallel for
    for (int d = 0; d < N; d++) {
        tokenize_ws_to_strings(documents[d], doc_term_strings[d]);
    }

    // 2b. 單執行緒構建全域詞彙表並轉換 IDs (Lock-Free for Vocab Build)
    unordered_map<string, int> vocab;
    vocab.reserve(200000);
    vector<string> id2word;
    id2word.reserve(200000);

    vector<vector<int>> doc_term_ids(N);

    for (int d = 0; d < N; d++) {
        const auto& strings = doc_term_strings[d];
        doc_term_ids[d].reserve(strings.size());

        for (const string& word : strings) {
            auto it = vocab.find(word);
            int id;
            if (it == vocab.end()) {
                id = (int)id2word.size();
                vocab.emplace(word, id);
                id2word.push_back(word);
            } else {
                id = it->second;
            }
            doc_term_ids[d].push_back(id);
        }
        // 釋放 doc_term_strings 的記憶體
        vector<string>().swap(doc_term_strings[d]); 
    }
    // 釋放整個 doc_term_strings 容器
    vector<vector<string>>().swap(doc_term_strings);


    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Tokenization Time: " << tokenize_duration << " seconds (Optimized Parallel)\n";

    const int V = (int)id2word.size();
    cout << "Vocabulary Size: " << V << "\n";

    // --------------------------------------------------------------------------------
    // Phase 3: Compute DF and IDF (Parallelized)
    // --------------------------------------------------------------------------------
    start_time = timing_clock_t::now();

    vector<int> df(V, 0);

    // OpenMP 並行化：計算 DF (Document Frequency)
    #pragma omp parallel 
    {
        // 每個執行緒有自己的 tmp 向量
        vector<int> tmp; 

        #pragma omp for
        for (int d = 0; d < N; d++) {
            const auto& ids = doc_term_ids[d];
            tmp.assign(ids.begin(), ids.end());
            sort(tmp.begin(), tmp.end());
            // 找出唯一詞彙 ID
            tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());

            for (int term_id : tmp) {
                // 使用 atomic 來安全地增加共享陣列的元素
                #pragma omp atomic
                df[term_id]++;
            }
        }
    }


    vector<double> idf(V, 0.0);

    // OpenMP 並行化：計算 IDF (Inverse Document Frequency)
    // 這裡沒有資料依賴，是純粹的並行計算
    #pragma omp parallel for
    for (int t = 0; t < V; t++) {
        if (df[t] > 0) idf[t] = log((double)N / (double)df[t]);
        else idf[t] = 0.0;
    }

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time: " << idf_duration << " seconds\n";

    // --------------------------------------------------------------------------------
    // Phase 4: Compute TF for each document (sparse) (Parallelized)
    // --------------------------------------------------------------------------------
    start_time = timing_clock_t::now();

    vector<vector<pair<int, double>>> all_tf_results(N);

    // OpenMP 並行化：將 N 個文件的 TF 計算分配給不同的執行緒
    #pragma omp parallel 
    {
        vector<int> tmp_local; // 每個執行緒有自己的臨時向量

        #pragma omp for
        for (int d = 0; d < N; d++) {
            const auto& ids = doc_term_ids[d];
            if (ids.empty()) continue;

            tmp_local.assign(ids.begin(), ids.end()); 
            sort(tmp_local.begin(), tmp_local.end());

            const double total = (double)tmp_local.size();
            vector<pair<int, double>> tf_sparse;
            tf_sparse.reserve(tmp_local.size() / 4 + 4);

            // 統計詞頻 (Count Frequency)
            size_t i = 0;
            while (i < tmp_local.size()) {
                int term_id = tmp_local[i];
                size_t j = i + 1;
                while (j < tmp_local.size() && tmp_local[j] == term_id) j++;

                // 計算正規化 TF
                double tf = (double)(j - i) / total;
                tf_sparse.emplace_back(term_id, tf);
                i = j;
            }

            all_tf_results[d] = std::move(tf_sparse);
        }
    }

    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";

    // --------------------------------------------------------------------------------
    // Phase 5a: Compute TF-IDF (compute only, no file IO) (Parallelized)
    // --------------------------------------------------------------------------------
    start_time = timing_clock_t::now();

    // 使用一個 vector<vector<string>> 儲存每個文件的輸出字串，避免鎖競爭。
    vector<vector<string>> out_lines_per_doc(N); 

    // OpenMP 並行化：計算 TF-IDF
    #pragma omp parallel for
    for (int d = 0; d < N; d++) {
        vector<pair<string, double>> tfidf_pairs;
        tfidf_pairs.reserve(all_tf_results[d].size());

        for (const auto& kv : all_tf_results[d]) {
            int term_id = kv.first;
            double tf_val = kv.second;
            // TF-IDF = TF * IDF
            double tfidf_val = tf_val * idf[term_id];
            tfidf_pairs.emplace_back(id2word[term_id], tfidf_val);
        }

        // 每個文件內部進行按詞彙字串排序
        sort(tfidf_pairs.begin(), tfidf_pairs.end(),
             [](const auto& a, const auto& b) {
                 return a.first < b.first;
             });
        
        // 將結果寫入該文件專屬的 vector
        for (const auto& p : tfidf_pairs) {
            out_lines_per_doc[d].push_back(to_string(d) + "," + p.first + "," + to_string(p.second) + "\n");
        }
    }

    // 單執行緒合併所有結果字串 (速度很快)
    vector<string> out_lines_serial;
    size_t total_lines = 0;
    for(const auto& lines : out_lines_per_doc) {
        total_lines += lines.size();
    }
    out_lines_serial.reserve(total_lines);
    for(auto& lines : out_lines_per_doc) {
        out_lines_serial.insert(out_lines_serial.end(), 
                                std::make_move_iterator(lines.begin()), 
                                std::make_move_iterator(lines.end()));
    }


    end_time = timing_clock_t::now();
    double tfidf_compute_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TF-IDFs Time: " << tfidf_compute_duration << " seconds\n";

    // Total time includes load, excludes CSV write
    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    // Phase 5b: Write CSV (IO only, excluded from total) (Unchanged - Serial)
    start_time = timing_clock_t::now();

    ofstream fout("openmp_optimized.csv"); // 輸出檔名改為 openmp_optimized.csv
    fout << "document_id,word,tfidf_value\n";
    for (const auto& line : out_lines_serial) {
        fout << line;
    }
    fout.close();

    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "TF-IDF saved to openmp_optimized.csv\n";

    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load, excluding write): "
         << total_excl_write << " seconds\n";
    cout << "------------------------------------------\n";

    return 0;
}