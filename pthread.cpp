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
#include <pthread.h> 

using namespace std;
namespace fs = std::filesystem;

using timing_clock_t = chrono::high_resolution_clock;
using timing_point_t = timing_clock_t::time_point;

// 全域 Mutex 用於保護詞彙表 (vocab, id2word) 的同時存取
pthread_mutex_t vocab_mutex = PTHREAD_MUTEX_INITIALIZER; 

// ------------------------------------------------------------
// Load .txt documents recursively from dataset folder
// ------------------------------------------------------------
static void load_20newsgroups(const string& dataset_path,
                             vector<string>& documents) {
    documents.clear();

    vector<fs::path> files;
    if (!fs::exists(dataset_path)) {
        cerr << "Dataset path not found: " << dataset_path << endl;
        return;
    }

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
// Fast whitespace tokenizer -> term ids (OPTIMIZED for Lock Contention)
// ------------------------------------------------------------
static inline void tokenize_ws_to_ids(const string& text,
                                      vector<int>& out_ids,
                                      unordered_map<string, int>& vocab,
                                      vector<string>& id2word) {
    out_ids.clear();
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

        int id;
        
        // --- 1. Lock-Free Read (Optimization: Check if word exists without locking) ---
        auto it = vocab.find(word);
        
        if (it == vocab.end()) {
            // --- 2. Word not found, acquire lock and re-check/insert ---
            pthread_mutex_lock(&vocab_mutex);

            // Double Check: Re-check inside the lock in case another thread just added it
            it = vocab.find(word);
            if (it == vocab.end()) {
                // Not found again, safely insert
                id = (int)id2word.size();
                vocab.emplace(word, id);
                id2word.push_back(word);
            } else {
                // Found by another thread
                id = it->second;
            }

            pthread_mutex_unlock(&vocab_mutex);
        } else {
            // --- 3. Word found outside lock (Fast path for existing vocabulary) ---
            id = it->second;
        }

        out_ids.push_back(id);
        i = j;
    }
}

// ------------------------------------------------------------
// Pthreads Structure and Functions
// ------------------------------------------------------------

// 執行緒參數結構體 for Phase 2: Tokenization
struct TokenThreadData {
    int start_doc_index;
    int end_doc_index;
    const vector<string>* documents;
    vector<vector<int>>* doc_term_ids;
    unordered_map<string, int>* vocab;
    vector<string>* id2word;
};

void* tokenize_thread_func(void* arg) {
    TokenThreadData* data = (TokenThreadData*)arg;
    for (int d = data->start_doc_index; d < data->end_doc_index; d++) {
        tokenize_ws_to_ids((*data->documents)[d], (*data->doc_term_ids)[d], 
                          *data->vocab, *data->id2word);
    }
    pthread_exit(NULL);
}


// 執行緒參數結構體 for Phase 3: DF Calculation
struct DFThreadData {
    int start_doc_index;
    int end_doc_index;
    const vector<vector<int>>* doc_term_ids;
    vector<int> df_local; // Local DF vector for each thread to accumulate results
    int V; // Vocabulary size
};

void* df_calc_thread_func(void* arg) {
    DFThreadData* data = (DFThreadData*)arg;
    
    data->df_local.assign(data->V, 0); 
    vector<int> tmp;

    for (int d = data->start_doc_index; d < data->end_doc_index; d++) {
        const auto& ids = (*data->doc_term_ids)[d];
        tmp.assign(ids.begin(), ids.end());
        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());

        for (int term_id : tmp) {
            if (term_id < data->V) {
                data->df_local[term_id]++;
            }
        }
    }
    pthread_exit(NULL);
}


// 執行緒參數結構體 for Phase 4/5a: TF/TF-IDF Calculation
struct TFIDFThreadData {
    int start_doc_index;
    int end_doc_index;
    const vector<vector<int>>* doc_term_ids;
    const vector<double>* idf;
    const vector<string>* id2word;
    vector<vector<pair<int, double>>>* all_tf_results;
    vector<string>* out_lines_local; // 每個執行緒本地的 TF-IDF 輸出 CSV 行
};

void* tfidf_calc_thread_func(void* arg) {
    TFIDFThreadData* data = (TFIDFThreadData*)arg;
    
    vector<int> tmp;
    vector<pair<string, double>> tfidf_pairs;
    
    for (int d = data->start_doc_index; d < data->end_doc_index; d++) {
        const auto& ids = (*data->doc_term_ids)[d];
        if (ids.empty()) continue;
        
        // --- Phase 4: Compute TF (Sparse) ---
        tmp.assign(ids.begin(), ids.end());
        sort(tmp.begin(), tmp.end());

        const double total = (double)tmp.size();
        vector<pair<int, double>> tf_sparse;
        tf_sparse.reserve(tmp.size() / 4 + 4);

        size_t i = 0;
        while (i < tmp.size()) {
            int term_id = tmp[i];
            size_t j = i + 1;
            while (j < tmp.size() && tmp[j] == term_id) j++;

            double tf = (double)(j - i) / total;
            tf_sparse.emplace_back(term_id, tf);
            i = j;
        }

        (*data->all_tf_results)[d] = std::move(tf_sparse);

        // --- Phase 5a: Compute TF-IDF ---
        tfidf_pairs.clear();
        tfidf_pairs.reserve((*data->all_tf_results)[d].size());

        for (const auto& kv : (*data->all_tf_results)[d]) {
            int term_id = kv.first;
            double tf_val = kv.second;
            double tfidf_val = tf_val * (*data->idf)[term_id]; 
            tfidf_pairs.emplace_back((*data->id2word)[term_id], tfidf_val); 
        }

        sort(tfidf_pairs.begin(), tfidf_pairs.end(),
             [](const auto& a, const auto& b) {
                 return a.first < b.first;
             });

        for (const auto& p : tfidf_pairs) {
            data->out_lines_local->push_back(to_string(d) + "," + p.first + "," + to_string(p.second) + "\n");
        }
    }

    pthread_exit(NULL);
}


// ------------------------------------------------------------
// Main function
// ------------------------------------------------------------
int main(int argc, char** argv) {
    timing_point_t t_total0 = timing_clock_t::now();
    timing_point_t start_time, end_time;

    // --- 解析命令列參數 ---
    int NUM_THREADS = 4; // 預設執行緒數量
    string dataset_path = "dataset";
    
    if (argc >= 2) {
        dataset_path = argv[1]; // argv[1] 為資料集路徑
    }
    
    if (argc >= 3) {
        try {
            // argv[2] 為執行緒數量
            NUM_THREADS = stoi(argv[2]); 
            if (NUM_THREADS <= 0) {
                 cerr << "Warning: Thread count must be positive. Defaulting to 4 threads.\n";
                 NUM_THREADS = 4;
            }
        } catch (const invalid_argument& e) {
            cerr << "Warning: Invalid thread count specified. Defaulting to 4 threads.\n";
        } catch (const out_of_range& e) {
            cerr << "Warning: Thread count out of range. Defaulting to 4 threads.\n";
        }
    }
    // ----------------------


    // Phase 1: Load documents
    start_time = timing_clock_t::now();
    vector<string> documents;
    load_20newsgroups(dataset_path, documents);
    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "--- Parallel TF-IDF Timing Report (Pthreads) ---\n";
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";
    cout << "Using " << NUM_THREADS << " threads.\n";

    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"\n";
        return 1;
    }

    const int N = (int)documents.size();

    // 為了動態執行緒數，我們需要使用動態分配的陣列或 std::vector 來儲存執行緒 ID 和參數
    vector<pthread_t> threads;
    threads.resize(NUM_THREADS);
    
    // 計算任務分配參數
    int chunk_size = N / NUM_THREADS;
    int remaining = N % NUM_THREADS;

    // ------------------------------------------------------------
    // Phase 2: Tokenization + build vocab (Parallelized & Lock-Optimized)
    // ------------------------------------------------------------
    start_time = timing_clock_t::now();

    unordered_map<string, int> vocab;
    vocab.reserve(200000);
    vector<string> id2word;
    id2word.reserve(200000);
    vector<vector<int>> doc_term_ids(N);

    vector<TokenThreadData> token_thread_data(NUM_THREADS);
    int current_doc = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        token_thread_data[i].start_doc_index = current_doc;
        token_thread_data[i].end_doc_index = current_doc + chunk_size + (i < remaining ? 1 : 0);
        token_thread_data[i].documents = &documents;
        token_thread_data[i].doc_term_ids = &doc_term_ids;
        token_thread_data[i].vocab = &vocab;
        token_thread_data[i].id2word = &id2word;

        pthread_create(&threads[i], NULL, tokenize_thread_func, &token_thread_data[i]);
        current_doc = token_thread_data[i].end_doc_index;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Tokenization Time: " << tokenize_duration << " seconds (Parallel & Optimized)\n";

    const int V = (int)id2word.size();
    cout << "Vocabulary Size: " << V << "\n";
    
    if (V == 0) {
        cout << "Error: Vocabulary is empty.\n";
        return 1;
    }

    // ------------------------------------------------------------
    // Phase 3: Compute DF (Parallelized) and IDF (Serial)
    // ------------------------------------------------------------
    start_time = timing_clock_t::now();

    vector<DFThreadData> df_thread_data(NUM_THREADS);
    
    current_doc = 0;
    // 重用 Phase 2 的任務分配邏輯和 threads 向量
    for (int i = 0; i < NUM_THREADS; i++) {
        df_thread_data[i].start_doc_index = current_doc;
        df_thread_data[i].end_doc_index = current_doc + chunk_size + (i < remaining ? 1 : 0);
        df_thread_data[i].doc_term_ids = &doc_term_ids;
        df_thread_data[i].V = V;

        pthread_create(&threads[i], NULL, df_calc_thread_func, &df_thread_data[i]);
        current_doc = df_thread_data[i].end_doc_index;
    }
    
    // 等待並合併本地 DF 結果
    vector<int> df(V, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        for (int t = 0; t < V; t++) {
            df[t] += df_thread_data[i].df_local[t];
        }
    }
    
    // Compute IDF (Serial)
    vector<double> idf(V, 0.0);
    for (int t = 0; t < V; t++) {
        if (df[t] > 0) idf[t] = log((double)N / (double)df[t]);
        else idf[t] = 0.0;
    }

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute DF/IDF Time: " << idf_duration << " seconds (DF Parallel, IDF Serial)\n";

    // ------------------------------------------------------------
    // Phase 4/5a: Compute TF and TF-IDF (Parallelized)
    // ------------------------------------------------------------
    start_time = timing_clock_t::now();

    vector<vector<pair<int, double>>> all_tf_results(N);
    vector<vector<string>> all_out_lines_local(NUM_THREADS);

    vector<TFIDFThreadData> tfidf_thread_data(NUM_THREADS);
    
    current_doc = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        tfidf_thread_data[i].start_doc_index = current_doc;
        tfidf_thread_data[i].end_doc_index = current_doc + chunk_size + (i < remaining ? 1 : 0);
        tfidf_thread_data[i].doc_term_ids = &doc_term_ids;
        tfidf_thread_data[i].idf = &idf;
        tfidf_thread_data[i].id2word = &id2word;
        tfidf_thread_data[i].all_tf_results = &all_tf_results;
        tfidf_thread_data[i].out_lines_local = &all_out_lines_local[i];

        pthread_create(&threads[i], NULL, tfidf_calc_thread_func, &tfidf_thread_data[i]);
        current_doc = tfidf_thread_data[i].end_doc_index;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 合併所有執行緒的輸出行
    vector<string> out_lines;
    size_t total_size = 0;
    for (const auto& local_lines : all_out_lines_local) {
        total_size += local_lines.size();
    }
    out_lines.reserve(total_size);
    for (auto& local_lines : all_out_lines_local) {
        out_lines.insert(out_lines.end(), 
                         make_move_iterator(local_lines.begin()), 
                         make_move_iterator(local_lines.end()));
    }

    end_time = timing_clock_t::now();
    double tfidf_compute_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TF-IDFs Time: " << tfidf_compute_duration << " seconds (Parallel)\n";

    // Total time includes load, excludes CSV write
    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    // ------------------------------------------------------------
    // Phase 5b: Write CSV (Output to pthread.csv)
    // ------------------------------------------------------------
    start_time = timing_clock_t::now();

    ofstream fout("pthread.csv");
    fout << "document_id,word,tfidf_value\n";
    for (const auto& line : out_lines) {
        fout << line;
    }
    fout.close();

    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "TF-IDF saved to pthread.csv\n";

    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load, excluding write): "
         << total_excl_write << " seconds\n";
    cout << "------------------------------------------\n";
    
    pthread_mutex_destroy(&vocab_mutex);

    return 0;
}