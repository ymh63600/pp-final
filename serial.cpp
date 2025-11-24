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

using namespace std;
namespace fs = std::filesystem;

using timing_clock_t = chrono::high_resolution_clock;
using timing_point_t = timing_clock_t::time_point;

// ------------------------------------------------------------
// Load .txt documents recursively from dataset folder
// Only *.txt
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
// Fast whitespace tokenizer -> term ids
// Semantics: whitespace split only
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

    // Phase 1: Load documents (IO, included in total)
    start_time = timing_clock_t::now();

    vector<string> documents;
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];
    load_20newsgroups(dataset_path, documents);

    end_time = timing_clock_t::now();
    double load_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "--- Serial TF-IDF Timing Report ---\n";
    cout << "Total Documents Loaded: " << documents.size() << "\n";
    cout << "Document Loading Time: " << load_duration << " seconds\n";

    if (documents.empty()) {
        cout << "Error: No documents found. Please check path \"" << dataset_path << "\"\n";
        return 1;
    }

    const int N = (int)documents.size();

    // Phase 2: Tokenization + build vocab + store term ids per doc
    start_time = timing_clock_t::now();

    unordered_map<string, int> vocab;
    vocab.reserve(200000);
    vector<string> id2word;
    id2word.reserve(200000);

    vector<vector<int>> doc_term_ids(N);

    for (int d = 0; d < N; d++) {
        tokenize_ws_to_ids(documents[d], doc_term_ids[d], vocab, id2word);
    }

    end_time = timing_clock_t::now();
    double tokenize_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Tokenization Time: " << tokenize_duration << " seconds\n";

    const int V = (int)id2word.size();
    cout << "Vocabulary Size: " << V << "\n";

    // Phase 3: Compute DF and IDF
    start_time = timing_clock_t::now();

    vector<int> df(V, 0);
    vector<int> tmp;

    for (int d = 0; d < N; d++) {
        const auto& ids = doc_term_ids[d];
        tmp.assign(ids.begin(), ids.end());
        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());

        for (int term_id : tmp) {
            df[term_id]++;
        }
    }

    vector<double> idf(V, 0.0);
    for (int t = 0; t < V; t++) {
        if (df[t] > 0) idf[t] = log((double)N / (double)df[t]);
        else idf[t] = 0.0;
    }

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time: " << idf_duration << " seconds\n";

    // Phase 4: Compute TF for each document (sparse)
    start_time = timing_clock_t::now();

    vector<vector<pair<int, double>>> all_tf_results(N);

    for (int d = 0; d < N; d++) {
        const auto& ids = doc_term_ids[d];
        if (ids.empty()) continue;

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

        all_tf_results[d] = std::move(tf_sparse);
    }

    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";

    // Phase 5a: Compute TF-IDF (compute only, no file IO)
    start_time = timing_clock_t::now();

    vector<string> out_lines;
    out_lines.reserve(1000000);

    vector<pair<string, double>> tfidf_pairs;
    for (int d = 0; d < N; d++) {
        tfidf_pairs.clear();
        tfidf_pairs.reserve(all_tf_results[d].size());

        for (const auto& kv : all_tf_results[d]) {
            int term_id = kv.first;
            double tf_val = kv.second;
            double tfidf_val = tf_val * idf[term_id];
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

    end_time = timing_clock_t::now();
    double tfidf_compute_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TF-IDFs Time: " << tfidf_compute_duration << " seconds\n";

    // Total time includes load, excludes CSV write
    timing_point_t t_total_end_excl_write = timing_clock_t::now();
    double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

    // Phase 5b: Write CSV (IO only, excluded from total)
    start_time = timing_clock_t::now();

    ofstream fout("serial.csv");
    fout << "document_id,word,tfidf_value\n";
    for (const auto& line : out_lines) {
        fout << line;
    }
    fout.close();

    end_time = timing_clock_t::now();
    double write_duration = chrono::duration<double>(end_time - start_time).count();

    cout << "CSV Write Time: " << write_duration << " seconds\n";
    cout << "TF-IDF saved to serial.csv\n";

    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load, excluding write): "
         << total_excl_write << " seconds\n";
    cout << "------------------------------------------\n";

    return 0;
}
