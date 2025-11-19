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

using namespace std;
namespace fs = std::filesystem;

// Type aliases for cleaner time measurement - RENAMED to avoid conflict
using timing_clock_t = chrono::high_resolution_clock; // ★ 改名
using timing_point_t = timing_clock_t::time_point;    // ★ 改名

// Tokenize text into words
vector<string> tokenize(const string& text) {
    // ... (computeTF function body remains unchanged) ...
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// Compute TF for a single document
map<string, double> computeTF(const vector<string>& words) {
    // ... (computeTF function body remains unchanged) ...
    map<string, double> tf;
    if (words.empty()) return tf;

    for (const string& w : words) {
        tf[w] += 1.0;
    }
    double total_words = words.size();
    for (auto& [word, count] : tf) {
        count /= total_words; // TF = Word Count / Total Words
    }
    return tf;
}

// Compute IDF across all documents
map<string, double> computeIDF(const vector<vector<string>>& docs) {
    // ... (computeIDF function body remains unchanged) ...
    map<string, double> idf;
    set<string> vocab;
    int N = docs.size();

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

    // IDF = log(N / df)
    for (const string& w : vocab) {
        idf[w] = log(N / idf.at(w)); 
    }

    return idf;
}

// Compute TF-IDF for a single document
map<string, double> computeTFIDF(const map<string, double>& tf, const map<string, double>& idf) {
    // ... (computeTFIDF function body remains unchanged) ...
    map<string, double> tfidf;
    for (const auto& [word, tf_val] : tf) {
        double idf_val = idf.count(word) ? idf.at(word) : 0.0;
        tfidf[word] = tf_val * idf_val;
    }
    return tfidf;
}

// Load 20-Newsgroups files
void load_20newsgroups(const string& root, vector<string>& documents) {
    // ... (load_20newsgroups function body remains unchanged) ...
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path());
            if (!fin.is_open()) continue;

            string line, content;
            while (getline(fin, line)) {
                content += line + " ";
            }
            documents.push_back(content);
        }
    }
}

// Main execution logic with timing
int main() {
    // 使用新的類型別名
    timing_point_t start_time, end_time; 

    // --- Phase 1: Load Documents ---
    start_time = timing_clock_t::now(); // ★ 使用 timing_clock_t

    vector<string> documents;
    string dataset_path = "dataset"; // Adjust this path if necessary
    load_20newsgroups(dataset_path, documents);

    end_time = timing_clock_t::now(); // ★ 使用 timing_clock_t
    double load_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "--- Serial TF-IDF Timing Report ---\n";
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

    // --- Phase 3: Compute IDF ---
    start_time = timing_clock_t::now();
    
    map<string, double> idf = computeIDF(tokenized_docs);

    end_time = timing_clock_t::now();
    double idf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute IDF Time: " << idf_duration << " seconds (Vocabulary Size: " << idf.size() << ")\n";

    // --- Phase 4: Compute All TFs ---
    start_time = timing_clock_t::now();

    // Store TF results for all documents
    vector<map<string, double>> all_tf_results;
    for (const auto& doc_tokens : tokenized_docs) {
        all_tf_results.push_back(computeTF(doc_tokens));
    }

    end_time = timing_clock_t::now();
    double tf_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "Compute All TFs Time: " << tf_duration << " seconds\n";


    // --- Phase 5: Compute All TF-IDFs ---
    start_time = timing_clock_t::now();
    
    // Store TF-IDF results (just for completion, we don't print them)
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
