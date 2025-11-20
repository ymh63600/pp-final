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
#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

// -----------------------------------------
// Tokenize text
// -----------------------------------------
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string w;
    while (ss >> w) tokens.push_back(w);
    return tokens;
}

// -----------------------------------------
// Compute TF for a single doc
// -----------------------------------------
map<string, double> computeTF(const vector<string>& words) {
    map<string, double> tf;
    if (words.empty()) return tf;

    for (auto& w : words) tf[w] += 1.0;
    double total = words.size();
    for (auto& [w, c] : tf) c /= total;
    return tf;
}

// -----------------------------------------
map<string, double> computeTFIDF(const map<string, double>& tf,
                                 const map<string, double>& idf)
{
    map<string, double> out;
    for (auto& [w, tfv] : tf) {
        double idfv = idf.count(w) ? idf.at(w) : 0.0;
        out[w] = tfv * idfv;
    }
    return out;
}

// -----------------------------------------
// Load dataset
// -----------------------------------------
void load_20newsgroups(const string& root, vector<string>& documents) {
    for (auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path());
            if (!fin.is_open()) continue;

            string content, line;
            while (getline(fin, line)) content += line + " ";
            documents.push_back(content);
        }
    }
}

// -----------------------------------------
// Compute IDF using OpenMP
// -----------------------------------------
map<string, double> computeIDF_parallel(const vector<vector<string>>& docs) {
    int N = docs.size();
    map<string, double> df;

    #pragma omp parallel
    {
        map<string, double> local_df;

        #pragma omp for
        for (int i = 0; i < N; i++) {
            set<string> seen(docs[i].begin(), docs[i].end());
            for (auto& w : seen) local_df[w] += 1.0;
        }

        #pragma omp critical
        {
            for (auto& [w, c] : local_df) df[w] += c;
        }
    }

    map<string, double> idf;
    for (auto& [w, dfc] : df) idf[w] = log(double(N) / dfc);
    return idf;
}

// ======================================================
// ========================== MAIN =======================
// ======================================================
int main(int argc, char* argv[]) {
    using Clock = chrono::high_resolution_clock;

    int num_threads = 8;
    if (argc > 1)               
        num_threads = atoi(argv[1]); 

    omp_set_num_threads(num_threads);


    cout << "--- OpenMP Parallel TF-IDF Timing Report ---\n";

    // -----------------------------------------
    // Load dataset
    // -----------------------------------------
    auto t0 = Clock::now();

    vector<string> documents;
    load_20newsgroups("dataset", documents);

    auto t1 = Clock::now();

    int N = documents.size();
    cout << "Total Documents Loaded: " << N << "\n";
    cout << "Document Loading Time: "
         << chrono::duration<double>(t1 - t0).count() << " seconds\n";

    if (N == 0) return 0;

    vector<vector<string>> tokenized(N);
    vector<map<string,double>> tf(N);
    vector<map<string,double>> tfidf(N);

    // ====================================================
    // 1. Tokenization
    // ====================================================
    auto t2 = Clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        tokenized[i] = tokenize(documents[i]);
    }

    auto t3 = Clock::now();
    cout << "Tokenization Time: "
         << chrono::duration<double>(t3 - t2).count() << " seconds\n";

    // ====================================================
    // 2. Compute IDF
    // ====================================================
    auto t4 = Clock::now();
    map<string,double> idf = computeIDF_parallel(tokenized);
    auto t5 = Clock::now();

    cout << "Compute IDF Time: "
         << chrono::duration<double>(t5 - t4).count()
         << " seconds (Vocabulary Size: " << idf.size() << ")\n";

    // ====================================================
    // 3. Compute All TFs
    // ====================================================
    auto t6 = Clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        tf[i] = computeTF(tokenized[i]);
    }

    auto t7 = Clock::now();
    cout << "Compute All TFs Time: "
         << chrono::duration<double>(t7 - t6).count() << " seconds\n";

    // ====================================================
    // 4. Compute All TF-IDFs
    // ====================================================
    auto t8 = Clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        tfidf[i] = computeTFIDF(tf[i], idf);
    }

    auto t9 = Clock::now();
    cout << "Compute All TF-IDFs Time: "
         << chrono::duration<double>(t9 - t8).count() << " seconds\n";

    // ====================================================
    // Summary
    // ====================================================
    cout << "------------------------------------------\n";
    cout << "Total Execution Time (including load): "
         << chrono::duration<double>(t9 - t0).count()
         << " seconds\n";
    cout << "------------------------------------------\n";

    return 0;
}
