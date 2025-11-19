#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <set>
#include <fstream>
#include <filesystem>
#include <pthread.h>
#include <chrono>

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

// ======================================================
// ================ pthread parallel part ===============
// ======================================================

struct ThreadTask {
    int tid;
    int start_idx;
    int end_idx;

    const vector<string>* docs;              // for tokenize
    const vector<vector<string>>* toks_in;   // for TF
    const map<string, double>* idf;          // for TF-IDF

    vector<vector<string>>* toks_out;
    vector<map<string, double>>* tf_out;
    vector<map<string, double>>* tfidf_out;
};

// -----------------------------------------
// Worker 1: Tokenize
// -----------------------------------------
void* worker_tokenize(void* arg) {
    ThreadTask* t = (ThreadTask*)arg;

    for (int i = t->start_idx; i < t->end_idx; i++) {
        t->toks_out->at(i) = tokenize(t->docs->at(i));
    }
    return nullptr;
}

// -----------------------------------------
// Worker 2: TF
// -----------------------------------------
void* worker_tf(void* arg) {
    ThreadTask* t = (ThreadTask*)arg;

    for (int i = t->start_idx; i < t->end_idx; i++) {
        t->tf_out->at(i) = computeTF(t->toks_in->at(i));
    }
    return nullptr;
}

// -----------------------------------------
// Worker 3: TF-IDF
// -----------------------------------------
void* worker_tfidf(void* arg) {
    ThreadTask* t = (ThreadTask*)arg;

    for (int i = t->start_idx; i < t->end_idx; i++) {
        t->tfidf_out->at(i) = computeTFIDF(t->tf_out->at(i), *(t->idf));
    }
    return nullptr;
}

// -----------------------------------------
// Start threads (generic)
// -----------------------------------------
template<typename Func>
void run_threads(Func func, int n_threads, int total,
                 vector<pthread_t>& threads, vector<ThreadTask>& tasks)
{
    threads.resize(n_threads);
    tasks.resize(n_threads);

    int chunk = (total + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; t++) {
        tasks[t].tid = t;
        tasks[t].start_idx = t * chunk;
        tasks[t].end_idx = min((t + 1) * chunk, total);

        pthread_create(&threads[t], nullptr, func, &tasks[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], nullptr);
}

// -----------------------------------------
// Parallel IDF with partial DF
// -----------------------------------------
map<string, double> computeIDF_parallel(
    const vector<vector<string>>& docs,
    int num_threads)
{
    int N = docs.size();
    vector<map<string, double>> partial_df(num_threads);
    vector<pthread_t> threads(num_threads);

    struct IDFTask {
        int tid;
        int s, e;
        const vector<vector<string>>* docs;
        vector<map<string, double>>* pdf;
    };

    auto df_worker = [](void* arg)->void* {
        IDFTask* t = (IDFTask*)arg;
        for (int i = t->s; i < t->e; i++) {
            set<string> seen;
            for (auto& w : t->docs->at(i)) seen.insert(w);
            for (auto& w : seen) (*t->pdf)[t->tid][w] += 1.0;
        }
        return nullptr;
    };

    int chunk = (N + num_threads - 1) / num_threads;
    vector<IDFTask> task(num_threads);

    for (int i = 0; i < num_threads; i++) {
        int s = i * chunk;
        int e = min((i+1) * chunk, N);
        task[i] = { i, s, e, &docs, &partial_df };
        pthread_create(&threads[i], nullptr, df_worker, &task[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], nullptr);

    // reduce all DF maps
    map<string, double> df;
    for (int t = 0; t < num_threads; t++)
        for (auto& [w, c] : partial_df[t]) df[w] += c;

    // compute IDF
    map<string, double> idf;
    for (auto& [w, dfc] : df)
        idf[w] = log(double(N) / dfc);

    return idf;
}

// ======================================================
// ========================== MAIN =======================
// ======================================================

int main() {
    using Clock = chrono::high_resolution_clock;

    int num_threads = 8;

    cout << "--- Parallel TF-IDF Timing Report ---\n";

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

    vector<pthread_t> threads;
    vector<ThreadTask> tasks;

    // ====================================================
    // 1. Tokenization
    // ====================================================
    tasks.clear();
    tasks.resize(num_threads);
    for (auto& t : tasks) {
        t.docs = &documents;
        t.toks_out = &tokenized;
    }

    auto t2 = Clock::now();
    run_threads(worker_tokenize, num_threads, N, threads, tasks);
    auto t3 = Clock::now();

    cout << "Tokenization Time: "
         << chrono::duration<double>(t3 - t2).count() << " seconds\n";

    // ====================================================
    // 2. Compute IDF
    // ====================================================
    auto t4 = Clock::now();

    map<string,double> idf = computeIDF_parallel(tokenized, num_threads);

    auto t5 = Clock::now();

    cout << "Compute IDF Time: "
         << chrono::duration<double>(t5 - t4).count()
         << " seconds (Vocabulary Size: " << idf.size() << ")\n";

    // ====================================================
    // 3. Compute All TFs
    // ====================================================
    tasks.clear();
    tasks.resize(num_threads);
    for (auto& t : tasks) {
        t.toks_in = &tokenized;
        t.tf_out = &tf;
    }

    auto t6 = Clock::now();
    run_threads(worker_tf, num_threads, N, threads, tasks);
    auto t7 = Clock::now();

    cout << "Compute All TFs Time: "
         << chrono::duration<double>(t7 - t6).count() << " seconds\n";

    // ====================================================
    // 4. Compute All TF-IDFs
    // ====================================================
    tasks.clear();
    tasks.resize(num_threads);
    for (auto& t : tasks) {
        t.tf_out = &tf;
        t.idf = &idf;
        t.tfidf_out = &tfidf;
    }

    auto t8 = Clock::now();
    run_threads(worker_tfidf, num_threads, N, threads, tasks);
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
