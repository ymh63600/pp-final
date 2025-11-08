#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <set>
#include <omp.h> // OpenMP

using namespace std;

// 將句子切成單詞
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// 計算 TF
map<string, double> computeTF(const vector<string>& words) {
    map<string, double> tf;
    for (const string& w : words) {
        tf[w] += 1.0;
    }
    for (auto& [word, count] : tf) {
        count /= words.size(); // TF = 詞頻 / 詞總數
    }
    return tf;
}

// 計算 IDF (多線程版本)
map<string, double> computeIDF(const vector<vector<string>>& docs) {
    map<string, double> idf;
    set<string> vocab;
    int N = docs.size();

    // 第一步：統計每個詞在哪些文件中出現
    vector<map<string, int>> local_counts(omp_get_max_threads());

    #pragma omp parallel for
    for (int i = 0; i < docs.size(); ++i) {
        int tid = omp_get_thread_num();
        set<string> seen(docs[i].begin(), docs[i].end());
        for (const auto& w : seen) {
            local_counts[tid][w] += 1;
        }
    }

    // 合併到全局 idf map
    for (auto& lc : local_counts) {
        for (auto& [w, cnt] : lc) {
            idf[w] += cnt;
        }
    }

    for (auto& [w, cnt] : idf) {
        idf[w] = log((double)N / cnt); // IDF = log(N / df)
    }

    return idf;
}

// 計算 TF-IDF
map<string, double> computeTFIDF(const map<string, double>& tf, const map<string, double>& idf) {
    map<string, double> tfidf;
    for (const auto& [word, tf_val] : tf) {
        tfidf[word] = tf_val * idf.at(word);
    }
    return tfidf;
}

int main() {
    vector<string> documents = {
        "this is a sample",
        "this is another example example",
        "one more sample document"
    };

    // 分詞
    vector<vector<string>> tokenized_docs(documents.size());
    #pragma omp parallel for
    for (int i = 0; i < documents.size(); ++i) {
        tokenized_docs[i] = tokenize(documents[i]);
    }

    // 計算 IDF
    map<string, double> idf = computeIDF(tokenized_docs);

    // 計算每個文件的 TF-IDF (並行)
    #pragma omp parallel for
    for (int i = 0; i < tokenized_docs.size(); ++i) {
        map<string, double> tf = computeTF(tokenized_docs[i]);
        map<string, double> tfidf = computeTFIDF(tf, idf);

        #pragma omp critical
        {
            cout << "Document " << i + 1 << " TF-IDF:\n";
            for (const auto& [word, score] : tfidf) {
                cout << "  " << word << ": " << score << "\n";
            }
            cout << endl;
        }
    }

    return 0;
}
