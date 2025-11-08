#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <set>

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

// 計算 IDF
map<string, double> computeIDF(const vector<vector<string>>& docs) {
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

    for (const string& w : vocab) {
        idf[w] = log(N / idf[w]);
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
    vector<vector<string>> tokenized_docs;
    for (const auto& doc : documents) {
        tokenized_docs.push_back(tokenize(doc));
    }

    // 計算 IDF
    map<string, double> idf = computeIDF(tokenized_docs);

    // 計算每個文件的 TF-IDF
    for (int i = 0; i < tokenized_docs.size(); ++i) {
        map<string, double> tf = computeTF(tokenized_docs[i]);
        map<string, double> tfidf = computeTFIDF(tf, idf);

        cout << "Document " << i + 1 << " TF-IDF:\n";
        for (const auto& [word, score] : tfidf) {
            cout << "  " << word << ": " << score << "\n";
        }
        cout << endl;
    }

    return 0;
}
