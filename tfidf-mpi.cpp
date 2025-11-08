#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <cmath>

using namespace std;

vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while (ss >> word) tokens.push_back(word);
    return tokens;
}

map<string, double> computeTF(const vector<string>& words) {
    map<string,double> tf;
    for(auto& w: words) tf[w]+=1.0;
    for(auto& [k,v]: tf) v/=words.size();
    return tf;
}

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    vector<string> docs;
    if(rank==0){
        docs = {"this is a sample",
                "this is another example example",
                "one more sample document"};
    }

    // 簡單：每個進程分配一個文檔
    string local_doc;
    if(rank < docs.size()) local_doc = docs[rank];
    else local_doc = "";

    vector<string> words = tokenize(local_doc);
    map<string,double> tf = computeTF(words);

    // 計算 DF
    map<string,int> local_df;
    set<string> seen(words.begin(), words.end());
    for(auto& w: seen) local_df[w]=1;

    // 收集所有詞的 df
    map<string,int> global_df;
    for(auto& [w,c]: local_df){
        int global_count;
        MPI_Allreduce(&c, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        global_df[w] = global_count;
    }

    // 計算 IDF
    map<string,double> idf;
    int N = docs.size();
    for(auto& [w,c]: global_df) idf[w] = log((double)N/c);

    // 計算 TF-IDF
    map<string,double> tfidf;
    for(auto& [w,tf_val]: tf) tfidf[w] = tf_val * idf[w];

    // 打印
    for(auto& [w,score]: tfidf) cout << "Rank " << rank << ": " << w << " = " << score << endl;

    MPI_Finalize();
    return 0;
}
