#include <mpi.h>
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

using Clock = chrono::high_resolution_clock;

// -----------------------------------------
// Tokenize: split by whitespace
// -----------------------------------------
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string w;
    while (ss >> w) {
        tokens.push_back(w);
    }
    return tokens;
}

// -----------------------------------------
// Compute TF for one document
// -----------------------------------------
map<string, double> computeTF(const vector<string>& words) {
    map<string, double> tf;
    if (words.empty()) {
        return tf;
    }

    for (const auto& w : words) {
        tf[w] += 1.0;
    }

    double total = static_cast<double>(words.size());
    for (auto& kv : tf) {
        kv.second /= total;
    }
    return tf;
}

// -----------------------------------------
// Compute TF-IDF for one document
// -----------------------------------------
map<string, double> computeTFIDF(const map<string, double>& tf,
                                 const map<string, double>& idf) {
    map<string, double> out;
    for (const auto& kv : tf) {
        const string& w = kv.first;
        double tfv = kv.second;
        auto it = idf.find(w);
        double idfv = (it != idf.end()) ? it->second : 0.0;
        out[w] = tfv * idfv;
    }
    return out;
}

// -----------------------------------------
// Load 20-newsgroups-like dataset
// -----------------------------------------
void load_20newsgroups(const string& root, vector<string>& documents) {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            ifstream fin(entry.path());
            if (!fin.is_open()) {
                continue;
            }
            string line;
            string content;
            while (getline(fin, line)) {
                content += line + " ";
            }
            documents.push_back(content);
        }
    }
}

// -----------------------------------------
// MPI-based IDF computation
// Each rank:
//   1) only looks at docs i where i % world_size == world_rank
//   2) builds local df map<string,double>
//   3) non-root ranks send local df to rank 0
//   4) rank 0 merges, computes global IDF, then broadcasts IDF to all ranks
// -----------------------------------------
void computeIDF_MPI(const vector<vector<string>>& tokenized,
                    int world_rank,
                    int world_size,
                    map<string, double>& out_idf) {
    int N = static_cast<int>(tokenized.size());

    // Step 1: local DF for docs assigned to this rank
    map<string, double> local_df;

    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) {
            continue;
        }

        const auto& words = tokenized[i];
        if (words.empty()) {
            continue;
        }

        set<string> seen(words.begin(), words.end());
        for (const auto& w : seen) {
            local_df[w] += 1.0;
        }
    }

    // Step 2: non-root ranks serialize local_df and send to rank 0
    ostringstream oss_local;
    for (const auto& kv : local_df) {
        oss_local << kv.first << " " << kv.second << "\n";
    }
    string local_str = oss_local.str();
    int local_len = static_cast<int>(local_str.size());

    if (world_rank == 0) {
        // Rank 0: start with its own local_df
        map<string, double> global_df = local_df;

        // Receive from other ranks
        for (int src = 1; src < world_size; ++src) {
            int recv_len = 0;
            MPI_Recv(&recv_len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_len > 0) {
                string buf;
                buf.resize(recv_len);
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                istringstream iss(buf);
                string w;
                double c;
                while (iss >> w >> c) {
                    global_df[w] += c;
                }
            }
        }

        // Compute global IDF
        out_idf.clear();
        for (const auto& kv : global_df) {
            const string& w = kv.first;
            double df = kv.second;
            if (df > 0.0) {
                out_idf[w] = log(static_cast<double>(N) / df);
            }
        }

        // Serialize IDF and broadcast to all ranks
        ostringstream oss_idf;
        for (const auto& kv : out_idf) {
            oss_idf << kv.first << " " << kv.second << "\n";
        }
        string idf_str = oss_idf.str();
        int idf_len = static_cast<int>(idf_str.size());

        MPI_Bcast(&idf_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (idf_len > 0) {
            MPI_Bcast(idf_str.data(), idf_len, MPI_CHAR, 0, MPI_COMM_WORLD);
        }
    } else {
        // Non-root: send local_df to rank 0
        MPI_Send(&local_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (local_len > 0) {
            MPI_Send(local_str.data(), local_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }

        // Receive global IDF from rank 0
        int idf_len = 0;
        MPI_Bcast(&idf_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        out_idf.clear();
        if (idf_len > 0) {
            string idf_str;
            idf_str.resize(idf_len);
            MPI_Bcast(idf_str.data(), idf_len, MPI_CHAR, 0, MPI_COMM_WORLD);

            istringstream iss(idf_str);
            string w;
            double val;
            while (iss >> w >> val) {
                out_idf[w] = val;
            }
        }
    }
}

// -----------------------------------------
// Main: MPI TF-IDF
// -----------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto t0 = Clock::now();

    // Phase 1: all ranks load the same dataset
    vector<string> documents;
    string dataset_path = "dataset";
    load_20newsgroups(dataset_path, documents);
    int N = static_cast<int>(documents.size());

    if (world_rank == 0) {
        cout << "--- MPI TF-IDF ---" << endl;
        cout << "World size (number of MPI processes): " << world_size << endl;
        cout << "Total documents loaded: " << N << endl;
    }

    if (N == 0) {
        if (world_rank == 0) {
            cerr << "No documents found in path: " << dataset_path << endl;
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = Clock::now();

    // Phase 2: tokenization (parallel over documents across ranks)
    vector<vector<string>> tokenized(N);

    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) {
            continue;
        }
        tokenized[i] = tokenize(documents[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = Clock::now();

    // Phase 3: compute IDF using MPI (parallel df + reduction + broadcast)
    map<string, double> idf;
    computeIDF_MPI(tokenized, world_rank, world_size, idf);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = Clock::now();

    // Phase 4: compute TF for documents assigned to this rank
    vector<map<string, double>> tf(N);
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) {
            continue;
        }
        tf[i] = computeTF(tokenized[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t4 = Clock::now();

    // Phase 5: compute TF-IDF for documents assigned to this rank
    vector<map<string, double>> tfidf(N);
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) {
            continue;
        }
        tfidf[i] = computeTFIDF(tf[i], idf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t5 = Clock::now();

    // Phase 6: send TF-IDF results to rank 0 and write mpi.csv
    // Each rank serializes its own assigned documents into lines.
    ostringstream oss_local;
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) {
            continue;
        }
        for (const auto& kv : tfidf[i]) {
            const string& w = kv.first;
            double val = kv.second;
            oss_local << i << "," << w << "," << val << "\n";
        }
    }

    string local_str = oss_local.str();
    int local_len = static_cast<int>(local_str.size());

    if (world_rank == 0) {
        // Rank 0: prepare to receive from others and write file
        ofstream fout("mpi.csv");
        fout << "document_id,word,tfidf_value\n";

        // First write its own data
        fout << local_str;

        // Receive from other ranks
        for (int src = 1; src < world_size; ++src) {
            int recv_len = 0;
            MPI_Recv(&recv_len, 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (recv_len > 0) {
                string buf;
                buf.resize(recv_len);
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                fout << buf;
            }
        }

        fout.close();
        cout << "TF-IDF saved to mpi.csv" << endl;
    } else {
        // Non-root: send local TF-IDF CSV fragment to rank 0
        MPI_Send(&local_len, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        if (local_len > 0) {
            MPI_Send(local_str.data(), local_len, MPI_CHAR, 0, 3, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t6 = Clock::now();

    if (world_rank == 0) {
        double t_load    = chrono::duration<double>(t1 - t0).count();
        double t_token   = chrono::duration<double>(t2 - t1).count();
        double t_idf     = chrono::duration<double>(t3 - t2).count();
        double t_tf      = chrono::duration<double>(t4 - t3).count();
        double t_tfidf   = chrono::duration<double>(t5 - t4).count();
        double t_output  = chrono::duration<double>(t6 - t5).count();
        double t_total   = chrono::duration<double>(t6 - t0).count();

        cout << "Timing (rank 0 approximate):" << endl;
        cout << "  Load documents:  " << t_load   << " s" << endl;
        cout << "  Tokenization:    " << t_token  << " s" << endl;
        cout << "  Compute IDF(MPI):" << t_idf    << " s" << endl;
        cout << "  Compute TF:      " << t_tf     << " s" << endl;
        cout << "  Compute TF-IDF:  " << t_tfidf  << " s" << endl;
        cout << "  Output CSV:      " << t_output << " s" << endl;
        cout << "  Total:           " << t_total  << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}
