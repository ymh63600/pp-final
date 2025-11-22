#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <numeric>   // for std::iota

using namespace std;
namespace fs = std::filesystem;

using Clock = chrono::high_resolution_clock;
using sv = std::string_view;

// -----------------------------------------
// string_view hash/eq for unordered containers
// -----------------------------------------
struct SvHashSimple {
    size_t operator()(sv x) const noexcept {
        return std::hash<sv>{}(x);
    }
};
struct SvEqSimple {
    bool operator()(sv a, sv b) const noexcept { return a == b; }
};

using TFMap  = std::unordered_map<sv, double, SvHashSimple, SvEqSimple>;
using DFMap  = std::unordered_map<sv, int,    SvHashSimple, SvEqSimple>;
using IDFMap = std::unordered_map<sv, double, SvHashSimple, SvEqSimple>;

// -----------------------------------------
// Fast whitespace tokenizer (string_view)
// Same semantics as stringstream >> w
// -----------------------------------------
static inline void tokenize_ws_sv(const string& text, vector<sv>& out_tokens) {
    out_tokens.clear();
    const char* s = text.c_str();
    size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        while (i < n && isspace((unsigned char)s[i])) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !isspace((unsigned char)s[j])) ++j;
        out_tokens.emplace_back(s + i, j - i);
        i = j;
    }
}

static inline vector<sv> tokenize_fast_sv(const string& text) {
    vector<sv> tokens;
    tokenize_ws_sv(text, tokens);
    return tokens;
}

// -----------------------------------------
// Compute TF for one document (unordered_map)
// -----------------------------------------
static inline TFMap computeTF(const vector<sv>& words) {
    TFMap tf;
    if (words.empty()) return tf;

    tf.reserve(words.size() / 2 + 8);

    for (const auto& w : words) {
        tf[w] += 1.0;
    }

    double total = (double)words.size();
    for (auto& kv : tf) {
        kv.second /= total;
    }
    return tf;
}

// -----------------------------------------
// Compute TF-IDF for one document
// -----------------------------------------
static inline TFMap computeTFIDF(const TFMap& tf, const IDFMap& idf) {
    TFMap out;
    out.reserve(tf.size());

    for (const auto& kv : tf) {
        sv w = kv.first;
        double tfv = kv.second;
        auto it = idf.find(w);
        double idfv = (it != idf.end()) ? it->second : 0.0;
        out[w] = tfv * idfv;
    }
    return out;
}

// -----------------------------------------
// Phase 1 helper: list all .txt files, sorted
// Only rank 0 calls this.
// -----------------------------------------
void list_txt_files_sorted(const string& root, vector<string>& out_paths) {
    out_paths.clear();
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            out_paths.push_back(entry.path().string());
        }
    }
    sort(out_paths.begin(), out_paths.end());
}

// -----------------------------------------
// Phase 1 helper: distribute doc_ids and paths
// Assignment rule: doc_id % world_size
// -----------------------------------------
void distribute_paths(const vector<string>& all_paths,
                      int world_rank,
                      int world_size,
                      vector<int>& local_doc_ids,
                      vector<string>& local_paths) {
    if (world_rank == 0) {
        int N = (int)all_paths.size();

        vector<vector<int>> ids_per_rank(world_size);
        vector<vector<string>> paths_per_rank(world_size);

        for (int i = 0; i < N; ++i) {
            int r = i % world_size;
            ids_per_rank[r].push_back(i);
            paths_per_rank[r].push_back(all_paths[i]);
        }

        for (int r = 1; r < world_size; ++r) {
            int k = (int)ids_per_rank[r].size();
            MPI_Send(&k, 1, MPI_INT, r, 10, MPI_COMM_WORLD);

            if (k > 0) {
                MPI_Send(ids_per_rank[r].data(), k, MPI_INT, r, 11, MPI_COMM_WORLD);

                ostringstream oss;
                for (const auto& p : paths_per_rank[r]) {
                    oss << p << "\n";
                }
                string packed = oss.str();
                int packed_len = (int)packed.size();

                MPI_Send(&packed_len, 1, MPI_INT, r, 12, MPI_COMM_WORLD);
                if (packed_len > 0) {
                    MPI_Send(packed.data(), packed_len, MPI_CHAR, r, 13, MPI_COMM_WORLD);
                }
            } else {
                int packed_len = 0;
                MPI_Send(&packed_len, 1, MPI_INT, r, 12, MPI_COMM_WORLD);
            }
        }

        local_doc_ids = std::move(ids_per_rank[0]);
        local_paths   = std::move(paths_per_rank[0]);

    } else {
        int k = 0;
        MPI_Recv(&k, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_doc_ids.clear();
        local_paths.clear();
        local_doc_ids.resize(k);
        local_paths.reserve(k);

        if (k > 0) {
            MPI_Recv(local_doc_ids.data(), k, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int packed_len = 0;
            MPI_Recv(&packed_len, 1, MPI_INT, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (packed_len > 0) {
                string packed;
                packed.resize(packed_len);
                MPI_Recv(packed.data(), packed_len, MPI_CHAR, 0, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                istringstream iss(packed);
                string line;
                while (getline(iss, line)) {
                    if (!line.empty()) local_paths.push_back(line);
                }
            }

            if ((int)local_paths.size() != k) {
                local_paths.resize(k);
            }
        }
    }
}

// -----------------------------------------
// Phase 1 helper: load local docs from paths
// -----------------------------------------
void load_documents_from_paths(const vector<string>& paths, vector<string>& documents) {
    documents.clear();
    documents.reserve(paths.size());
    for (const auto& p : paths) {
        ifstream fin(p);
        if (!fin.is_open()) {
            documents.push_back("");
            continue;
        }

        string line;
        string content;
        while (getline(fin, line)) {
            content += line;
            content.push_back(' ');
        }
        documents.push_back(std::move(content));
    }
}

// -----------------------------------------
// MPI-based IDF computation (optimized local DF)
// tokenized is size N, only local indices filled.
// vocab_storage: output vocab strings on every rank (backing store)
// out_idf uses string_view keys to vocab_storage.
// -----------------------------------------
void computeIDF_MPI(const vector<vector<sv>>& tokenized,
                    int world_rank,
                    int world_size,
                    vector<string>& vocab_storage,
                    IDFMap& out_idf) {
    int N = (int)tokenized.size();

    DFMap local_df;
    local_df.reserve(200000);

    // local DF
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) continue;
        const auto& words = tokenized[i];
        if (words.empty()) continue;

        unordered_set<sv, SvHashSimple, SvEqSimple> seen;
        seen.reserve(words.size());

        for (auto w : words) seen.insert(w);
        for (auto w : seen) local_df[w] += 1;
    }

    // pack local df to string
    ostringstream oss_local;
    for (const auto& kv : local_df) {
        oss_local << kv.first << " " << kv.second << "\n";
    }
    string local_str = oss_local.str();
    int local_len = (int)local_str.size();

    if (world_rank == 0) {
        // global df on root (owning strings)
        unordered_map<string, double> global_df;
        global_df.reserve(local_df.size() * world_size + 1024);

        for (const auto& kv : local_df) {
            global_df[string(kv.first)] += (double)kv.second;
        }

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

        // build vocab + idf arrays
        vocab_storage.clear();
        vocab_storage.reserve(global_df.size());

        vector<double> idf_vals;
        idf_vals.reserve(global_df.size());

        for (const auto& kv : global_df) {
            const string& w = kv.first;
            double df = kv.second;
            if (df > 0.0) {
                vocab_storage.push_back(w);
                idf_vals.push_back(log((double)N / df));
            }
        }

        // sort vocab to make broadcast deterministic
        vector<int> perm(vocab_storage.size());
        iota(perm.begin(), perm.end(), 0);
        sort(perm.begin(), perm.end(),
             [&](int a, int b){ return vocab_storage[a] < vocab_storage[b]; });

        vector<string> vocab_sorted;
        vector<double> idf_sorted;
        vocab_sorted.reserve(vocab_storage.size());
        idf_sorted.reserve(vocab_storage.size());

        for (int idx : perm) {
            vocab_sorted.push_back(vocab_storage[idx]);
            idf_sorted.push_back(idf_vals[idx]);
        }
        vocab_storage.swap(vocab_sorted);
        idf_vals.swap(idf_sorted);

        int V = (int)vocab_storage.size();
        MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

        ostringstream oss_vocab;
        for (const auto& w : vocab_storage) oss_vocab << w << "\n";
        string vocab_str = oss_vocab.str();
        int vocab_len = (int)vocab_str.size();

        MPI_Bcast(&vocab_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (vocab_len > 0) MPI_Bcast(vocab_str.data(), vocab_len, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (V > 0) MPI_Bcast(idf_vals.data(), V, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // build idf map with string_view referencing vocab_storage
        out_idf.clear();
        out_idf.reserve(V * 2 + 8);
        for (int i = 0; i < V; ++i) {
            out_idf[sv(vocab_storage[i])] = idf_vals[i];
        }

    } else {
        // send local df to root
        MPI_Send(&local_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (local_len > 0) MPI_Send(local_str.data(), local_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

        int V = 0;
        MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int vocab_len = 0;
        MPI_Bcast(&vocab_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vocab_storage.clear();
        vocab_storage.reserve(V);

        if (vocab_len > 0) {
            string vocab_str;
            vocab_str.resize(vocab_len);
            MPI_Bcast(vocab_str.data(), vocab_len, MPI_CHAR, 0, MPI_COMM_WORLD);

            istringstream iss_vocab(vocab_str);
            string w;
            while (getline(iss_vocab, w)) {
                if (!w.empty()) vocab_storage.push_back(w);
            }
        }

        vector<double> idf_vals(V);
        if (V > 0) MPI_Bcast(idf_vals.data(), V, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        out_idf.clear();
        out_idf.reserve(V * 2 + 8);

        int realV = (int)vocab_storage.size();
        int useV = (realV < V) ? realV : V;
        for (int i = 0; i < useV; ++i) {
            out_idf[sv(vocab_storage[i])] = idf_vals[i];
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

    // Phase 1: rank 0 builds file list, each rank loads only its docs
    string dataset_path = "dataset";
    vector<string> all_paths;
    int N = 0;

    if (world_rank == 0) {
        list_txt_files_sorted(dataset_path, all_paths);
        N = (int)all_paths.size();
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    vector<int> local_doc_ids;
    vector<string> local_paths;
    distribute_paths(all_paths, world_rank, world_size, local_doc_ids, local_paths);

    vector<string> local_docs;
    load_documents_from_paths(local_paths, local_docs);

    vector<string> documents;
    documents.resize(N);
    for (int j = 0; j < (int)local_docs.size(); ++j) {
        int doc_id = local_doc_ids[j];
        documents[doc_id] = std::move(local_docs[j]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = Clock::now();

    // Phase 2: tokenization (parallel over documents across ranks)
    vector<vector<sv>> tokenized(N);

    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) continue;
        tokenized[i] = tokenize_fast_sv(documents[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = Clock::now();

    // Phase 3: compute IDF using MPI
    vector<string> vocab_storage;
    IDFMap idf;
    computeIDF_MPI(tokenized, world_rank, world_size, vocab_storage, idf);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = Clock::now();

    // Phase 4: compute TF for documents assigned to this rank
    vector<TFMap> tf(N);
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) continue;
        tf[i] = computeTF(tokenized[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t4 = Clock::now();

    // Phase 5: compute TF-IDF for documents assigned to this rank
    vector<TFMap> tfidf(N);
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) continue;
        tfidf[i] = computeTFIDF(tf[i], idf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t5 = Clock::now();

    // Phase 6: send TF-IDF results to rank 0 and write mpi.csv
    ostringstream oss_local;
    for (int i = 0; i < N; ++i) {
        if (i % world_size != world_rank) continue;

        // sort by word to match old map order
        vector<pair<sv,double>> items;
        items.reserve(tfidf[i].size());
        for (const auto& kv : tfidf[i]) items.push_back(kv);
        sort(items.begin(), items.end(),
             [](const auto& a, const auto& b){ return a.first < b.first; });

        for (const auto& kv : items) {
            oss_local << i << "," << kv.first << "," << kv.second << "\n";
        }
    }

    string local_str = oss_local.str();
    int local_len = (int)local_str.size();

    if (world_rank == 0) {
        ofstream fout("mpi.csv");
        fout << "document_id,word,tfidf_value\n";

        fout << local_str;

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
        MPI_Send(&local_len, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        if (local_len > 0) MPI_Send(local_str.data(), local_len, MPI_CHAR, 0, 3, MPI_COMM_WORLD);
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
        double t_total   = chrono::duration<double>(t5 - t0).count();

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
