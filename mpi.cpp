#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cctype>

using namespace std;
namespace fs = std::filesystem;

using Clock = chrono::high_resolution_clock;

// -----------------------------------------
// Fast whitespace tokenizer -> local word list
// Same semantics as stringstream >> w (whitespace split only)
// -----------------------------------------
static inline void tokenize_ws(const string& text, vector<string>& out_tokens) {
    out_tokens.clear();
    const char* s = text.data();
    size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        while (i < n && isspace((unsigned char)s[i])) {
            ++i;
        }
        if (i >= n) break;

        size_t j = i;
        while (j < n && !isspace((unsigned char)s[j])) {
            ++j;
        }

        out_tokens.emplace_back(text.substr(i, j - i));
        i = j;
    }
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
        int N = static_cast<int>(all_paths.size());

        vector<vector<int>> ids_per_rank(world_size);
        vector<vector<string>> paths_per_rank(world_size);

        for (int i = 0; i < N; ++i) {
            int r = i % world_size;
            ids_per_rank[r].push_back(i);
            paths_per_rank[r].push_back(all_paths[i]);
        }

        for (int r = 1; r < world_size; ++r) {
            int k = static_cast<int>(ids_per_rank[r].size());
            MPI_Send(&k, 1, MPI_INT, r, 10, MPI_COMM_WORLD);

            if (k > 0) {
                MPI_Send(ids_per_rank[r].data(), k, MPI_INT, r, 11, MPI_COMM_WORLD);

                ostringstream oss;
                for (const auto& p : paths_per_rank[r]) {
                    oss << p << "\n";
                }
                string packed = oss.str();
                int packed_len = static_cast<int>(packed.size());

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
        ifstream fin(p, ios::in | ios::binary);
        if (!fin.is_open()) {
            documents.push_back("");
            continue;
        }

        string content;
        fin.seekg(0, ios::end);
        size_t sz = (size_t)fin.tellg();
        fin.seekg(0, ios::beg);
        content.resize(sz);
        fin.read(&content[0], sz);
        documents.push_back(std::move(content));
    }
}

// -----------------------------------------
// Pack/unpack newline-separated word lists
// -----------------------------------------
static inline string pack_words_newline(const vector<string>& words) {
    ostringstream oss;
    for (const auto& w : words) oss << w << "\n";
    return oss.str();
}

static inline void unpack_words_newline(const string& packed, vector<string>& out_words) {
    out_words.clear();
    istringstream iss(packed);
    string w;
    while (getline(iss, w)) {
        if (!w.empty()) out_words.push_back(w);
    }
}

// -----------------------------------------
// Main: MPI TF-IDF optimized
// Total includes load and MPI gather, excludes disk write of mpi.csv
// -----------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto t_total0 = Clock::now();

    // Phase 1: rank 0 builds file list, each rank loads only its docs
    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];

    vector<string> all_paths;
    int N = 0;

    if (world_rank == 0) {
        list_txt_files_sorted(dataset_path, all_paths);
        N = (int)all_paths.size();
        cout << "--- MPI TF-IDF (optimized) ---\n";
        cout << "World size: " << world_size << "\n";
        cout << "Total documents: " << N << "\n";
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N == 0) {
        if (world_rank == 0) {
            cerr << "No documents found in path: " << dataset_path << "\n";
        }
        MPI_Finalize();
        return 0;
    }

    vector<int> local_doc_ids;
    vector<string> local_paths;
    distribute_paths(all_paths, world_rank, world_size, local_doc_ids, local_paths);

    vector<string> local_docs;
    load_documents_from_paths(local_paths, local_docs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = Clock::now();

    // Phase 2: local tokenization + build local vocab + local term ids
    unordered_map<string, int> local_vocab;
    local_vocab.reserve(200000);

    vector<vector<int>> doc_term_ids(N);
    vector<string> tokens;
    tokens.reserve(1024);

    for (int idx = 0; idx < (int)local_docs.size(); ++idx) {
        int doc_id = local_doc_ids[idx];
        const string& text = local_docs[idx];

        tokenize_ws(text, tokens);

        vector<int> local_ids;
        local_ids.reserve(tokens.size());

        for (const auto& w : tokens) {
            auto it = local_vocab.find(w);
            int id;
            if (it == local_vocab.end()) {
                id = (int)local_vocab.size();
                local_vocab.emplace(w, id);
            } else {
                id = it->second;
            }
            local_ids.push_back(id);
        }

        doc_term_ids[doc_id] = std::move(local_ids);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = Clock::now();

    // Phase 2.5: gather all unique words to rank 0 -> build global vocab (lexicographic)
    vector<string> local_words;
    local_words.reserve(local_vocab.size());
    for (const auto& kv : local_vocab) local_words.push_back(kv.first);

    string local_packed = pack_words_newline(local_words);
    int local_len = (int)local_packed.size();

    vector<string> global_words;
    if (world_rank == 0) {
        unordered_set<string> global_set;
        global_set.reserve(1000000);

        for (const auto& w : local_words) global_set.insert(w);

        for (int src = 1; src < world_size; ++src) {
            int recv_len = 0;
            MPI_Recv(&recv_len, 1, MPI_INT, src, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_len > 0) {
                string buf;
                buf.resize(recv_len);
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<string> recv_words;
                unpack_words_newline(buf, recv_words);
                for (const auto& w : recv_words) global_set.insert(w);
            }
        }

        global_words.assign(global_set.begin(), global_set.end());
        sort(global_words.begin(), global_words.end());
    } else {
        MPI_Send(&local_len, 1, MPI_INT, 0, 20, MPI_COMM_WORLD);
        if (local_len > 0) {
            MPI_Send(local_packed.data(), local_len, MPI_CHAR, 0, 21, MPI_COMM_WORLD);
        }
    }

    int V = 0;
    if (world_rank == 0) V = (int)global_words.size();
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

    string global_packed;
    int global_len = 0;
    if (world_rank == 0) {
        global_packed = pack_words_newline(global_words);
        global_len = (int)global_packed.size();
    }
    MPI_Bcast(&global_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) global_packed.resize(global_len);
    if (global_len > 0) {
        MPI_Bcast(global_packed.data(), global_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    if (world_rank != 0) {
        unpack_words_newline(global_packed, global_words);
    }

    unordered_map<string, int> global_map;
    global_map.reserve(V * 2);
    for (int i = 0; i < V; ++i) {
        global_map.emplace(global_words[i], i);
    }

    vector<string> local_id2word(local_vocab.size());
    for (const auto& kv : local_vocab) {
        local_id2word[kv.second] = kv.first;
    }

    for (int doc_id : local_doc_ids) {
        auto& ids = doc_term_ids[doc_id];
        for (int& lid : ids) {
            const string& w = local_id2word[lid];
            lid = global_map[w];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = Clock::now();

    // Phase 3: local DF over global ids, then reduce to global DF
    vector<int> local_df(V, 0);
    vector<int> tmp_ids;

    for (int doc_id : local_doc_ids) {
        const auto& ids = doc_term_ids[doc_id];
        if (ids.empty()) continue;

        tmp_ids.assign(ids.begin(), ids.end());
        sort(tmp_ids.begin(), tmp_ids.end());
        tmp_ids.erase(unique(tmp_ids.begin(), tmp_ids.end()), tmp_ids.end());

        for (int tid : tmp_ids) local_df[tid] += 1;
    }

    vector<int> global_df(V, 0);
    MPI_Reduce(local_df.data(), global_df.data(), V, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    vector<double> idf(V, 0.0);
    if (world_rank == 0) {
        for (int t = 0; t < V; ++t) {
            if (global_df[t] > 0) idf[t] = log((double)N / (double)global_df[t]);
            else idf[t] = 0.0;
        }
    }
    MPI_Bcast(idf.data(), V, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t4 = Clock::now();

    // Phase 4: compute TF (sparse) for local docs
    vector<vector<pair<int, double>>> tf_sparse(N);

    for (int doc_id : local_doc_ids) {
        const auto& ids = doc_term_ids[doc_id];
        if (ids.empty()) continue;

        tmp_ids.assign(ids.begin(), ids.end());
        sort(tmp_ids.begin(), tmp_ids.end());

        double total = (double)tmp_ids.size();
        vector<pair<int, double>> tfv;
        tfv.reserve(tmp_ids.size() / 4 + 4);

        size_t i = 0;
        while (i < tmp_ids.size()) {
            int tid = tmp_ids[i];
            size_t j = i + 1;
            while (j < tmp_ids.size() && tmp_ids[j] == tid) ++j;

            double tf = (double)(j - i) / total;
            tfv.emplace_back(tid, tf);
            i = j;
        }

        tf_sparse[doc_id] = std::move(tfv);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t5 = Clock::now();

    // Phase 5: compute TF-IDF and pack local CSV lines (compute only)
    ostringstream oss_local;
    for (int doc_id : local_doc_ids) {
        const auto& tfv = tf_sparse[doc_id];
        for (const auto& kv : tfv) {
            int tid = kv.first;
            double val = kv.second * idf[tid];
            oss_local << doc_id << "," << global_words[tid] << "," << val << "\n";
        }
    }

    auto t_pack_end = Clock::now();
    string out_local = oss_local.str();
    int out_len = (int)out_local.size();

    // Phase 6: gather CSV to rank 0 (communication included in total)
    vector<string> gathered; // only rank 0 uses
    if (world_rank == 0) {
        gathered.reserve(world_size);
        gathered.push_back(out_local);

        for (int src = 1; src < world_size; ++src) {
            int recv_len = 0;
            MPI_Recv(&recv_len, 1, MPI_INT, src, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_len > 0) {
                string buf;
                buf.resize(recv_len);
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 31, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                gathered.push_back(std::move(buf));
            } else {
                gathered.push_back(string());
            }
        }
    } else {
        MPI_Send(&out_len, 1, MPI_INT, 0, 30, MPI_COMM_WORLD);
        if (out_len > 0) {
            MPI_Send(out_local.data(), out_len, MPI_CHAR, 0, 31, MPI_COMM_WORLD);
        }
    }

    // End of total excluding disk write
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_total_end_excl_write = Clock::now();

    // Disk write time measured separately and excluded from total
    double write_time = 0.0;
    if (world_rank == 0) {
        auto t_write0 = Clock::now();

        ofstream fout("mpi.csv");
        fout << "document_id,word,tfidf_value\n";
        for (const auto& part : gathered) {
            fout << part;
        }
        fout.close();

        auto t_write1 = Clock::now();
        write_time = chrono::duration<double>(t_write1 - t_write0).count();
        cout << "TF-IDF saved to mpi.csv\n";
        cout << "CSV Write Time: " << write_time << " s\n";
    }

    if (world_rank == 0) {
        double t_load    = chrono::duration<double>(t1 - t_total0).count();
        double t_token   = chrono::duration<double>(t2 - t1).count();
        double t_vocab   = chrono::duration<double>(t3 - t2).count();
        double t_idf     = chrono::duration<double>(t4 - t3).count();
        double t_tf      = chrono::duration<double>(t5 - t4).count();
        double t_tfidf   = chrono::duration<double>(t_pack_end - t5).count();
        double t_total   = chrono::duration<double>(t_total_end_excl_write - t_total0).count();

        cout << "Timing (rank 0 approximate):\n";
        cout << "  Load documents:      " << t_load  << " s\n";
        cout << "  Tokenization(local): " << t_token << " s\n";
        cout << "  Build global vocab:  " << t_vocab << " s\n";
        cout << "  Compute IDF(MPI):    " << t_idf   << " s\n";
        cout << "  Compute TF:          " << t_tf    << " s\n";
        cout << "  Compute TF-IDF:      " << t_tfidf << " s\n";
        cout << "  Total (incl load, excl write): " << t_total << " s\n";
    }

    MPI_Finalize();
    return 0;
}
