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

static inline void tokenize_ws(const string& text, vector<string>& out_tokens) {
    out_tokens.clear();
    const char* s = text.data();
    size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        while (i < n && isspace((unsigned char)s[i])) ++i;
        if (i >= n) break;

        size_t j = i;
        while (j < n && !isspace((unsigned char)s[j])) ++j;

        out_tokens.emplace_back(text.substr(i, j - i));
        i = j;
    }
}

void list_txt_files_sorted(const string& root, vector<string>& out_paths) {
    out_paths.clear();
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            out_paths.push_back(entry.path().string());
        }
    }
    sort(out_paths.begin(), out_paths.end());
}

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
                for (const auto& p : paths_per_rank[r]) oss << p << "\n";
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

            if ((int)local_paths.size() != k) local_paths.resize(k);
        }
    }
}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto t_total0 = Clock::now();

    string dataset_path = "dataset";
    if (argc >= 2) dataset_path = argv[1];

    vector<string> all_paths;
    int N = 0;

    if (world_rank == 0) {
        list_txt_files_sorted(dataset_path, all_paths);
        N = (int)all_paths.size();
        cout << "MPI TF-IDF optimized\n";
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

    unordered_map<string, int> local_vocab;
    local_vocab.reserve(200000);

    vector<vector<int>> local_doc_term_ids(local_docs.size());
    vector<string> tokens;
    tokens.reserve(1024);

    for (int i = 0; i < (int)local_docs.size(); ++i) {
        tokenize_ws(local_docs[i], tokens);

        vector<int> ids;
        ids.reserve(tokens.size());

        for (const auto& w : tokens) {
            auto it = local_vocab.find(w);
            int id;
            if (it == local_vocab.end()) {
                id = (int)local_vocab.size();
                local_vocab.emplace(w, id);
            } else {
                id = it->second;
            }
            ids.push_back(id);
        }
        local_doc_term_ids[i] = std::move(ids);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = Clock::now();

    vector<string> local_words;
    local_words.reserve(local_vocab.size());
    for (const auto& kv : local_vocab) local_words.push_back(kv.first);

    string local_packed = pack_words_newline(local_words);
    int local_len = (int)local_packed.size();

    vector<int> recv_lens;
    vector<int> displs;
    vector<char> recv_buf;

    if (world_rank == 0) {
        recv_lens.resize(world_size);
    }

    MPI_Gather(&local_len, 1, MPI_INT,
               world_rank == 0 ? recv_lens.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    vector<string> global_words;

    if (world_rank == 0) {
        displs.resize(world_size);
        int total_len = 0;
        for (int r = 0; r < world_size; ++r) {
            displs[r] = total_len;
            total_len += recv_lens[r];
        }
        recv_buf.resize(total_len);
    }

    MPI_Gatherv(local_len > 0 ? local_packed.data() : nullptr, local_len, MPI_CHAR,
                world_rank == 0 ? recv_buf.data() : nullptr,
                world_rank == 0 ? recv_lens.data() : nullptr,
                world_rank == 0 ? displs.data() : nullptr,
                MPI_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        unordered_set<string> global_set;
        global_set.reserve(1000000);

        for (int r = 0; r < world_size; ++r) {
            int len = recv_lens[r];
            if (len == 0) continue;
            string part(recv_buf.data() + displs[r], len);

            vector<string> words_r;
            unpack_words_newline(part, words_r);
            for (const auto& w : words_r) global_set.insert(w);
        }

        global_words.assign(global_set.begin(), global_set.end());
        sort(global_words.begin(), global_words.end());
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
    for (int i = 0; i < V; ++i) global_map.emplace(global_words[i], i);

    vector<string> local_id2word(local_vocab.size());
    for (const auto& kv : local_vocab) local_id2word[kv.second] = kv.first;

    for (int i = 0; i < (int)local_doc_term_ids.size(); ++i) {
        auto& ids = local_doc_term_ids[i];
        for (int& lid : ids) {
            const string& w = local_id2word[lid];
            lid = global_map[w];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = Clock::now();

    vector<int> local_df(V, 0);
    vector<int> tmp_ids;

    for (int i = 0; i < (int)local_doc_term_ids.size(); ++i) {
        const auto& ids = local_doc_term_ids[i];
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

    ostringstream oss_local;
    double tf_time_local = 0.0;
    double tfidf_time_local = 0.0;

    for (int i = 0; i < (int)local_doc_term_ids.size(); ++i) {
        int doc_id = local_doc_ids[i];
        const auto& ids = local_doc_term_ids[i];
        if (ids.empty()) continue;

        auto tf0 = Clock::now();

        tmp_ids.assign(ids.begin(), ids.end());
        sort(tmp_ids.begin(), tmp_ids.end());

        double total = (double)tmp_ids.size();

        auto tf1 = Clock::now();
        tf_time_local += chrono::duration<double>(tf1 - tf0).count();

        auto tfidf0 = Clock::now();

        size_t p = 0;
        while (p < tmp_ids.size()) {
            int tid = tmp_ids[p];
            size_t q = p + 1;
            while (q < tmp_ids.size() && tmp_ids[q] == tid) ++q;

            double tf = (double)(q - p) / total;
            double val = tf * idf[tid];

            oss_local << doc_id << "," << global_words[tid] << "," << val << "\n";
            p = q;
        }

        auto tfidf1 = Clock::now();
        tfidf_time_local += chrono::duration<double>(tfidf1 - tfidf0).count();
    }

    string out_local = oss_local.str();
    int out_len = (int)out_local.size();

    double tf_time_max = 0.0;
    double tfidf_time_max = 0.0;
    MPI_Reduce(&tf_time_local, &tf_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tfidf_time_local, &tfidf_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        recv_lens.assign(world_size, 0);
    }

    MPI_Gather(&out_len, 1, MPI_INT,
               world_rank == 0 ? recv_lens.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        displs.assign(world_size, 0);
        int total_len = 0;
        for (int r = 0; r < world_size; ++r) {
            displs[r] = total_len;
            total_len += recv_lens[r];
        }
        recv_buf.assign(total_len, 0);
    }

    MPI_Gatherv(out_len > 0 ? out_local.data() : nullptr, out_len, MPI_CHAR,
                world_rank == 0 ? recv_buf.data() : nullptr,
                world_rank == 0 ? recv_lens.data() : nullptr,
                world_rank == 0 ? displs.data() : nullptr,
                MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_total_end_excl_write = Clock::now();

    double write_time = 0.0;
    if (world_rank == 0) {
        auto t_write0 = Clock::now();

        ofstream fout("mpi.csv");
        fout << "document_id,word,tfidf_value\n";
        if (!recv_buf.empty()) {
            fout.write(recv_buf.data(), (streamsize)recv_buf.size());
        }
        fout.close();

        auto t_write1 = Clock::now();
        write_time = chrono::duration<double>(t_write1 - t_write0).count();
    }

    if (world_rank == 0) {
        double t_doc_load = chrono::duration<double>(t1 - t_total0).count();
        double t_token    = chrono::duration<double>(t3 - t1).count();
        double t_idf      = chrono::duration<double>(t4 - t3).count();

        double total_excl_write = chrono::duration<double>(t_total_end_excl_write - t_total0).count();
        double remaining_after_idf = chrono::duration<double>(t_total_end_excl_write - t4).count();

        double t_tf = tf_time_max;
        double t_tfidf = remaining_after_idf - t_tf;
        if (t_tfidf < 0.0) t_tfidf = 0.0;

        cout << "--- MPI TF-IDF Timing Report ---\n";
        cout << "Total Documents Loaded: " << N << "\n";
        cout << "Document Loading Time: " << t_doc_load << " seconds\n";
        cout << "Tokenization Time: " << t_token << " seconds\n";
        cout << "Vocabulary Size: " << V << "\n";
        cout << "Compute IDF Time: " << t_idf << " seconds\n";
        cout << "Compute All TFs Time: " << t_tf << " seconds\n";
        cout << "Compute All TF-IDFs Time: " << t_tfidf << " seconds\n";
        cout << "CSV Write Time: " << write_time << " seconds\n";
        cout << "TF-IDF saved to mpi.csv\n";
        cout << "------------------------------------------\n";
        cout << "Total Execution Time (including load, excluding write): "
             << total_excl_write << " seconds\n";
        cout << "------------------------------------------\n";
    }

    MPI_Finalize();
    return 0;
}
