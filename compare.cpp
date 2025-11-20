#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

map<string, map<string,string>> read_csv_string(const string& filename) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Failed to open " << filename << "\n";
        return {};
    }

    string line;
    getline(fin, line);

    map<string, map<string,string>> data;
    int line_no = 1;

    while (getline(fin, line)) {
        line_no++;
        stringstream ss(line);
        string doc, word, val_s;

        if (!getline(ss, doc, ',') || !getline(ss, word, ',') || !getline(ss, val_s, ',')) {
            cerr << "Warning: invalid format at line " << line_no << ": " << line << "\n";
            continue;
        }

        // 去掉開頭結尾空格
        val_s.erase(0, val_s.find_first_not_of(" \t\r\n"));
        val_s.erase(val_s.find_last_not_of(" \t\r\n") + 1);

        data[doc][word] = val_s;
    }

    return data;
}

// 字串比對函式
bool compare_csv_maps_string(const map<string,map<string,string>>& a,
                             const map<string,map<string,string>>& b) {
    if (a.size() != b.size()) return false;
    for (auto& [doc, amap] : a) {
        if (b.find(doc) == b.end()) return false;
        auto& bmap = b.at(doc);
        if (amap.size() != bmap.size()) return false;

        for (auto& [word, val] : amap) {
            if (bmap.find(word) == bmap.end()) return false;
            if (val != bmap.at(word)) return false;
        }
    }
    return true;
}

// ====================== MAIN ======================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " file1.csv file2.csv\n";
        return 1;
    }

    auto data1 = read_csv_string(argv[1]);
    auto data2 = read_csv_string(argv[2]);

    bool same = compare_csv_maps_string(data1, data2);
    if (same)
        cout << "The CSV files are identical.\n";
    else
        cout << "The CSV files are different.\n";

    return 0;
}
