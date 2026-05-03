#ifndef GBT_IO_H
#define GBT_IO_H

#include "gbt.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

inline void gbt_write_string(ofstream& f, const string& s) {
    int len = (int)s.size();
    f.write(reinterpret_cast<const char*>(&len), sizeof(int));
    f.write(s.data(), len);
}

inline string gbt_read_string(ifstream& f) {
    int len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(int));
    string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

template<typename T>
void gbt_write_pod(ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
void gbt_read_pod(ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
}

inline void save_gbt_node(ofstream& f, const GBTNode& n) {
    gbt_write_pod(f, n.is_leaf);
    gbt_write_pod(f, n.feature_idx);
    gbt_write_pod(f, n.threshold);
    gbt_write_pod(f, n.left);
    gbt_write_pod(f, n.right);
    gbt_write_pod(f, n.leaf_value);
}

inline void load_gbt_node(ifstream& f, GBTNode& n) {
    gbt_read_pod(f, n.is_leaf);
    gbt_read_pod(f, n.feature_idx);
    gbt_read_pod(f, n.threshold);
    gbt_read_pod(f, n.left);
    gbt_read_pod(f, n.right);
    gbt_read_pod(f, n.leaf_value);
}

inline bool save_gbt(const GBT& gbt, const string& path) {
    ofstream f(path, ios::binary);
    if (!f.is_open()) {
        cerr << "[GBT-IO] Cannot write " << path << "\n";
        return false;
    }
    cout << "[GBT-IO] Saving GBT to " << path << "...\n";

    int magic = 0xCB71;
    gbt_write_pod(f, magic);
    gbt_write_pod(f, gbt.n_estimators);
    gbt_write_pod(f, gbt.max_depth);
    gbt_write_pod(f, gbt.min_samples_leaf);
    gbt_write_pod(f, gbt.learning_rate);
    gbt_write_pod(f, gbt.subsample);
    gbt_write_pod(f, gbt.n_random_features);
    gbt_write_pod(f, gbt.initial_value);
    gbt_write_pod(f, gbt.feature_builder.reference_year);

    int n_areas = (int)gbt.feature_builder.area_order.size();
    gbt_write_pod(f, n_areas);
    for (const auto& a : gbt.feature_builder.area_order) {
        gbt_write_string(f, a);
    }

    int n_trees = (int)gbt.trees.size();
    gbt_write_pod(f, n_trees);
    for (const auto& t : gbt.trees) {
        int n_nodes = (int)t.nodes.size();
        gbt_write_pod(f, n_nodes);
        for (const auto& n : t.nodes) {
            save_gbt_node(f, n);
        }
    }

    f.close();
    cout << "[GBT-IO] Saved " << n_trees << " trees, " << n_areas << " areas.\n";
    return true;
}

inline bool load_gbt(GBT& gbt, const string& path) {
    ifstream f(path, ios::binary);
    if (!f.is_open()) {
        cerr << "[GBT-IO] Cannot open " << path << "\n";
        return false;
    }
    cout << "[GBT-IO] Loading GBT from " << path << "...\n";

    int magic = 0;
    gbt_read_pod(f, magic);
    if (magic != 0xCB71) {
        cerr << "[GBT-IO] Bad magic.\n";
        return false;
    }

    gbt_read_pod(f, gbt.n_estimators);
    gbt_read_pod(f, gbt.max_depth);
    gbt_read_pod(f, gbt.min_samples_leaf);
    gbt_read_pod(f, gbt.learning_rate);
    gbt_read_pod(f, gbt.subsample);
    gbt_read_pod(f, gbt.n_random_features);
    gbt_read_pod(f, gbt.initial_value);
    gbt_read_pod(f, gbt.feature_builder.reference_year);

    int n_areas = 0;
    gbt_read_pod(f, n_areas);
    vector<string> areas;
    areas.reserve(n_areas);
    for (int i = 0; i < n_areas; ++i) {
        areas.push_back(gbt_read_string(f));
    }
    gbt.feature_builder.set_areas(areas);

    int n_trees = 0;
    gbt_read_pod(f, n_trees);
    gbt.trees.clear();
    gbt.trees.reserve(n_trees);
    for (int t = 0; t < n_trees; ++t) {
        GBTTree tree;
        int n_nodes = 0;
        gbt_read_pod(f, n_nodes);
        tree.nodes.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            load_gbt_node(f, tree.nodes[i]);
        }
        gbt.trees.push_back(move(tree));
    }

    f.close();
    cout << "[GBT-IO] Loaded " << n_trees << " trees, " << n_areas << " areas.\n";
    return true;
}

#endif