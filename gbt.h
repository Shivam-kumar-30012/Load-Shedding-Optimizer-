#ifndef GBT_H
#define GBT_H

#include "gbt_features.h"
#include "data_io.h"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <limits>

using namespace std;

// =============================================================================
// Gradient Boosted Trees for regression.
// Each tree fits the residual of the prior ensemble.
//   prediction = initial_value + sum( learning_rate * tree_k(x) )
// Trees are simple regression trees: leaves store a single predicted value.
// =============================================================================

struct GBTNode {
    bool is_leaf = false;
    int feature_idx = -1;
    double threshold = 0.0;
    int left = -1;
    int right = -1;
    double leaf_value = 0.0;
};

struct GBTTree {
    vector<GBTNode> nodes;

    double predict(const vector<double>& x) const {
        if (nodes.empty()) return 0.0;
        int idx = 0;
        while (!nodes[idx].is_leaf) {
            const auto& n = nodes[idx];
            idx = (x[n.feature_idx] <= n.threshold) ? n.left : n.right;
            if (idx < 0) return 0.0;
        }
        return nodes[idx].leaf_value;
    }
};

class GBT {
public:
    int n_estimators;
    int max_depth;
    int min_samples_leaf;
    double learning_rate;
    double subsample;
    int n_random_features;

    // Safety buffer for deployment forecasts.
    // Real grid operators add reserve margin to forecasts to handle outliers
    // (heat waves, cold snaps) that the model can't predict from time alone.
    // 1.20 = plan for 20% above predicted demand.
    static constexpr double SAFETY_BUFFER = 1.40;

    double initial_value = 0.0;
    vector<GBTTree> trees;
    GBTFeatureBuilder feature_builder;

    mt19937 rng;

    GBT(int n_est = 150, int max_d = 6, int min_leaf = 20,
        double lr = 0.05, double subs = 0.7, int n_feat = 8,
        unsigned seed = 42)
        : n_estimators(n_est), max_depth(max_d), min_samples_leaf(min_leaf),
          learning_rate(lr), subsample(subs), n_random_features(n_feat), rng(seed) {}

    // Predict a single sample (raw, no buffer)
    double predict_one(const vector<double>& x) const {
        double s = initial_value;
        for (const auto& t : trees) s += learning_rate * t.predict(x);
        return s;
    }

    // Predict for a (timestamp, area) — applies safety buffer for deployment use
    double predict(time_t t, const string& area_id) const {
        auto x = feature_builder.build(t, area_id);
        double v = predict_one(x);
        if (v < 0) v = 0.0;
        return v * SAFETY_BUFFER;
    }

    // Build a 24-hour forecast matrix for all areas on target_date
    vector<vector<double>> forecast_matrix(
        const vector<string>& area_order,
        time_t target_date,
        double /*unused*/ = 1.0) const
    {
        tm* tm_info = localtime(&target_date);
        tm midnight = *tm_info;
        midnight.tm_hour = 0; midnight.tm_min = 0; midnight.tm_sec = 0;
        time_t day_start = mktime(&midnight);

        int n_areas = (int)area_order.size();
        vector<vector<double>> matrix(n_areas, vector<double>(24, 0.0));
        for (int i = 0; i < n_areas; ++i) {
            for (int h = 0; h < 24; ++h) {
                time_t hour_t = day_start + h * 3600;
                matrix[i][h] = predict(hour_t, area_order[i]);
            }
        }
        return matrix;
    }

    // Train from raw records
    void train(const vector<Record>& records,
               const vector<string>& area_order,
               bool verbose = true)
    {
        feature_builder.set_areas(area_order);

        vector<vector<double>> X;
        vector<double> y;
        X.reserve(records.size());
        y.reserve(records.size());
        for (const auto& r : records) {
            if (feature_builder.area_to_idx.find(r.area_id) ==
                feature_builder.area_to_idx.end()) continue;
            X.push_back(feature_builder.build(r.timestamp, r.area_id));
            y.push_back(r.demand_mw);
        }

        if (X.empty()) {
            cerr << "[GBT] No training samples.\n";
            return;
        }

        if (verbose) {
            cout << "\n[GBT] Training " << n_estimators << " trees on "
                 << X.size() << " samples, " << X[0].size() << " features.\n";
        }

        initial_value = accumulate(y.begin(), y.end(), 0.0) / y.size();

        vector<double> pred(y.size(), initial_value);

        trees.clear();
        trees.reserve(n_estimators);

        int n_total = (int)X.size();
        int n_sub = max(1, (int)(subsample * n_total));
        uniform_int_distribution<int> row_dist(0, n_total - 1);

        for (int t = 0; t < n_estimators; ++t) {
            vector<double> residuals(n_total);
            for (int i = 0; i < n_total; ++i) residuals[i] = y[i] - pred[i];

            vector<int> bag;
            bag.reserve(n_sub);
            for (int i = 0; i < n_sub; ++i) bag.push_back(row_dist(rng));

            GBTTree tree;
            build_tree(tree, X, residuals, bag);
            trees.push_back(move(tree));

            const auto& tree_built = trees.back();
            for (int i = 0; i < n_total; ++i) {
                pred[i] += learning_rate * tree_built.predict(X[i]);
            }

            if (verbose) {
                int pct = (int)((double)(t + 1) / n_estimators * 100.0);
                if ((t + 1) % 10 == 0 || t == n_estimators - 1) {
                    double mae = 0.0;
                    for (int i = 0; i < n_total; ++i)
                        mae += fabs(y[i] - pred[i]);
                    mae /= n_total;
                    cout << "[GBT] " << pct << "% (" << (t + 1) << "/"
                         << n_estimators << " trees) | train MAE=" << mae << "\n";
                }
            }
        }
        if (verbose) cout << "[GBT] Training complete.\n";
    }

private:
    void build_tree(GBTTree& tree,
                    const vector<vector<double>>& X,
                    const vector<double>& targets,
                    const vector<int>& indices)
    {
        tree.nodes.clear();
        tree.nodes.reserve(64);
        build_node(tree, X, targets, indices, 0);
    }

    int build_node(GBTTree& tree,
                   const vector<vector<double>>& X,
                   const vector<double>& targets,
                   const vector<int>& indices,
                   int depth)
    {
        int node_idx = (int)tree.nodes.size();
        tree.nodes.push_back(GBTNode());

        double sum = 0.0;
        for (int i : indices) sum += targets[i];
        double mean = indices.empty() ? 0.0 : sum / indices.size();

        if ((int)indices.size() <= min_samples_leaf || depth >= max_depth) {
            tree.nodes[node_idx].is_leaf = true;
            tree.nodes[node_idx].leaf_value = mean;
            return node_idx;
        }

        int n_feats = X.empty() ? 0 : (int)X[0].size();
        if (n_feats == 0) {
            tree.nodes[node_idx].is_leaf = true;
            tree.nodes[node_idx].leaf_value = mean;
            return node_idx;
        }

        uniform_int_distribution<int> feat_dist(0, n_feats - 1);
        double best_gain = 0.0;
        int best_feat = -1;
        double best_thresh = 0.0;

        double total_sum_sq = 0.0;
        for (int i : indices) {
            double d = targets[i] - mean;
            total_sum_sq += d * d;
        }

        for (int trial = 0; trial < n_random_features; ++trial) {
            int f = feat_dist(rng);
            uniform_int_distribution<int> idx_dist(0, (int)indices.size() - 1);
            double thresh = X[indices[idx_dist(rng)]][f];

            double left_sum = 0.0, right_sum = 0.0;
            int left_n = 0, right_n = 0;
            for (int i : indices) {
                if (X[i][f] <= thresh) { left_sum += targets[i]; left_n++; }
                else                   { right_sum += targets[i]; right_n++; }
            }
            if (left_n < min_samples_leaf || right_n < min_samples_leaf) continue;

            double left_mean = left_sum / left_n;
            double right_mean = right_sum / right_n;
            double left_ss = 0.0, right_ss = 0.0;
            for (int i : indices) {
                if (X[i][f] <= thresh) {
                    double d = targets[i] - left_mean; left_ss += d * d;
                } else {
                    double d = targets[i] - right_mean; right_ss += d * d;
                }
            }
            double gain = total_sum_sq - (left_ss + right_ss);
            if (gain > best_gain) {
                best_gain = gain;
                best_feat = f;
                best_thresh = thresh;
            }
        }

        if (best_feat < 0) {
            tree.nodes[node_idx].is_leaf = true;
            tree.nodes[node_idx].leaf_value = mean;
            return node_idx;
        }

        vector<int> left_idx, right_idx;
        left_idx.reserve(indices.size() / 2);
        right_idx.reserve(indices.size() / 2);
        for (int i : indices) {
            if (X[i][best_feat] <= best_thresh) left_idx.push_back(i);
            else                                 right_idx.push_back(i);
        }

        tree.nodes[node_idx].is_leaf = false;
        tree.nodes[node_idx].feature_idx = best_feat;
        tree.nodes[node_idx].threshold = best_thresh;

        int left_child = build_node(tree, X, targets, left_idx, depth + 1);
        int right_child = build_node(tree, X, targets, right_idx, depth + 1);

        tree.nodes[node_idx].left = left_child;
        tree.nodes[node_idx].right = right_child;
        return node_idx;
    }
};

#endif