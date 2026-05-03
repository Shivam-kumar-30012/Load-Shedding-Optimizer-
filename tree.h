#ifndef TREE_H
#define TREE_H

#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

using namespace std;

// One training sample for the QRF
// features = [hour, day_of_week, month, area_idx_onehot_0..n_areas-1]
struct Sample {
    vector<double> features;
    double target;       // demand_mw
};

// One node in the decision tree
struct TreeNode {
    bool is_leaf = false;
    int feature_idx = -1;       // which feature to split on (internal node)
    double threshold = 0.0;     // split: left = <= threshold, right = > threshold
    int left_child = -1;        // index into nodes vector (-1 if leaf)
    int right_child = -1;
    vector<double> leaf_targets; // training targets that reached this leaf (for quantile prediction)
};

class Tree {
public:
    vector<TreeNode> nodes;     // node 0 is root
    int max_depth;
    int min_samples_leaf;
    int n_random_features;      // try this many random features per split
    int n_random_thresholds;    // try this many random thresholds per feature

    mt19937 rng;

    Tree(int max_d = 12, int min_leaf = 20, int n_feat = 5, int n_thresh = 5, unsigned seed = 42)
        : max_depth(max_d), min_samples_leaf(min_leaf),
          n_random_features(n_feat), n_random_thresholds(n_thresh),
          rng(seed) {}

    // Compute variance of targets in a sample subset
    double variance(const vector<int>& indices, const vector<Sample>& data) {
        if (indices.empty()) return 0.0;
        double mean = 0.0;
        for (int idx : indices) mean += data[idx].target;
        mean /= indices.size();
        double var = 0.0;
        for (int idx : indices) {
            double d = data[idx].target - mean;
            var += d * d;
        }
        return var / indices.size();
    }

    // Recursively build tree, returns node index
    int build_node(const vector<int>& indices, const vector<Sample>& data, int depth) {
        TreeNode node;

        // Stopping conditions: max depth or too few samples
        if (depth >= max_depth || (int)indices.size() <= min_samples_leaf) {
            node.is_leaf = true;
            node.leaf_targets.reserve(indices.size());
            for (int idx : indices) node.leaf_targets.push_back(data[idx].target);
            int my_idx = nodes.size();
            nodes.push_back(node);
            return my_idx;
        }

        int n_features = data[0].features.size();
        double best_score = numeric_limits<double>::max();
        int best_feature = -1;
        double best_threshold = 0.0;
        vector<int> best_left, best_right;

        // Try `n_random_features` random features
        uniform_int_distribution<int> feat_dist(0, n_features - 1);

        for (int f_try = 0; f_try < n_random_features; ++f_try) {
            int f = feat_dist(rng);

            // Find min and max of this feature in the subset
            double f_min = numeric_limits<double>::max();
            double f_max = -numeric_limits<double>::max();
            for (int idx : indices) {
                double v = data[idx].features[f];
                if (v < f_min) f_min = v;
                if (v > f_max) f_max = v;
            }
            if (f_min == f_max) continue; // can't split on a constant feature

            uniform_real_distribution<double> thresh_dist(f_min, f_max);

            // Try `n_random_thresholds` random thresholds
            for (int t_try = 0; t_try < n_random_thresholds; ++t_try) {
                double thresh = thresh_dist(rng);
                vector<int> left_idx, right_idx;
                left_idx.reserve(indices.size());
                right_idx.reserve(indices.size());

                for (int idx : indices) {
                    if (data[idx].features[f] <= thresh) left_idx.push_back(idx);
                    else right_idx.push_back(idx);
                }

                if ((int)left_idx.size() < min_samples_leaf || (int)right_idx.size() < min_samples_leaf)
                    continue;

                // Score = weighted variance of children (lower is better)
                double total = (double)indices.size();
                double score = (left_idx.size() / total) * variance(left_idx, data)
                             + (right_idx.size() / total) * variance(right_idx, data);

                if (score < best_score) {
                    best_score = score;
                    best_feature = f;
                    best_threshold = thresh;
                    best_left = left_idx;
                    best_right = right_idx;
                }
            }
        }

        // If no valid split found, make this a leaf
        if (best_feature == -1) {
            node.is_leaf = true;
            node.leaf_targets.reserve(indices.size());
            for (int idx : indices) node.leaf_targets.push_back(data[idx].target);
            int my_idx = nodes.size();
            nodes.push_back(node);
            return my_idx;
        }

        // Internal node: reserve slot, recurse, then fill in
        int my_idx = nodes.size();
        nodes.push_back(node);  // placeholder

        int left_id = build_node(best_left, data, depth + 1);
        int right_id = build_node(best_right, data, depth + 1);

        // Refill (push_back invalidated reference)
        nodes[my_idx].is_leaf = false;
        nodes[my_idx].feature_idx = best_feature;
        nodes[my_idx].threshold = best_threshold;
        nodes[my_idx].left_child = left_id;
        nodes[my_idx].right_child = right_id;

        return my_idx;
    }

    // Train: build tree from sample indices
    void fit(const vector<Sample>& data, const vector<int>& sample_indices) {
        nodes.clear();
        if (sample_indices.empty()) return;
        build_node(sample_indices, data, 0);
    }

    // Drop a sample down the tree, return the leaf's target list
    const vector<double>& predict_leaf(const vector<double>& features) const {
        int idx = 0;
        while (!nodes[idx].is_leaf) {
            if (features[nodes[idx].feature_idx] <= nodes[idx].threshold)
                idx = nodes[idx].left_child;
            else
                idx = nodes[idx].right_child;
        }
        return nodes[idx].leaf_targets;
    }
};

#endif // TREE_H