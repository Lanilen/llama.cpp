// A LightGBM text model (Booster.save_model) evaluated without LightGBM: numerical splits that go left when x <= the
// threshold (decision_type 2, default left, no missing-value handling: what a model trained on dense rows has), leaves
// summed over the trees, a sigmoid for objective=binary. The fork uses it for small learned decisions on the host
// (DSpark's exit model, --spec-draft-exit-model); a model with any other split type is refused.
#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct lgbm_tiny {
    struct tree {
        std::vector<int>    feat, left, right;
        std::vector<double> thr, leaf;
    };
    std::vector<tree> trees;
    int  n_features = 0;
    bool sigmoid    = false;

    template <typename T> static std::vector<T> nums(const std::string & s) {
        std::vector<T> v;
        std::istringstream is(s);
        T x;
        while (is >> x) {
            v.push_back(x);
        }
        return v;
    }

    // false (and err) if the file is not a model this evaluator reads
    bool load(const std::string & path, std::string & err) {
        std::ifstream f(path);
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        std::string line;
        tree * cur = nullptr;
        while (std::getline(f, line)) {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            if (k == "max_feature_idx") {
                n_features = std::stoi(v) + 1;
            } else if (k == "objective") {
                sigmoid = v.rfind("binary", 0) == 0;
            } else if (k == "Tree") {
                trees.emplace_back();
                cur = &trees.back();
            } else if (cur && k == "split_feature") {
                cur->feat = nums<int>(v);
            } else if (cur && k == "threshold") {
                cur->thr = nums<double>(v);
            } else if (cur && k == "left_child") {
                cur->left = nums<int>(v);
            } else if (cur && k == "right_child") {
                cur->right = nums<int>(v);
            } else if (cur && k == "leaf_value") {
                cur->leaf = nums<double>(v);
            } else if (cur && k == "decision_type") {
                for (int d : nums<int>(v)) {
                    if (d != 2) {
                        err = "decision_type " + std::to_string(d) + " is not supported (numerical, default left only)";
                        return false;
                    }
                }
            } else if (cur && k == "num_cat" && std::stoi(v) != 0) {
                err = "categorical splits are not supported";
                return false;
            }
        }
        if (trees.empty() || n_features == 0) {
            err = "no trees in " + path;
            return false;
        }
        return true;
    }

    // the raw score, or with objective=binary the probability
    double predict(const double * x) const {
        double s = 0.0;
        for (const tree & t : trees) {
            if (t.feat.empty()) {
                s += t.leaf.empty() ? 0.0 : t.leaf[0];
                continue;
            }
            int n = 0;
            while (n >= 0) {
                n = x[t.feat[n]] <= t.thr[n] ? t.left[n] : t.right[n];
            }
            s += t.leaf[-n - 1];
        }
        return sigmoid ? 1.0 / (1.0 + std::exp(-s)) : s;
    }
};
