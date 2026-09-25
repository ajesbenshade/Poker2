#pragma once

// Tabular storage shared by the CFR solvers. String keys are fine for the toy
// games; the HUNL engine will swap in dense arrays indexed by
// (bucket, action-sequence id) behind the same regret-matching logic.

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace poker2 {

// sigma(a) proportional to max(regret(a), 0); uniform if no positive regret.
inline void regret_matching(const std::vector<double>& regret, std::vector<double>& out) {
  const size_t n = regret.size();
  out.resize(n);
  double positive_sum = 0.0;
  for (double r : regret) positive_sum += std::max(r, 0.0);
  for (size_t i = 0; i < n; ++i) {
    out[i] = positive_sum > 0.0 ? std::max(regret[i], 0.0) / positive_sum : 1.0 / n;
  }
}

// Action probabilities per infoset, indexed like the state's legal_actions().
class TabularPolicy {
 public:
  void set(const std::string& key, std::vector<double> probs) { table_[key] = std::move(probs); }

  // Unknown infosets play uniformly.
  std::vector<double> get(const std::string& key, size_t num_actions) const {
    auto it = table_.find(key);
    if (it == table_.end()) return std::vector<double>(num_actions, 1.0 / num_actions);
    return it->second;
  }

  size_t size() const { return table_.size(); }
  const std::unordered_map<std::string, std::vector<double>>& entries() const { return table_; }

 private:
  std::unordered_map<std::string, std::vector<double>> table_;
};

struct InfosetNode {
  std::vector<double> regret;
  std::vector<double> strategy_sum;
};

class InfosetTable {
 public:
  // References stay valid across later insertions (unordered_map node stability),
  // so solvers may hold one across recursive calls.
  InfosetNode& get(const std::string& key, size_t num_actions) {
    auto [it, inserted] = nodes_.try_emplace(key);
    if (inserted) {
      it->second.regret.assign(num_actions, 0.0);
      it->second.strategy_sum.assign(num_actions, 0.0);
    }
    return it->second;
  }

  size_t size() const { return nodes_.size(); }

  TabularPolicy average_policy() const {
    TabularPolicy policy;
    for (const auto& [key, node] : nodes_) {
      const size_t n = node.strategy_sum.size();
      double total = 0.0;
      for (double s : node.strategy_sum) total += s;
      std::vector<double> probs(n, 1.0 / n);
      if (total > 0.0) {
        for (size_t i = 0; i < n; ++i) probs[i] = node.strategy_sum[i] / total;
      }
      policy.set(key, std::move(probs));
    }
    return policy;
  }

  TabularPolicy current_policy() const {
    TabularPolicy policy;
    std::vector<double> probs;
    for (const auto& [key, node] : nodes_) {
      regret_matching(node.regret, probs);
      policy.set(key, probs);
    }
    return policy;
  }

 private:
  std::unordered_map<std::string, InfosetNode> nodes_;
};

}  // namespace poker2
