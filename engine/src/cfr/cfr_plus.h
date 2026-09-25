#pragma once

// CFR+ with alternating updates and linearly weighted averaging
// (Tammelin et al. 2015). Walks the full game tree every iteration, so it is
// only practical for small games. It serves as the exact reference solver
// that the sampling solvers and the best-response code are checked against.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cfr/tables.h"
#include "core/game.h"

namespace poker2 {

template <class Game>
class CfrPlusSolver {
 public:
  using State = typename Game::State;

  void iterate() {
    ++iteration_;
    for (int player = 0; player < 2; ++player) {
      traverse(Game::initial_state(), player, 1.0, 1.0);
      // An infoset is reached through several histories per traversal. Its
      // strategy must stay fixed for the whole traversal and the CFR+ floor
      // applies to the summed regret, so updates are deferred until here.
      for (auto& [node, delta] : pending_regret_) {
        for (size_t i = 0; i < delta.size(); ++i) {
          const double updated = node->regret[i] + delta[i];
          node->regret[i] = updated > 0.0 ? updated : 0.0;
        }
      }
      pending_regret_.clear();
    }
  }

  int64_t iteration() const { return iteration_; }
  size_t num_infosets() const { return table_.size(); }
  TabularPolicy average_policy() const { return table_.average_policy(); }

 private:
  // Returns the traverser's expected value at `state` under the current profile.
  // reach_self: traverser's reach; reach_other: opponent * chance reach.
  double traverse(const State& state, int traverser, double reach_self, double reach_other) {
    if (state.is_terminal()) return state.utility(traverser);

    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      double value = 0.0;
      for (const Outcome& o : outcomes) {
        State child = state;
        child.apply(o.action);
        value += o.prob * traverse(child, traverser, reach_self, reach_other * o.prob);
      }
      return value;
    }

    std::vector<int> actions;
    state.legal_actions(actions);
    const size_t n = actions.size();
    InfosetNode& node = table_.get(state.infoset_key(), n);
    std::vector<double> sigma;
    regret_matching(node.regret, sigma);

    if (state.current_player() != traverser) {
      double value = 0.0;
      for (size_t i = 0; i < n; ++i) {
        State child = state;
        child.apply(actions[i]);
        value += sigma[i] * traverse(child, traverser, reach_self, reach_other * sigma[i]);
      }
      return value;
    }

    std::vector<double> action_values(n);
    double node_value = 0.0;
    for (size_t i = 0; i < n; ++i) {
      State child = state;
      child.apply(actions[i]);
      action_values[i] = traverse(child, traverser, reach_self * sigma[i], reach_other);
      node_value += sigma[i] * action_values[i];
    }

    const double avg_weight = static_cast<double>(iteration_) * reach_self;
    std::vector<double>& delta = pending_regret_[&node];
    delta.resize(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      delta[i] += reach_other * (action_values[i] - node_value);
      node.strategy_sum[i] += avg_weight * sigma[i];
    }
    return node_value;
  }

  InfosetTable table_;
  std::unordered_map<InfosetNode*, std::vector<double>> pending_regret_;
  int64_t iteration_ = 0;
};

}  // namespace poker2
