#pragma once

// External-sampling Monte Carlo CFR (Lanctot et al. 2009) with optional
// Linear CFR weighting (Brown & Sandholm 2019). This is the algorithm the
// HUNL blueprint will use: per iteration and per traverser, sample chance and
// opponent actions, explore every traverser action.
//
// Linear weighting: iteration t contributes with weight min(t, linear_weight_cap).
// A cap of 0 disables weighting (plain MCCFR). Once t passes the cap the weight
// stays constant, which matches Pluribus switching off discounting after its
// warm-up period.

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "cfr/tables.h"
#include "core/game.h"

namespace poker2 {

struct MccfrOptions {
  uint64_t seed = 1;
  int64_t linear_weight_cap = INT64_MAX;
};

template <class Game>
class ExternalSamplingMccfr {
 public:
  using State = typename Game::State;

  explicit ExternalSamplingMccfr(MccfrOptions options = {})
      : options_(options), rng_(options.seed) {}

  void iterate() {
    ++iteration_;
    weight_ = options_.linear_weight_cap <= 0
                  ? 1.0
                  : static_cast<double>(std::min(iteration_, options_.linear_weight_cap));
    for (int player = 0; player < 2; ++player) traverse(Game::initial_state(), player);
  }

  int64_t iteration() const { return iteration_; }
  size_t num_infosets() const { return table_.size(); }
  TabularPolicy average_policy() const { return table_.average_policy(); }

 private:
  // Returns a sampled estimate of the traverser's counterfactual value.
  double traverse(State state, int traverser) {
    std::vector<Outcome> outcomes;
    while (state.is_chance()) {
      state.chance_outcomes(outcomes);
      state.apply(outcomes[sample_index(outcomes)].action);
    }
    if (state.is_terminal()) return state.utility(traverser);

    std::vector<int> actions;
    state.legal_actions(actions);
    const size_t n = actions.size();
    InfosetNode& node = table_.get(state.infoset_key(), n);
    std::vector<double> sigma;
    regret_matching(node.regret, sigma);

    if (state.current_player() != traverser) {
      // Opponent reach is sampled, so accumulating sigma here gives an
      // unbiased average-strategy update ("simple" averaging).
      for (size_t i = 0; i < n; ++i) node.strategy_sum[i] += weight_ * sigma[i];
      state.apply(actions[sample_index(sigma)]);
      return traverse(std::move(state), traverser);
    }

    std::vector<double> action_values(n);
    double node_value = 0.0;
    for (size_t i = 0; i < n; ++i) {
      State child = state;
      child.apply(actions[i]);
      action_values[i] = traverse(std::move(child), traverser);
      node_value += sigma[i] * action_values[i];
    }
    for (size_t i = 0; i < n; ++i) node.regret[i] += weight_ * (action_values[i] - node_value);
    return node_value;
  }

  size_t sample_index(const std::vector<double>& probs) {
    double r = uniform_(rng_);
    for (size_t i = 0; i + 1 < probs.size(); ++i) {
      r -= probs[i];
      if (r < 0.0) return i;
    }
    return probs.size() - 1;
  }

  size_t sample_index(const std::vector<Outcome>& outcomes) {
    double r = uniform_(rng_);
    for (size_t i = 0; i + 1 < outcomes.size(); ++i) {
      r -= outcomes[i].prob;
      if (r < 0.0) return i;
    }
    return outcomes.size() - 1;
  }

  MccfrOptions options_;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
  InfosetTable table_;
  int64_t iteration_ = 0;
  double weight_ = 1.0;
};

}  // namespace poker2
