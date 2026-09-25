#pragma once

// Exact evaluation for games small enough to enumerate: expected value of a
// profile, best-response value, and exploitability. Exploitability is the
// yardstick every solver change is judged by; average utility is not.

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cfr/tables.h"
#include "core/game.h"

namespace poker2 {

// Expected value for `player` when both seats follow `policy`.
template <class Game>
double expected_value(const TabularPolicy& policy, int player) {
  using State = typename Game::State;
  std::function<double(const State&)> value = [&](const State& state) -> double {
    if (state.is_terminal()) return state.utility(player);
    double total = 0.0;
    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      for (const Outcome& o : outcomes) {
        State child = state;
        child.apply(o.action);
        total += o.prob * value(child);
      }
      return total;
    }
    std::vector<int> actions;
    state.legal_actions(actions);
    const std::vector<double> sigma = policy.get(state.infoset_key(), actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
      if (sigma[i] == 0.0) continue;
      State child = state;
      child.apply(actions[i]);
      total += sigma[i] * value(child);
    }
    return total;
  };
  return value(Game::initial_state());
}

// Value `br_player` achieves by best-responding to the opponent's part of
// `policy`. Requires perfect recall: the best action at an infoset maximizes
// the opponent-and-chance-reach-weighted value summed over its histories, and
// deeper best-response choices never depend on shallower ones.
template <class Game>
class BestResponse {
 public:
  using State = typename Game::State;

  BestResponse(const TabularPolicy& policy, int br_player)
      : policy_(policy), br_player_(br_player) {
    collect(Game::initial_state(), 1.0);
  }

  double value() { return value(Game::initial_state()); }

  // Best-response policy for br_player (deterministic).
  TabularPolicy policy() {
    TabularPolicy out;
    for (const auto& [key, histories] : infosets_) {
      std::vector<int> actions;
      histories.front().first.legal_actions(actions);
      std::vector<double> probs(actions.size(), 0.0);
      probs[best_action_index(key)] = 1.0;
      out.set(key, std::move(probs));
    }
    return out;
  }

 private:
  void collect(const State& state, double reach) {
    if (state.is_terminal()) return;
    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      for (const Outcome& o : outcomes) {
        State child = state;
        child.apply(o.action);
        collect(child, reach * o.prob);
      }
      return;
    }
    std::vector<int> actions;
    state.legal_actions(actions);
    std::vector<double> sigma(actions.size(), 1.0);
    if (state.current_player() == br_player_) {
      infosets_[state.infoset_key()].emplace_back(state, reach);
    } else {
      sigma = policy_.get(state.infoset_key(), actions.size());
    }
    for (size_t i = 0; i < actions.size(); ++i) {
      State child = state;
      child.apply(actions[i]);
      collect(child, reach * sigma[i]);
    }
  }

  double value(const State& state) {
    if (state.is_terminal()) return state.utility(br_player_);
    const std::string history = state.history_string();
    if (auto it = value_cache_.find(history); it != value_cache_.end()) return it->second;

    double total = 0.0;
    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      for (const Outcome& o : outcomes) {
        State child = state;
        child.apply(o.action);
        total += o.prob * value(child);
      }
    } else {
      std::vector<int> actions;
      state.legal_actions(actions);
      if (state.current_player() == br_player_) {
        State child = state;
        child.apply(actions[best_action_index(state.infoset_key())]);
        total = value(child);
      } else {
        const std::vector<double> sigma = policy_.get(state.infoset_key(), actions.size());
        for (size_t i = 0; i < actions.size(); ++i) {
          if (sigma[i] == 0.0) continue;
          State child = state;
          child.apply(actions[i]);
          total += sigma[i] * value(child);
        }
      }
    }
    value_cache_.emplace(history, total);
    return total;
  }

  size_t best_action_index(const std::string& key) {
    if (auto it = best_action_cache_.find(key); it != best_action_cache_.end()) return it->second;
    const auto& histories = infosets_.at(key);
    std::vector<int> actions;
    histories.front().first.legal_actions(actions);
    size_t best = 0;
    double best_value = -1e300;
    for (size_t i = 0; i < actions.size(); ++i) {
      double total = 0.0;
      for (const auto& [state, reach] : histories) {
        State child = state;
        child.apply(actions[i]);
        total += reach * value(child);
      }
      if (total > best_value + 1e-12) {
        best_value = total;
        best = i;
      }
    }
    best_action_cache_.emplace(key, best);
    return best;
  }

  const TabularPolicy& policy_;
  int br_player_;
  std::unordered_map<std::string, std::vector<std::pair<State, double>>> infosets_;
  std::unordered_map<std::string, double> value_cache_;
  std::unordered_map<std::string, size_t> best_action_cache_;
};

struct ExploitabilityReport {
  double br_value[2];
  // Sum of what each player gains by deviating to a best response.
  // Equals br_value[0] + br_value[1] because the games are zero-sum.
  double nash_conv() const { return br_value[0] + br_value[1]; }
  // Average gain per seat, in chips per hand. 0 at a Nash equilibrium.
  double exploitability() const { return nash_conv() / 2.0; }
};

template <class Game>
ExploitabilityReport exploitability(const TabularPolicy& policy) {
  ExploitabilityReport report{};
  for (int player = 0; player < 2; ++player) {
    report.br_value[player] = BestResponse<Game>(policy, player).value();
  }
  return report;
}

// Every decision infoset in the game, for sanity checks.
template <class Game>
std::set<std::string> enumerate_infosets() {
  using State = typename Game::State;
  std::set<std::string> keys;
  std::function<void(const State&)> walk = [&](const State& state) {
    if (state.is_terminal()) return;
    std::vector<int> moves;
    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      for (const Outcome& o : outcomes) moves.push_back(o.action);
    } else {
      keys.insert(state.infoset_key());
      state.legal_actions(moves);
    }
    for (int move : moves) {
      State child = state;
      child.apply(move);
      walk(child);
    }
  };
  walk(Game::initial_state());
  return keys;
}

}  // namespace poker2
