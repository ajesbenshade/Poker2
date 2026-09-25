#pragma once

// Maps a betting state to the small set of actions the blueprint considers.
//
// Raise sizes are fractions of the pot after calling: raising "1.0" means
// calling and then raising by the resulting pot. Sizes below the legal
// minimum snap up to the minimum raise, sizes at or above the stack become
// all-in, and duplicate amounts are dropped.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "holdem/game.h"

namespace poker2::holdem {

struct ActionAbstraction {
  // raise_sizes[street][level]: pot fractions for the (level + 1)-th raise on that
  // street. Levels past the last entry reuse the last entry.
  std::array<std::vector<std::vector<double>>, 4> raise_sizes;
  int max_raises_per_street = 3;  // no raising at all once reached
  bool allow_all_in = true;       // offer all-in whenever raising is allowed
  bool allow_open_limp = true;    // small blind may just call as its first action

  // Small blind folds or goes all-in; big blind folds or calls. Exact
  // exploitability is computable for this game, which validates the trainer.
  static ActionAbstraction push_fold() {
    ActionAbstraction a;
    a.allow_open_limp = false;
    a.max_raises_per_street = 1;
    return a;
  }

  // First proposal for the blueprint; tune with poker2_tree before training.
  static ActionAbstraction blueprint() {
    ActionAbstraction a;
    a.raise_sizes[kPreflop] = {{0.5, 0.75, 1.0, 1.5, 2.5}, {0.5, 1.0, 1.5}, {0.5, 1.0}};
    a.raise_sizes[kFlop] = {{0.33, 0.5, 0.75, 1.0, 1.5}, {0.5, 1.0}, {1.0}};
    a.raise_sizes[kTurn] = {{0.5, 0.75, 1.0, 1.5}, {0.5, 1.0}, {1.0}};
    a.raise_sizes[kRiver] = {{0.33, 0.5, 0.75, 1.0, 1.5}, {0.5, 1.0}, {1.0}};
    return a;
  }

  int32_t raise_to_for_fraction(const HunlState& s, double fraction) const {
    const int opp = 1 - s.to_act();
    const double pot_after_call = static_cast<double>(s.pot() + s.to_call());
    return s.street_contribution(opp) + static_cast<int32_t>(std::lround(fraction * pot_after_call));
  }

  // Actions in a fixed order: fold (if facing a bet), check/call, raises ascending.
  void actions(const HunlState& s, std::vector<Action>& out) const {
    out.clear();
    if (s.is_terminal()) return;
    if (s.can_fold()) out.push_back(Action::fold());
    const bool open_limp = s.street() == kPreflop && s.to_act() == 0 && s.raises_this_street() == 0;
    if (allow_open_limp || !open_limp) out.push_back(Action::check_call());
    if (!s.can_raise() || s.raises_this_street() >= max_raises_per_street) return;

    const auto& levels = raise_sizes[s.street()];
    const int32_t lo = s.min_raise_to(), hi = s.max_raise_to();
    std::vector<int32_t> amounts;
    if (!levels.empty()) {
      const size_t level = std::min<size_t>(s.raises_this_street(), levels.size() - 1);
      for (double f : levels[level]) {
        amounts.push_back(std::clamp(raise_to_for_fraction(s, f), lo, hi));
      }
    }
    if (allow_all_in) amounts.push_back(hi);
    std::sort(amounts.begin(), amounts.end());
    amounts.erase(std::unique(amounts.begin(), amounts.end()), amounts.end());
    for (int32_t amount : amounts) out.push_back(Action::raise_to(amount));
  }
};

struct TreeStats {
  // Per street: decision nodes and total actions over those nodes.
  std::array<uint64_t, 4> decision_nodes{};
  std::array<uint64_t, 4> action_slots{};
  uint64_t terminal_nodes = 0;
};

// Walks the full abstract betting tree.
inline void count_tree(const HunlState& s, const ActionAbstraction& abstraction, TreeStats& stats) {
  if (s.is_terminal()) {
    ++stats.terminal_nodes;
    return;
  }
  std::vector<Action> actions;
  abstraction.actions(s, actions);
  ++stats.decision_nodes[s.street()];
  stats.action_slots[s.street()] += actions.size();
  for (const Action& a : actions) {
    HunlState child = s;
    child.apply(a);
    count_tree(child, abstraction, stats);
  }
}

}  // namespace poker2::holdem
