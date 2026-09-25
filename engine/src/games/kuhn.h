#pragma once

// Kuhn poker: 3-card deck (J, Q, K), 1-chip ante, one bet of 1 chip.
// Equilibrium value for player 0 is -1/18.

#include <string>
#include <vector>

#include "core/game.h"

namespace poker2 {

class KuhnState {
 public:
  static constexpr int kPass = 0;
  static constexpr int kBet = 1;

  bool is_terminal() const {
    return history_ == "pp" || history_ == "bp" || history_ == "bb" || history_ == "pbp" ||
           history_ == "pbb";
  }

  bool is_chance() const { return cards_[1] < 0; }

  int current_player() const {
    return is_chance() ? kChancePlayer : static_cast<int>(history_.size() % 2);
  }

  void chance_outcomes(std::vector<Outcome>& out) const {
    out.clear();
    const int remaining = cards_[0] < 0 ? 3 : 2;
    for (int card = 0; card < 3; ++card) {
      if (card != cards_[0]) out.push_back({card, 1.0 / remaining});
    }
  }

  void legal_actions(std::vector<int>& out) const { out.assign({kPass, kBet}); }

  void apply(int action) {
    if (is_chance()) {
      (cards_[0] < 0 ? cards_[0] : cards_[1]) = action;
    } else {
      history_.push_back(action == kPass ? 'p' : 'b');
    }
  }

  double utility(int player) const {
    const double showdown = cards_[0] > cards_[1] ? 1.0 : -1.0;
    double u0 = 0.0;
    if (history_ == "pp") u0 = showdown;
    else if (history_ == "bb" || history_ == "pbb") u0 = 2.0 * showdown;
    else if (history_ == "bp") u0 = 1.0;
    else if (history_ == "pbp") u0 = -1.0;
    return player == 0 ? u0 : -u0;
  }

  std::string infoset_key() const {
    return std::to_string(cards_[current_player()]) + history_;
  }

  std::string history_string() const {
    return std::to_string(cards_[0]) + std::to_string(cards_[1]) + ":" + history_;
  }

 private:
  int cards_[2] = {-1, -1};
  std::string history_;
};

struct Kuhn {
  using State = KuhnState;
  static State initial_state() { return State(); }
  static const char* name() { return "kuhn"; }
};

}  // namespace poker2
