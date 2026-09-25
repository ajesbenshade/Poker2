#pragma once

// Leduc hold'em: 6-card deck (J, Q, K in two suits), 1-chip ante, one private
// card each, one public card dealt after the first betting round.
// Fixed bet sizes of 2 (round 1) and 4 (round 2), at most 2 raises per round,
// player 0 acts first in both rounds. A pair with the public card wins,
// otherwise the higher rank wins; equal ranks split.
//
// Suits never matter, so infoset keys use ranks only. That is lossless and
// gives the textbook 288 information sets.

#include <string>
#include <vector>

#include "core/game.h"

namespace poker2 {

class LeducState {
 public:
  static constexpr int kFold = 0;
  static constexpr int kCall = 1;  // check when nothing to call
  static constexpr int kRaise = 2;  // bet when nothing to call
  static constexpr int kNumCards = 6;
  static constexpr int kMaxRaisesPerRound = 2;

  static int rank(int card) { return card / 2; }

  bool is_terminal() const { return finished_; }

  bool is_chance() const {
    return !finished_ && (private_[1] < 0 || (round_ == 1 && public_ < 0));
  }

  int current_player() const { return is_chance() ? kChancePlayer : player_; }

  void chance_outcomes(std::vector<Outcome>& out) const {
    out.clear();
    int used = 0;
    for (int card : {private_[0], private_[1], public_}) used += card >= 0;
    const double prob = 1.0 / (kNumCards - used);
    for (int card = 0; card < kNumCards; ++card) {
      if (card != private_[0] && card != private_[1] && card != public_) {
        out.push_back({card, prob});
      }
    }
  }

  void legal_actions(std::vector<int>& out) const {
    out.clear();
    if (to_call() > 0) out.push_back(kFold);
    out.push_back(kCall);
    if (raises_ < kMaxRaisesPerRound) out.push_back(kRaise);
  }

  void apply(int action) {
    if (is_chance()) {
      if (private_[0] < 0) private_[0] = action;
      else if (private_[1] < 0) private_[1] = action;
      else public_ = action;
      return;
    }

    std::string& history = history_[round_];
    const int owed = to_call();
    switch (action) {
      case kFold:
        history.push_back('f');
        folder_ = player_;
        finished_ = true;
        return;
      case kCall: {
        // A call closes the round; so does a check that follows a check.
        const bool closes_round = owed > 0 || !history.empty();
        history.push_back('c');
        contrib_[player_] += owed;
        if (closes_round) end_round();
        else player_ = 1 - player_;
        return;
      }
      case kRaise:
        history.push_back('r');
        contrib_[player_] = contrib_[1 - player_] + (round_ == 0 ? 2 : 4);
        ++raises_;
        player_ = 1 - player_;
        return;
    }
  }

  double utility(int player) const {
    const int opponent = 1 - player;
    if (folder_ >= 0) return folder_ == player ? -contrib_[player] : contrib_[opponent];
    const int mine = strength(private_[player]);
    const int theirs = strength(private_[opponent]);
    if (mine > theirs) return contrib_[opponent];
    if (mine < theirs) return -contrib_[player];
    return 0.0;
  }

  std::string infoset_key() const {
    std::string key(1, static_cast<char>('0' + rank(private_[player_])));
    if (public_ >= 0) key.push_back(static_cast<char>('0' + rank(public_)));
    key.push_back(':');
    key += history_[0];
    key.push_back('/');
    key += history_[1];
    return key;
  }

  std::string history_string() const {
    std::string s;
    for (int card : {private_[0], private_[1], public_}) s += std::to_string(card) + ",";
    return s + history_[0] + "/" + history_[1];
  }

 private:
  int to_call() const { return contrib_[1 - player_] - contrib_[player_]; }

  int strength(int card) const {
    return rank(card) == rank(public_) ? 10 + rank(card) : rank(card);
  }

  void end_round() {
    if (round_ == 0) {
      round_ = 1;
      raises_ = 0;
      player_ = 0;
    } else {
      finished_ = true;
    }
  }

  int private_[2] = {-1, -1};
  int public_ = -1;
  int round_ = 0;
  int player_ = 0;
  int raises_ = 0;
  int contrib_[2] = {1, 1};
  int folder_ = -1;
  bool finished_ = false;
  std::string history_[2];
};

struct Leduc {
  using State = LeducState;
  static State initial_state() { return State(); }
  static const char* name() { return "leduc"; }
};

}  // namespace poker2
