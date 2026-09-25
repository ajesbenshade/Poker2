#pragma once

// Heads-up no-limit hold'em betting rules, independent of cards.
//
// Defaults match Slumbot: 20,000-chip stacks, 50/100 blinds.
// Seat 0 posts the small blind, is the button, and acts first preflop.
// Seat 1 posts the big blind and acts first on every later street.
//
// Raise amounts are "raise to" totals for the current street, which is also
// how Slumbot action strings express them (b300 = raise to 300 this street).
// Both seats start with equal stacks, so a call is always affordable and there
// are never side pots; once a player is all-in the opponent can only call or fold.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "holdem/cards.h"
#include "holdem/evaluator.h"

namespace poker2::holdem {

struct GameConfig {
  int32_t stack = 20000;
  int32_t small_blind = 50;
  int32_t big_blind = 100;
};

enum class ActionType : uint8_t { kFold, kCheckCall, kRaise };

struct Action {
  ActionType type;
  int32_t amount = 0;  // kRaise only: street total after raising

  static Action fold() { return {ActionType::kFold, 0}; }
  static Action check_call() { return {ActionType::kCheckCall, 0}; }
  static Action raise_to(int32_t amount) { return {ActionType::kRaise, amount}; }
  bool operator==(const Action& o) const { return type == o.type && amount == o.amount; }
};

enum Street : int8_t { kPreflop = 0, kFlop = 1, kTurn = 2, kRiver = 3 };

class HunlState {
 public:
  explicit HunlState(const GameConfig& config = GameConfig()) : config_(config) {
    if (config.big_blind <= 0 || config.small_blind <= 0 || config.stack < config.big_blind) {
      throw std::invalid_argument("invalid game config");
    }
    contrib_[0] = street_contrib_[0] = config.small_blind;
    contrib_[1] = street_contrib_[1] = config.big_blind;
    last_raise_size_ = config.big_blind;
  }

  const GameConfig& config() const { return config_; }
  bool is_terminal() const { return finished_; }
  bool is_showdown() const { return finished_ && folder_ < 0; }
  int folder() const { return folder_; }
  int street() const { return street_; }
  int to_act() const { return to_act_; }
  int raises_this_street() const { return raises_; }
  int32_t contribution(int seat) const { return contrib_[seat]; }
  int32_t street_contribution(int seat) const { return street_contrib_[seat]; }
  int32_t pot() const { return contrib_[0] + contrib_[1]; }
  int32_t remaining(int seat) const { return config_.stack - contrib_[seat]; }

  int32_t to_call() const { return street_contrib_[1 - to_act_] - street_contrib_[to_act_]; }
  bool can_fold() const { return !finished_ && to_call() > 0; }
  bool can_raise() const {
    return !finished_ && remaining(to_act_) > to_call() && remaining(1 - to_act_) > 0;
  }
  // Smallest legal raise-to. A player without chips for a full raise may only go all-in.
  int32_t min_raise_to() const {
    const int32_t full = street_contrib_[1 - to_act_] + std::max(last_raise_size_, config_.big_blind);
    return std::min(full, max_raise_to());
  }
  int32_t max_raise_to() const { return street_contrib_[to_act_] + remaining(to_act_); }

  bool is_legal(const Action& a) const {
    if (finished_) return false;
    switch (a.type) {
      case ActionType::kFold: return can_fold();
      case ActionType::kCheckCall: return true;
      case ActionType::kRaise:
        return can_raise() && a.amount >= min_raise_to() && a.amount <= max_raise_to();
    }
    return false;
  }

  void apply(const Action& a) {
    if (!is_legal(a)) throw std::logic_error("illegal action");
    const int me = to_act_, opp = 1 - me;
    switch (a.type) {
      case ActionType::kFold:
        folder_ = static_cast<int8_t>(me);
        finished_ = true;
        return;
      case ActionType::kCheckCall: {
        const int32_t amount = to_call();
        contrib_[me] += amount;
        street_contrib_[me] += amount;
        acted_[me] = true;
        if (acted_[opp]) close_street();
        else to_act_ = static_cast<int8_t>(opp);
        return;
      }
      case ActionType::kRaise: {
        const int32_t increment = a.amount - street_contrib_[opp];
        if (increment >= last_raise_size_) last_raise_size_ = increment;  // short all-ins don't count
        contrib_[me] += a.amount - street_contrib_[me];
        street_contrib_[me] = a.amount;
        ++raises_;
        acted_[me] = true;
        acted_[opp] = false;
        to_act_ = static_cast<int8_t>(opp);
        return;
      }
    }
  }

  // Net chips won by `seat` at a terminal state.
  int32_t payoff(int seat, const Deal& deal) const {
    if (!finished_) throw std::logic_error("payoff of a non-terminal state");
    if (folder_ >= 0) return folder_ == seat ? -contrib_[seat] : contrib_[folder_];
    const uint32_t mine = evaluate_seat(deal, seat);
    const uint32_t theirs = evaluate_seat(deal, 1 - seat);
    if (mine > theirs) return contrib_[1 - seat];
    if (mine < theirs) return -contrib_[seat];
    return 0;
  }

  bool operator==(const HunlState& o) const {
    return contrib_[0] == o.contrib_[0] && contrib_[1] == o.contrib_[1] &&
           street_contrib_[0] == o.street_contrib_[0] &&
           street_contrib_[1] == o.street_contrib_[1] && last_raise_size_ == o.last_raise_size_ &&
           street_ == o.street_ && to_act_ == o.to_act_ && folder_ == o.folder_ &&
           raises_ == o.raises_ && acted_[0] == o.acted_[0] && acted_[1] == o.acted_[1] &&
           finished_ == o.finished_;
  }

 private:
  void close_street() {
    if (street_ == kRiver || remaining(0) == 0 || remaining(1) == 0) {
      finished_ = true;  // showdown, dealing out any remaining board cards
      return;
    }
    ++street_;
    street_contrib_[0] = street_contrib_[1] = 0;
    last_raise_size_ = config_.big_blind;
    raises_ = 0;
    acted_[0] = acted_[1] = false;
    to_act_ = 1;
  }

  GameConfig config_;
  int32_t contrib_[2];
  int32_t street_contrib_[2];
  int32_t last_raise_size_;
  int8_t street_ = kPreflop;
  int8_t to_act_ = 0;
  int8_t folder_ = -1;
  int8_t raises_ = 0;  // voluntary bets/raises this street; blinds don't count
  bool acted_[2] = {false, false};
  bool finished_ = false;
};

// Slumbot-style action strings: k = check, c = call, f = fold, bN = raise to N
// on this street, '/' between streets. Example: "b300c/kb450c/kk/b900f".
// TODO(day 27): confirm the exact format against the live Slumbot API,
// including what it sends after an all-in is called.

inline std::string action_token(const HunlState& state, const Action& a) {
  switch (a.type) {
    case ActionType::kFold: return "f";
    case ActionType::kCheckCall: return state.to_call() > 0 ? "c" : "k";
    case ActionType::kRaise: return "b" + std::to_string(a.amount);
  }
  return "?";
}

// Applies `a` and appends its token (plus '/' when a new street starts).
inline void apply_and_record(HunlState& state, const Action& a, std::string& history) {
  history += action_token(state, a);
  const int street = state.street();
  state.apply(a);
  if (!state.is_terminal() && state.street() != street) history += '/';
}

inline HunlState state_from_actions(const std::string& history,
                                    const GameConfig& config = GameConfig()) {
  HunlState state(config);
  size_t i = 0;
  while (i < history.size()) {
    const char c = history[i];
    if (c == '/') {
      ++i;
      continue;
    }
    Action a = Action::check_call();
    if (c == 'f') a = Action::fold();
    else if (c == 'b') {
      size_t end = i + 1;
      while (end < history.size() && history[end] >= '0' && history[end] <= '9') ++end;
      if (end == i + 1) throw std::invalid_argument("bet without amount in: " + history);
      a = Action::raise_to(std::stoi(history.substr(i + 1, end - i - 1)));
      i = end - 1;
    } else if ((c == 'k' && state.to_call() != 0) || (c == 'c' && state.to_call() == 0)) {
      throw std::invalid_argument("check/call mismatch in: " + history);
    } else if (c != 'k' && c != 'c') {
      throw std::invalid_argument("bad action character in: " + history);
    }
    if (!state.is_legal(a)) throw std::invalid_argument("illegal action in: " + history);
    state.apply(a);
    ++i;
  }
  return state;
}

}  // namespace poker2::holdem
