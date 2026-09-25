// HUNL betting rules, payoffs, action strings, and the action abstraction.

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "holdem/action_abstraction.h"
#include "holdem/game.h"
#include "test_framework.h"

using namespace poker2::holdem;
using namespace poker2::testing;

namespace {

bool throws_invalid(const std::string& history, const GameConfig& config = GameConfig()) {
  try {
    state_from_actions(history, config);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

Deal deal_from(const std::string& sb, const std::string& bb, const std::string& board) {
  const auto s = parse_cards(sb), b = parse_cards(bb), d = parse_cards(board);
  Deal deal;
  deal.hole[0][0] = s[0];
  deal.hole[0][1] = s[1];
  deal.hole[1][0] = b[0];
  deal.hole[1][1] = b[1];
  for (int i = 0; i < 5; ++i) deal.board[i] = d[i];
  return deal;
}

// A random legal action with a mix of min raises, all-ins, and in-between sizes.
Action random_action(const HunlState& s, std::mt19937_64& rng) {
  const int roll = static_cast<int>(rng() % 100);
  if (s.can_fold() && roll < 15) return Action::fold();
  if (s.can_raise() && roll >= 70) {
    const int32_t lo = s.min_raise_to(), hi = s.max_raise_to();
    const int pick = static_cast<int>(rng() % 3);
    if (pick == 0) return Action::raise_to(lo);
    if (pick == 1) return Action::raise_to(hi);
    return Action::raise_to(lo + static_cast<int32_t>(rng() % (hi - lo + 1)));
  }
  return Action::check_call();
}

}  // namespace

TEST("game: blinds, first actor, and opening raise limits") {
  const HunlState s;
  check_eq(s.pot(), 150, "pot after blinds");
  check_eq(s.to_act(), 0, "small blind acts first preflop");
  check_eq(s.to_call(), 50, "small blind owes 50");
  check_eq(s.min_raise_to(), 200, "minimum open is a raise to 2bb");
  check_eq(s.max_raise_to(), 20000, "maximum open is all-in");
  check(!s.is_legal(Action::raise_to(199)), "raise below minimum is illegal");
  check(!s.is_legal(Action::raise_to(20001)), "raise above stack is illegal");
}

TEST("game: street transitions and who acts") {
  HunlState s = state_from_actions("c");
  check_eq(s.to_act(), 1, "big blind gets the option after a limp");
  check(!s.can_fold(), "big blind cannot fold when nothing is owed");
  s = state_from_actions("ck");
  check_eq(s.street(), static_cast<int>(kFlop), "limp-check reaches the flop");
  check_eq(s.to_act(), 1, "big blind acts first postflop");
  check_eq(s.pot(), 200, "pot after limp-check");
  check_eq(s.min_raise_to(), 100, "minimum postflop bet is 1bb");
  s = state_from_actions("cb300c");
  check_eq(s.street(), static_cast<int>(kFlop), "big blind raise over a limp, then call");
  check_eq(s.pot(), 600, "pot after 3bb raise called");
  s = state_from_actions("b300c/kk/kk");
  check_eq(s.street(), static_cast<int>(kRiver), "check-check closes the flop and turn");
  s = state_from_actions("b300c/kk/kk/kk");
  check(s.is_showdown(), "river check-check goes to showdown");
}

TEST("game: minimum re-raise tracks the last raise increment") {
  check_eq(state_from_actions("b300").min_raise_to(), 500, "raise to 300 (+200) -> min 500");
  check_eq(state_from_actions("b300b900").min_raise_to(), 1500, "raise to 900 (+600) -> min 1500");
  check_eq(state_from_actions("b300c/b100").min_raise_to(), 200, "bet 100 -> min raise to 200");
  check_eq(state_from_actions("b300c/b450b1350").min_raise_to(), 2250, "900 increment carries");
}

TEST("game: all-ins, short all-ins, and payoffs") {
  HunlState s = state_from_actions("b20000");
  check(!s.can_raise(), "cannot raise an all-in");
  s = state_from_actions("b20000c");
  check(s.is_showdown(), "called all-in runs out to showdown");
  check_eq(s.pot(), 40000, "all-in pot");

  GameConfig shallow;
  shallow.stack = 1000;
  s = state_from_actions("b300b900", shallow);
  check_eq(s.min_raise_to(), 1000, "short stack's only raise is all-in");
  check_eq(s.max_raise_to(), 1000, "all-in amount");
  s = state_from_actions("b300b900b1000", shallow);
  check(!s.can_raise(), "no raising after a short all-in");
  check(s.can_fold(), "can still fold to a short all-in");

  const Deal deal = deal_from("AsAh", "KsKh", "2c3d7h9sJd");
  check_eq(state_from_actions("f").payoff(0, deal), -50, "small blind folds");
  check_eq(state_from_actions("b300f").payoff(0, deal), 100, "big blind folds to a raise");
  check_eq(state_from_actions("b300c/kk/kk/kk").payoff(0, deal), 300, "aces win at showdown");
  check_eq(state_from_actions("b300c/kk/kk/kk").payoff(1, deal), -300, "kings lose at showdown");
  const Deal chop = deal_from("2s3h", "4d5c", "AsKsQsJsTs");
  check_eq(state_from_actions("b20000c").payoff(0, chop), 0, "royal flush on board splits");
}

TEST("game: illegal action strings are rejected") {
  check(throws_invalid("k"), "small blind cannot check preflop");
  check(throws_invalid("cf"), "cannot fold when nothing is owed");
  check(throws_invalid("b150"), "raise below minimum");
  check(throws_invalid("b30000"), "raise above stack");
  check(throws_invalid("b20000b20000"), "raise after all-in");
  check(throws_invalid("b300c/c"), "call with nothing to call");
  check(throws_invalid("fk"), "action after the hand ended");
}

TEST("game: random playouts keep invariants and round-trip action strings") {
  std::mt19937_64 rng(5);
  int failures = 0;
  for (int hand = 0; hand < 200000; ++hand) {
    HunlState s;
    std::string history;
    int steps = 0;
    while (!s.is_terminal() && steps++ < 200) {
      apply_and_record(s, random_action(s, rng), history);
      failures += s.contribution(0) > s.config().stack || s.contribution(1) > s.config().stack;
    }
    if (!s.is_terminal()) {
      ++failures;
      continue;
    }
    if (s.is_showdown()) failures += s.contribution(0) != s.contribution(1);
    const Deal deal = sample_deal(rng);
    failures += s.payoff(0, deal) + s.payoff(1, deal) != 0;
    failures += !(state_from_actions(history) == s);
  }
  check_eq(failures, 0, "invariant or round-trip failures");
}

TEST("abstraction: opening actions for the blueprint") {
  const ActionAbstraction abs = ActionAbstraction::blueprint();
  std::vector<Action> actions;
  abs.actions(HunlState(), actions);
  // Pot after calling is 200: 0.5 -> 200, 0.75 -> 250, 1.0 -> 300, 1.5 -> 400, 2.5 -> 600.
  const std::vector<Action> expected = {Action::fold(),         Action::check_call(),
                                        Action::raise_to(200),  Action::raise_to(250),
                                        Action::raise_to(300),  Action::raise_to(400),
                                        Action::raise_to(600),  Action::raise_to(20000)};
  check(actions == expected, "fold, call, five open sizes, all-in");
}

TEST("abstraction: sizes snap to legal limits and dedupe") {
  ActionAbstraction abs;
  abs.raise_sizes[kPreflop] = {{0.1, 0.2, 150.0, 200.0}};  // two below min, two above stack
  std::vector<Action> actions;
  abs.actions(HunlState(), actions);
  const std::vector<Action> expected = {Action::fold(), Action::check_call(),
                                        Action::raise_to(200), Action::raise_to(20000)};
  check(actions == expected, "below-min snaps to min raise, above-stack becomes one all-in");

  abs.max_raises_per_street = 2;
  abs.actions(state_from_actions("b200b400"), actions);
  check(actions.size() == 2, "only fold and call once the raise cap is reached");
}

TEST("abstraction: every abstract action is legal in random abstract playouts") {
  const ActionAbstraction abs = ActionAbstraction::blueprint();
  std::mt19937_64 rng(17);
  std::vector<Action> actions;
  int illegal = 0;
  for (int hand = 0; hand < 100000; ++hand) {
    HunlState s;
    while (!s.is_terminal()) {
      abs.actions(s, actions);
      for (const Action& a : actions) illegal += !s.is_legal(a);
      s.apply(actions[rng() % actions.size()]);
    }
  }
  check_eq(illegal, 0, "illegal abstract actions");
}
