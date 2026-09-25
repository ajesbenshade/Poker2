// Blueprint: betting tree, trainer, checkpoints, and evaluation.

#include <array>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "abstraction/card_abstraction.h"
#include "blueprint/betting_tree.h"
#include "blueprint/evaluation.h"
#include "blueprint/push_fold.h"
#include "blueprint/strategy.h"
#include "blueprint/trainer.h"
#include "test_framework.h"

using namespace poker2;
using namespace poker2::blueprint;
using namespace poker2::testing;
using holdem::Card;

namespace {

// Preflop buckets are the 169 classes; later streets get one bucket each.
abstraction::CardAbstraction coarse_abstraction(int max_street) {
  abstraction::CardAbstraction a;
  std::vector<uint16_t> identity(169);
  std::iota(identity.begin(), identity.end(), 0);
  a.set_table(0, identity, 169);
  for (int s = 1; s <= max_street; ++s) {
    a.set_table(s, std::vector<uint16_t>(a.indexer(s).size(), 0), 1);
  }
  return a;
}

TableLayout coarse_layout() {
  TableLayout layout;
  layout.buckets = {169, 1, 1, 1};
  layout.store_average = {true, true, true, true};
  return layout;
}

holdem::GameConfig stack_of(int chips) {
  holdem::GameConfig c;
  c.stack = chips;
  return c;
}

// Small postflop game: 10bb stacks, pot-size bets, two raises per street.
holdem::ActionAbstraction small_abstraction() {
  holdem::ActionAbstraction a;
  for (int s = 0; s < 4; ++s) a.raise_sizes[s] = {{1.0}};
  a.max_raises_per_street = 2;
  return a;
}

std::array<double, 169> strategy_by_class(const BettingTree& tree, const StrategyTables& tables,
                                          uint32_t node_id, int action) {
  std::array<double, 169> out{};
  const TreeNode& n = tree.node(node_id);
  float sigma[BettingTree::kMaxActions];
  for (int c = 0; c < 169; ++c) {
    tables.play_strategy(n, c, sigma);
    out[c] = sigma[action];
  }
  return out;
}

}  // namespace

TEST("blueprint tree: matches the abstraction's tree statistics") {
  const holdem::ActionAbstraction abs = holdem::ActionAbstraction::blueprint();
  const BettingTree tree(abs);
  holdem::TreeStats stats;
  holdem::count_tree(holdem::HunlState(), abs, stats);
  uint64_t nodes = 0;
  for (int s = 0; s < 4; ++s) {
    check_eq(tree.street_nodes(s), stats.decision_nodes[s], "decision nodes on street " + std::to_string(s));
    check_eq(tree.street_slots(s), stats.action_slots[s], "action slots on street " + std::to_string(s));
    nodes += stats.decision_nodes[s];
  }
  check_eq(tree.num_nodes(), nodes, "total decision nodes");
  check_eq(tree.num_terminals(), stats.terminal_nodes, "terminal nodes");
}

TEST("blueprint tree: children replay the stored actions") {
  const BettingTree tree(holdem::ActionAbstraction::blueprint());
  std::mt19937_64 rng(4);
  int failures = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const uint32_t id = static_cast<uint32_t>(rng() % tree.num_nodes());
    const TreeNode& n = tree.node(id);
    const int a = static_cast<int>(rng() % n.num_actions);
    holdem::HunlState s = tree.state(id);
    failures += s.street() != n.street || s.to_act() != n.player;
    s.apply(tree.action(n, a));
    const int32_t child = tree.child(n, a);
    if (child >= 0) {
      failures += !(tree.state(static_cast<uint32_t>(child)) == s);
    } else {
      const TerminalNode& t = tree.terminal(child);
      failures += !s.is_terminal() || t.contrib[0] != s.contribution(0) ||
                  t.contrib[1] != s.contribution(1) || t.folder != s.folder();
    }
  }
  check_eq(failures, 0, "replay mismatches");
}

TEST("blueprint tree: push/fold shape") {
  const BettingTree tree(holdem::ActionAbstraction::push_fold(), stack_of(1000));
  check_eq(tree.num_nodes(), size_t{2}, "two decision nodes");
  check_eq(tree.num_terminals(), size_t{3}, "fold, fold to push, call");
  check_eq(static_cast<int>(tree.node(0).num_actions), 2, "small blind: fold or all-in");
  check(tree.action(tree.node(0), 1) == holdem::Action::raise_to(1000), "the raise is all-in");
}

TEST("blueprint trainer: terminal values match the rules engine") {
  const BettingTree tree(holdem::ActionAbstraction::blueprint());
  const abstraction::CardAbstraction cards = coarse_abstraction(0);
  std::mt19937_64 rng(8);
  int failures = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const holdem::Deal deal = holdem::sample_deal(rng);
    const DealView v = view_deal(deal, cards, 0);
    holdem::HunlState s;
    int32_t ref = 0;
    while (ref >= 0) {
      const TreeNode& n = tree.node(static_cast<uint32_t>(ref));
      const int a = static_cast<int>(rng() % n.num_actions);
      s.apply(tree.action(n, a));
      ref = tree.child(n, a);
    }
    for (int seat = 0; seat < 2; ++seat) {
      failures += terminal_value(tree.terminal(ref), v, seat) != s.payoff(seat, deal);
    }
  }
  check_eq(failures, 0, "payoff mismatches");
}

TEST("blueprint trainer: push/fold converges to equilibrium (exact exploitability)") {
  const holdem::GameConfig config = stack_of(1000);  // 10 big blinds
  const BettingTree tree(holdem::ActionAbstraction::push_fold(), config);
  const abstraction::CardAbstraction cards = coarse_abstraction(0);
  StrategyTables tables(tree, coarse_layout());
  TrainerOptions opt;
  opt.threads = 8;
  BlueprintTrainer trainer(tree, cards, tables, opt);
  const PushFoldEvaluator judge(config.stack, config.small_blind, config.big_blind, 400, 3, 16);

  auto exploitability_mbb = [&] {
    const PushFoldReport r = judge.evaluate(strategy_by_class(tree, tables, 0, 1),
                                            strategy_by_class(tree, tables, 1, 1));
    return r.exploitability() / config.big_blind * 1000.0;
  };
  trainer.run(20000, 0);
  const double early = exploitability_mbb();
  for (int epoch = 1; epoch <= 40; ++epoch) trainer.run(100000, epoch);
  const double late = exploitability_mbb();
  std::printf("    push/fold exploitability: %.1f mbb/hand after 20K iterations, %.2f after 4.02M\n",
              early, late);
  check(late < early / 5.0, "exploitability falls at least 5x");
  check_below(late, 5.0, "exploitability in mbb/hand after 4M iterations");

  // Sanity: AA always pushes and always calls; 72o never calls a 10bb shove.
  const auto push = strategy_by_class(tree, tables, 0, 1);
  const auto call = strategy_by_class(tree, tables, 1, 1);
  const holdem::HandIndexer preflop({2});
  const auto aa = holdem::parse_cards("AsAh"), seven_deuce = holdem::parse_cards("7s2h");
  check(push[preflop.index(aa.data())] > 0.99, "AA pushes");
  check(call[preflop.index(aa.data())] > 0.99, "AA calls");
  check(call[preflop.index(seven_deuce.data())] < 0.01, "72o folds to a shove");
}

TEST("blueprint trainer: single-threaded runs are reproducible") {
  const BettingTree tree(small_abstraction(), stack_of(1000));
  const abstraction::CardAbstraction cards = coarse_abstraction(3);
  StrategyTables a(tree, coarse_layout()), b(tree, coarse_layout());
  BlueprintTrainer ta(tree, cards, a, TrainerOptions{}), tb(tree, cards, b, TrainerOptions{});
  ta.run(5000, 0);
  tb.run(5000, 0);
  check(a == b, "identical tables for the same seed");
}

TEST("blueprint strategy: checkpoint round trip and mismatch detection") {
  const BettingTree tree(small_abstraction(), stack_of(1000));
  const abstraction::CardAbstraction cards = coarse_abstraction(3);
  StrategyTables tables(tree, coarse_layout());
  BlueprintTrainer trainer(tree, cards, tables, TrainerOptions{});
  trainer.run(3000, 0);
  const std::string file = "/tmp/poker2_checkpoint_test.bin";
  tables.save(file, 3000);
  StrategyTables loaded(tree, coarse_layout());
  check_eq(loaded.load(file), uint64_t{3000}, "iteration count restored");
  check(loaded == tables, "tables restored exactly");

  TableLayout other = coarse_layout();
  other.buckets[1] = 2;
  StrategyTables wrong(tree, other);
  bool threw = false;
  try {
    wrong.load(file);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "checkpoint for a different layout is rejected");
  std::remove(file.c_str());
}

TEST("blueprint trainer: board-blind postflop game converges (exact exploitability)") {
  // Bucket = preflop class on every street: perfect recall, board-blind, so the
  // exact best response over all combo pairs applies. Four streets, pot-size
  // bets, two raises per street, all-ins.
  const holdem::GameConfig config = stack_of(1000);
  const BettingTree tree(small_abstraction(), config);
  const abstraction::CardAbstraction cards = abstraction::CardAbstraction::preflop_classes_only();
  TableLayout layout;
  layout.buckets = {169, 169, 169, 169};
  StrategyTables tables(tree, layout);
  TrainerOptions opt;
  opt.threads = 16;
  BlueprintTrainer trainer(tree, cards, tables, opt);
  const AllInEquity eq(400, 3, 16);
  auto exploitability_mbb = [&] {
    return board_blind_exploitability(tree, tables, eq).exploitability() / config.big_blind * 1000.0;
  };
  trainer.run(100000, 0);
  const double early = exploitability_mbb();
  for (int epoch = 1; epoch <= 39; ++epoch) trainer.run(100000, epoch);
  const double late = exploitability_mbb();
  std::printf("    board-blind exploitability: %.1f mbb/hand after 100K iterations, %.1f after 4M\n",
              early, late);
  // Measured: ~299 at 100K, ~33 at 4M, ~11 at 16M, ~1.3 at 256M.
  check(late < early / 5.0, "exploitability falls at least 5x");
  check_below(late, 45.0, "exploitability in mbb/hand after 4M iterations");
}

TEST("blueprint evaluation: LBR and head-to-head against baseline agents") {
  const holdem::GameConfig config = stack_of(1000);
  const BettingTree tree(small_abstraction(), config);
  const abstraction::CardAbstraction cards = coarse_abstraction(3);
  StrategyTables tables(tree, coarse_layout());
  LbrOptions lbr;
  lbr.hands = 40000;
  lbr.threads = 16;
  const MatchResult untrained = local_best_response(tree, tables, cards, lbr);

  TrainerOptions opt;
  opt.threads = 16;
  BlueprintTrainer trainer(tree, cards, tables, opt);
  for (int epoch = 0; epoch < 10; ++epoch) trainer.run(100000, epoch);
  const MatchResult trained = local_best_response(tree, tables, cards, lbr);
  std::printf("    LBR: untrained %.0f +/- %.0f mbb/hand, trained %.0f +/- %.0f\n",
              untrained.mbb_per_hand, untrained.stderr_mbb, trained.mbb_per_hand, trained.stderr_mbb);
  check(untrained.mbb_per_hand > 5 * untrained.stderr_mbb, "LBR clearly beats the uniform strategy");
  // No "training lowers LBR" check: with one postflop bucket this game has
  // imperfect recall, and LBR is a heuristic bound; convergence is checked
  // exactly by the board-blind test instead.

  const MatchResult vs_random =
      head_to_head(tree, cards, blueprint_agent(tables), random_agent(), 40000, 5, 16);
  const MatchResult vs_caller =
      head_to_head(tree, cards, blueprint_agent(tables), calling_agent(tree), 40000, 6, 16);
  std::printf("    head-to-head: vs random %.0f +/- %.0f, vs always-call %.0f +/- %.0f mbb/hand\n",
              vs_random.mbb_per_hand, vs_random.stderr_mbb, vs_caller.mbb_per_hand, vs_caller.stderr_mbb);
  check(vs_random.mbb_per_hand > 3 * vs_random.stderr_mbb, "beats the random agent");
  check(vs_caller.mbb_per_hand > 3 * vs_caller.stderr_mbb, "beats the always-call agent");
}
