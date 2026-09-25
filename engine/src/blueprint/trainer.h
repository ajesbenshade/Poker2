#pragma once

// Multithreaded external-sampling MCCFR for the HUNL blueprint.
//
// One iteration samples a deal, looks up both players' buckets, and traverses
// the tree once per player: every action of the traverser is explored, chance
// and the opponent are sampled. This is the algorithm validated on Leduc
// (cfr/mccfr.h), specialized to dense tables:
//   - regrets are floored at `regret_floor` so pruned actions can recover;
//   - with pruning on, traverser actions with regret below `prune_threshold`
//     are skipped with probability `prune_probability`, except on the river
//     and for actions that end the hand (as in Pluribus);
//   - Linear CFR weighting is done by StrategyTables::scale between epochs.

#include <array>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "abstraction/card_abstraction.h"
#include "blueprint/betting_tree.h"
#include "blueprint/strategy.h"
#include "core/parallel.h"
#include "holdem/evaluator.h"

namespace poker2::blueprint {

struct TrainerOptions {
  uint64_t seed = 1;
  int threads = 1;
  float regret_floor = -3.1e8f;
  bool pruning = false;
  float prune_threshold = -3.0e8f;
  double prune_probability = 0.95;
};

// Everything about a sampled deal that the traversal needs.
struct DealView {
  std::array<std::array<uint16_t, 4>, 2> bucket;  // [seat][street]
  uint32_t strength[2];
};

inline DealView view_deal(const holdem::Deal& deal, const abstraction::CardAbstraction& cards,
                          int max_street) {
  DealView v{};
  for (int seat = 0; seat < 2; ++seat) {
    for (int s = 0; s <= max_street; ++s) {
      v.bucket[seat][s] = static_cast<uint16_t>(cards.bucket(s, deal.hole[seat], deal.board));
    }
    v.strength[seat] = holdem::evaluate_seat(deal, seat);
  }
  return v;
}

inline float terminal_value(const TerminalNode& t, const DealView& v, int player) {
  if (t.folder >= 0) return t.folder == player ? -t.contrib[player] : t.contrib[t.folder];
  if (v.strength[player] > v.strength[1 - player]) return t.contrib[1 - player];
  if (v.strength[player] < v.strength[1 - player]) return -t.contrib[player];
  return 0.0f;
}

class BlueprintTrainer {
 public:
  BlueprintTrainer(const BettingTree& tree, const abstraction::CardAbstraction& cards,
                   StrategyTables& tables, TrainerOptions options)
      : tree_(tree), cards_(cards), tables_(tables), options_(options) {}

  // Runs `iterations` iterations split across threads. `epoch` makes each
  // call's random streams distinct and reproducible.
  void run(uint64_t iterations, uint64_t epoch) {
    const int threads = std::max(1, options_.threads);
    parallel_ranges(iterations, threads, [&](uint64_t begin, uint64_t end, int t) {
      std::mt19937_64 rng(options_.seed * 0x9E3779B97F4A7C15ull + epoch * 1000003ull + t);
      std::uniform_real_distribution<double> uniform(0.0, 1.0);
      for (uint64_t i = begin; i < end; ++i) {
        const holdem::Deal deal = holdem::sample_deal(rng);
        const DealView view = view_deal(deal, cards_, tree_.max_street());
        for (int player = 0; player < 2; ++player) {
          const bool prune = options_.pruning && uniform(rng) < options_.prune_probability;
          traverse(0, player, view, rng, prune);
        }
      }
    });
  }

 private:
  float traverse(int32_t ref, int player, const DealView& v, std::mt19937_64& rng, bool prune) {
    if (ref < 0) return terminal_value(tree_.terminal(ref), v, player);
    const TreeNode& n = tree_.node(static_cast<uint32_t>(ref));
    const int acting = n.player;
    const int bucket = v.bucket[acting][n.street];
    const int count = n.num_actions;
    float* regret = tables_.regret(n, bucket);
    float sigma[BettingTree::kMaxActions];
    regret_matching(regret, count, sigma);

    if (acting != player) {
      tables_.add_average(n, bucket, sigma);
      double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      int chosen = count - 1;
      for (int a = 0; a < count - 1; ++a) {
        r -= sigma[a];
        if (r < 0.0) {
          chosen = a;
          break;
        }
      }
      return traverse(tree_.child(n, chosen), player, v, rng, prune);
    }

    float values[BettingTree::kMaxActions];
    bool explored[BettingTree::kMaxActions];
    float node_value = 0.0f;
    for (int a = 0; a < count; ++a) {
      const int32_t child = tree_.child(n, a);
      explored[a] = !(prune && n.street != holdem::kRiver && child >= 0 &&
                      load_relaxed(regret[a]) < options_.prune_threshold);
      if (!explored[a]) continue;  // sigma[a] is 0 here: its regret is negative
      values[a] = traverse(child, player, v, rng, prune);
      node_value += sigma[a] * values[a];
    }
    for (int a = 0; a < count; ++a) {
      if (!explored[a]) continue;
      const double updated = static_cast<double>(load_relaxed(regret[a])) + (values[a] - node_value);
      const double floored = updated > options_.regret_floor ? updated : options_.regret_floor;
      store_relaxed(regret[a], static_cast<float>(floored));
    }
    return node_value;
  }

  const BettingTree& tree_;
  const abstraction::CardAbstraction& cards_;
  StrategyTables& tables_;
  TrainerOptions options_;
};

}  // namespace poker2::blueprint
