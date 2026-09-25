#pragma once

// Exact exploitability for games whose strategies never depend on the board.
//
// If neither player's strategy depends on the board, a showdown's expected
// value depends only on the two hole-card combos' all-in equity. Exact best
// responses can then be computed over all 1326 x 1225 card-compatible combo
// pairs with an all-in equity matrix (estimated by Monte Carlo).
//
//   PushFoldEvaluator:  the push/fold game (ActionAbstraction::push_fold).
//   board_blind_exploitability():  any betting tree, via range vectors, for
//     CardAbstraction::preflop_classes_only() (bucket = preflop class on every
//     street). The best responder gets the same information as the strategy
//     (preflop class and betting history), so this is exploitability within the
//     abstract game and must go to zero as training converges. The abstract game
//     must be perfect-recall: with one postflop bucket, players forget their hand
//     after the flop, CFR loses its guarantee, and a bottom-up best response is no
//     longer exact (measured exploitability rose 50 -> 118 mbb/hand that way).

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "blueprint/betting_tree.h"
#include "blueprint/strategy.h"
#include "core/parallel.h"
#include "holdem/evaluator.h"
#include "holdem/isomorphism.h"

namespace poker2::blueprint {

// All-in equity between every pair of hole-card combos.
class AllInEquity {
 public:
  static constexpr int kCombos = 1326;

  AllInEquity(int boards_per_pair, uint64_t seed, int threads) : equity_(kCombos * kCombos, 0.5f) {
    int i = 0;
    for (holdem::Card a = 0; a < 52; ++a)
      for (holdem::Card b = a + 1; b < 52; ++b) combo_[i++] = {a, b};
    const holdem::HandIndexer preflop({2});
    for (int c = 0; c < kCombos; ++c) class_[c] = static_cast<int>(preflop.index(combo_[c].data()));

    parallel_for(kCombos, threads, 8, [&](uint64_t begin, uint64_t end, int) {
      for (uint64_t x = begin; x < end; ++x) {
        std::mt19937_64 rng(seed * 1315423911ull + x);
        for (uint64_t y = x + 1; y < kCombos; ++y) {
          if (overlap(x, y)) continue;
          holdem::Card deck[48];
          int n = 0;
          for (holdem::Card c = 0; c < 52; ++c) {
            if (c != combo_[x][0] && c != combo_[x][1] && c != combo_[y][0] && c != combo_[y][1]) {
              deck[n++] = c;
            }
          }
          uint32_t score = 0;
          for (int k = 0; k < boards_per_pair; ++k) {
            for (int j = 0; j < 5; ++j) std::swap(deck[j], deck[j + rng() % (n - j)]);
            const holdem::Card me[7] = {combo_[x][0], combo_[x][1], deck[0], deck[1], deck[2], deck[3], deck[4]};
            const holdem::Card op[7] = {combo_[y][0], combo_[y][1], deck[0], deck[1], deck[2], deck[3], deck[4]};
            const uint32_t a = holdem::evaluate(me, 7), b = holdem::evaluate(op, 7);
            score += a > b ? 2 : (a == b ? 1 : 0);
          }
          const float eq = score / (2.0f * boards_per_pair);
          equity_[x * kCombos + y] = eq;
          equity_[y * kCombos + x] = 1.0f - eq;
        }
      }
    });
  }

  bool overlap(uint64_t x, uint64_t y) const {
    return combo_[x][0] == combo_[y][0] || combo_[x][0] == combo_[y][1] ||
           combo_[x][1] == combo_[y][0] || combo_[x][1] == combo_[y][1];
  }
  float equity(int x, int y) const { return equity_[static_cast<size_t>(x) * kCombos + y]; }
  holdem::Card card(int combo, int k) const { return combo_[combo][k]; }
  int preflop_class(int combo) const { return class_[combo]; }

 private:
  std::array<std::array<holdem::Card, 2>, kCombos> combo_;
  std::array<int, kCombos> class_;
  std::vector<float> equity_;
};

struct PushFoldReport {
  double br_small_blind;  // chips per hand the small blind wins by best-responding
  double br_big_blind;    // same for the big blind
  double value_small_blind;
  double exploitability() const { return (br_small_blind + br_big_blind) / 2.0; }
};

class PushFoldEvaluator {
 public:
  PushFoldEvaluator(int stack, int small_blind, int big_blind, int boards_per_pair, uint64_t seed,
                    int threads)
      : stack_(stack), small_blind_(small_blind), big_blind_(big_blind),
        eq_(boards_per_pair, seed, threads) {}

  // push[c], call[c]: probabilities per preflop class.
  PushFoldReport evaluate(const std::array<double, 169>& push, const std::array<double, 169>& call) const {
    constexpr int kN = AllInEquity::kCombos;
    PushFoldReport r{};
    const double s = stack_;
    for (int i = 0; i < kN; ++i) {
      double ev_push = 0.0;
      int n = 0;
      for (int j = 0; j < kN; ++j) {
        if (eq_.overlap(i, j)) continue;
        const double c = call[eq_.preflop_class(j)];
        ev_push += (1.0 - c) * big_blind_ + c * s * (2.0 * eq_.equity(i, j) - 1.0);
        ++n;
      }
      ev_push /= n;
      const double ev_fold = -small_blind_;
      const double p = push[eq_.preflop_class(i)];
      r.br_small_blind += std::max(ev_push, ev_fold);
      r.value_small_blind += p * ev_push + (1.0 - p) * ev_fold;
    }
    r.br_small_blind /= kN;
    r.value_small_blind /= kN;

    for (int j = 0; j < kN; ++j) {
      double walk = 0.0, call_ev = 0.0, fold_ev = 0.0;
      int n = 0;
      for (int i = 0; i < kN; ++i) {
        if (eq_.overlap(i, j)) continue;
        const double p = push[eq_.preflop_class(i)];
        walk += (1.0 - p) * small_blind_;
        call_ev += p * s * (2.0 * eq_.equity(j, i) - 1.0);
        fold_ev += p * -static_cast<double>(big_blind_);
        ++n;
      }
      r.br_big_blind += (walk + std::max(call_ev, fold_ev)) / n;
    }
    r.br_big_blind /= kN;
    return r;
  }

 private:
  int stack_, small_blind_, big_blind_;
  AllInEquity eq_;
};

// ---------------------------------------------------------------------------
// Board-blind games on any betting tree.

namespace detail {

using Range = std::vector<double>;  // weight per combo

// Best-response values for `br` (per combo, not yet normalized) against the
// opponent's fixed strategy, given the opponent's reach per combo.
inline Range br_values(const BettingTree& tree, const StrategyTables& tables, const AllInEquity& eq,
                       int32_t ref, int br, const Range& opp_reach) {
  constexpr int kN = AllInEquity::kCombos;
  Range v(kN, 0.0);
  if (ref < 0) {
    const TerminalNode& t = tree.terminal(ref);
    const int opp = 1 - br;
    if (t.folder >= 0) {
      // Compatible reach = total - combos sharing either card (+ the combo itself,
      // subtracted twice).
      const double payoff = t.folder == br ? -t.contrib[br] : t.contrib[opp];
      std::array<double, 52> card_sum{};
      double total = 0.0;
      for (int j = 0; j < kN; ++j) {
        card_sum[eq.card(j, 0)] += opp_reach[j];
        card_sum[eq.card(j, 1)] += opp_reach[j];
        total += opp_reach[j];
      }
      for (int i = 0; i < kN; ++i) {
        v[i] = (total - card_sum[eq.card(i, 0)] - card_sum[eq.card(i, 1)] + opp_reach[i]) * payoff;
      }
      return v;
    }
    const double stake = t.contrib[br];  // contributions are equal at showdown
    parallel_ranges(kN, 8, [&](uint64_t begin, uint64_t end, int) {
      for (uint64_t i = begin; i < end; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kN; ++j) {
          if (opp_reach[j] > 0.0 && !eq.overlap(i, j)) sum += opp_reach[j] * (2.0 * eq.equity(i, j) - 1.0);
        }
        v[i] = sum * stake;
      }
    });
    return v;
  }

  const TreeNode& n = tree.node(static_cast<uint32_t>(ref));
  float sigma[BettingTree::kMaxActions];
  if (n.player != br) {
    // Opponent acts: split its reach by its strategy (bucket = preflop class).
    std::vector<Range> child_reach(n.num_actions, Range(kN, 0.0));
    for (int j = 0; j < kN; ++j) {
      if (opp_reach[j] <= 0.0) continue;
      tables.play_strategy(n, eq.preflop_class(j), sigma);
      for (int a = 0; a < n.num_actions; ++a) child_reach[a][j] = opp_reach[j] * sigma[a];
    }
    for (int a = 0; a < n.num_actions; ++a) {
      const Range child = br_values(tree, tables, eq, tree.child(n, a), br, child_reach[a]);
      for (int i = 0; i < kN; ++i) v[i] += child[i];
    }
    return v;
  }

  // Best responder acts: one choice per information set (its bucket).
  std::vector<Range> child_values(n.num_actions);
  for (int a = 0; a < n.num_actions; ++a) {
    child_values[a] = br_values(tree, tables, eq, tree.child(n, a), br, opp_reach);
  }
  std::vector<std::vector<double>> bucket_total(169, std::vector<double>(n.num_actions, 0.0));
  for (int i = 0; i < kN; ++i) {
    for (int a = 0; a < n.num_actions; ++a) bucket_total[eq.preflop_class(i)][a] += child_values[a][i];
  }
  std::vector<int> best(169, 0);
  for (int b = 0; b < 169; ++b) {
    for (int a = 1; a < n.num_actions; ++a) {
      if (bucket_total[b][a] > bucket_total[b][best[b]] + 1e-9) best[b] = a;
    }
  }
  for (int i = 0; i < kN; ++i) {
    v[i] = child_values[best[eq.preflop_class(i)]][i];
  }
  return v;
}

}  // namespace detail

struct BoardBlindReport {
  double br_value[2];  // chips per hand each seat wins by best-responding
  double exploitability() const { return (br_value[0] + br_value[1]) / 2.0; }
};

// Requires tables trained with CardAbstraction::preflop_classes_only().
inline BoardBlindReport board_blind_exploitability(const BettingTree& tree, const StrategyTables& tables,
                                                   const AllInEquity& eq) {
  constexpr int kN = AllInEquity::kCombos;
  BoardBlindReport r{};
  for (int br = 0; br < 2; ++br) {
    const detail::Range values = detail::br_values(tree, tables, eq, 0, br, detail::Range(kN, 1.0));
    double total = 0.0;
    for (double x : values) total += x;
    r.br_value[br] = total / (static_cast<double>(kN) * 1225.0);  // 1225 = C(50, 2) compatible combos
  }
  return r;
}

}  // namespace poker2::blueprint
