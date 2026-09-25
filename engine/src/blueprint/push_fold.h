#pragma once

// Exact exploitability for the push-or-fold game (ActionAbstraction::push_fold):
// the small blind folds or goes all-in, the big blind folds or calls.
//
// Strategies are per preflop class (169). Best responses are computed exactly
// over all 1326 x 1225 card-compatible hand pairs, using an all-in equity
// matrix estimated by Monte Carlo. This gives an end-to-end check of the
// blueprint trainer on a game whose equilibrium can actually be measured.

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "core/parallel.h"
#include "holdem/evaluator.h"
#include "holdem/isomorphism.h"

namespace poker2::blueprint {

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
      : stack_(stack), small_blind_(small_blind), big_blind_(big_blind), equity_(1326 * 1326, 0.5f) {
    int i = 0;
    for (holdem::Card a = 0; a < 52; ++a)
      for (holdem::Card b = a + 1; b < 52; ++b) combo_[i++] = {a, b};
    const holdem::HandIndexer preflop({2});
    for (int c = 0; c < 1326; ++c) class_[c] = static_cast<int>(preflop.index(combo_[c].data()));

    parallel_for(1326, threads, 8, [&](uint64_t begin, uint64_t end, int) {
      for (uint64_t x = begin; x < end; ++x) {
        std::mt19937_64 rng(seed * 1315423911ull + x);
        for (uint64_t y = x + 1; y < 1326; ++y) {
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
          equity_[x * 1326 + y] = eq;
          equity_[y * 1326 + x] = 1.0f - eq;
        }
      }
    });
  }

  // push[c], call[c]: probabilities per preflop class.
  PushFoldReport evaluate(const std::array<double, 169>& push, const std::array<double, 169>& call) const {
    PushFoldReport r{};
    const double s = stack_;
    for (int i = 0; i < 1326; ++i) {
      double ev_push = 0.0;
      int n = 0;
      for (int j = 0; j < 1326; ++j) {
        if (overlap(i, j)) continue;
        const double c = call[class_[j]];
        ev_push += (1.0 - c) * big_blind_ + c * s * (2.0 * equity_[i * 1326 + j] - 1.0);
        ++n;
      }
      ev_push /= n;
      const double ev_fold = -small_blind_;
      const double p = push[class_[i]];
      r.br_small_blind += std::max(ev_push, ev_fold);
      r.value_small_blind += p * ev_push + (1.0 - p) * ev_fold;
    }
    r.br_small_blind /= 1326;
    r.value_small_blind /= 1326;

    for (int j = 0; j < 1326; ++j) {
      double walk = 0.0, call_ev = 0.0, fold_ev = 0.0;
      int n = 0;
      for (int i = 0; i < 1326; ++i) {
        if (overlap(i, j)) continue;
        const double p = push[class_[i]];
        walk += (1.0 - p) * small_blind_;
        call_ev += p * s * (2.0 * equity_[j * 1326 + i] - 1.0);
        fold_ev += p * -static_cast<double>(big_blind_);
        ++n;
      }
      r.br_big_blind += (walk + std::max(call_ev, fold_ev)) / n;
    }
    r.br_big_blind /= 1326;
    return r;
  }

 private:
  bool overlap(uint64_t x, uint64_t y) const {
    return combo_[x][0] == combo_[y][0] || combo_[x][0] == combo_[y][1] ||
           combo_[x][1] == combo_[y][0] || combo_[x][1] == combo_[y][1];
  }

  int stack_, small_blind_, big_blind_;
  std::array<std::array<holdem::Card, 2>, 1326> combo_;
  std::array<int, 1326> class_;
  std::vector<float> equity_;
};

}  // namespace poker2::blueprint
