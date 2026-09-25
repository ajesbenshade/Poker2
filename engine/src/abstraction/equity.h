#pragma once

// Hand-strength computations used to build the card abstraction.
//
// Equity is P(win) + P(tie) / 2 against an opponent hand drawn uniformly from
// the unseen cards. "Opponent-cluster hand strength" (OCHS, Johanson et al.
// 2013) splits that into one equity per group of opponent preflop hands.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "holdem/evaluator.h"
#include "holdem/isomorphism.h"

namespace poker2::abstraction {

using holdem::Card;

constexpr int kMaxOpponentClusters = 16;

// Maps every hole-card pair to an opponent cluster id in [0, num_clusters).
struct OpponentClusters {
  int num_clusters = 1;
  std::array<std::array<uint8_t, 52>, 52> of_pair{};  // symmetric; diagonal unused
};

struct RiverStrength {
  double equity;                                    // vs a uniformly random hand
  std::array<double, kMaxOpponentClusters> by_cluster;  // equity vs each cluster
};

// Exact river strength: enumerates all C(45, 2) = 990 opponent hands.
inline RiverStrength river_strength(const Card hole[2], const Card board[5],
                                    const OpponentClusters& clusters) {
  Card cards[7] = {hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]};
  const uint32_t mine = holdem::evaluate(cards, 7);

  uint64_t dead = 0;
  for (Card c : cards) dead |= uint64_t{1} << c;
  Card rest[45];
  int n = 0;
  for (Card c = 0; c < 52; ++c) {
    if (!(dead >> c & 1)) rest[n++] = c;
  }

  std::array<uint32_t, kMaxOpponentClusters> score{}, count{};
  uint32_t total_score = 0, total_count = 0;
  for (int i = 0; i < n; ++i) {
    cards[0] = rest[i];
    for (int j = i + 1; j < n; ++j) {
      cards[1] = rest[j];
      const uint32_t theirs = holdem::evaluate(cards, 7);
      const uint32_t s = mine > theirs ? 2 : (mine == theirs ? 1 : 0);
      const int c = clusters.of_pair[rest[i]][rest[j]];
      score[c] += s;
      ++count[c];
      total_score += s;
      ++total_count;
    }
  }

  RiverStrength out{};
  out.equity = total_score / (2.0 * total_count);
  for (int c = 0; c < clusters.num_clusters; ++c) {
    out.by_cluster[c] = count[c] ? score[c] / (2.0 * count[c]) : 0.5;
  }
  return out;
}

inline double river_equity(const Card hole[2], const Card board[5]) {
  static const OpponentClusters single;  // everything in cluster 0
  return river_strength(hole, board, single).equity;
}

// Monte Carlo preflop equity vs a random hand (all-in to the river).
inline double preflop_equity(const Card hole[2], int samples, uint64_t seed) {
  std::mt19937_64 rng(seed);
  uint64_t score = 0;
  Card deck[52];
  int n = 0;
  for (Card c = 0; c < 52; ++c) {
    if (c != hole[0] && c != hole[1]) deck[n++] = c;
  }
  for (int s = 0; s < samples; ++s) {
    for (int i = 0; i < 7; ++i) std::swap(deck[i], deck[i + rng() % (n - i)]);
    const Card me[7] = {hole[0], hole[1], deck[2], deck[3], deck[4], deck[5], deck[6]};
    const Card opp[7] = {deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6]};
    const uint32_t a = holdem::evaluate(me, 7), b = holdem::evaluate(opp, 7);
    score += a > b ? 2 : (a == b ? 1 : 0);
  }
  return score / (2.0 * samples);
}

// Number of raw hole-card combos in a preflop class: pairs 6, suited 4, offsuit 12.
inline int preflop_combos(const Card hole[2]) {
  if (holdem::rank_of(hole[0]) == holdem::rank_of(hole[1])) return 6;
  return holdem::suit_of(hole[0]) == holdem::suit_of(hole[1]) ? 4 : 12;
}

// Groups the 169 preflop classes into `num_clusters` bands of preflop equity,
// each holding about the same number of raw combos.
inline OpponentClusters make_opponent_clusters(int num_clusters, int samples, uint64_t seed,
                                               std::vector<double>* equities_out = nullptr) {
  const holdem::HandIndexer preflop({2});
  std::vector<double> equity(169);
  std::vector<int> combos(169);
  for (int i = 0; i < 169; ++i) {
    Card hole[2];
    preflop.unindex(i, hole);
    equity[i] = preflop_equity(hole, samples, seed + i);
    combos[i] = preflop_combos(hole);
  }
  std::vector<int> order(169);
  for (int i = 0; i < 169; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return equity[a] < equity[b]; });

  std::vector<uint8_t> cluster_of_class(169);
  int cumulative = 0;
  for (int cls : order) {
    const double middle = cumulative + combos[cls] / 2.0;
    cluster_of_class[cls] =
        static_cast<uint8_t>(std::min(num_clusters - 1, static_cast<int>(middle * num_clusters / 1326)));
    cumulative += combos[cls];
  }

  OpponentClusters out;
  out.num_clusters = num_clusters;
  for (Card a = 0; a < 52; ++a) {
    for (Card b = 0; b < 52; ++b) {
      if (a == b) continue;
      const Card hole[2] = {a, b};
      out.of_pair[a][b] = cluster_of_class[preflop.index(hole)];
    }
  }
  if (equities_out) *equities_out = equity;
  return out;
}

// 16-bit fixed point for equities in [0, 1].
inline uint16_t quantize(double x) {
  return static_cast<uint16_t>(std::lround(std::clamp(x, 0.0, 1.0) * 65535.0));
}
inline float dequantize(uint16_t q) { return q / 65535.0f; }

}  // namespace poker2::abstraction
