#pragma once

// Clustering features for each street, all derived from one river equity
// table (equity vs a random hand for every {2,5} river class).
//
//   river: OCHS vector, equity vs each opponent preflop cluster (stored table)
//   turn:  the 46 river equities over every possible river card, sorted
//   flop:  the 1,081 river equities over every (turn, river) runout, sorted
//          and averaged into kFlopFeatureDim quantile bins
//
// For sorted equal-size samples, squared Euclidean distance is the squared
// 2-Wasserstein distance between the equity distributions, so k-means on these
// features clusters hands by the shape of their equity distribution rather than
// only its mean (distribution-aware abstraction, Johanson et al. 2013).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "abstraction/equity.h"
#include "holdem/isomorphism.h"

namespace poker2::abstraction {

constexpr int kTurnFeatureDim = 46;
constexpr int kFlopRunouts = 1081;  // C(47, 2)
constexpr int kFlopFeatureDim = 50;

class RiverEquityTable {
 public:
  RiverEquityTable() : indexer_({2, 5}) {}
  explicit RiverEquityTable(std::vector<uint16_t> values)
      : indexer_({2, 5}), values_(std::move(values)) {}

  const holdem::HandIndexer& indexer() const { return indexer_; }
  const std::vector<uint16_t>& values() const { return values_; }

  float operator()(const Card hole[2], const Card board[5]) const { return lookup(hole, board); }

  float lookup(const Card hole[2], const Card board[5]) const {
    const Card cards[7] = {hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]};
    return dequantize(values_[indexer_.index(cards)]);
  }

 private:
  holdem::HandIndexer indexer_;
  std::vector<uint16_t> values_;
};

// `river(hole, board5)` returns river equity; RiverEquityTable::lookup in
// production, exact computation in tests.

// Writes kTurnFeatureDim floats; returns turn equity (their mean).
template <class RiverLookup>
double turn_feature(const RiverLookup& river, const Card hole[2], const Card board[4], float* out) {
  uint64_t dead = 0;
  for (int i = 0; i < 2; ++i) dead |= uint64_t{1} << hole[i];
  for (int i = 0; i < 4; ++i) dead |= uint64_t{1} << board[i];
  Card river_board[5] = {board[0], board[1], board[2], board[3], 0};
  int n = 0;
  double sum = 0.0;
  for (Card c = 0; c < 52; ++c) {
    if (dead >> c & 1) continue;
    river_board[4] = c;
    out[n] = river(hole, river_board);
    sum += out[n++];
  }
  std::sort(out, out + n);
  return sum / n;
}

// Writes kFlopFeatureDim floats; returns flop equity (mean over all runouts).
template <class RiverLookup>
double flop_feature(const RiverLookup& river, const Card hole[2], const Card board[3], float* out) {
  uint64_t dead = 0;
  for (int i = 0; i < 2; ++i) dead |= uint64_t{1} << hole[i];
  for (int i = 0; i < 3; ++i) dead |= uint64_t{1} << board[i];
  Card rest[47];
  int n = 0;
  for (Card c = 0; c < 52; ++c) {
    if (!(dead >> c & 1)) rest[n++] = c;
  }
  float runouts[kFlopRunouts];
  int m = 0;
  double sum = 0.0;
  Card river_board[5] = {board[0], board[1], board[2], 0, 0};
  for (int i = 0; i < n; ++i) {
    river_board[3] = rest[i];
    for (int j = i + 1; j < n; ++j) {
      river_board[4] = rest[j];
      runouts[m] = river(hole, river_board);
      sum += runouts[m++];
    }
  }
  std::sort(runouts, runouts + m);
  for (int b = 0; b < kFlopFeatureDim; ++b) {
    const int lo = b * m / kFlopFeatureDim, hi = (b + 1) * m / kFlopFeatureDim;
    double bin = 0.0;
    for (int i = lo; i < hi; ++i) bin += runouts[i];
    out[b] = static_cast<float>(bin / (hi - lo));
  }
  return sum / m;
}

}  // namespace poker2::abstraction
