#pragma once

// Hand evaluator for 5 to 7 cards, table-free, built on 13-bit rank masks.
//
// Returns a 32-bit strength: higher is better, equal means a tie.
// Layout: category << 20 | five 4-bit ranks in order of significance.
//
// Flush is checked before quads and full house because neither can coexist
// with a flush in 7 or fewer cards.

#include <cstdint>

#include "holdem/cards.h"

namespace poker2::holdem {

enum HandCategory : uint32_t {
  kHighCard = 0,
  kPair = 1,
  kTwoPair = 2,
  kTrips = 3,
  kStraight = 4,
  kFlush = 5,
  kFullHouse = 6,
  kQuads = 7,
  kStraightFlush = 8,
};

constexpr const char* kCategoryNames[] = {"high card", "pair",       "two pair",
                                          "trips",     "straight",   "flush",
                                          "full house", "quads",     "straight flush"};

inline HandCategory category_of(uint32_t strength) {
  return static_cast<HandCategory>(strength >> 20);
}

namespace detail {

inline int highest_bit(uint32_t mask) { return 31 - __builtin_clz(mask); }

// Packs the top `count` ranks of `mask` (highest first) starting at nibble `slot`.
inline uint32_t pack_top(uint32_t mask, int count, int slot) {
  uint32_t out = 0;
  for (int i = 0; i < count; ++i, --slot) {
    const int r = highest_bit(mask);
    out |= static_cast<uint32_t>(r) << (4 * slot);
    mask &= ~(1u << r);
  }
  return out;
}

// High rank of the best straight in a rank mask, or -1. Handles the wheel (A-5).
inline int straight_high(uint32_t ranks) {
  // Bit r+1 = rank r present; bit 0 = ace playing low.
  const uint32_t m = (ranks << 1) | ((ranks >> 12) & 1u);
  const uint32_t runs = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
  return runs ? highest_bit(runs) + 3 : -1;
}

inline uint32_t make_strength(HandCategory category, uint32_t ranks) {
  return (static_cast<uint32_t>(category) << 20) | ranks;
}

}  // namespace detail

inline uint32_t evaluate(const Card* cards, int n) {
  using namespace detail;
  uint32_t suit_mask[4] = {0, 0, 0, 0};
  for (int i = 0; i < n; ++i) suit_mask[suit_of(cards[i])] |= 1u << rank_of(cards[i]);

  for (uint32_t m : suit_mask) {
    if (__builtin_popcount(m) >= 5) {
      const int high = straight_high(m);
      if (high >= 0) return make_strength(kStraightFlush, static_cast<uint32_t>(high) << 16);
      return make_strength(kFlush, pack_top(m, 5, 4));
    }
  }

  // Ranks held at least once / twice / three times / four times.
  uint32_t ones = 0, twos = 0, threes = 0, fours = 0;
  for (uint32_t m : suit_mask) {
    fours |= threes & m;
    threes |= twos & m;
    twos |= ones & m;
    ones |= m;
  }

  if (fours) {
    const int quad = highest_bit(fours);
    return make_strength(kQuads, (static_cast<uint32_t>(quad) << 16) |
                                     pack_top(ones & ~(1u << quad), 1, 3));
  }

  const uint32_t trips = threes;          // exactly three (no quads here)
  const uint32_t pairs = twos & ~threes;  // exactly two
  if (trips && (__builtin_popcount(trips) >= 2 || pairs)) {
    const int trip = highest_bit(trips);
    const int pair = highest_bit((trips & ~(1u << trip)) | pairs);
    return make_strength(kFullHouse,
                         (static_cast<uint32_t>(trip) << 16) | (static_cast<uint32_t>(pair) << 12));
  }

  const int high = straight_high(ones);
  if (high >= 0) return make_strength(kStraight, static_cast<uint32_t>(high) << 16);

  if (trips) {
    const int trip = highest_bit(trips);
    return make_strength(kTrips, (static_cast<uint32_t>(trip) << 16) |
                                     pack_top(ones & ~(1u << trip), 2, 3));
  }

  if (__builtin_popcount(pairs) >= 2) {
    const int p1 = highest_bit(pairs);
    const int p2 = highest_bit(pairs & ~(1u << p1));
    const uint32_t kicker_mask = ones & ~(1u << p1) & ~(1u << p2);
    return make_strength(kTwoPair, (static_cast<uint32_t>(p1) << 16) |
                                       (static_cast<uint32_t>(p2) << 12) |
                                       pack_top(kicker_mask, 1, 2));
  }

  if (pairs) {
    const int p = highest_bit(pairs);
    return make_strength(kPair,
                         (static_cast<uint32_t>(p) << 16) | pack_top(ones & ~(1u << p), 3, 3));
  }

  return make_strength(kHighCard, pack_top(ones, 5, 4));
}

// Strength of a seat's best hand at showdown.
inline uint32_t evaluate_seat(const Deal& deal, int seat) {
  const Card cards[7] = {deal.hole[seat][0], deal.hole[seat][1], deal.board[0], deal.board[1],
                         deal.board[2],      deal.board[3],      deal.board[4]};
  return evaluate(cards, 7);
}

}  // namespace poker2::holdem
