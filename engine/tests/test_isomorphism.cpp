// Suit-isomorphism indexer. Reference class counts are from Waugh (2013),
// "A Fast and Optimal Hand Isomorphism Algorithm":
//   board as one group (used for abstraction): {2} 169, {2,3} 1,286,792,
//     {2,4} 13,960,050, {2,5} 123,156,254
//   board dealt street by street (perfect recall): {2,3,1} 55,190,538,
//     {2,3,1,1} 2,428,287,420

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "holdem/isomorphism.h"
#include "test_framework.h"

using namespace poker2::holdem;
using namespace poker2::testing;

namespace {

// Board cards form one unordered group: hand strength ignores which street a card came on.
const std::vector<std::vector<int>> kStreets = {{2}, {2, 3}, {2, 4}, {2, 5}};
const char* kStreetNames[] = {"preflop", "flop", "turn", "river"};

// Brute-force canonical form: lexicographically smallest group-sorted card
// list over all 24 suit relabelings.
std::vector<Card> canonical(const std::vector<Card>& cards, const std::vector<int>& groups) {
  std::array<int, 4> perm = {0, 1, 2, 3};
  std::vector<Card> best;
  do {
    std::vector<Card> mapped;
    size_t pos = 0;
    for (int size : groups) {
      std::vector<Card> group;
      for (int i = 0; i < size; ++i, ++pos) {
        group.push_back(make_card(rank_of(cards[pos]), perm[suit_of(cards[pos])]));
      }
      std::sort(group.begin(), group.end());
      mapped.insert(mapped.end(), group.begin(), group.end());
    }
    if (best.empty() || mapped < best) best = mapped;
  } while (std::next_permutation(perm.begin(), perm.end()));
  return best;
}

std::vector<Card> random_hand(std::mt19937_64& rng, int n) {
  std::array<Card, 52> deck;
  for (int i = 0; i < 52; ++i) deck[i] = static_cast<Card>(i);
  for (int i = 0; i < n; ++i) std::swap(deck[i], deck[i + rng() % (52 - i)]);
  return std::vector<Card>(deck.begin(), deck.begin() + n);
}

bool all_distinct(const std::vector<Card>& cards) {
  uint64_t seen = 0;
  for (Card c : cards) {
    if (c >= 52 || (seen >> c & 1)) return false;
    seen |= uint64_t{1} << c;
  }
  return true;
}

// Round-trips every index in [begin, end) with the given stride.
int round_trip_failures(const HandIndexer& indexer, uint64_t begin, uint64_t end, uint64_t stride) {
  std::vector<Card> cards(indexer.num_cards());
  int failures = 0;
  for (uint64_t i = begin; i < end; i += stride) {
    indexer.unindex(i, cards.data());
    failures += !all_distinct(cards) || indexer.index(cards.data()) != i;
  }
  return failures;
}

}  // namespace

TEST("isomorphism: class counts match the published values") {
  const uint64_t expected[] = {169, 1286792, 13960050, 123156254};
  for (int s = 0; s < 4; ++s) {
    check_eq(HandIndexer(kStreets[s]).size(), expected[s], std::string(kStreetNames[s]) + " classes");
  }
  check_eq(HandIndexer({2, 3, 1}).size(), uint64_t{55190538}, "perfect-recall turn classes");
  check_eq(HandIndexer({2, 3, 1, 1}).size(), uint64_t{2428287420}, "perfect-recall river classes");
}

TEST("isomorphism: all 1,326 hole-card pairs cover exactly 169 classes") {
  const HandIndexer indexer({2});
  std::vector<int> hits(169, 0);
  for (Card a = 0; a < 52; ++a)
    for (Card b = a + 1; b < 52; ++b) {
      const Card cards[2] = {a, b};
      ++hits[indexer.index(cards)];
    }
  // Pairs: 6 combos each, suited: 4, offsuit: 12.
  int pairs = 0, suited = 0, offsuit = 0;
  for (int h : hits) {
    pairs += h == 6;
    suited += h == 4;
    offsuit += h == 12;
  }
  check_eq(pairs, 13, "pair classes with 6 combos");
  check_eq(suited, 78, "suited classes with 4 combos");
  check_eq(offsuit, 78, "offsuit classes with 12 combos");
}

TEST("isomorphism: index/unindex round-trip (preflop, flop exhaustive; turn, river sampled)") {
  check_eq(round_trip_failures(HandIndexer(kStreets[0]), 0, 169, 1), 0, "preflop failures");
  check_eq(round_trip_failures(HandIndexer(kStreets[1]), 0, 1286792, 1), 0, "flop failures");
  check_eq(round_trip_failures(HandIndexer(kStreets[2]), 0, 13960050, 97), 0, "turn failures");
  check_eq(round_trip_failures(HandIndexer(kStreets[3]), 0, 123156254, 997), 0, "river failures");
  check_eq(round_trip_failures(HandIndexer({2, 3, 1}), 0, 55190538, 397), 0,
           "perfect-recall turn failures");
  check_eq(round_trip_failures(HandIndexer({2, 3, 1, 1}), 0, 2428287420, 19997), 0,
           "perfect-recall river failures");
}

TEST("isomorphism: index is invariant to suit relabeling and in-group order") {
  std::mt19937_64 rng(99);
  for (int s = 0; s < 4; ++s) {
    const HandIndexer indexer(kStreets[s]);
    int failures = 0;
    for (int trial = 0; trial < 50000; ++trial) {
      std::vector<Card> hand = random_hand(rng, indexer.num_cards());
      std::array<int, 4> perm = {0, 1, 2, 3};
      std::shuffle(perm.begin(), perm.end(), rng);
      std::vector<Card> other;
      for (Card c : hand) other.push_back(make_card(rank_of(c), perm[suit_of(c)]));
      size_t pos = 0;
      for (int size : kStreets[s]) {
        std::shuffle(other.begin() + pos, other.begin() + pos + size, rng);
        pos += size;
      }
      failures += indexer.index(hand.data()) != indexer.index(other.data());
    }
    check_eq(failures, 0, std::string(kStreetNames[s]) + " relabeling failures");
  }
}

TEST("isomorphism: representative is isomorphic to the original hand") {
  std::mt19937_64 rng(7);
  for (int s = 0; s < 4; ++s) {
    const HandIndexer indexer(kStreets[s]);
    int failures = 0;
    std::vector<Card> rep(indexer.num_cards());
    for (int trial = 0; trial < 20000; ++trial) {
      const std::vector<Card> hand = random_hand(rng, indexer.num_cards());
      indexer.unindex(indexer.index(hand.data()), rep.data());
      failures += canonical(hand, kStreets[s]) != canonical(rep, kStreets[s]);
    }
    check_eq(failures, 0, std::string(kStreetNames[s]) + " representative failures");
  }
}

SLOW_TEST("isomorphism: all 25,989,600 flop deals hit every class") {
  const HandIndexer indexer({2, 3});
  std::vector<uint8_t> hit(indexer.size(), 0);
  Card c[5];
  for (c[0] = 0; c[0] < 52; ++c[0])
    for (c[1] = c[0] + 1; c[1] < 52; ++c[1])
      for (c[2] = 0; c[2] < 52; ++c[2])
        for (c[3] = c[2] + 1; c[3] < 52; ++c[3])
          for (c[4] = c[3] + 1; c[4] < 52; ++c[4]) {
            if (c[2] == c[0] || c[2] == c[1] || c[3] == c[0] || c[3] == c[1] || c[4] == c[0] ||
                c[4] == c[1])
              continue;
            hit[indexer.index(c)] = 1;
          }
  uint64_t covered = 0;
  for (uint8_t h : hit) covered += h;
  check_eq(covered, indexer.size(), "flop classes covered");
}

SLOW_TEST("isomorphism: exhaustive turn and river round-trip") {
  check_eq(round_trip_failures(HandIndexer(kStreets[2]), 0, 13960050, 1), 0, "turn failures");
  check_eq(round_trip_failures(HandIndexer(kStreets[3]), 0, 123156254, 1), 0, "river failures");
}
