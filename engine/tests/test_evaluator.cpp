// Hand evaluator against published poker combinatorics.
// Reference counts: Wikipedia "Poker probability" (5-card and 7-card tables).

#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "holdem/evaluator.h"
#include "test_framework.h"

using namespace poker2::holdem;
using namespace poker2::testing;

namespace {

uint32_t eval(const std::string& cards) {
  const std::vector<Card> c = parse_cards(cards);
  return evaluate(c.data(), static_cast<int>(c.size()));
}

}  // namespace

TEST("evaluator: hand-picked orderings") {
  check(category_of(eval("AsKsQsJsTs")) == kStraightFlush, "royal flush category");
  check(category_of(eval("As2s3s4s5s")) == kStraightFlush, "steel wheel category");
  check(eval("As2s3s4s5s") < eval("2s3s4s5s6s"), "steel wheel is the lowest straight flush");
  check(eval("AsAhAdAc2s") > eval("KsKhKdKcAs"), "quad aces beat quad kings");
  check(eval("KsKhKdKcAs") > eval("KsKhKdKcQs"), "quads kicker matters");
  check(eval("3s3h3d2s2h") > eval("2s2h2dAsAh"), "full house ranked by trips first");
  check(eval("AsQs9s5s3s") > eval("AsQs9s5s2s"), "flush compares all five cards");
  check(eval("As2d3c4h5s") < eval("2d3c4h5s6h"), "wheel is the lowest straight");
  check(eval("TsJdQcKhAs") > eval("9sTdJcQhKs"), "broadway beats king-high straight");
  check(eval("AsAhKdKcQs") > eval("AsAhKdKc2s"), "two pair kicker matters");
  check(eval("AsAhKdKcQs") > eval("AsAhQdQcKs"), "two pair ranked by top pair then second");
  check(eval("AsAh7d5c3s") > eval("AsAh7d5c2s"), "pair uses three kickers");
  check(eval("AsKh9d5c3s") == eval("AdKc9h5s3d"), "suits never matter without a flush");
  // 7-card specifics.
  check(eval("AsAhAdKsKhKd2c") == eval("AsAhAdKsKh3c4d"), "two trips make the best full house");
  check(eval("AsAhKdKcQsQh2d") == eval("AsAhKdKc2sQh3d"),
        "third pair only counts as a kicker");
  check(eval("2s3s4s5s6s7s8s") == eval("4s5s6s7s8sAhAd"), "7-card straight flush uses top five");
  check(category_of(eval("AsKsQsJs9s8h7h")) == kFlush, "flush beats straight when both exist");
}

TEST("evaluator: all 2,598,960 five-card hands match published counts") {
  const std::array<uint64_t, 9> expected_count = {1302540, 1098240, 123552, 54912, 10200,
                                                  5108,    3744,    624,    40};
  const std::array<size_t, 9> expected_distinct = {1277, 2860, 858, 858, 10, 1277, 156, 156, 10};
  std::array<uint64_t, 9> count{};
  std::array<std::set<uint32_t>, 9> distinct;
  Card c[5];
  for (c[0] = 0; c[0] < 52; ++c[0])
    for (c[1] = c[0] + 1; c[1] < 52; ++c[1])
      for (c[2] = c[1] + 1; c[2] < 52; ++c[2])
        for (c[3] = c[2] + 1; c[3] < 52; ++c[3])
          for (c[4] = c[3] + 1; c[4] < 52; ++c[4]) {
            const uint32_t s = evaluate(c, 5);
            ++count[category_of(s)];
            distinct[category_of(s)].insert(s);
          }
  size_t total_distinct = 0;
  for (int k = 0; k < 9; ++k) {
    check_eq(count[k], expected_count[k], std::string(kCategoryNames[k]) + " count");
    check_eq(distinct[k].size(), expected_distinct[k], std::string(kCategoryNames[k]) + " classes");
    total_distinct += distinct[k].size();
  }
  check_eq(total_distinct, size_t{7462}, "distinct 5-card hand values");
}

TEST("evaluator: 7-card result equals best of 21 five-card subsets") {
  std::mt19937_64 rng(2024);
  int mismatches = 0;
  for (int trial = 0; trial < 300000; ++trial) {
    std::array<Card, 52> deck;
    for (int i = 0; i < 52; ++i) deck[i] = static_cast<Card>(i);
    for (int i = 0; i < 7; ++i) std::swap(deck[i], deck[i + rng() % (52 - i)]);
    uint32_t best = 0;
    for (int skip1 = 0; skip1 < 7; ++skip1)
      for (int skip2 = skip1 + 1; skip2 < 7; ++skip2) {
        Card five[5];
        int n = 0;
        for (int i = 0; i < 7; ++i)
          if (i != skip1 && i != skip2) five[n++] = deck[i];
        best = std::max(best, evaluate(five, 5));
      }
    mismatches += evaluate(deck.data(), 7) != best;
  }
  check_eq(mismatches, 0, "7-card vs best-of-subsets mismatches");
}

SLOW_TEST("evaluator: all 133,784,560 seven-card hands match published counts") {
  const std::array<uint64_t, 9> expected_count = {23294460, 58627800, 31433400, 6461620, 6180020,
                                                  4047644,  3473184,  224848,   41584};
  const std::array<size_t, 9> expected_distinct = {407, 1470, 763, 575, 10, 1277, 156, 156, 10};
  std::array<uint64_t, 9> count{};
  std::vector<uint8_t> seen(9u << 20, 0);  // one flag per possible strength value
  Card c[7];
  for (c[0] = 0; c[0] < 52; ++c[0])
    for (c[1] = c[0] + 1; c[1] < 52; ++c[1])
      for (c[2] = c[1] + 1; c[2] < 52; ++c[2])
        for (c[3] = c[2] + 1; c[3] < 52; ++c[3])
          for (c[4] = c[3] + 1; c[4] < 52; ++c[4])
            for (c[5] = c[4] + 1; c[5] < 52; ++c[5])
              for (c[6] = c[5] + 1; c[6] < 52; ++c[6]) {
                const uint32_t s = evaluate(c, 7);
                ++count[category_of(s)];
                seen[s] = 1;
              }
  std::array<size_t, 9> distinct{};
  for (uint32_t s = 0; s < seen.size(); ++s) distinct[s >> 20] += seen[s];
  size_t total_distinct = 0;
  for (int k = 0; k < 9; ++k) {
    check_eq(count[k], expected_count[k], std::string(kCategoryNames[k]) + " count");
    check_eq(distinct[k], expected_distinct[k], std::string(kCategoryNames[k]) + " classes");
    total_distinct += distinct[k];
  }
  check_eq(total_distinct, size_t{4824}, "distinct 7-card hand values");
}
