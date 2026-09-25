// Card abstraction building blocks: equity, OCHS clusters, features, k-means,
// table files, and bucket lookup.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "abstraction/card_abstraction.h"
#include "abstraction/equity.h"
#include "abstraction/features.h"
#include "abstraction/kmeans.h"
#include "abstraction/table_file.h"
#include "test_framework.h"

using namespace poker2::abstraction;
using namespace poker2::holdem;
using namespace poker2::testing;

namespace {

double river_eq(const std::string& hole, const std::string& board) {
  const auto h = parse_cards(hole), b = parse_cards(board);
  return river_equity(h.data(), b.data());
}

struct ExactRiver {
  float operator()(const Card hole[2], const Card board[5]) const {
    return static_cast<float>(river_equity(hole, board));
  }
};

// Independent brute force: all (runout, opponent) combinations for a turn or flop hand.
double brute_force_equity(const std::vector<Card>& hole, const std::vector<Card>& board) {
  uint64_t dead = 0;
  for (Card c : hole) dead |= uint64_t{1} << c;
  for (Card c : board) dead |= uint64_t{1} << c;
  std::vector<Card> rest;
  for (Card c = 0; c < 52; ++c)
    if (!(dead >> c & 1)) rest.push_back(c);
  const int need = 5 - static_cast<int>(board.size());
  uint64_t score = 0, total = 0;
  std::vector<Card> full(board);
  std::function<void(size_t, int)> runout = [&](size_t start, int left) {
    if (left == 0) {
      Card me[7] = {hole[0], hole[1], full[0], full[1], full[2], full[3], full[4]};
      const uint32_t mine = evaluate(me, 7);
      uint64_t used = dead;
      for (Card c : full) used |= uint64_t{1} << c;
      for (Card a = 0; a < 52; ++a) {
        if (used >> a & 1) continue;
        for (Card b = a + 1; b < 52; ++b) {
          if (used >> b & 1) continue;
          me[0] = a;
          me[1] = b;
          const uint32_t theirs = evaluate(me, 7);
          score += mine > theirs ? 2 : (mine == theirs ? 1 : 0);
          ++total;
        }
      }
      return;
    }
    for (size_t i = start; i < rest.size(); ++i) {
      full.push_back(rest[i]);
      runout(i + 1, left - 1);
      full.pop_back();
    }
  };
  runout(0, need);
  return score / (2.0 * total);
}

}  // namespace

TEST("equity: exact river values") {
  check_near(river_eq("AsKs", "QsJsTs2d3c"), 1.0, 1e-12, "royal flush always wins");
  check_near(river_eq("2c3d", "AsKsQsJsTs"), 0.5, 1e-12, "royal flush on board always splits");
  // Quad aces on board with a king kicker in hand: only another king ties.
  // 45 unseen cards, 3 kings among them: ties = C(3,1)*42 + C(3,2) = 129 of 990 hands.
  check_near(river_eq("Kd2c", "AsAhAdAc3h"), 1.0 - 129.0 / 990.0 / 2.0, 1e-12,
             "quads on board, king kicker");
}

TEST("equity: river equity is invariant to suit relabeling") {
  std::mt19937_64 rng(3);
  int failures = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const Deal d = sample_deal(rng);
    std::array<int, 4> perm = {0, 1, 2, 3};
    std::shuffle(perm.begin(), perm.end(), rng);
    auto map = [&](Card c) { return make_card(rank_of(c), perm[suit_of(c)]); };
    const Card hole2[2] = {map(d.hole[0][1]), map(d.hole[0][0])};
    Card board2[5];
    for (int i = 0; i < 5; ++i) board2[i] = map(d.board[4 - i]);
    failures += river_equity(d.hole[0], d.board) != river_equity(hole2, board2);
  }
  check_eq(failures, 0, "relabeling failures");
}

TEST("equity: preflop equities match published values") {
  const auto aa = parse_cards("AsAh"), kk = parse_cards("KsKh"), ako = parse_cards("AsKd");
  // Published all-in equity vs a random hand: AA 85.2%, KK 82.4%, AKo 65.3%.
  check_near(preflop_equity(aa.data(), 200000, 1), 0.852, 0.005, "AA vs random");
  check_near(preflop_equity(kk.data(), 200000, 2), 0.824, 0.005, "KK vs random");
  check_near(preflop_equity(ako.data(), 200000, 3), 0.653, 0.005, "AKo vs random");
}

TEST("equity: opponent clusters are balanced equity bands") {
  const OpponentClusters oc = make_opponent_clusters(8, 20000, 1);
  std::array<int, 8> combos{};
  for (Card a = 0; a < 52; ++a)
    for (Card b = a + 1; b < 52; ++b) ++combos[oc.of_pair[a][b]];
  int largest_gap = 0;
  for (int c : combos) largest_gap = std::max(largest_gap, std::abs(c - 1326 / 8));
  check(largest_gap <= 12, "each cluster holds ~166 combos (within one offsuit class)");
  check_eq(static_cast<int>(oc.of_pair[parse_card("As")][parse_card("Ah")]), 7, "AA in top cluster");
  check_eq(static_cast<int>(oc.of_pair[parse_card("3s")][parse_card("2h")]), 0, "32o in bottom cluster");
  check(oc.of_pair[parse_card("Ks")][parse_card("Qd")] == oc.of_pair[parse_card("Qh")][parse_card("Kc")],
        "cluster depends only on the preflop class");
}

TEST("features: turn and flop features agree with brute-force equity") {
  const ExactRiver exact;
  float turn[kTurnFeatureDim], flop[kFlopFeatureDim];
  const auto h1 = parse_cards("8h9h"), b1 = parse_cards("6h7hKc2s");
  const double turn_eq = turn_feature(exact, h1.data(), b1.data(), turn);
  check_near(turn_eq, brute_force_equity(h1, b1), 1e-6, "turn feature mean = turn equity");
  check(std::is_sorted(turn, turn + kTurnFeatureDim), "turn feature sorted");

  const auto h2 = parse_cards("AhKd"), b2 = parse_cards("Kc7s2d");
  const double flop_eq = flop_feature(exact, h2.data(), b2.data(), flop);
  check_near(flop_eq, brute_force_equity(h2, b2), 1e-6, "flop feature mean = flop equity");
  check(std::is_sorted(flop, flop + kFlopFeatureDim), "flop quantile bins sorted");
}

TEST("kmeans: recovers well-separated clusters deterministically") {
  std::mt19937_64 rng(11);
  std::normal_distribution<float> noise(0.0f, 0.05f);
  const float truth[3][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  std::vector<float> points;
  for (int i = 0; i < 30000; ++i) {
    const auto& t = truth[i % 3];
    points.push_back(t[0] + noise(rng));
    points.push_back(t[1] + noise(rng));
  }
  KMeansOptions opt;
  opt.k = 3;
  opt.threads = 4;
  opt.seed = 5;
  const KMeansResult r = kmeans_fit(points, 2, opt);
  int matched = 0;
  for (const auto& t : truth) {
    for (int c = 0; c < 3; ++c) {
      matched += std::hypot(r.centers[c * 2] - t[0], r.centers[c * 2 + 1] - t[1]) < 0.02;
    }
  }
  check_eq(matched, 3, "each true center recovered");
  check(1.0 - r.inertia / r.total_variance > 0.98, "explained variance above 0.98");
  const KMeansResult again = kmeans_fit(points, 2, opt);
  check(again.centers == r.centers, "same seed and threads give identical centers");
}

TEST("kmeans: more clusters than distinct points stays valid") {
  std::vector<float> points;
  for (int i = 0; i < 100; ++i) points.push_back(static_cast<float>(i % 4));  // 4 distinct values
  KMeansOptions opt;
  opt.k = 10;
  opt.threads = 2;
  const KMeansResult r = kmeans_fit(points, 1, opt);
  check_eq(r.k, 10, "k kept");
  check(r.inertia < 1e-9, "every point sits on a center");
  std::vector<float> tiny = {1.0f, 2.0f};
  opt.k = 5;
  check_eq(kmeans_fit(tiny, 1, opt).k, 2, "k clipped to the number of points");
}

TEST("table files: round trip and shape validation") {
  const std::string file = "/tmp/poker2_table_test.bin";
  std::vector<uint16_t> values(300);
  for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<uint16_t>(i * 7);
  write_table(file, kRiverOchsTable, 100, 3, 3, values);
  uint32_t aux = 0;
  check(read_table(file, kRiverOchsTable, 100, 3, &aux) == values, "values round-trip");
  check_eq(aux, 3u, "aux round-trips");
  bool threw = false;
  try {
    read_table(file, kBucketTable, 100, 3);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "wrong kind rejected");
  threw = false;
  try {
    read_table(file, kRiverOchsTable, 101, 3);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "wrong row count rejected");
  std::remove(file.c_str());
}

TEST("card abstraction: lookups go through the isomorphism index") {
  CardAbstraction abs;
  for (int s = 0; s < 4; ++s) {
    std::vector<uint16_t> table(abs.indexer(s).size());
    for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<uint16_t>(i % 1009);
    abs.set_table(s, std::move(table), 1009);
  }
  std::mt19937_64 rng(21);
  int failures = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const Deal d = sample_deal(rng);
    std::array<int, 4> perm = {0, 1, 2, 3};
    std::shuffle(perm.begin(), perm.end(), rng);
    auto map = [&](Card c) { return make_card(rank_of(c), perm[suit_of(c)]); };
    const Card hole2[2] = {map(d.hole[0][1]), map(d.hole[0][0])};
    const Card board2[5] = {map(d.board[2]), map(d.board[1]), map(d.board[0]), map(d.board[3]),
                            map(d.board[4])};
    for (int s = 1; s < 3; ++s) failures += abs.bucket(s, d.hole[0], d.board) != abs.bucket(s, hole2, board2);
    const Card cards[7] = {d.hole[0][0], d.hole[0][1], d.board[0], d.board[1],
                           d.board[2],   d.board[3],   d.board[4]};
    failures += abs.bucket(3, d.hole[0], d.board) != static_cast<int>(abs.indexer(3).index(cards) % 1009);
  }
  check_eq(failures, 0, "bucket lookup failures");
}
