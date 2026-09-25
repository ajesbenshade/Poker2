// Builds the card abstraction.
//
//   poker2_abstraction [--out DIR] [--stages preflop,equity,river,turn,flop,report]
//                      [--buckets FLOP,TURN,RIVER] [--clusters K] [--sample N]
//                      [--iters N] [--threads N] [--seed S]
//
// Stages (each reads what earlier stages wrote to --out):
//   preflop  169 lossless buckets; opponent clusters for OCHS (opponent_clusters.txt)
//   equity   river equity and OCHS for all 123,156,254 river classes (~2 GB)
//   river    k-means on OCHS vectors
//   turn     k-means on sorted river-equity distributions (46 values)
//   flop     k-means on sorted runout-equity distributions (50 quantile bins)
//   report   bucket occupancy and equity variance explained, on random deals

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "abstraction/card_abstraction.h"
#include "abstraction/equity.h"
#include "abstraction/features.h"
#include "abstraction/kmeans.h"
#include "abstraction/table_file.h"
#include "core/parallel.h"

using namespace poker2;
using namespace poker2::abstraction;
using holdem::Card;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  std::string out;
  std::vector<std::string> stages = {"preflop", "equity", "river", "turn", "flop", "report"};
  int buckets[4] = {169, 2000, 2000, 1500};
  int clusters = 8;
  int sample = 1000000;
  int iters = 50;
  int threads = default_threads();
  uint64_t seed = 1;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: poker2_abstraction [--out DIR] [--stages a,b,...] [--buckets F,T,R]\n"
               "                          [--clusters K] [--sample N] [--iters N]\n"
               "                          [--threads N] [--seed S]\n");
  std::exit(2);
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(item);
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  const char* home = std::getenv("HOME");
  a.out = std::string(home ? home : ".") + "/poker2-data/abstraction";
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    const std::string flag = argv[i];
    if (flag == "--out") a.out = next();
    else if (flag == "--stages") a.stages = split(next());
    else if (flag == "--buckets") {
      const auto b = split(next());
      if (b.size() != 3) usage();
      for (int s = 0; s < 3; ++s) a.buckets[s + 1] = std::atoi(b[s].c_str());
    } else if (flag == "--clusters") a.clusters = std::atoi(next().c_str());
    else if (flag == "--sample") a.sample = std::atoi(next().c_str());
    else if (flag == "--iters") a.iters = std::atoi(next().c_str());
    else if (flag == "--threads") a.threads = std::atoi(next().c_str());
    else if (flag == "--seed") a.seed = std::strtoull(next().c_str(), nullptr, 10);
    else usage();
  }
  if (a.clusters < 1 || a.clusters > kMaxOpponentClusters || a.sample < 1 || a.threads < 1) usage();
  for (int s = 1; s < 4; ++s) {
    if (a.buckets[s] < 1 || a.buckets[s] > 65535) usage();
  }
  return a;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Runs fn over [0, n) in parallel and prints progress every 5%.
template <class F>
void run_with_progress(const char* what, uint64_t n, int threads, uint64_t chunk, F&& fn) {
  std::atomic<uint64_t> done{0};
  std::atomic<int> last_reported{0};
  const auto start = Clock::now();
  parallel_for(n, threads, chunk, [&](uint64_t b, uint64_t e, int t) {
    fn(b, e, t);
    const uint64_t now = done.fetch_add(e - b) + (e - b);
    const int pct = static_cast<int>(now * 20 / n) * 5;
    int prev = last_reported.load();
    while (pct > prev && !last_reported.compare_exchange_weak(prev, pct)) {
    }
    if (pct > prev) {
      std::printf("  %s: %3d%%  (%.0fs)\n", what, pct, seconds_since(start));
      std::fflush(stdout);
    }
  });
}

std::string path(const Args& a, const char* name) { return a.out + "/" + name; }

OpponentClusters opponent_clusters(const Args& a, std::vector<double>* equities = nullptr) {
  return make_opponent_clusters(a.clusters, 200000, a.seed, equities);
}

void stage_preflop(const Args& a) {
  const holdem::HandIndexer preflop({2});
  std::vector<double> equity;
  const OpponentClusters clusters = opponent_clusters(a, &equity);
  FILE* f = std::fopen(path(a, "opponent_clusters.txt").c_str(), "w");
  if (!f) throw std::runtime_error("cannot write opponent_clusters.txt");
  std::fprintf(f, "# class  equity_vs_random  opponent_cluster\n");
  for (int i = 0; i < 169; ++i) {
    Card hole[2];
    preflop.unindex(i, hole);
    std::fprintf(f, "%s %.4f %d\n", holdem::cards_to_string({hole[0], hole[1]}).c_str(), equity[i],
                 clusters.of_pair[hole[0]][hole[1]]);
  }
  std::fclose(f);
  std::vector<uint16_t> identity(169);
  std::iota(identity.begin(), identity.end(), 0);
  write_table(path(a, street_file_name(0)), kBucketTable, 169, 1, 169, identity);
}

void stage_equity(const Args& a) {
  const OpponentClusters clusters = opponent_clusters(a);
  const holdem::HandIndexer river({2, 5});
  const uint64_t n = river.size();
  const int k = clusters.num_clusters;
  std::vector<uint16_t> equity(n), ochs(n * k);
  run_with_progress("river equity", n, a.threads, 1 << 14, [&](uint64_t b, uint64_t e, int) {
    Card cards[7];
    for (uint64_t i = b; i < e; ++i) {
      river.unindex(i, cards);
      const RiverStrength s = river_strength(cards, cards + 2, clusters);
      equity[i] = quantize(s.equity);
      for (int c = 0; c < k; ++c) ochs[i * k + c] = quantize(s.by_cluster[c]);
    }
  });
  write_table(path(a, "river_equity.bin"), kRiverEquityTable, n, 1, 0, equity);
  write_table(path(a, "river_ochs.bin"), kRiverOchsTable, n, k, k, ochs);
}

RiverEquityTable load_equity(const Args& a) {
  const holdem::HandIndexer river({2, 5});
  return RiverEquityTable(read_table(path(a, "river_equity.bin"), kRiverEquityTable, river.size(), 1));
}

// Relabels centers so bucket ids increase with the mean of the center vector.
void sort_centers(KMeansResult& r) {
  std::vector<int> order(r.k);
  std::iota(order.begin(), order.end(), 0);
  auto mean = [&](int c) {
    double s = 0.0;
    for (int d = 0; d < r.dim; ++d) s += r.centers[static_cast<size_t>(c) * r.dim + d];
    return s / r.dim;
  };
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return mean(x) < mean(y); });
  std::vector<float> sorted(r.centers.size());
  for (int i = 0; i < r.k; ++i) {
    std::copy_n(r.centers.begin() + static_cast<size_t>(order[i]) * r.dim, r.dim,
                sorted.begin() + static_cast<size_t>(i) * r.dim);
  }
  r.centers = std::move(sorted);
}

// Shared clustering driver. `sample_feature(rng, out)` writes the feature of a
// random raw deal; `class_feature(index, out)` writes the feature of a class.
void cluster_street(const Args& a, int street, uint64_t num_classes, int dim,
                    const std::function<void(std::mt19937_64&, float*)>& sample_feature,
                    const std::function<void(uint64_t, float*)>& class_feature) {
  const char* names[] = {"preflop", "flop", "turn", "river"};
  auto start = Clock::now();
  std::vector<float> points(static_cast<size_t>(a.sample) * dim);
  parallel_ranges(a.sample, a.threads, [&](uint64_t b, uint64_t e, int t) {
    std::mt19937_64 rng(a.seed * 1000003 + street * 101 + t);
    for (uint64_t i = b; i < e; ++i) sample_feature(rng, points.data() + i * dim);
  });
  std::printf("  %s: %d sample features in %.0fs\n", names[street], a.sample, seconds_since(start));

  start = Clock::now();
  KMeansOptions opt;
  opt.k = a.buckets[street];
  opt.max_iterations = a.iters;
  opt.seed = a.seed + street;
  opt.threads = a.threads;
  KMeansResult r = kmeans_fit(points, dim, opt);
  sort_centers(r);
  std::printf("  %s: k-means k=%d, %d iterations, %.0fs, explained variance %.4f, reseeded %d\n",
              names[street], r.k, r.iterations, seconds_since(start),
              1.0 - r.inertia / r.total_variance, r.reseeded);

  start = Clock::now();
  std::vector<uint16_t> buckets(num_classes);
  run_with_progress(names[street], num_classes, a.threads, 1 << 12,
                    [&](uint64_t b, uint64_t e, int) {
                      std::vector<float> f(dim);
                      for (uint64_t i = b; i < e; ++i) {
                        class_feature(i, f.data());
                        buckets[i] = static_cast<uint16_t>(nearest_center(r.centers.data(), r.k, dim, f.data()));
                      }
                    });
  write_table(path(a, street_file_name(street)), kBucketTable, num_classes, 1, r.k, buckets);
  std::printf("  %s: assigned %lu classes in %.0fs\n", names[street], num_classes, seconds_since(start));
}

void stage_river(const Args& a) {
  const holdem::HandIndexer river({2, 5});
  uint32_t k = 0;
  const std::vector<uint16_t> ochs =
      read_table(path(a, "river_ochs.bin"), kRiverOchsTable, river.size(), a.clusters, &k);
  const int dim = static_cast<int>(k);
  cluster_street(
      a, 3, river.size(), dim,
      [&](std::mt19937_64& rng, float* out) {
        const holdem::Deal d = holdem::sample_deal(rng);
        const Card cards[7] = {d.hole[0][0], d.hole[0][1], d.board[0], d.board[1],
                               d.board[2],   d.board[3],   d.board[4]};
        const uint64_t idx = river.index(cards);
        for (int c = 0; c < dim; ++c) out[c] = dequantize(ochs[idx * dim + c]);
      },
      [&](uint64_t idx, float* out) {
        for (int c = 0; c < dim; ++c) out[c] = dequantize(ochs[idx * dim + c]);
      });
}

void stage_turn(const Args& a) {
  const RiverEquityTable table = load_equity(a);
  const holdem::HandIndexer turn({2, 4});
  cluster_street(
      a, 2, turn.size(), kTurnFeatureDim,
      [&](std::mt19937_64& rng, float* out) {
        const holdem::Deal d = holdem::sample_deal(rng);
        turn_feature(table, d.hole[0], d.board, out);
      },
      [&](uint64_t idx, float* out) {
        Card cards[6];
        turn.unindex(idx, cards);
        turn_feature(table, cards, cards + 2, out);
      });
}

void stage_flop(const Args& a) {
  const RiverEquityTable table = load_equity(a);
  const holdem::HandIndexer flop({2, 3});
  cluster_street(
      a, 1, flop.size(), kFlopFeatureDim,
      [&](std::mt19937_64& rng, float* out) {
        const holdem::Deal d = holdem::sample_deal(rng);
        flop_feature(table, d.hole[0], d.board, out);
      },
      [&](uint64_t idx, float* out) {
        Card cards[5];
        flop.unindex(idx, cards);
        flop_feature(table, cards, cards + 2, out);
      });
}

// Equity of a random deal's seat-0 hand on `street`, from the river table.
double street_equity(const RiverEquityTable& table, int street, const holdem::Deal& d) {
  float scratch[kFlopFeatureDim > kTurnFeatureDim ? kFlopFeatureDim : kTurnFeatureDim];
  if (street == 3) return table.lookup(d.hole[0], d.board);
  if (street == 2) return turn_feature(table, d.hole[0], d.board, scratch);
  return flop_feature(table, d.hole[0], d.board, scratch);
}

void stage_report(const Args& a) {
  const CardAbstraction abs = CardAbstraction::load(a.out);
  const RiverEquityTable table = load_equity(a);
  const char* names[] = {"preflop", "flop", "turn", "river"};
  const int samples = std::min(a.sample, 500000);
  std::printf("  %-7s %8s %10s %12s %12s %12s\n", "street", "buckets", "unseen", "max share",
              "median share", "equity R^2");
  for (int street = 1; street < 4; ++street) {
    const int k = abs.num_buckets(street);
    std::vector<int> bucket(samples);
    std::vector<double> equity(samples);
    parallel_ranges(samples, a.threads, [&](uint64_t b, uint64_t e, int t) {
      std::mt19937_64 rng(a.seed * 7919 + street * 31 + t);
      for (uint64_t i = b; i < e; ++i) {
        const holdem::Deal d = holdem::sample_deal(rng);
        bucket[i] = abs.bucket(street, d.hole[0], d.board);
        equity[i] = street_equity(table, street, d);
      }
    });
    std::vector<double> sum(k, 0.0);
    std::vector<int> count(k, 0);
    double total = 0.0;
    for (int i = 0; i < samples; ++i) {
      sum[bucket[i]] += equity[i];
      ++count[bucket[i]];
      total += equity[i];
    }
    const double mean = total / samples;
    double within = 0.0, overall = 0.0;
    for (int i = 0; i < samples; ++i) {
      const double bucket_mean = sum[bucket[i]] / count[bucket[i]];
      within += (equity[i] - bucket_mean) * (equity[i] - bucket_mean);
      overall += (equity[i] - mean) * (equity[i] - mean);
    }
    std::vector<int> sorted = count;
    std::sort(sorted.begin(), sorted.end());
    const int unseen = static_cast<int>(std::count(count.begin(), count.end(), 0));
    std::printf("  %-7s %8d %10d %11.3f%% %11.4f%% %12.4f\n", names[street], k, unseen,
                100.0 * sorted.back() / samples, 100.0 * sorted[k / 2] / samples,
                1.0 - within / overall);
  }

  // Spot checks: bucket ids are ordered by strength, so strong hands should
  // land near the top and weak hands near the bottom.
  struct Spot {
    int street;
    const char* hole;
    const char* board;
    const char* note;
  };
  const Spot spots[] = {
      {3, "AsKs", "QsJsTs2d3c", "royal flush"},
      {3, "7c2d", "AsKsQh9h4c", "seven high, no draw left"},
      {2, "AhAd", "AsKd7c2h", "top set"},
      {2, "8h9h", "6h7hKc2s", "open-ended straight flush draw"},
      {2, "3c2d", "AsKdQh9c", "nothing"},
      {1, "AhKh", "QhJhTh", "flopped royal flush"},
      {1, "7c2d", "AsKsQh", "nothing"},
  };
  std::printf("  spot checks (bucket / buckets on street):\n");
  for (const Spot& s : spots) {
    const auto hole = holdem::parse_cards(s.hole), board = holdem::parse_cards(s.board);
    std::printf("    %-6s %s | %-10s -> %5d / %d  (%s)\n", names[s.street], s.hole, s.board,
                abs.bucket(s.street, hole.data(), board.data()), abs.num_buckets(s.street), s.note);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  std::filesystem::create_directories(a.out);
  std::printf("output: %s  threads: %d  buckets: %d/%d/%d/%d  sample: %d  seed: %lu\n",
              a.out.c_str(), a.threads, a.buckets[0], a.buckets[1], a.buckets[2], a.buckets[3],
              a.sample, a.seed);
  const std::pair<const char*, void (*)(const Args&)> stages[] = {
      {"preflop", stage_preflop}, {"equity", stage_equity}, {"river", stage_river},
      {"turn", stage_turn},       {"flop", stage_flop},     {"report", stage_report}};
  for (const std::string& name : a.stages) {
    bool found = false;
    for (const auto& [stage_name, fn] : stages) {
      if (name != stage_name) continue;
      found = true;
      std::printf("== %s\n", stage_name);
      std::fflush(stdout);
      const auto start = Clock::now();
      fn(a);
      std::printf("== %s done in %.0fs\n", stage_name, seconds_since(start));
      std::fflush(stdout);
    }
    if (!found) {
      std::fprintf(stderr, "unknown stage: %s\n", name.c_str());
      return 2;
    }
  }
  return 0;
}
