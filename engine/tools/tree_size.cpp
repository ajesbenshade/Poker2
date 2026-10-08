// Sizes the blueprint's abstract betting tree and its table memory.
//
//   poker2_tree [--buckets PRE,FLOP,TURN,RIVER]
//
// Memory model (blueprint/strategy.h defaults): one float regret per (node,
// action, bucket) on every street, plus an average-strategy entry that is double
// on preflop and flop and float on turn and river.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "holdem/action_abstraction.h"

using namespace poker2::holdem;

int main(int argc, char** argv) {
  unsigned long long buckets[4] = {169, 2000, 2000, 1500};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--buckets") && i + 1 < argc) {
      if (std::sscanf(argv[++i], "%llu,%llu,%llu,%llu", &buckets[0], &buckets[1], &buckets[2],
                      &buckets[3]) != 4) {
        std::fprintf(stderr, "--buckets expects PRE,FLOP,TURN,RIVER\n");
        return 2;
      }
    } else {
      std::fprintf(stderr, "usage: poker2_tree [--buckets PRE,FLOP,TURN,RIVER]\n");
      return 2;
    }
  }

  const ActionAbstraction abstraction = ActionAbstraction::blueprint();
  TreeStats stats;
  const auto start = std::chrono::steady_clock::now();
  count_tree(HunlState(), abstraction, stats);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  const char* names[] = {"preflop", "flop", "turn", "river"};
  double total_gb = 0.0;
  unsigned long long total_infosets = 0;
  std::printf("%-8s %14s %14s %8s %16s %10s\n", "street", "nodes", "action slots", "buckets",
              "infosets", "GB");
  for (int s = 0; s < 4; ++s) {
    const unsigned long long nodes = stats.decision_nodes[s], slots = stats.action_slots[s];
    const unsigned long long infosets = nodes * buckets[s];
    const double bytes_per_slot = s <= kFlop ? 12.0 : 8.0;
    const double gb = static_cast<double>(slots) * buckets[s] * bytes_per_slot / 1e9;
    total_gb += gb;
    total_infosets += infosets;
    std::printf("%-8s %14llu %14llu %8llu %16llu %10.2f\n", names[s], nodes, slots, buckets[s], infosets, gb);
  }
  std::printf("%-8s %14s %14s %8s %16llu %10.2f\n", "total", "", "", "", total_infosets, total_gb);
  std::printf("terminal nodes: %llu  (walked in %.1fs)\n",
              static_cast<unsigned long long>(stats.terminal_nodes), secs);
  return 0;
}
