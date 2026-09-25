// Sizes the blueprint's abstract betting tree and its table memory.
//
//   poker2_tree [--buckets PRE,FLOP,TURN,RIVER]
//
// Memory model: one int32 regret per (node, action, bucket) on every street,
// plus one float average-strategy entry on preflop and flop only (later streets
// are played with real-time search).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "holdem/action_abstraction.h"

using namespace poker2::holdem;

int main(int argc, char** argv) {
  uint64_t buckets[4] = {169, 2000, 2000, 1500};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--buckets") && i + 1 < argc) {
      if (std::sscanf(argv[++i], "%lu,%lu,%lu,%lu", &buckets[0], &buckets[1], &buckets[2],
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
  uint64_t total_infosets = 0;
  std::printf("%-8s %14s %14s %8s %16s %10s\n", "street", "nodes", "action slots", "buckets",
              "infosets", "GB");
  for (int s = 0; s < 4; ++s) {
    const uint64_t infosets = stats.decision_nodes[s] * buckets[s];
    const double bytes_per_slot = s <= kFlop ? 8.0 : 4.0;
    const double gb = static_cast<double>(stats.action_slots[s]) * buckets[s] * bytes_per_slot / 1e9;
    total_gb += gb;
    total_infosets += infosets;
    std::printf("%-8s %14lu %14lu %8lu %16lu %10.2f\n", names[s], stats.decision_nodes[s],
                stats.action_slots[s], buckets[s], infosets, gb);
  }
  std::printf("%-8s %14s %14s %8s %16lu %10.2f\n", "total", "", "", "", total_infosets, total_gb);
  std::printf("terminal nodes: %lu  (walked in %.1fs)\n", stats.terminal_nodes, secs);
  return 0;
}
