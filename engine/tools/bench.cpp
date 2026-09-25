// Single-thread throughput of the hold'em building blocks.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "holdem/action_abstraction.h"
#include "holdem/isomorphism.h"

using namespace poker2::holdem;
using Clock = std::chrono::steady_clock;

namespace {

template <class F>
void bench(const char* name, uint64_t iterations, F&& f) {
  const auto start = Clock::now();
  uint64_t sink = 0;
  for (uint64_t i = 0; i < iterations; ++i) sink += f(i);
  const double secs = std::chrono::duration<double>(Clock::now() - start).count();
  std::printf("%-34s %8.1f M/s   (checksum %lu)\n", name, iterations / secs / 1e6, sink);
}

}  // namespace

int main() {
  std::mt19937_64 rng(1);
  constexpr int kHands = 1 << 20;
  std::vector<Deal> deals(kHands);
  for (Deal& d : deals) d = sample_deal(rng);

  bench("sample_deal", 20000000, [&](uint64_t) { return sample_deal(rng).board[0]; });

  bench("evaluate 7 cards", 50000000, [&](uint64_t i) {
    return evaluate_seat(deals[i & (kHands - 1)], 0);
  });

  const HandIndexer river({2, 5});
  bench("river isomorphism index", 20000000, [&](uint64_t i) {
    const Deal& d = deals[i & (kHands - 1)];
    const Card cards[7] = {d.hole[0][0], d.hole[0][1], d.board[0], d.board[1],
                           d.board[2],   d.board[3],   d.board[4]};
    return river.index(cards);
  });

  const HandIndexer flop({2, 3});
  bench("flop isomorphism index", 20000000, [&](uint64_t i) {
    const Deal& d = deals[i & (kHands - 1)];
    const Card cards[5] = {d.hole[0][0], d.hole[0][1], d.board[0], d.board[1], d.board[2]};
    return flop.index(cards);
  });

  const ActionAbstraction abs = ActionAbstraction::blueprint();
  std::vector<Action> actions;
  bench("random abstract hand playout", 2000000, [&](uint64_t) {
    HunlState s;
    uint64_t steps = 0;
    while (!s.is_terminal()) {
      abs.actions(s, actions);
      s.apply(actions[rng() % actions.size()]);
      ++steps;
    }
    return steps;
  });
  return 0;
}
