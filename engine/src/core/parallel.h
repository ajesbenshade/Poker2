#pragma once

// Minimal std::thread helpers (no OpenMP dependency).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace poker2 {

inline int default_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n ? static_cast<int>(n) : 1;
}

// Dynamic scheduling over [0, n) in chunks: fn(begin, end, thread_id).
// Use for independent per-item work; results must not depend on scheduling.
template <class F>
void parallel_for(uint64_t n, int threads, uint64_t chunk, F&& fn) {
  threads = std::max(1, threads);
  std::atomic<uint64_t> next{0};
  auto worker = [&](int tid) {
    for (;;) {
      const uint64_t begin = next.fetch_add(chunk);
      if (begin >= n) break;
      fn(begin, std::min(n, begin + chunk), tid);
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(worker, t);
  worker(0);
  for (std::thread& t : pool) t.join();
}

// Static partition into `threads` contiguous ranges: fn(begin, end, thread_id).
// Thread t always gets the same range, so per-thread partial results merged in
// thread order are deterministic for a fixed thread count.
template <class F>
void parallel_ranges(uint64_t n, int threads, F&& fn) {
  threads = std::max(1, threads);
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    const uint64_t begin = n * t / threads, end = n * (t + 1) / threads;
    if (t == threads - 1) fn(begin, end, t);
    else pool.emplace_back([&fn, begin, end, t] { fn(begin, end, t); });
  }
  for (std::thread& t : pool) t.join();
}

}  // namespace poker2
