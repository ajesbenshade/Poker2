#pragma once

// Regret and average-strategy tables for the blueprint.
//
// All training threads read and write these arrays concurrently without locks
// (as Pluribus did). Accesses go through relaxed std::atomic_ref, which costs
// nothing on x86 and keeps the races well-defined; an occasional lost update is
// harmless noise for MCCFR.
//
// Precision, chosen so the full blueprint fits in memory:
//   - Regrets are float. Updates are chip-sized, so float resolution relative to
//     the sums is ample (exact board-blind check: float and double tables agree,
//     1.27 vs 1.28 mbb/hand at 256M iterations).
//   - Average strategies add probabilities (<= 1) per visit. Past 2^24 (~16.7M)
//     a float sum cannot absorb even 1.0, and a multi-day run visits each preflop
//     bucket billions of times, so preflop and flop averages are double. Turn and
//     river infosets number ~900M and are each visited rarely, so float is safe.
// Full blueprint (169/2000/2000/1500 buckets): ~22.6 GB, vs ~45 GB all-double.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "blueprint/betting_tree.h"
#include "core/parallel.h"

namespace poker2::blueprint {

template <class T>
T load_relaxed(const T& x) {
  return std::atomic_ref<T>(const_cast<T&>(x)).load(std::memory_order_relaxed);
}
template <class T>
void store_relaxed(T& x, T v) {
  std::atomic_ref<T>(x).store(v, std::memory_order_relaxed);
}

// Positive-regret matching over n actions; uniform when no regret is positive.
inline void regret_matching(const float* regret, int n, float* out) {
  double positive[BettingTree::kMaxActions];
  double total = 0.0;
  for (int a = 0; a < n; ++a) {
    const double r = load_relaxed(regret[a]);
    positive[a] = r > 0.0 ? r : 0.0;
    total += positive[a];
  }
  for (int a = 0; a < n; ++a) {
    out[a] = total > 0.0 ? static_cast<float>(positive[a] / total) : 1.0f / n;
  }
}

struct TableLayout {
  std::array<int, 4> buckets{};
  // Average strategies on every street by default: MCCFR's current strategy does
  // not converge, and reading the turn and river from it roughly doubled LBR.
  std::array<bool, 4> store_average{true, true, true, true};
  // Streets whose averages are double rather than float (see top of file).
  std::array<bool, 4> double_average{true, true, false, false};

  // Bytes per average entry on a street: 0 (not stored), 4 or 8.
  int average_bytes(int street) const {
    return store_average[street] ? (double_average[street] ? 8 : 4) : 0;
  }

  // Table memory a layout needs for a tree, without allocating it.
  uint64_t memory_bytes(const BettingTree& tree) const {
    uint64_t total = 0;
    for (int s = 0; s <= tree.max_street(); ++s) {
      const uint64_t entries = tree.street_slots(s) * static_cast<uint64_t>(buckets[s]);
      total += entries * (sizeof(float) + average_bytes(s));
    }
    return total;
  }
};

// Checkpoint file header, shared by StrategyTables and CheckpointReader.
// After it, for each street 0..3: float regrets, then double averages, then
// float averages (each array present only when that street stores it).
struct CheckpointHeader {
  char magic[8];
  uint32_t version;
  uint32_t pad;
  uint64_t iterations;
  uint64_t num_nodes;
  uint64_t num_terminals;
  uint64_t street_slots[4];
  int32_t buckets[4];
  uint8_t average_bytes[4];  // per street: 0 (none), 4 (float), 8 (double); regrets are float
  uint32_t pad2;

  // The header a checkpoint for this tree and layout must have.
  static CheckpointHeader expected(const BettingTree& tree, const TableLayout& layout, uint64_t iterations) {
    CheckpointHeader h{};
    std::memcpy(h.magic, "P2BLUEP", 8);
    h.version = 3;
    h.iterations = iterations;
    h.num_nodes = tree.num_nodes();
    h.num_terminals = tree.num_terminals();
    for (int s = 0; s < 4; ++s) {
      h.street_slots[s] = tree.street_slots(s);
      h.buckets[s] = layout.buckets[s];
      h.average_bytes[s] = static_cast<uint8_t>(s <= tree.max_street() ? layout.average_bytes(s) : 0);
    }
    return h;
  }

  // Same tree, layout and precision (ignores the iteration count).
  bool matches(const CheckpointHeader& o) const {
    return std::memcmp(magic, o.magic, 8) == 0 && version == o.version && num_nodes == o.num_nodes &&
           num_terminals == o.num_terminals &&
           std::memcmp(street_slots, o.street_slots, sizeof(street_slots)) == 0 &&
           std::memcmp(buckets, o.buckets, sizeof(buckets)) == 0 &&
           std::memcmp(average_bytes, o.average_bytes, sizeof(average_bytes)) == 0;
  }
};

class StrategyTables {
 public:
  StrategyTables(const BettingTree& tree, const TableLayout& layout) : tree_(tree), layout_(layout) {
    for (int s = 0; s <= tree.max_street(); ++s) {
      const size_t size = tree.street_slots(s) * static_cast<size_t>(layout.buckets[s]);
      regret_[s].assign(size, 0.0f);
      if (layout.average_bytes(s) == 8) average_d_[s].assign(size, 0.0);
      if (layout.average_bytes(s) == 4) average_f_[s].assign(size, 0.0f);
    }
  }

  const BettingTree& tree() const { return tree_; }
  const TableLayout& layout() const { return layout_; }
  bool has_average(int street) const { return !average_d_[street].empty() || !average_f_[street].empty(); }

  // For diagnostics: make play_strategy ignore stored averages on these streets.
  void set_play_current(int street, bool current) { play_current_[street] = current; }

  size_t offset(const TreeNode& n, int bucket) const {
    return n.slot_offset * layout_.buckets[n.street] + static_cast<size_t>(bucket) * n.num_actions;
  }
  float* regret(const TreeNode& n, int bucket) { return regret_[n.street].data() + offset(n, bucket); }
  const float* regret(const TreeNode& n, int bucket) const {
    return regret_[n.street].data() + offset(n, bucket);
  }

  // Adds `sigma` to the average-strategy sums at (n, bucket), if stored.
  void add_average(const TreeNode& n, int bucket, const float* sigma) {
    const size_t o = offset(n, bucket);
    if (!average_d_[n.street].empty()) {
      double* s = average_d_[n.street].data() + o;
      for (int a = 0; a < n.num_actions; ++a) store_relaxed(s[a], load_relaxed(s[a]) + sigma[a]);
    } else if (!average_f_[n.street].empty()) {
      float* s = average_f_[n.street].data() + o;
      for (int a = 0; a < n.num_actions; ++a) store_relaxed(s[a], load_relaxed(s[a]) + sigma[a]);
    }
  }

  void current_strategy(const TreeNode& n, int bucket, float* out) const {
    regret_matching(regret(n, bucket), n.num_actions, out);
  }

  // Normalized average strategy where stored; the current strategy elsewhere.
  void play_strategy(const TreeNode& n, int bucket, float* out) const {
    if (!has_average(n.street) || play_current_[n.street]) {
      current_strategy(n, bucket, out);
      return;
    }
    const size_t o = offset(n, bucket);
    double sums[BettingTree::kMaxActions];
    double total = 0.0;
    for (int a = 0; a < n.num_actions; ++a) {
      sums[a] = !average_d_[n.street].empty() ? load_relaxed(average_d_[n.street][o + a])
                                              : load_relaxed(average_f_[n.street][o + a]);
      total += sums[a];
    }
    if (total <= 0.0) {
      current_strategy(n, bucket, out);
      return;
    }
    for (int a = 0; a < n.num_actions; ++a) out[a] = static_cast<float>(sums[a] / total);
  }

  // Linear CFR discounting: scales every accumulated regret and average by `factor`.
  // Call only while no training threads are running.
  void scale(double factor, int threads = 1) {
    auto scale_table = [&](auto& table) {
      using T = typename std::decay_t<decltype(table)>::value_type;
      const T f = static_cast<T>(factor);
      parallel_ranges(table.size(), threads, [&](uint64_t b, uint64_t e, int) {
        for (uint64_t i = b; i < e; ++i) table[i] *= f;
      });
    };
    for (int s = 0; s < 4; ++s) {
      scale_table(regret_[s]);
      scale_table(average_d_[s]);
      scale_table(average_f_[s]);
    }
  }

  uint64_t memory_bytes() const {
    uint64_t total = 0;
    for (int s = 0; s < 4; ++s) {
      total += regret_[s].size() * sizeof(float) + average_d_[s].size() * sizeof(double) +
               average_f_[s].size() * sizeof(float);
    }
    return total;
  }

  void save(const std::string& path, uint64_t iterations) const {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    const CheckpointHeader h = CheckpointHeader::expected(tree_, layout_, iterations);
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    for (int s = 0; s < 4 && ok; ++s) {
      ok = write_all(f, regret_[s]) && write_all(f, average_d_[s]) && write_all(f, average_f_[s]);
    }
    if (std::fclose(f) != 0 || !ok) throw std::runtime_error("write failed: " + tmp);
    // std::filesystem::rename replaces an existing file on every platform (std::rename
    // fails on Windows when the target exists).
    std::filesystem::rename(tmp, path);
  }

  // Returns the saved iteration count. Throws if the file was written for a
  // different tree, bucket layout, or precision.
  uint64_t load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    CheckpointHeader saved{};
    bool ok = std::fread(&saved, sizeof(saved), 1, f) == 1;
    if (!ok || !saved.matches(CheckpointHeader::expected(tree_, layout_, 0))) {
      std::fclose(f);
      throw std::runtime_error("checkpoint does not match this tree, layout and precision: " + path);
    }
    for (int s = 0; s < 4 && ok; ++s) {
      ok = read_all(f, regret_[s]) && read_all(f, average_d_[s]) && read_all(f, average_f_[s]);
    }
    std::fclose(f);
    if (!ok) throw std::runtime_error("truncated checkpoint: " + path);
    return saved.iterations;
  }

  bool operator==(const StrategyTables& o) const {
    return regret_ == o.regret_ && average_d_ == o.average_d_ && average_f_ == o.average_f_;
  }

 private:
  template <class T>
  static bool write_all(FILE* f, const std::vector<T>& v) {
    return v.empty() || std::fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
  }
  template <class T>
  static bool read_all(FILE* f, std::vector<T>& v) {
    return v.empty() || std::fread(v.data(), sizeof(T), v.size(), f) == v.size();
  }

  const BettingTree& tree_;
  TableLayout layout_;
  std::array<std::vector<float>, 4> regret_;
  std::array<std::vector<double>, 4> average_d_;
  std::array<std::vector<float>, 4> average_f_;
  std::array<bool, 4> play_current_{};
};

}  // namespace poker2::blueprint
