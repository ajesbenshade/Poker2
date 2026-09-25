#pragma once

// Regret and average-strategy tables for the blueprint.
//
// All training threads read and write these arrays concurrently without locks
// (as Pluribus did). Accesses go through relaxed std::atomic_ref, which costs
// nothing on x86 and keeps the races well-defined; an occasional lost update is
// harmless noise for MCCFR.
//
// Table values are double by default. At blueprint scale, float32 stops
// accumulating: past 2^24 (~16.7M) adding 1.0 is lost entirely, and an 18-day
// run visits each preflop bucket ~2 billion times. Up to 256M iterations the
// exact board-blind check shows no float/double difference (1.27 vs 1.28
// mbb/hand), so -DPOKER2_TABLE_VALUE=float is an option for short runs where
// memory matters.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "blueprint/betting_tree.h"
#include "core/parallel.h"

#ifndef POKER2_TABLE_VALUE
#define POKER2_TABLE_VALUE double
#endif

namespace poker2::blueprint {

using TableValue = POKER2_TABLE_VALUE;

template <class T>
T load_relaxed(const T& x) {
  return std::atomic_ref<T>(const_cast<T&>(x)).load(std::memory_order_relaxed);
}
template <class T>
void store_relaxed(T& x, T v) {
  std::atomic_ref<T>(x).store(v, std::memory_order_relaxed);
}

// Positive-regret matching over n actions; uniform when no regret is positive.
inline void regret_matching(const TableValue* regret, int n, float* out) {
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
};

class StrategyTables {
 public:
  StrategyTables(const BettingTree& tree, const TableLayout& layout) : tree_(tree), layout_(layout) {
    for (int s = 0; s <= tree.max_street(); ++s) {
      const size_t size = tree.street_slots(s) * static_cast<size_t>(layout.buckets[s]);
      regret_[s].assign(size, TableValue(0));
      if (layout.store_average[s]) average_[s].assign(size, TableValue(0));
    }
  }

  const BettingTree& tree() const { return tree_; }
  const TableLayout& layout() const { return layout_; }
  bool has_average(int street) const { return !average_[street].empty(); }

  // For diagnostics: make play_strategy ignore stored averages on these streets.
  void set_play_current(int street, bool current) { play_current_[street] = current; }

  size_t offset(const TreeNode& n, int bucket) const {
    return n.slot_offset * layout_.buckets[n.street] + static_cast<size_t>(bucket) * n.num_actions;
  }
  TableValue* regret(const TreeNode& n, int bucket) { return regret_[n.street].data() + offset(n, bucket); }
  const TableValue* regret(const TreeNode& n, int bucket) const {
    return regret_[n.street].data() + offset(n, bucket);
  }
  TableValue* average(const TreeNode& n, int bucket) {
    return average_[n.street].data() + offset(n, bucket);
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
    const TableValue* s = average_[n.street].data() + offset(n, bucket);
    double total = 0.0;
    for (int a = 0; a < n.num_actions; ++a) total += load_relaxed(s[a]);
    if (total <= 0.0) {
      current_strategy(n, bucket, out);
      return;
    }
    for (int a = 0; a < n.num_actions; ++a) out[a] = static_cast<float>(load_relaxed(s[a]) / total);
  }

  // Linear CFR discounting: scales every accumulated regret and average by `factor`.
  // Call only while no training threads are running.
  void scale(double factor, int threads = 1) {
    const TableValue f = static_cast<TableValue>(factor);
    auto scale_table = [&](std::vector<TableValue>& table) {
      parallel_ranges(table.size(), threads, [&](uint64_t b, uint64_t e, int) {
        for (uint64_t i = b; i < e; ++i) table[i] *= f;
      });
    };
    for (auto& table : regret_) scale_table(table);
    for (auto& table : average_) scale_table(table);
  }

  uint64_t memory_bytes() const {
    uint64_t total = 0;
    for (int s = 0; s < 4; ++s) total += (regret_[s].size() + average_[s].size()) * sizeof(TableValue);
    return total;
  }

  void save(const std::string& path, uint64_t iterations) const {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    Header h = header(iterations);
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    for (int s = 0; s < 4 && ok; ++s) {
      ok = write_all(f, regret_[s]) && write_all(f, average_[s]);
    }
    if (std::fclose(f) != 0 || !ok) throw std::runtime_error("write failed: " + tmp);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed: " + path);
  }

  // Returns the saved iteration count. Throws if the file was written for a
  // different tree or bucket layout.
  uint64_t load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    Header saved{};
    const Header expected = header(0);
    bool ok = std::fread(&saved, sizeof(saved), 1, f) == 1;
    if (!ok || std::memcmp(saved.magic, expected.magic, 8) != 0 || saved.version != expected.version ||
        saved.value_bytes != expected.value_bytes ||
        saved.num_nodes != expected.num_nodes || saved.num_terminals != expected.num_terminals ||
        std::memcmp(saved.street_slots, expected.street_slots, sizeof(saved.street_slots)) != 0 ||
        std::memcmp(saved.buckets, expected.buckets, sizeof(saved.buckets)) != 0 ||
        std::memcmp(saved.store_average, expected.store_average, sizeof(saved.store_average)) != 0) {
      std::fclose(f);
      throw std::runtime_error("checkpoint does not match this tree and layout: " + path);
    }
    for (int s = 0; s < 4 && ok; ++s) ok = read_all(f, regret_[s]) && read_all(f, average_[s]);
    std::fclose(f);
    if (!ok) throw std::runtime_error("truncated checkpoint: " + path);
    return saved.iterations;
  }

  bool operator==(const StrategyTables& o) const { return regret_ == o.regret_ && average_ == o.average_; }

 private:
  struct Header {
    char magic[8];
    uint32_t version;
    uint32_t value_bytes;  // sizeof(TableValue) the file was written with
    uint64_t iterations;
    uint64_t num_nodes;
    uint64_t num_terminals;
    uint64_t street_slots[4];
    int32_t buckets[4];
    uint8_t store_average[4];
    uint32_t pad2;
  };

  Header header(uint64_t iterations) const {
    Header h{};
    std::memcpy(h.magic, "P2BLUEP", 8);
    h.version = 2;
    h.value_bytes = sizeof(TableValue);
    h.iterations = iterations;
    h.num_nodes = tree_.num_nodes();
    h.num_terminals = tree_.num_terminals();
    for (int s = 0; s < 4; ++s) {
      h.street_slots[s] = tree_.street_slots(s);
      h.buckets[s] = layout_.buckets[s];
      h.store_average[s] = !average_[s].empty();
    }
    return h;
  }

  static bool write_all(FILE* f, const std::vector<TableValue>& v) {
    return v.empty() || std::fwrite(v.data(), sizeof(TableValue), v.size(), f) == v.size();
  }
  static bool read_all(FILE* f, std::vector<TableValue>& v) {
    return v.empty() || std::fread(v.data(), sizeof(TableValue), v.size(), f) == v.size();
  }

  const BettingTree& tree_;
  TableLayout layout_;
  std::array<std::vector<TableValue>, 4> regret_;
  std::array<std::vector<TableValue>, 4> average_;
  std::array<bool, 4> play_current_{};
};

}  // namespace poker2::blueprint
