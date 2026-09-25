#pragma once

// Regret and average-strategy tables for the blueprint.
//
// All training threads read and write these arrays concurrently without locks
// (as Pluribus did). Accesses go through relaxed std::atomic_ref, which costs
// nothing on x86 and keeps the races well-defined; an occasional lost update is
// harmless noise for MCCFR.

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

namespace poker2::blueprint {

inline float load_relaxed(const float& x) {
  return std::atomic_ref<float>(const_cast<float&>(x)).load(std::memory_order_relaxed);
}
inline void store_relaxed(float& x, float v) {
  std::atomic_ref<float>(x).store(v, std::memory_order_relaxed);
}

// Positive-regret matching over n actions; uniform when no regret is positive.
inline void regret_matching(const float* regret, int n, float* out) {
  float total = 0.0f;
  for (int a = 0; a < n; ++a) {
    const float r = load_relaxed(regret[a]);
    out[a] = r > 0.0f ? r : 0.0f;
    total += out[a];
  }
  if (total > 0.0f) {
    for (int a = 0; a < n; ++a) out[a] /= total;
  } else {
    for (int a = 0; a < n; ++a) out[a] = 1.0f / n;
  }
}

struct TableLayout {
  std::array<int, 4> buckets{};
  std::array<bool, 4> store_average{true, true, false, false};
};

class StrategyTables {
 public:
  StrategyTables(const BettingTree& tree, const TableLayout& layout) : tree_(tree), layout_(layout) {
    for (int s = 0; s <= tree.max_street(); ++s) {
      const size_t size = tree.street_slots(s) * static_cast<size_t>(layout.buckets[s]);
      regret_[s].assign(size, 0.0f);
      if (layout.store_average[s]) average_[s].assign(size, 0.0f);
    }
  }

  const BettingTree& tree() const { return tree_; }
  const TableLayout& layout() const { return layout_; }
  bool has_average(int street) const { return !average_[street].empty(); }

  size_t offset(const TreeNode& n, int bucket) const {
    return n.slot_offset * layout_.buckets[n.street] + static_cast<size_t>(bucket) * n.num_actions;
  }
  float* regret(const TreeNode& n, int bucket) { return regret_[n.street].data() + offset(n, bucket); }
  const float* regret(const TreeNode& n, int bucket) const {
    return regret_[n.street].data() + offset(n, bucket);
  }
  float* average(const TreeNode& n, int bucket) { return average_[n.street].data() + offset(n, bucket); }

  void current_strategy(const TreeNode& n, int bucket, float* out) const {
    regret_matching(regret(n, bucket), n.num_actions, out);
  }

  // Normalized average strategy where stored; the current strategy elsewhere.
  void play_strategy(const TreeNode& n, int bucket, float* out) const {
    if (!has_average(n.street)) {
      current_strategy(n, bucket, out);
      return;
    }
    const float* s = average_[n.street].data() + offset(n, bucket);
    float total = 0.0f;
    for (int a = 0; a < n.num_actions; ++a) total += load_relaxed(s[a]);
    if (total <= 0.0f) {
      current_strategy(n, bucket, out);
      return;
    }
    for (int a = 0; a < n.num_actions; ++a) out[a] = load_relaxed(s[a]) / total;
  }

  // Linear CFR discounting: scales every accumulated regret and average by `factor`.
  // Call only while no training threads are running.
  void scale(float factor, int threads = 1) {
    auto scale_table = [&](std::vector<float>& table) {
      parallel_ranges(table.size(), threads, [&](uint64_t b, uint64_t e, int) {
        for (uint64_t i = b; i < e; ++i) table[i] *= factor;
      });
    };
    for (auto& table : regret_) scale_table(table);
    for (auto& table : average_) scale_table(table);
  }

  uint64_t memory_bytes() const {
    uint64_t total = 0;
    for (int s = 0; s < 4; ++s) total += (regret_[s].size() + average_[s].size()) * sizeof(float);
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
    uint32_t pad;
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
    h.version = 1;
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

  static bool write_all(FILE* f, const std::vector<float>& v) {
    return v.empty() || std::fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
  }
  static bool read_all(FILE* f, std::vector<float>& v) {
    return v.empty() || std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
  }

  const BettingTree& tree_;
  TableLayout layout_;
  std::array<std::vector<float>, 4> regret_;
  std::array<std::vector<float>, 4> average_;
};

}  // namespace poker2::blueprint
