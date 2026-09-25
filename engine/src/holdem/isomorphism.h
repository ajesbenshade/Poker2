#pragma once

// Perfect index over suit-isomorphism classes of hold'em hands.
//
// Cards are dealt in ordered groups (hole 2, flop 3, turn 1, river 1); order
// within a group does not matter and suits can be relabeled freely. For each
// suit, its "column" is the tuple of rank sets it holds in each group, and its
// "shape" is the tuple of those set sizes. Two hands are isomorphic exactly
// when their multisets of four columns are equal, so a class is:
//   1. a configuration: the multiset of the four shapes, then
//   2. for each distinct shape, a multiset of that many columns.
// The index is configuration offset + mixed-radix combination of the
// per-shape multiset ranks. Columns are ranked with the combinatorial number
// system: each group's rank set is a combination of the ranks not already
// used by earlier groups of the same suit.
//
// Group sizes are free. For card abstraction the board is one group, since
// hand strength ignores which street a board card arrived on:
//   {2} 169, {2,3} 1,286,792, {2,4} 13,960,050, {2,5} 123,156,254.
// Dealing the board street by street ({2,3,1}, {2,3,1,1}) keeps that order and
// gives 55,190,538 and 2,428,287,420 classes (Waugh 2013).

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "holdem/cards.h"

namespace poker2::holdem {

namespace detail {

inline uint64_t binomial(int64_t n, int64_t k) {
  if (k < 0 || n < k) return 0;
  uint64_t result = 1;
  for (int64_t i = 1; i <= k; ++i) {
    result = result * static_cast<uint64_t>(n - k + i) / static_cast<uint64_t>(i);
  }
  return result;
}

// Largest p in [lo, hi] with binomial(p, k) <= r. Requires binomial(lo, k) <= r.
inline int64_t largest_with_binomial_at_most(uint64_t r, int64_t k, int64_t lo, int64_t hi) {
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo + 1) / 2;
    if (binomial(mid, k) <= r) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

}  // namespace detail

class HandIndexer {
 public:
  static constexpr int kMaxGroups = 4;

  // group_sizes: e.g. {2} preflop, {2,3} flop, {2,4} turn, {2,5} river.
  explicit HandIndexer(std::vector<int> group_sizes) : sizes_(std::move(group_sizes)) {
    if (sizes_.empty() || static_cast<int>(sizes_.size()) > kMaxGroups) {
      throw std::invalid_argument("HandIndexer supports 1 to 4 groups");
    }
    for (int s : sizes_) total_cards_ += s;
    std::array<int, kMaxGroups> remaining{};
    for (size_t g = 0; g < sizes_.size(); ++g) remaining[g] = sizes_[g];
    std::array<ShapeKey, kNumSuits> chosen{};
    enumerate_configs(0, kMaxShapeKey, remaining, chosen);
  }

  int num_groups() const { return static_cast<int>(sizes_.size()); }
  int num_cards() const { return total_cards_; }
  uint64_t size() const { return size_; }

  // `cards` holds num_cards() cards laid out group by group.
  uint64_t index(const Card* cards) const {
    SuitColumns cols = columns_of(cards);
    std::array<int, kNumSuits> order = {0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (cols.shape[a] != cols.shape[b]) return cols.shape[a] > cols.shape[b];
      return cols.index[a] < cols.index[b];
    });

    uint64_t key = 0;
    for (int s : order) key = (key << 12) | cols.shape[s];
    const Config& config = configs_[config_lookup_.at(key)];

    uint64_t idx = 0;
    int pos = 0;
    for (const ShapeGroup& group : config.groups) {
      // Multiset rank: sorted a_1 <= ... <= a_m, b_k = a_k + k - 1, rank = sum C(b_k, k).
      uint64_t rank = 0;
      for (int k = 1; k <= group.multiplicity; ++k, ++pos) {
        rank += detail::binomial(static_cast<int64_t>(cols.index[order[pos]]) + k - 1, k);
      }
      idx = idx * group.multiset_count + rank;
    }
    return config.offset + idx;
  }

  // Writes a canonical representative of class `idx` (cards group by group).
  void unindex(uint64_t idx, Card* cards) const {
    if (idx >= size_) throw std::out_of_range("hand index out of range");
    auto it = std::upper_bound(offsets_.begin(), offsets_.end(), idx);
    const Config& config = configs_[static_cast<size_t>(it - offsets_.begin()) - 1];
    uint64_t rem = idx - config.offset;

    std::vector<uint64_t> ranks(config.groups.size());
    for (size_t d = config.groups.size(); d-- > 0;) {
      ranks[d] = rem % config.groups[d].multiset_count;
      rem /= config.groups[d].multiset_count;
    }

    std::array<std::array<uint32_t, kMaxGroups>, kNumSuits> masks{};
    int suit = 0;
    for (size_t d = 0; d < config.groups.size(); ++d) {
      const ShapeGroup& group = config.groups[d];
      std::vector<uint64_t> column_indices(group.multiplicity);
      uint64_t r = ranks[d];
      for (int k = group.multiplicity; k >= 1; --k) {
        const int64_t upper = static_cast<int64_t>(group.column_count) + k - 2;
        const int64_t b = detail::largest_with_binomial_at_most(r, k, k - 1, upper);
        r -= detail::binomial(b, k);
        column_indices[k - 1] = static_cast<uint64_t>(b - (k - 1));
      }
      for (uint64_t column : column_indices) {
        decode_column(group.shape, column, masks[suit]);
        ++suit;
      }
    }

    int out = 0;
    for (int g = 0; g < num_groups(); ++g) {
      for (int s = 0; s < kNumSuits; ++s) {
        for (uint32_t m = masks[s][g]; m; m &= m - 1) {
          cards[out++] = make_card(__builtin_ctz(m), s);
        }
      }
    }
  }

 private:
  // Shape: 3 bits per group of the suit's card count in that group.
  using ShapeKey = uint32_t;
  static constexpr ShapeKey kMaxShapeKey = 0xFFF;

  struct ShapeGroup {
    ShapeKey shape;
    int multiplicity;
    uint64_t column_count;    // distinct columns with this shape
    uint64_t multiset_count;  // C(column_count + multiplicity - 1, multiplicity)
  };

  struct Config {
    std::vector<ShapeGroup> groups;  // distinct shapes, in descending shape order
    uint64_t offset;
  };

  struct SuitColumns {
    std::array<ShapeKey, kNumSuits> shape;
    std::array<uint64_t, kNumSuits> index;
  };

  int count_in_group(ShapeKey shape, int g) const { return (shape >> (3 * g)) & 7; }

  // Radix for group g of a column: choose n_g ranks from those unused by earlier groups.
  uint64_t group_radix(ShapeKey shape, int g) const {
    int used = 0;
    for (int j = 0; j < g; ++j) used += count_in_group(shape, j);
    return detail::binomial(kNumRanks - used, count_in_group(shape, g));
  }

  uint64_t column_count(ShapeKey shape) const {
    uint64_t count = 1;
    for (int g = 0; g < num_groups(); ++g) count *= group_radix(shape, g);
    return count;
  }

  SuitColumns columns_of(const Card* cards) const {
    std::array<std::array<uint32_t, kMaxGroups>, kNumSuits> masks{};
    int pos = 0;
    for (int g = 0; g < num_groups(); ++g) {
      for (int i = 0; i < sizes_[g]; ++i, ++pos) {
        masks[suit_of(cards[pos])][g] |= 1u << rank_of(cards[pos]);
      }
    }
    SuitColumns cols{};
    for (int s = 0; s < kNumSuits; ++s) {
      uint32_t used = 0;
      ShapeKey shape = 0;
      uint64_t idx = 0;
      for (int g = 0; g < num_groups(); ++g) {
        const uint32_t m = masks[s][g];
        const int n = __builtin_popcount(m);
        shape |= static_cast<ShapeKey>(n) << (3 * g);
        // Colex rank of m among the ranks not in `used`.
        uint64_t sub = 0;
        int k = 1;
        for (uint32_t rest = m; rest; rest &= rest - 1, ++k) {
          const int r = __builtin_ctz(rest);
          const int position = r - __builtin_popcount(used & ((1u << r) - 1));
          sub += detail::binomial(position, k);
        }
        idx = idx * detail::binomial(kNumRanks - __builtin_popcount(used), n) + sub;
        used |= m;
      }
      cols.shape[s] = shape;
      cols.index[s] = idx;
    }
    return cols;
  }

  void decode_column(ShapeKey shape, uint64_t column,
                     std::array<uint32_t, kMaxGroups>& masks) const {
    std::array<uint64_t, kMaxGroups> subs{};
    for (int g = num_groups() - 1; g >= 0; --g) {
      const uint64_t radix = group_radix(shape, g);
      subs[g] = column % radix;
      column /= radix;
    }
    uint32_t used = 0;
    for (int g = 0; g < num_groups(); ++g) {
      const int n = count_in_group(shape, g);
      const int available = kNumRanks - __builtin_popcount(used);
      uint64_t r = subs[g];
      uint32_t m = 0;
      for (int k = n; k >= 1; --k) {
        const int64_t position = detail::largest_with_binomial_at_most(r, k, k - 1, available - 1);
        r -= detail::binomial(position, k);
        // Map the position among unused ranks back to an actual rank.
        int seen = -1;
        for (int rank = 0; rank < kNumRanks; ++rank) {
          if (used & (1u << rank)) continue;
          if (++seen == position) {
            m |= 1u << rank;
            break;
          }
        }
      }
      masks[g] = m;
      used |= m;
    }
  }

  // Chooses shapes for suits in non-increasing order so each multiset appears once.
  void enumerate_configs(int suit, ShapeKey max_shape, std::array<int, kMaxGroups>& remaining,
                         std::array<ShapeKey, kNumSuits>& chosen) {
    if (suit == kNumSuits) {
      for (int g = 0; g < num_groups(); ++g) {
        if (remaining[g] != 0) return;
      }
      add_config(chosen);
      return;
    }
    for (ShapeKey shape = max_shape + 1; shape-- > 0;) {
      bool valid = true;
      for (int g = 0; g < kMaxGroups && valid; ++g) {
        const int n = count_in_group(shape, g);
        if (g >= num_groups() ? n != 0 : n > remaining[g]) valid = false;
      }
      if (!valid) continue;
      for (int g = 0; g < num_groups(); ++g) remaining[g] -= count_in_group(shape, g);
      chosen[suit] = shape;
      enumerate_configs(suit + 1, shape, remaining, chosen);
      for (int g = 0; g < num_groups(); ++g) remaining[g] += count_in_group(shape, g);
    }
  }

  void add_config(const std::array<ShapeKey, kNumSuits>& shapes) {
    Config config;
    config.offset = size_;
    uint64_t key = 0;
    uint64_t count = 1;
    for (int s = 0; s < kNumSuits; ++s) {
      key = (key << 12) | shapes[s];
      if (s > 0 && shapes[s] == shapes[s - 1]) {
        ++config.groups.back().multiplicity;
      } else {
        config.groups.push_back({shapes[s], 1, column_count(shapes[s]), 0});
      }
    }
    for (ShapeGroup& group : config.groups) {
      group.multiset_count =
          detail::binomial(static_cast<int64_t>(group.column_count) + group.multiplicity - 1,
                           group.multiplicity);
      count *= group.multiset_count;
    }
    config_lookup_.emplace(key, configs_.size());
    offsets_.push_back(config.offset);
    configs_.push_back(std::move(config));
    size_ += count;
  }

  std::vector<int> sizes_;
  int total_cards_ = 0;
  uint64_t size_ = 0;
  std::vector<Config> configs_;
  std::vector<uint64_t> offsets_;
  std::unordered_map<uint64_t, size_t> config_lookup_;
};

}  // namespace poker2::holdem
