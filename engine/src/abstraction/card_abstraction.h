#pragma once

// Runtime card abstraction: maps (street, hole cards, board) to a bucket.
// Preflop is lossless (the 169 classes); later streets read bucket tables
// indexed by suit-isomorphism class, board treated as one group.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "abstraction/table_file.h"
#include "holdem/cards.h"
#include "holdem/isomorphism.h"

namespace poker2::abstraction {

inline const char* street_file_name(int street) {
  static const char* names[] = {"buckets_preflop.bin", "buckets_flop.bin", "buckets_turn.bin",
                                "buckets_river.bin"};
  return names[street];
}

class CardAbstraction {
 public:
  CardAbstraction()
      : indexers_{holdem::HandIndexer({2}), holdem::HandIndexer({2, 3}),
                  holdem::HandIndexer({2, 4}), holdem::HandIndexer({2, 5})} {}

  static CardAbstraction load(const std::string& dir) {
    CardAbstraction a;
    for (int s = 0; s < 4; ++s) {
      uint32_t buckets = 0;
      a.tables_[s] = read_table(dir + "/" + street_file_name(s), kBucketTable,
                                a.indexers_[s].size(), 1, &buckets);
      a.num_buckets_[s] = static_cast<int>(buckets);
    }
    return a;
  }

  // For tests and tools: install a table directly.
  void set_table(int street, std::vector<uint16_t> table, int num_buckets) {
    tables_[street] = std::move(table);
    num_buckets_[street] = num_buckets;
  }

  const holdem::HandIndexer& indexer(int street) const { return indexers_[street]; }
  int num_buckets(int street) const { return num_buckets_[street]; }

  // board must hold 0, 3, 4 or 5 cards for streets 0..3.
  int bucket(int street, const holdem::Card hole[2], const holdem::Card* board) const {
    static constexpr int kBoardCards[] = {0, 3, 4, 5};
    holdem::Card cards[7] = {hole[0], hole[1]};
    for (int i = 0; i < kBoardCards[street]; ++i) cards[2 + i] = board[i];
    return tables_[street][indexers_[street].index(cards)];
  }

 private:
  std::array<holdem::HandIndexer, 4> indexers_;
  std::array<std::vector<uint16_t>, 4> tables_;
  std::array<int, 4> num_buckets_{};
};

}  // namespace poker2::abstraction
