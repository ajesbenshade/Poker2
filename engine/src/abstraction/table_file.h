#pragma once

// Binary files for abstraction tables: a fixed header followed by raw
// little-endian uint16 values (rows * cols).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker2::abstraction {

struct TableHeader {
  char magic[8];      // "P2TABLE\0"
  uint32_t version;   // 1
  uint32_t kind;      // TableKind
  uint64_t rows;      // number of hand classes
  uint32_t cols;      // values per row
  uint32_t aux;       // buckets for bucket tables, clusters for OCHS tables
};

enum TableKind : uint32_t { kBucketTable = 1, kRiverEquityTable = 2, kRiverOchsTable = 3 };

inline void write_table(const std::string& path, TableKind kind, uint64_t rows, uint32_t cols,
                        uint32_t aux, const std::vector<uint16_t>& values) {
  if (values.size() != rows * cols) throw std::logic_error("table size mismatch for " + path);
  TableHeader h{};
  std::memcpy(h.magic, "P2TABLE", 8);
  h.version = 1;
  h.kind = kind;
  h.rows = rows;
  h.cols = cols;
  h.aux = aux;
  const std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + tmp);
  const bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 &&
                  std::fwrite(values.data(), sizeof(uint16_t), values.size(), f) == values.size();
  if (std::fclose(f) != 0 || !ok) throw std::runtime_error("write failed: " + tmp);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed: " + path);
}

inline std::vector<uint16_t> read_table(const std::string& path, TableKind kind, uint64_t rows,
                                        uint32_t cols, uint32_t* aux_out = nullptr) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  TableHeader h{};
  if (std::fread(&h, sizeof(h), 1, f) != 1 || std::memcmp(h.magic, "P2TABLE", 8) != 0 ||
      h.version != 1) {
    std::fclose(f);
    throw std::runtime_error("not a Poker2 table: " + path);
  }
  if (h.kind != kind || h.rows != rows || h.cols != cols) {
    std::fclose(f);
    throw std::runtime_error("unexpected table shape in " + path);
  }
  std::vector<uint16_t> values(rows * cols);
  const bool ok = std::fread(values.data(), sizeof(uint16_t), values.size(), f) == values.size();
  std::fclose(f);
  if (!ok) throw std::runtime_error("truncated table: " + path);
  if (aux_out) *aux_out = h.aux;
  return values;
}

inline bool file_exists(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

}  // namespace poker2::abstraction
