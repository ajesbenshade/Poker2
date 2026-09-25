#pragma once

// Card representation: id = rank * 4 + suit, rank 0..12 = 2..A, suit 0..3 = s,h,d,c.

#include <cctype>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker2::holdem {

using Card = uint8_t;

constexpr int kNumCards = 52;
constexpr int kNumRanks = 13;
constexpr int kNumSuits = 4;
constexpr char kRankChars[] = "23456789TJQKA";
constexpr char kSuitChars[] = "shdc";

constexpr int rank_of(Card c) { return c >> 2; }
constexpr int suit_of(Card c) { return c & 3; }
constexpr Card make_card(int rank, int suit) { return static_cast<Card>(rank * 4 + suit); }

inline std::string card_to_string(Card c) {
  return {kRankChars[rank_of(c)], kSuitChars[suit_of(c)]};
}

inline Card parse_card(const std::string& s) {
  if (s.size() != 2) throw std::invalid_argument("bad card: " + s);
  const std::string ranks = kRankChars, suits = kSuitChars;
  const auto r = ranks.find(static_cast<char>(std::toupper(static_cast<unsigned char>(s[0]))));
  const auto u = suits.find(static_cast<char>(std::tolower(static_cast<unsigned char>(s[1]))));
  if (r == std::string::npos || u == std::string::npos) throw std::invalid_argument("bad card: " + s);
  return make_card(static_cast<int>(r), static_cast<int>(u));
}

// "AsKd" -> {As, Kd}
inline std::vector<Card> parse_cards(const std::string& s) {
  if (s.size() % 2) throw std::invalid_argument("bad card list: " + s);
  std::vector<Card> out;
  for (size_t i = 0; i < s.size(); i += 2) out.push_back(parse_card(s.substr(i, 2)));
  return out;
}

inline std::string cards_to_string(const std::vector<Card>& cards) {
  std::string out;
  for (Card c : cards) out += card_to_string(c);
  return out;
}

// All 9 cards of a heads-up hand: two hole cards per seat and a 5-card board.
struct Deal {
  Card hole[2][2];
  Card board[5];
};

// Samples a uniformly random deal with a partial Fisher-Yates shuffle.
template <class Rng>
Deal sample_deal(Rng& rng) {
  Card deck[kNumCards];
  for (int i = 0; i < kNumCards; ++i) deck[i] = static_cast<Card>(i);
  for (int i = 0; i < 9; ++i) {
    std::uniform_int_distribution<int> pick(i, kNumCards - 1);
    const int j = pick(rng);
    const Card tmp = deck[i];
    deck[i] = deck[j];
    deck[j] = tmp;
  }
  Deal d;
  d.hole[0][0] = deck[0];
  d.hole[0][1] = deck[1];
  d.hole[1][0] = deck[2];
  d.hole[1][1] = deck[3];
  for (int i = 0; i < 5; ++i) d.board[i] = deck[4 + i];
  return d;
}

}  // namespace poker2::holdem
