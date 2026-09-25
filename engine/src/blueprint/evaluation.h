#pragma once

// Evaluating a blueprint during training.
//
// local_best_response(): LBR (Lisy & Bowling 2017), the standard lower bound on
//   HUNL exploitability. LBR tracks the blueprint's range from its actions and,
//   at each of its own postflop decisions, picks the action with the best
//   expected value assuming the hand then checks down (for raises it uses the
//   blueprint's fold probability and calling range at the next node). Preflop
//   it always calls. This version stays inside the betting abstraction, so it
//   measures exploitability within the abstraction. Higher is worse.
//
// head_to_head(): duplicate match (every deal played twice with seats swapped)
//   between two agents.

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "abstraction/card_abstraction.h"
#include "blueprint/betting_tree.h"
#include "blueprint/strategy.h"
#include "blueprint/trainer.h"
#include "core/parallel.h"
#include "holdem/evaluator.h"

namespace poker2::blueprint {

using holdem::Card;

struct MatchResult {
  double mbb_per_hand = 0.0;  // for the first agent / for LBR
  double stderr_mbb = 0.0;
  uint64_t hands = 0;
};

namespace detail {

struct Accumulator {
  double sum = 0.0, sum_sq = 0.0;
  uint64_t n = 0;
  void add(double x) {
    sum += x;
    sum_sq += x * x;
    ++n;
  }
};

inline MatchResult summarize(const std::vector<Accumulator>& parts, double big_blind,
                             uint64_t hands_per_sample) {
  Accumulator total;
  for (const Accumulator& p : parts) {
    total.sum += p.sum;
    total.sum_sq += p.sum_sq;
    total.n += p.n;
  }
  MatchResult r;
  if (total.n == 0) return r;
  const double mean = total.sum / total.n;
  const double var = total.n > 1 ? (total.sum_sq - total.n * mean * mean) / (total.n - 1) : 0.0;
  const double scale = 1000.0 / big_blind / hands_per_sample;
  r.mbb_per_hand = mean * scale;
  r.stderr_mbb = std::sqrt(std::max(var, 0.0) / total.n) * scale;
  r.hands = total.n * hands_per_sample;
  return r;
}

inline int sample_index(const float* probs, int n, std::mt19937_64& rng) {
  double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
  for (int a = 0; a < n - 1; ++a) {
    r -= probs[a];
    if (r < 0.0) return a;
  }
  return n - 1;
}

}  // namespace detail

// An agent picks an action index at a decision node.
using Agent = std::function<int(const TreeNode&, uint32_t node_id, const DealView&, int seat,
                                std::mt19937_64&)>;

inline Agent blueprint_agent(const StrategyTables& tables) {
  return [&tables](const TreeNode& n, uint32_t, const DealView& v, int seat, std::mt19937_64& rng) {
    float sigma[BettingTree::kMaxActions];
    tables.play_strategy(n, v.bucket[seat][n.street], sigma);
    return detail::sample_index(sigma, n.num_actions, rng);
  };
}

inline Agent calling_agent(const BettingTree& tree) {
  return [&tree](const TreeNode& n, uint32_t, const DealView&, int, std::mt19937_64&) {
    for (int a = 0; a < n.num_actions; ++a) {
      if (tree.action(n, a).type == holdem::ActionType::kCheckCall) return a;
    }
    return 0;
  };
}

inline Agent random_agent() {
  return [](const TreeNode& n, uint32_t, const DealView&, int, std::mt19937_64& rng) {
    return static_cast<int>(rng() % n.num_actions);
  };
}

inline float play_hand(const BettingTree& tree, const DealView& v, const Agent& seat0,
                       const Agent& seat1, int for_seat, std::mt19937_64& rng) {
  int32_t ref = 0;
  while (ref >= 0) {
    const TreeNode& n = tree.node(static_cast<uint32_t>(ref));
    const Agent& agent = n.player == 0 ? seat0 : seat1;
    ref = tree.child(n, agent(n, static_cast<uint32_t>(ref), v, n.player, rng));
  }
  return terminal_value(tree.terminal(ref), v, for_seat);
}

inline MatchResult head_to_head(const BettingTree& tree, const abstraction::CardAbstraction& cards,
                                const Agent& a, const Agent& b, uint64_t deals, uint64_t seed,
                                int threads) {
  std::vector<detail::Accumulator> parts(std::max(1, threads));
  parallel_ranges(deals, threads, [&](uint64_t begin, uint64_t end, int t) {
    std::mt19937_64 rng(seed * 7777 + t);
    for (uint64_t i = begin; i < end; ++i) {
      const DealView v = view_deal(holdem::sample_deal(rng), cards, tree.max_street());
      const float first = play_hand(tree, v, a, b, 0, rng);
      const float second = play_hand(tree, v, b, a, 1, rng);
      parts[t].add(first + second);
    }
  });
  return detail::summarize(parts, tree.config().big_blind, 2);
}

// ---------------------------------------------------------------------------
// Local best response

namespace detail {

struct Combos {
  std::array<std::array<Card, 2>, 1326> cards;
  Combos() {
    int i = 0;
    for (Card a = 0; a < 52; ++a)
      for (Card b = a + 1; b < 52; ++b) cards[i++] = {a, b};
  }
};

inline const Combos& combos() {
  static const Combos c;
  return c;
}

// Equity of `hole` against each opponent combo on a partial board, averaged
// over runouts (exact on turn and river, `rollouts` samples on the flop).
inline void equity_vs_combos(const Card hole[2], const Card* board, int board_cards,
                             const std::array<float, 1326>& weight, int rollouts,
                             std::mt19937_64& rng, std::array<float, 1326>& equity) {
  const auto& all = combos().cards;
  std::array<float, 1326> score{};
  std::array<uint16_t, 1326> count{};
  uint64_t dead = (uint64_t{1} << hole[0]) | (uint64_t{1} << hole[1]);
  for (int i = 0; i < board_cards; ++i) dead |= uint64_t{1} << board[i];
  std::vector<Card> deck;
  for (Card c = 0; c < 52; ++c)
    if (!(dead >> c & 1)) deck.push_back(c);

  auto accumulate = [&](const Card full_board[5]) {
    uint64_t used = dead;
    for (int i = board_cards; i < 5; ++i) used |= uint64_t{1} << full_board[i];
    Card cards[7] = {hole[0], hole[1], full_board[0], full_board[1], full_board[2], full_board[3],
                     full_board[4]};
    const uint32_t mine = holdem::evaluate(cards, 7);
    for (int h = 0; h < 1326; ++h) {
      if (weight[h] <= 0.0f) continue;
      if ((used >> all[h][0] & 1) || (used >> all[h][1] & 1)) continue;
      cards[0] = all[h][0];
      cards[1] = all[h][1];
      const uint32_t theirs = holdem::evaluate(cards, 7);
      score[h] += mine > theirs ? 1.0f : (mine == theirs ? 0.5f : 0.0f);
      ++count[h];
    }
  };

  Card full[5] = {};
  for (int i = 0; i < board_cards; ++i) full[i] = board[i];
  if (board_cards == 5) {
    accumulate(full);
  } else if (board_cards == 4) {
    for (Card r : deck) {
      full[4] = r;
      accumulate(full);
    }
  } else {
    for (int k = 0; k < rollouts; ++k) {
      const size_t x = rng() % deck.size();
      size_t y = rng() % (deck.size() - 1);
      if (y >= x) ++y;
      full[3] = deck[x];
      full[4] = deck[y];
      accumulate(full);
    }
  }
  for (int h = 0; h < 1326; ++h) equity[h] = count[h] ? score[h] / count[h] : 0.5f;
}

inline double weighted_equity(const std::array<float, 1326>& w, const std::array<float, 1326>& eq) {
  double num = 0.0, den = 0.0;
  for (int h = 0; h < 1326; ++h) {
    num += static_cast<double>(w[h]) * eq[h];
    den += w[h];
  }
  return den > 0.0 ? num / den : 0.5;
}

}  // namespace detail

struct LbrOptions {
  uint64_t hands = 20000;
  uint64_t seed = 1;
  int threads = 1;
  int flop_rollouts = 24;
};

inline MatchResult local_best_response(const BettingTree& tree, const StrategyTables& tables,
                                       const abstraction::CardAbstraction& cards,
                                       const LbrOptions& opt) {
  static constexpr int kBoardCards[] = {0, 3, 4, 5};
  const auto& all = detail::combos().cards;
  std::vector<detail::Accumulator> parts(std::max(1, opt.threads));

  parallel_ranges(opt.hands, opt.threads, [&](uint64_t begin, uint64_t end, int t) {
    std::mt19937_64 rng(opt.seed * 104729 + t);
    std::array<float, 1326> weight, equity, calling;
    std::array<uint16_t, 1326> bucket_of;
    float sigma[BettingTree::kMaxActions];

    for (uint64_t hand = begin; hand < end; ++hand) {
      const int lbr = static_cast<int>(hand & 1), opp = 1 - lbr;
      const holdem::Deal deal = holdem::sample_deal(rng);
      const DealView v = view_deal(deal, cards, tree.max_street());
      const uint64_t lbr_cards = (uint64_t{1} << deal.hole[lbr][0]) | (uint64_t{1} << deal.hole[lbr][1]);
      for (int h = 0; h < 1326; ++h) {
        weight[h] = ((lbr_cards >> all[h][0] & 1) || (lbr_cards >> all[h][1] & 1)) ? 0.0f : 1.0f;
      }

      int prepared_street = -1;
      int32_t ref = 0;
      while (ref >= 0) {
        const uint32_t id = static_cast<uint32_t>(ref);
        const TreeNode& n = tree.node(id);
        if (n.street != prepared_street) {
          prepared_street = n.street;
          uint64_t board_mask = 0;
          for (int i = 0; i < kBoardCards[n.street]; ++i) board_mask |= uint64_t{1} << deal.board[i];
          for (int h = 0; h < 1326; ++h) {
            if (weight[h] <= 0.0f) continue;
            if ((board_mask >> all[h][0] & 1) || (board_mask >> all[h][1] & 1)) {
              weight[h] = 0.0f;
              continue;
            }
            bucket_of[h] = static_cast<uint16_t>(cards.bucket(n.street, all[h].data(), deal.board));
          }
        }

        if (n.player == opp) {
          tables.play_strategy(n, v.bucket[opp][n.street], sigma);
          const int a = detail::sample_index(sigma, n.num_actions, rng);
          for (int h = 0; h < 1326; ++h) {
            if (weight[h] <= 0.0f) continue;
            tables.play_strategy(n, bucket_of[h], sigma);
            weight[h] *= sigma[a];
          }
          ref = tree.child(n, a);
          continue;
        }

        int choice = 0;
        if (n.street == holdem::kPreflop) {
          for (int a = 0; a < n.num_actions; ++a) {
            if (tree.action(n, a).type == holdem::ActionType::kCheckCall) choice = a;
          }
        } else {
          detail::equity_vs_combos(deal.hole[lbr], deal.board, kBoardCards[n.street], weight,
                                   opt.flop_rollouts, rng, equity);
          const HunlState& s = tree.state(id);
          double best = -1e18;
          for (int a = 0; a < n.num_actions; ++a) {
            const holdem::Action& act = tree.action(n, a);
            double value;
            if (act.type == holdem::ActionType::kFold) {
              value = -s.contribution(lbr);
            } else if (act.type == holdem::ActionType::kCheckCall) {
              const double c = std::max(s.contribution(0), s.contribution(1));
              value = c * (2.0 * detail::weighted_equity(weight, equity) - 1.0);
            } else {
              const int32_t child = tree.child(n, a);
              if (child < 0) continue;  // a raise always gives the opponent a decision
              const TreeNode& reply = tree.node(static_cast<uint32_t>(child));
              const HunlState& cs = tree.state(static_cast<uint32_t>(child));
              double fold_weight = 0.0, total_weight = 0.0;
              for (int h = 0; h < 1326; ++h) {
                calling[h] = 0.0f;
                if (weight[h] <= 0.0f) continue;
                tables.play_strategy(reply, bucket_of[h], sigma);
                fold_weight += weight[h] * sigma[0];  // fold is always action 0 when facing a bet
                total_weight += weight[h];
                calling[h] = weight[h] * (1.0f - sigma[0]);
              }
              const double fold_prob = total_weight > 0.0 ? fold_weight / total_weight : 0.0;
              const double c = cs.contribution(lbr);
              value = fold_prob * cs.contribution(opp) +
                      (1.0 - fold_prob) * c * (2.0 * detail::weighted_equity(calling, equity) - 1.0);
            }
            if (value > best + 1e-9) {
              best = value;
              choice = a;
            }
          }
        }
        ref = tree.child(n, choice);
      }
      parts[t].add(terminal_value(tree.terminal(ref), v, lbr));
    }
  });
  return detail::summarize(parts, tree.config().big_blind, 1);
}

}  // namespace poker2::blueprint
