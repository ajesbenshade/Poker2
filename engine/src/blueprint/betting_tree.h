#pragma once

// The abstract betting tree, flattened into arrays for fast traversal.
//
// Decision nodes are numbered in depth-first order (root = 0). A child
// reference >= 0 is a decision node; a negative reference r is terminal
// number ~r. Each decision node owns a contiguous block of "slots" (one per
// action) on its street; strategy tables are laid out per street as
// [slot_offset * buckets + bucket * num_actions + action].

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "holdem/action_abstraction.h"
#include "holdem/game.h"

namespace poker2::blueprint {

using holdem::Action;
using holdem::HunlState;

struct TreeNode {
  int8_t street;
  int8_t player;
  uint8_t num_actions;
  uint32_t first_child;   // index into the children / actions arrays
  uint64_t slot_offset;   // actions of earlier nodes on the same street
};

struct TerminalNode {
  int32_t contrib[2];
  int8_t folder;  // -1 at showdown
};

class BettingTree {
 public:
  static constexpr int kMaxActions = 16;

  BettingTree(const holdem::ActionAbstraction& abstraction,
              const holdem::GameConfig& config = holdem::GameConfig())
      : config_(config) {
    build(HunlState(config), abstraction);
  }

  const holdem::GameConfig& config() const { return config_; }
  size_t num_nodes() const { return nodes_.size(); }
  size_t num_terminals() const { return terminals_.size(); }
  const TreeNode& node(uint32_t i) const { return nodes_[i]; }
  const HunlState& state(uint32_t i) const { return states_[i]; }
  const TerminalNode& terminal(int32_t ref) const { return terminals_[~ref]; }
  int32_t child(const TreeNode& n, int a) const { return children_[n.first_child + a]; }
  const Action& action(const TreeNode& n, int a) const { return actions_[n.first_child + a]; }
  uint64_t street_slots(int street) const { return street_slots_[street]; }
  uint64_t street_nodes(int street) const { return street_nodes_[street]; }
  int max_street() const { return max_street_; }

 private:
  int32_t build(const HunlState& s, const holdem::ActionAbstraction& abstraction) {
    if (s.is_terminal()) {
      TerminalNode t{{s.contribution(0), s.contribution(1)}, static_cast<int8_t>(s.folder())};
      terminals_.push_back(t);
      return ~static_cast<int32_t>(terminals_.size() - 1);
    }
    std::vector<Action> actions;
    abstraction.actions(s, actions);
    if (actions.size() > kMaxActions) throw std::logic_error("too many actions at one node");

    const uint32_t id = static_cast<uint32_t>(nodes_.size());
    TreeNode n{};
    n.street = static_cast<int8_t>(s.street());
    n.player = static_cast<int8_t>(s.to_act());
    n.num_actions = static_cast<uint8_t>(actions.size());
    n.first_child = static_cast<uint32_t>(children_.size());
    n.slot_offset = street_slots_[s.street()];
    street_slots_[s.street()] += actions.size();
    ++street_nodes_[s.street()];
    max_street_ = std::max(max_street_, s.street());
    nodes_.push_back(n);
    states_.push_back(s);
    children_.resize(children_.size() + actions.size());
    actions_.insert(actions_.end(), actions.begin(), actions.end());

    for (size_t a = 0; a < actions.size(); ++a) {
      HunlState child = s;
      child.apply(actions[a]);
      const int32_t ref = build(child, abstraction);
      children_[nodes_[id].first_child + a] = ref;
    }
    return static_cast<int32_t>(id);
  }

  holdem::GameConfig config_;
  std::vector<TreeNode> nodes_;
  std::vector<HunlState> states_;
  std::vector<TerminalNode> terminals_;
  std::vector<int32_t> children_;
  std::vector<Action> actions_;
  std::array<uint64_t, 4> street_slots_{};
  std::array<uint64_t, 4> street_nodes_{};
  int max_street_ = 0;
};

}  // namespace poker2::blueprint
