#pragma once

// Game interface shared by every solver in the engine.
//
// A Game type provides:
//   using State = ...;
//   static State initial_state();
//   static const char* name();
//
// A State provides:
//   bool is_terminal() const;
//   bool is_chance() const;
//   int current_player() const;                       // 0 or 1 at decision nodes
//   void chance_outcomes(std::vector<Outcome>&) const;  // chance nodes only
//   void legal_actions(std::vector<int>&) const;        // decision nodes only
//   void apply(int action_or_outcome);
//   double utility(int player) const;                   // terminal nodes only
//   std::string infoset_key() const;                    // acting player's view
//   std::string history_string() const;                 // unique per history
//
// Games are two-player zero-sum. Infoset keys must be unique across both
// players (e.g. derivable from the betting history) so one policy table can
// hold the whole strategy profile.

#include <string>
#include <vector>

namespace poker2 {

constexpr int kChancePlayer = -1;

struct Outcome {
  int action;
  double prob;
};

}  // namespace poker2
