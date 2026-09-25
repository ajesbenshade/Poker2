// CFR core on Kuhn and Leduc.
// Expected values come from the literature or from game theory (the Kuhn
// best-response values are derived by hand), not from this code's own output.
// The one exception is the MCCFR convergence bound, which is noted inline.

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "cfr/cfr_plus.h"
#include "cfr/evaluation.h"
#include "cfr/mccfr.h"
#include "games/kuhn.h"
#include "games/leduc.h"
#include "test_framework.h"

using namespace poker2;
using namespace poker2::testing;

namespace {

// Walks every terminal history and checks utilities are zero-sum.
template <class Game>
bool all_terminals_zero_sum() {
  using State = typename Game::State;
  bool ok = true;
  std::function<void(const State&)> walk = [&](const State& state) {
    if (state.is_terminal()) {
      ok &= std::fabs(state.utility(0) + state.utility(1)) < 1e-12;
      return;
    }
    std::vector<int> moves;
    if (state.is_chance()) {
      std::vector<Outcome> outcomes;
      state.chance_outcomes(outcomes);
      double total = 0.0;
      for (const Outcome& o : outcomes) {
        moves.push_back(o.action);
        total += o.prob;
      }
      ok &= std::fabs(total - 1.0) < 1e-12;
    } else {
      state.legal_actions(moves);
    }
    for (int move : moves) {
      State child = state;
      child.apply(move);
      walk(child);
    }
  };
  walk(Game::initial_state());
  return ok;
}

TabularPolicy uniform_policy() { return TabularPolicy(); }  // unknown keys play uniform

}  // namespace

TEST("kuhn: rules are zero-sum with valid chance") {
  check(all_terminals_zero_sum<Kuhn>(), "kuhn zero-sum / chance probabilities");
  check(enumerate_infosets<Kuhn>().size() == 12, "kuhn has 12 infosets");
}

TEST("leduc: rules are zero-sum with valid chance") {
  check(all_terminals_zero_sum<Leduc>(), "leduc zero-sum / chance probabilities");
  check_eq(enumerate_infosets<Leduc>().size(), size_t{288}, "leduc has 288 infosets");
}

TEST("kuhn: best response vs uniform (exact reference value)") {
  // Uniform-random Kuhn has NashConv 11/12: BR values 1/2 (P0) and 5/12 (P1).
  const ExploitabilityReport r = exploitability<Kuhn>(uniform_policy());
  check_near(r.br_value[0], 0.5, 1e-9, "P0 best response to uniform");
  check_near(r.br_value[1], 5.0 / 12.0, 1e-9, "P1 best response to uniform");
  check_near(expected_value<Kuhn>(uniform_policy(), 0), 0.125, 1e-9, "uniform self-play value");
}

TEST("kuhn: CFR+ converges to the known equilibrium") {
  CfrPlusSolver<Kuhn> solver;
  for (int i = 0; i < 2000; ++i) solver.iterate();
  const TabularPolicy avg = solver.average_policy();
  check_below(exploitability<Kuhn>(avg).exploitability(), 1e-3, "exploitability");
  check_near(expected_value<Kuhn>(avg, 0), -1.0 / 18.0, 2e-3, "game value for P0");
  // P1's equilibrium strategy is unique (action 1 = bet/call):
  check_near(avg.get("0b", 2)[1], 0.0, 1e-2, "P1 with J folds to a bet");
  check_near(avg.get("1b", 2)[1], 1.0 / 3.0, 2e-2, "P1 with Q calls a bet 1/3");
  check_near(avg.get("2b", 2)[1], 1.0, 1e-2, "P1 with K calls a bet");
  check_near(avg.get("0p", 2)[1], 1.0 / 3.0, 2e-2, "P1 with J bluffs 1/3 after a check");
  check_near(avg.get("1p", 2)[1], 0.0, 1e-2, "P1 with Q checks behind");
  check_near(avg.get("2p", 2)[1], 1.0, 1e-2, "P1 with K bets after a check");
  // P0's family is parameterized by alpha = P(bet with J) in [0, 1/3]; K bets 3*alpha.
  const double alpha = avg.get("0", 2)[1];
  check(alpha >= -1e-3 && alpha <= 1.0 / 3.0 + 2e-2, "P0 J-bet frequency within [0, 1/3]");
  check_near(avg.get("2", 2)[1], 3.0 * alpha, 3e-2, "P0 K bets 3 * alpha");
}

TEST("leduc: CFR+ converges to the published game value") {
  CfrPlusSolver<Leduc> solver;
  for (int i = 0; i < 1000; ++i) solver.iterate();
  const TabularPolicy avg = solver.average_policy();
  check_below(exploitability<Leduc>(avg).exploitability(), 2e-3, "exploitability");
  // Published Leduc game value for the first player: -0.0856.
  check_near(expected_value<Leduc>(avg, 0), -0.0856, 2e-3, "game value for P0");
}

TEST("kuhn: external-sampling MCCFR converges") {
  ExternalSamplingMccfr<Kuhn> solver({/*seed=*/7});
  for (int i = 0; i < 100000; ++i) solver.iterate();
  const TabularPolicy avg = solver.average_policy();
  check_below(exploitability<Kuhn>(avg).exploitability(), 5e-3, "exploitability");
  check_near(expected_value<Kuhn>(avg, 0), -1.0 / 18.0, 5e-3, "game value for P0");
}

TEST("leduc: external-sampling MCCFR converges and improves") {
  ExternalSamplingMccfr<Leduc> solver({/*seed=*/11});
  for (int i = 0; i < 10000; ++i) solver.iterate();
  const double early = exploitability<Leduc>(solver.average_policy()).exploitability();
  for (int i = 0; i < 290000; ++i) solver.iterate();
  const TabularPolicy avg = solver.average_policy();
  const double late = exploitability<Leduc>(avg).exploitability();
  check(late < early / 3.0, "exploitability drops at least 3x from 10k to 300k iterations");
  // Sampling noise makes this seed-dependent. Measured: ~0.027 at 300k, falling
  // roughly as 1/sqrt(T) to ~0.006-0.008 at 3M across seeds 11-13.
  check_below(late, 0.04, "exploitability after 300k iterations");
  check_near(expected_value<Leduc>(avg, 0), -0.0856, 0.02, "game value for P0");
}

TEST("mccfr: same seed gives identical results") {
  ExternalSamplingMccfr<Leduc> a({/*seed=*/3}), b({/*seed=*/3});
  for (int i = 0; i < 2000; ++i) {
    a.iterate();
    b.iterate();
  }
  const auto pa = a.average_policy(), pb = b.average_policy();
  bool same = pa.size() == pb.size();
  for (const auto& [key, probs] : pa.entries()) same &= pb.get(key, probs.size()) == probs;
  check(same, "deterministic under a fixed seed");
}
