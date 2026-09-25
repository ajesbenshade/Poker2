// Runs a solver on a toy game and reports exploitability over time.
//
//   poker2_solve --game leduc --algo mccfr --iters 1000000 --eval-every 100000 --csv out.csv

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cfr/cfr_plus.h"
#include "cfr/evaluation.h"
#include "cfr/mccfr.h"
#include "games/kuhn.h"
#include "games/leduc.h"

using namespace poker2;

namespace {

struct Args {
  std::string game = "leduc";
  std::string algo = "mccfr";
  long long iters = 100000;
  long long eval_every = 10000;
  unsigned long long seed = 1;
  long long linear_cap = -1;  // -1: always linear
  std::string csv;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: poker2_solve [--game kuhn|leduc] [--algo mccfr|cfrplus] [--iters N]\n"
               "                    [--eval-every N] [--seed N] [--linear-cap N] [--csv PATH]\n"
               "  --linear-cap: stop increasing the Linear CFR weight after N iterations\n"
               "                (0 = unweighted MCCFR, default = always linear)\n");
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--game")) a.game = next();
    else if (!std::strcmp(argv[i], "--algo")) a.algo = next();
    else if (!std::strcmp(argv[i], "--iters")) a.iters = std::atoll(next());
    else if (!std::strcmp(argv[i], "--eval-every")) a.eval_every = std::atoll(next());
    else if (!std::strcmp(argv[i], "--seed")) a.seed = std::strtoull(next(), nullptr, 10);
    else if (!std::strcmp(argv[i], "--linear-cap")) a.linear_cap = std::atoll(next());
    else if (!std::strcmp(argv[i], "--csv")) a.csv = next();
    else usage();
  }
  if (a.iters <= 0 || a.eval_every <= 0) usage();
  return a;
}

template <class Game, class Solver>
int run(Solver& solver, const Args& args) {
  FILE* csv = args.csv.empty() ? nullptr : std::fopen(args.csv.c_str(), "w");
  if (!args.csv.empty() && !csv) {
    std::perror(args.csv.c_str());
    return 1;
  }
  if (csv) std::fprintf(csv, "iteration,seconds,infosets,exploitability,value_p0\n");

  std::printf("game=%s algo=%s iters=%lld seed=%llu\n", Game::name(), args.algo.c_str(),
              args.iters, args.seed);
  std::printf("%12s %10s %9s %16s %10s\n", "iteration", "seconds", "infosets", "exploitability",
              "value_p0");

  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  double eval_seconds = 0.0;  // excluded from the reported solve time
  for (long long it = 1; it <= args.iters; ++it) {
    solver.iterate();
    if (it % args.eval_every != 0 && it != args.iters) continue;

    const auto eval_start = Clock::now();
    const double solve_seconds =
        std::chrono::duration<double>(eval_start - start).count() - eval_seconds;
    const TabularPolicy avg = solver.average_policy();
    const double expl = exploitability<Game>(avg).exploitability();
    const double value = expected_value<Game>(avg, 0);
    std::printf("%12lld %10.2f %9zu %16.6f %10.5f\n", it, solve_seconds, solver.num_infosets(),
                expl, value);
    std::fflush(stdout);
    if (csv) {
      std::fprintf(csv, "%lld,%.3f,%zu,%.9f,%.9f\n", it, solve_seconds, solver.num_infosets(),
                   expl, value);
      std::fflush(csv);
    }
    eval_seconds += std::chrono::duration<double>(Clock::now() - eval_start).count();
  }
  if (csv) std::fclose(csv);
  return 0;
}

template <class Game>
int dispatch(const Args& args) {
  if (args.algo == "cfrplus") {
    CfrPlusSolver<Game> solver;
    return run<Game>(solver, args);
  }
  if (args.algo == "mccfr") {
    MccfrOptions options;
    options.seed = args.seed;
    if (args.linear_cap >= 0) options.linear_weight_cap = args.linear_cap;
    ExternalSamplingMccfr<Game> solver(options);
    return run<Game>(solver, args);
  }
  usage();
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  if (args.game == "kuhn") return dispatch<Kuhn>(args);
  if (args.game == "leduc") return dispatch<Leduc>(args);
  usage();
}
