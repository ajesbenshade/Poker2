// Trains the HUNL blueprint.
//
//   poker2_train --abstraction DIR --run DIR [--profile blueprint|small|push_fold]
//                [--hours H] [--max-iterations N] [--epoch N]
//                [--linear-minutes M | --linear-until N] [--prune-after-minutes M | --prune-after N]
//                [--checkpoint-minutes M] [--eval-minutes M] [--lbr-hands N] [--h2h-deals N]
//                [--average-streets 0123] [--threads N] [--seed S] [--resume]
//
// Work is split into epochs of --epoch iterations. Between epochs:
//   - Linear CFR: during the first --linear-minutes of training time (or while
//     iterations <= --linear-until), all regrets and averages are scaled by
//     k/(k+1) after epoch k (Pluribus-style discounting);
//   - pruning switches on after --prune-after-minutes (or --prune-after iterations);
//   - checkpoints and evaluations run on their own timers.
// Training time excludes evaluation and checkpoints and survives --resume.
// Prefer the minute-based options: throughput depends on the abstraction and on
// how far training has progressed, so iteration counts are hard to size ahead.
// Ctrl-C or SIGTERM finishes the current epoch, checkpoints, and exits.
//
// Files in --run: config.txt, state.txt, train.log, metrics.csv, checkpoint.bin

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <unistd.h>

#include "abstraction/card_abstraction.h"
#include "blueprint/betting_tree.h"
#include "blueprint/evaluation.h"
#include "blueprint/strategy.h"
#include "blueprint/trainer.h"
#include "core/parallel.h"

using namespace poker2;
using namespace poker2::blueprint;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// A crash in a detached run otherwise leaves no trace, so fatal signals are
// recorded in train.log (with async-signal-safe calls only) before dying.
int g_log_fd = -1;
void on_fatal(int sig) {
  char msg[] = "[fatal] poker2_train killed by signal    \n";
  msg[sizeof(msg) - 4] = static_cast<char>('0' + sig / 10);
  msg[sizeof(msg) - 3] = static_cast<char>('0' + sig % 10);
  if (g_log_fd >= 0) (void)!write(g_log_fd, msg, sizeof(msg) - 1);
  (void)!write(2, msg, sizeof(msg) - 1);
  std::signal(sig, SIG_DFL);
  std::raise(sig);
}

struct Args {
  std::string abstraction_dir;
  std::string run_dir;
  std::string profile = "blueprint";
  double hours = 24.0;
  uint64_t max_iterations = 0;  // 0 = no limit
  uint64_t epoch = 1000000;
  uint64_t linear_until = 0;
  uint64_t prune_after = 0;  // 0 = never
  double linear_minutes = 0.0;
  double prune_after_minutes = 0.0;  // 0 = never
  double checkpoint_minutes = 60.0;
  double eval_minutes = 60.0;
  uint64_t lbr_hands = 20000;
  uint64_t h2h_deals = 20000;
  std::string average_streets = "0123";  // streets that store average strategies
  int threads = default_threads();
  uint64_t seed = 1;
  bool resume = false;
};

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: poker2_train --abstraction DIR --run DIR [--profile blueprint|small|push_fold]\n"
               "                    [--hours H] [--max-iterations N] [--epoch N]\n"
               "                    [--linear-minutes M | --linear-until N]\n"
               "                    [--prune-after-minutes M | --prune-after N]\n"
               "                    [--checkpoint-minutes M] [--eval-minutes M]\n"
               "                    [--lbr-hands N] [--h2h-deals N] [--threads N] [--seed S] [--resume]\n"
               "                    [--average-streets 0123]  (streets storing averages; default all)\n");
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    const std::string f = argv[i];
    if (f == "--abstraction") a.abstraction_dir = next();
    else if (f == "--run") a.run_dir = next();
    else if (f == "--profile") a.profile = next();
    else if (f == "--hours") a.hours = std::atof(next().c_str());
    else if (f == "--max-iterations") a.max_iterations = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--epoch") a.epoch = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--linear-until") a.linear_until = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--prune-after") a.prune_after = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--linear-minutes") a.linear_minutes = std::atof(next().c_str());
    else if (f == "--prune-after-minutes") a.prune_after_minutes = std::atof(next().c_str());
    else if (f == "--checkpoint-minutes") a.checkpoint_minutes = std::atof(next().c_str());
    else if (f == "--eval-minutes") a.eval_minutes = std::atof(next().c_str());
    else if (f == "--lbr-hands") a.lbr_hands = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--h2h-deals") a.h2h_deals = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--average-streets") a.average_streets = next();
    else if (f == "--threads") a.threads = std::atoi(next().c_str());
    else if (f == "--seed") a.seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (f == "--resume") a.resume = true;
    else usage();
  }
  if (a.abstraction_dir.empty() || a.run_dir.empty() || a.epoch == 0 || a.threads < 1) usage();
  return a;
}

holdem::ActionAbstraction make_profile(const std::string& name) {
  if (name == "blueprint") return holdem::ActionAbstraction::blueprint();
  if (name == "push_fold") return holdem::ActionAbstraction::push_fold();
  if (name == "small") {
    // Fewer sizes for quick pilots: half pot, pot, all-in.
    holdem::ActionAbstraction a;
    for (int s = 0; s < 4; ++s) a.raise_sizes[s] = {{0.5, 1.0}, {1.0}};
    return a;
  }
  std::fprintf(stderr, "unknown profile: %s\n", name.c_str());
  std::exit(2);
}

// Everything that must match between a run and its resumed continuation.
std::string config_text(const Args& a, const TableLayout& layout) {
  std::ostringstream s;
  s << "profile " << a.profile << "\n"
    << "abstraction " << a.abstraction_dir << "\n"
    << "buckets " << layout.buckets[0] << " " << layout.buckets[1] << " " << layout.buckets[2] << " "
    << layout.buckets[3] << "\n"
    << "epoch " << a.epoch << "\n"
    << "linear_until " << a.linear_until << "\n"
    << "prune_after " << a.prune_after << "\n"
    << "linear_minutes " << a.linear_minutes << "\n"
    << "prune_after_minutes " << a.prune_after_minutes << "\n"
    << "seed " << a.seed << "\n"
    << "average_bytes " << layout.average_bytes(0) << " " << layout.average_bytes(1) << " "
    << layout.average_bytes(2) << " " << layout.average_bytes(3) << "\n";
  return s.str();
}

class Log {
 public:
  explicit Log(const std::string& path) : file_(std::fopen(path.c_str(), "a")) {}
  ~Log() {
    if (file_) std::fclose(file_);
  }
  template <class... T>
  void operator()(const char* fmt, T... args) {
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    char line[1024];
    if constexpr (sizeof...(args) == 0) {
      std::snprintf(line, sizeof(line), "%s", fmt);
    } else {
      std::snprintf(line, sizeof(line), fmt, args...);
    }
    std::printf("[%s] %s\n", stamp, line);
    std::fflush(stdout);
    if (file_) {
      std::fprintf(file_, "[%s] %s\n", stamp, line);
      std::fflush(file_);
    }
  }

 private:
  FILE* file_;
};

double minutes_since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count() / 60.0;
}

// MemAvailable from /proc/meminfo, in bytes (0 if unknown).
uint64_t available_memory() {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  uint64_t kb = 0;
  while (in >> key >> kb >> unit) {
    if (key == "MemAvailable:") return kb * 1024;
  }
  return 0;
}

double read_trained_seconds(const std::string& path) {
  std::ifstream in(path);
  std::string key;
  double value = 0.0;
  while (in >> key >> value) {
    if (key == "trained_seconds") return value;
  }
  return 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  std::filesystem::create_directories(args.run_dir);
  Log log(args.run_dir + "/train.log");
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGHUP, SIG_IGN);  // keep training if the launching session goes away
  g_log_fd = open((args.run_dir + "/train.log").c_str(), O_WRONLY | O_APPEND);
  for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) std::signal(sig, on_fatal);

  log("loading card abstraction from %s", args.abstraction_dir.c_str());
  const abstraction::CardAbstraction cards = abstraction::CardAbstraction::load(args.abstraction_dir);
  const BettingTree tree(make_profile(args.profile));
  TableLayout layout;
  for (int s = 0; s < 4; ++s) {
    layout.buckets[s] = cards.num_buckets(s);
    layout.store_average[s] = args.average_streets.find(static_cast<char>('0' + s)) != std::string::npos;
  }

  const std::string config = config_text(args, layout);
  const std::string config_path = args.run_dir + "/config.txt";
  const std::string checkpoint_path = args.run_dir + "/checkpoint.bin";
  if (args.resume) {
    std::ifstream in(config_path);
    std::stringstream saved;
    saved << in.rdbuf();
    if (saved.str() != config) {
      log("config.txt does not match these arguments; refusing to resume");
      return 1;
    }
  } else {
    if (std::filesystem::exists(checkpoint_path)) {
      log("%s already exists; pass --resume or use a new --run directory", checkpoint_path.c_str());
      return 1;
    }
    std::ofstream(config_path) << config;
  }

  const uint64_t needed = layout.memory_bytes(tree);
  const uint64_t available = available_memory();
  if (available && needed + (2ull << 30) > available) {
    log("tables need %.1f GB but only %.1f GB is available (keeping 2 GB spare); "
        "raise WSL memory in .wslconfig or use a smaller abstraction",
        needed / 1e9, available / 1e9);
    return 1;
  }
  StrategyTables tables(tree, layout);
  log("tree: %zu decision nodes, %zu terminals; buckets %d/%d/%d/%d; tables %.2f GB",
      tree.num_nodes(), tree.num_terminals(), layout.buckets[0], layout.buckets[1], layout.buckets[2],
      layout.buckets[3], tables.memory_bytes() / 1e9);
  const std::string state_path = args.run_dir + "/state.txt";
  uint64_t iterations = 0;
  double trained_seconds = 0.0;  // time spent in training epochs, across resumes
  if (args.resume) {
    iterations = tables.load(checkpoint_path);
    trained_seconds = read_trained_seconds(state_path);
    log("resumed from %s at %lu iterations, %.0f training minutes", checkpoint_path.c_str(), iterations,
        trained_seconds / 60.0);
  }

  const bool new_metrics = !std::filesystem::exists(args.run_dir + "/metrics.csv");
  std::ofstream metrics(args.run_dir + "/metrics.csv", std::ios::app);
  if (new_metrics) {
    metrics << "wall_minutes,iterations,iterations_per_second,lbr_mbb,lbr_stderr,"
               "vs_call_mbb,vs_call_stderr,vs_random_mbb,vs_random_stderr\n";
  }

  const auto start = Clock::now();
  auto last_checkpoint = Clock::now(), last_eval = Clock::now();
  double rate = 0.0;

  auto checkpoint = [&] {
    const auto t = Clock::now();
    tables.save(checkpoint_path, iterations);
    std::ofstream(state_path) << "iterations " << iterations << "\ntrained_seconds " << trained_seconds << "\n";
    log("checkpoint at %lu iterations, %.0f training minutes (%.0fs)", iterations, trained_seconds / 60.0,
        minutes_since(t) * 60.0);
    last_checkpoint = Clock::now();
  };
  auto evaluate = [&] {
    const auto t = Clock::now();
    LbrOptions lbr;
    lbr.hands = args.lbr_hands;
    lbr.seed = args.seed + iterations;
    lbr.threads = args.threads;
    const MatchResult l = local_best_response(tree, tables, cards, lbr);
    const MatchResult c = head_to_head(tree, cards, blueprint_agent(tables), calling_agent(tree),
                                       args.h2h_deals, args.seed + 1, args.threads);
    const MatchResult r = head_to_head(tree, cards, blueprint_agent(tables), random_agent(),
                                       args.h2h_deals, args.seed + 2, args.threads);
    log("eval at %lu iterations: LBR %.0f +/- %.0f mbb/hand | vs always-call %+.0f +/- %.0f | "
        "vs random %+.0f +/- %.0f  (%.0fs)",
        iterations, l.mbb_per_hand, l.stderr_mbb, c.mbb_per_hand, c.stderr_mbb, r.mbb_per_hand,
        r.stderr_mbb, minutes_since(t) * 60.0);
    metrics << minutes_since(start) << "," << iterations << "," << rate << "," << l.mbb_per_hand << ","
            << l.stderr_mbb << "," << c.mbb_per_hand << "," << c.stderr_mbb << "," << r.mbb_per_hand
            << "," << r.stderr_mbb << std::endl;
    last_eval = Clock::now();
  };

  if (!args.resume) evaluate();  // baseline for the untrained strategy

  while (!g_stop) {
    if (args.max_iterations && iterations >= args.max_iterations) break;
    if (minutes_since(start) >= args.hours * 60.0) break;

    TrainerOptions opt;
    opt.seed = args.seed;
    opt.threads = args.threads;
    const double trained_minutes = trained_seconds / 60.0;
    opt.pruning = (args.prune_after > 0 && iterations >= args.prune_after) ||
                  (args.prune_after_minutes > 0 && trained_minutes >= args.prune_after_minutes);
    BlueprintTrainer trainer(tree, cards, tables, opt);
    const uint64_t epoch_index = iterations / args.epoch;
    const auto t = Clock::now();
    trainer.run(args.epoch, epoch_index);
    const double epoch_seconds = minutes_since(t) * 60.0;
    iterations += args.epoch;
    trained_seconds += epoch_seconds;
    rate = args.epoch / epoch_seconds;

    const bool linear = (args.linear_until > 0 && iterations <= args.linear_until) ||
                        (args.linear_minutes > 0 && trained_minutes < args.linear_minutes);
    if (linear) {
      const double k = static_cast<double>(epoch_index + 1);
      tables.scale(k / (k + 1.0), args.threads);
    }
    log("epoch %lu: %lu iterations, %.0f it/s, %.0f training minutes%s%s", epoch_index + 1, iterations,
        rate, trained_seconds / 60.0, opt.pruning ? ", pruning" : "", linear ? ", linear discount" : "");

    if (minutes_since(last_checkpoint) >= args.checkpoint_minutes) checkpoint();
    if (minutes_since(last_eval) >= args.eval_minutes) evaluate();
  }

  log(g_stop ? "stop requested" : "time or iteration limit reached");
  checkpoint();
  evaluate();
  log("exiting normally");
  return 0;
}
