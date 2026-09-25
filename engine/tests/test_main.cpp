// Test runner. Usage: poker2_tests [--slow] [name-substring]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "test_framework.h"

int main(int argc, char** argv) {
  using namespace poker2::testing;
  bool run_slow = false;
  std::string filter;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--slow")) run_slow = true;
    else filter = argv[i];
  }

  int skipped = 0;
  for (const TestCase& test : registry()) {
    if (!filter.empty() && std::string(test.name).find(filter) == std::string::npos) continue;
    if (test.slow && !run_slow) {
      ++skipped;
      continue;
    }
    const int failures_before = failures();
    const auto start = std::chrono::steady_clock::now();
    test.fn();
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("[%s] %s (%.2fs)\n", failures() == failures_before ? " OK " : "FAIL", test.name,
                secs);
    std::fflush(stdout);
  }

  std::printf("\n%d checks, %d failures", checks(), failures());
  if (skipped) std::printf(", %d slow tests skipped (run with --slow)", skipped);
  std::printf("\n");
  return failures() == 0 ? 0 : 1;
}
