#pragma once

// Minimal dependency-free test framework. Tests self-register with TEST(name);
// slow ones use SLOW_TEST(name) and only run with --slow.

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace poker2::testing {

struct TestCase {
  const char* name;
  bool slow;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, bool slow, std::function<void()> fn) {
    registry().push_back({name, slow, std::move(fn)});
  }
};

inline int& failures() {
  static int count = 0;
  return count;
}

inline int& checks() {
  static int count = 0;
  return count;
}

inline void check(bool ok, const std::string& what) {
  ++checks();
  if (!ok) {
    ++failures();
    std::printf("    FAIL: %s\n", what.c_str());
  }
}

inline void check_near(double actual, double expected, double tol, const std::string& what) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s: got %.6f, expected %.6f +/- %.6f", what.c_str(), actual,
                expected, tol);
  check(std::fabs(actual - expected) <= tol, buf);
}

inline void check_below(double actual, double limit, const std::string& what) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s: got %.6f, limit %.6f", what.c_str(), actual, limit);
  check(actual < limit, buf);
}

template <class A, class B>
void check_eq(const A& actual, const B& expected, const std::string& what) {
  const bool ok = actual == expected;
  if (ok) {
    check(true, what);
  } else {
    check(false, what + ": got " + std::to_string(actual) + ", expected " +
                     std::to_string(expected));
  }
}

}  // namespace poker2::testing

#define POKER2_CONCAT_INNER(a, b) a##b
#define POKER2_CONCAT(a, b) POKER2_CONCAT_INNER(a, b)
#define POKER2_REGISTER(name, slow)                                                   \
  static void POKER2_CONCAT(test_fn_, __LINE__)();                                    \
  static ::poker2::testing::Registrar POKER2_CONCAT(test_reg_, __LINE__)(             \
      name, slow, POKER2_CONCAT(test_fn_, __LINE__));                                 \
  static void POKER2_CONCAT(test_fn_, __LINE__)()
#define TEST(name) POKER2_REGISTER(name, false)
#define SLOW_TEST(name) POKER2_REGISTER(name, true)
