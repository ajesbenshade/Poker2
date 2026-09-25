# Build inside WSL (Ubuntu 24.04, g++ 13):
#   make -C engine test        build and run the fast test suite
#   make -C engine test-slow   also run exhaustive tests (133M-hand enumerations etc.)
#   make -C engine             build everything into engine/build/

CXX ?= g++
CXXFLAGS ?= -O2 -march=native -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -Isrc

BUILD := build
HEADERS := $(wildcard src/*/*.h)
TEST_SOURCES := $(wildcard tests/*.cpp)
TOOLS := poker2_solve poker2_tree poker2_bench

.PHONY: all test test-slow clean

all: $(BUILD)/poker2_tests $(addprefix $(BUILD)/,$(TOOLS))

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/poker2_tests: $(TEST_SOURCES) tests/test_framework.h $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SOURCES) -o $@

$(BUILD)/poker2_solve: tools/solve.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD)/poker2_tree: tools/tree_size.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD)/poker2_bench: tools/bench.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test: $(BUILD)/poker2_tests
	./$(BUILD)/poker2_tests

test-slow: $(BUILD)/poker2_tests
	./$(BUILD)/poker2_tests --slow

clean:
	rm -rf $(BUILD)
