# Build inside WSL (Ubuntu 24.04, g++ 13):
#   make -C engine test      build and run the test suite
#   make -C engine           build everything into engine/build/

CXX ?= g++
CXXFLAGS ?= -O2 -march=native -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -Isrc

BUILD := build
HEADERS := $(wildcard src/*/*.h)

.PHONY: all test clean

all: $(BUILD)/poker2_tests $(BUILD)/poker2_solve

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/poker2_tests: tests/test_main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD)/poker2_solve: tools/solve.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test: $(BUILD)/poker2_tests
	./$(BUILD)/poker2_tests

clean:
	rm -rf $(BUILD)
