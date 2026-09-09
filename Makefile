CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Icpp/include
BUILD_DIR := build
LIB_SOURCES := \
	cpp/src/config.cpp \
	cpp/src/manifest.cpp \
	cpp/src/runtime.cpp \
	cpp/src/weights.cpp
CLI_SOURCES := $(LIB_SOURCES) cpp/src/main.cpp
TEST_SOURCES := $(LIB_SOURCES) tests/test_cpp.cpp

.PHONY: all clean test test-cpp test-python verify

all: $(BUILD_DIR)/onebit-llm

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/onebit-llm: $(CLI_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CLI_SOURCES) -o $@

$(BUILD_DIR)/test-cpp: $(TEST_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_SOURCES) -o $@

test-python:
	PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_python.py'

test-cpp: $(BUILD_DIR)/test-cpp
	$(BUILD_DIR)/test-cpp

verify: $(BUILD_DIR)/onebit-llm
	PYTHONPATH=python python3 -m onebit_llm describe > $(BUILD_DIR)/python.describe
	$(BUILD_DIR)/onebit-llm describe > $(BUILD_DIR)/cpp.describe
	diff -u $(BUILD_DIR)/python.describe $(BUILD_DIR)/cpp.describe

test: test-python test-cpp verify

clean:
	rm -rf $(BUILD_DIR)
