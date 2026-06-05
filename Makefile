CXX ?= c++
MODE ?= debug

BUILD_DIR := build/$(MODE)
TEST_BINARY := $(BUILD_DIR)/local_gguf_engine_tests

ENGINE_SOURCES := \
	src/model_format/gguf_reader.cpp

TEST_SOURCES := \
	tests/test_main.cpp

COMMON_FLAGS := -std=c++20 -Iinclude -Wall -Wextra -Wpedantic -Wconversion -Wshadow
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS ?= -O3 -DNDEBUG -mcpu=apple-m4

ifeq ($(MODE),release)
	CXXFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
else
	CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
endif

.PHONY: test release-test clean

test: $(TEST_BINARY)
	./$(TEST_BINARY)

release-test:
	$(MAKE) MODE=release test

$(TEST_BINARY): $(ENGINE_SOURCES) $(TEST_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(TEST_SOURCES) -o $(TEST_BINARY)

clean:
	rm -rf build
