CXX ?= c++
MODE ?= debug

BUILD_DIR := build/$(MODE)
TEST_BINARY := $(BUILD_DIR)/local_gguf_engine_tests
GGUF_INSPECT_BINARY := $(BUILD_DIR)/gguf-inspect
BENCHMARK_BINARY := $(BUILD_DIR)/local-gguf-bench

ENGINE_SOURCES := \
	src/engine/llama_layer.cpp \
	src/engine/llama_model.cpp \
	src/engine/model.cpp \
	src/engine/sampling.cpp \
	src/model_format/gguf_reader.cpp \
	src/model_format/gguf_tensor_bridge.cpp \
	src/tokenizer/tokenizer.cpp \
	src/tensor/dtype.cpp \
	src/tensor/f16.cpp \
	src/tensor/layout.cpp \
	src/tensor/ops/attention.cpp \
	src/tensor/ops/elementwise.cpp \
	src/tensor/ops/matmul.cpp \
	src/tensor/ops/normalization.cpp \
	src/tensor/ops/reduction.cpp \
	src/tensor/ops/rope.cpp \
	src/tensor/ops/softmax.cpp \
	src/tensor/quantization.cpp \
	src/tensor/shape.cpp \
	src/tensor/tensor.cpp \
	src/tensor/tensor_error.cpp \
	src/tensor/typed_access.cpp

FRONTEND_CLI_SOURCES := \
	src/frontends/cli/gguf_inspect.cpp

GGUF_INSPECT_SOURCES := \
	src/frontends/cli/gguf_inspect_main.cpp

BENCHMARK_SOURCES := \
	benchmarks/benchmark_harness.cpp

BENCHMARK_MAIN_SOURCES := \
	benchmarks/benchmark_main.cpp

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

.PHONY: test release-test gguf-inspect benchmark run-benchmark clean

test: $(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(BENCHMARK_BINARY)
	./$(TEST_BINARY)

release-test:
	$(MAKE) MODE=release test

gguf-inspect: $(GGUF_INSPECT_BINARY)

benchmark: $(BENCHMARK_BINARY)

run-benchmark: $(BENCHMARK_BINARY)
	./$(BENCHMARK_BINARY)

$(TEST_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(BENCHMARK_SOURCES) $(TEST_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(BENCHMARK_SOURCES) $(TEST_SOURCES) -o $(TEST_BINARY)

$(GGUF_INSPECT_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES) -o $(GGUF_INSPECT_BINARY)

$(BENCHMARK_BINARY): $(ENGINE_SOURCES) $(BENCHMARK_SOURCES) $(BENCHMARK_MAIN_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(BENCHMARK_SOURCES) $(BENCHMARK_MAIN_SOURCES) -o $(BENCHMARK_BINARY)

clean:
	rm -rf build
