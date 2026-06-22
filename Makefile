CXX ?= c++
MODE ?= debug

BUILD_DIR := build/$(MODE)
TEST_BINARY := $(BUILD_DIR)/local_gguf_engine_tests
GGUF_INSPECT_BINARY := $(BUILD_DIR)/gguf-inspect
GENERATE_BINARY := $(BUILD_DIR)/local-gguf-generate
BENCHMARK_BINARY := $(BUILD_DIR)/local-gguf-bench

ENGINE_SOURCES := \
	src/engine/gemma2_model.cpp \
	src/engine/gemma2_reference.cpp \
	src/engine/kv_cache.cpp \
	src/engine/llama_layer.cpp \
	src/engine/llama_model.cpp \
	src/engine/model.cpp \
	src/engine/sampling.cpp \
	src/model_format/gguf_reader.cpp \
	src/model_format/gguf_tensor_bridge.cpp \
	src/runtime/backend_error.cpp \
	src/tokenizer/tokenizer.cpp \
	src/tensor/backend.cpp \
	src/tensor/backend_buffer.mm \
	src/tensor/backends/cpu_backend.cpp \
	src/tensor/backends/metal/metal_benchmark.mm \
	src/tensor/backends/metal/metal_buffer.mm \
	src/tensor/backends/metal/metal_context.mm \
	src/tensor/backends/metal/metal_llama_ops.mm \
	src/tensor/backends/metal/metal_tensor_ops.mm \
	src/tensor/backends/metal_backend.mm \
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
	src/frontends/cli/generate.cpp \
	src/frontends/cli/gguf_inspect.cpp

GGUF_INSPECT_SOURCES := \
	src/frontends/cli/gguf_inspect_main.cpp

GENERATE_SOURCES := \
	src/frontends/cli/generate_main.cpp

BENCHMARK_SOURCES := \
	benchmarks/benchmark_harness.cpp \
	benchmarks/engine/benchmark_synthetic_llama.cpp \
	benchmarks/engine/kv/benchmark_resident_kv_decode.cpp \
	benchmarks/frontends/cli/benchmark_cli.cpp \
	benchmarks/model_format/benchmark_model_load.cpp \
	benchmarks/support/benchmark_support.cpp \
	benchmarks/tensor/benchmark_projection.cpp \
	benchmarks/tensor/benchmark_synthetic_core.cpp

BENCHMARK_MAIN_SOURCES := \
	benchmarks/frontends/cli/benchmark_main.cpp

TEST_INCLUDE_FLAGS := \
	-Itests/benchmarks \
	-Itests/engine \
	-Itests/frontends/cli \
	-Itests/model_format \
	-Itests/support \
	-Itests/tensor \
	-Itests/tokenizer

TEST_SOURCES := \
	tests/benchmarks/benchmark_tests.cpp \
	tests/engine/engine_model_tests.cpp \
	tests/engine/gemma2_reference_tests.cpp \
	tests/engine/llama_attention_layer_tests.cpp \
	tests/engine/llama_backend_tests.cpp \
	tests/engine/llama_block_tests.cpp \
	tests/engine/llama_feed_forward_layer_tests.cpp \
	tests/engine/llama_generation_tests.cpp \
	tests/engine/llama_kv_cache_tests.cpp \
	tests/engine/llama_model_tests.cpp \
	tests/engine/sampling_tests.cpp \
	tests/frontends/cli/generate_cli_tests.cpp \
	tests/frontends/cli/gguf_inspect_tests.cpp \
	tests/model_format/gguf_reader_tests.cpp \
	tests/model_format/gguf_tensor_bridge_tests.cpp \
	tests/support/llama_test_support.cpp \
	tests/support/tensor_test_support.cpp \
	tests/support/test_support.cpp \
	tests/tensor/tensor_backend_tests.cpp \
	tests/tensor/tensor_tests.cpp \
	tests/tokenizer/tokenizer_tests.cpp \
	tests/test_main.cpp

COMMON_FLAGS := -std=c++20 -Iinclude -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fobjc-arc
METAL_LDFLAGS := -framework Foundation -framework Metal
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS ?= -O3 -DNDEBUG -mcpu=apple-m4

ifeq ($(MODE),release)
	CXXFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
else
	CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
endif

.PHONY: test release-test gguf-inspect generate benchmark bench run-benchmark clean

test: $(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(GENERATE_BINARY) $(BENCHMARK_BINARY)
	./$(TEST_BINARY)

release-test:
	$(MAKE) MODE=release test

gguf-inspect: $(GGUF_INSPECT_BINARY)

generate: $(GENERATE_BINARY)

benchmark: $(BENCHMARK_BINARY)

bench: benchmark

run-benchmark: $(BENCHMARK_BINARY)
	./$(BENCHMARK_BINARY)

$(TEST_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(BENCHMARK_SOURCES) $(TEST_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_INCLUDE_FLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(BENCHMARK_SOURCES) $(TEST_SOURCES) $(METAL_LDFLAGS) -o $(TEST_BINARY)

$(GGUF_INSPECT_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES) $(METAL_LDFLAGS) -o $(GGUF_INSPECT_BINARY)

$(GENERATE_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GENERATE_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GENERATE_SOURCES) $(METAL_LDFLAGS) -o $(GENERATE_BINARY)

$(BENCHMARK_BINARY): $(ENGINE_SOURCES) $(BENCHMARK_SOURCES) $(BENCHMARK_MAIN_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(BENCHMARK_SOURCES) $(BENCHMARK_MAIN_SOURCES) $(METAL_LDFLAGS) -o $(BENCHMARK_BINARY)

clean:
	rm -rf build
