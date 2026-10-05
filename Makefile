CXX ?= c++
MODE ?= debug
ENGINE_BUILD := $(shell git rev-parse HEAD 2>/dev/null)$(shell test -z "$$(git status --porcelain 2>/dev/null)" || printf '%s' '-dirty')

BUILD_DIR := build/$(MODE)
TEST_BINARY := $(BUILD_DIR)/local_gguf_engine_tests
GGUF_INSPECT_BINARY := $(BUILD_DIR)/gguf-inspect
GENERATE_BINARY := $(BUILD_DIR)/local-gguf-generate
BENCHMARK_BINARY := $(BUILD_DIR)/local-gguf-bench

ENGINE_SOURCES := \
	src/benchmarks/feature_capture_benchmark.cpp \
	src/benchmarks/immutable_weight_benchmark.mm \
	src/benchmarks/output_head_benchmark.cpp \
	src/benchmarks/quick_benchmark.cpp \
	src/benchmarks/target_verifier_benchmark.cpp \
	src/benchmarks/width_two_verifier_benchmark.cpp \
	src/engine/gemma2_chat_session.cpp \
	src/engine/feature_bank.cpp \
	src/engine/gemma2_model.cpp \
	src/engine/gemma2_metal_plan.cpp \
	src/engine/gemma2_reference.cpp \
	src/engine/gemma2_runtime.cpp \
	src/engine/gemma_target_runtime.cpp \
	src/engine/chain_chat.mm \
	src/engine/standalone_chat.cpp \
	src/engine/standalone_draft_artifact.cpp \
	src/engine/gemma3_draft.mm \
	src/engine/model.cpp \
	src/engine/prepared_gemma2_plan.cpp \
	src/engine/prepared_plan_test_hooks.cpp \
	src/engine/retrieval_proposer.cpp \
	src/engine/sampling.cpp \
	src/engine/speculative_controller.cpp \
	src/engine/target_verifier.cpp \
	src/engine/target_verifier_episode.cpp \
	src/engine/target_verifier_runtime.cpp \
	src/engine/target_verifier_test_hooks.cpp \
	src/model_format/gguf_reader.cpp \
	src/model_format/gguf_tensor_bridge.cpp \
	src/runtime/backend_error.cpp \
	src/runtime/sha256.cpp \
	src/tokenizer/tokenizer.cpp \
	src/tensor/backend.cpp \
	src/tensor/backend_buffer.mm \
	src/tensor/backends/cpu_backend.cpp \
	src/tensor/backends/metal/metal_buffer.mm \
	src/tensor/backends/metal/metal_command_context.mm \
	src/tensor/backends/metal/metal_context.mm \
	src/tensor/backends/metal/metal_tensor_ops.mm \
	src/tensor/backends/metal_backend.mm \
	src/tensor/dtype.cpp \
	src/tensor/f16.cpp \
	src/tensor/kquant_packing.cpp \
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
	src/frontends/cli/benchmark_main.cpp

TEST_INCLUDE_FLAGS := \
	-Itests/benchmarks \
	-Itests/engine \
	-Itests/frontends/cli \
	-Itests/model_format \
	-Itests/support \
	-Itests/tensor \
	-Itests/tokenizer

# Fault injection is compiled into the test binary only. Production binaries
# have no test hook or alternate device-selection path.
TEST_BUILD_FLAGS := -DLOCAL_GGUF_ENGINE_TESTING=1

TEST_SOURCES := \
	tests/benchmarks/quick_benchmark_tests.cpp \
	tests/engine/gemma2_metal_plan_tests.cpp \
	tests/engine/standalone_draft_artifact_tests.cpp \
	tests/engine/standalone_chat_tests.cpp \
	tests/engine/gemma3_draft_tests.cpp \
	tests/engine/gemma3_model_plan_tests.cpp \
	tests/engine/gemma3_target_runtime_tests.cpp \
	tests/tensor/gemma3_metal_tests.cpp \
	tests/engine/engine_model_tests.cpp \
	tests/engine/feature_bank_tests.cpp \
	tests/engine/gemma2_reference_tests.cpp \
	tests/engine/retrieval_proposer_tests.cpp \
	tests/engine/sampling_tests.cpp \
	tests/engine/lossless_sampling_tests.cpp \
	tests/engine/speculative_controller_tests.cpp \
	tests/frontends/cli/generate_cli_tests.cpp \
	tests/frontends/cli/gguf_inspect_tests.cpp \
	tests/model_format/gguf_reader_tests.cpp \
	tests/model_format/gguf_tensor_bridge_tests.cpp \
	tests/support/tensor_test_support.cpp \
	tests/support/gemma2_reference_test_support.cpp \
	tests/support/test_support.cpp \
	tests/tensor/tensor_backend_tests.cpp \
	tests/tensor/tensor_tests.cpp \
	tests/tokenizer/tokenizer_tests.cpp \
	tests/test_main.cpp

COMMON_FLAGS := -std=c++20 -Iinclude -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fobjc-arc \
	-DLOCAL_GGUF_ENGINE_BUILD=\"$(ENGINE_BUILD)\"
METAL_LDFLAGS := -framework Foundation -framework Metal
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS ?= -O3 -DNDEBUG -mcpu=apple-m4

ifeq ($(MODE),release)
	CXXFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
else
	CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
endif

.PHONY: test release-test gguf-inspect generate benchmark reference-benchmark clean

# Embedded Metal source is a header too. Rebuild when a numeric kernel or
# public contract changes, even if no .cpp/.mm timestamp changed.
ENGINE_HEADERS := $(wildcard include/local_gguf_engine/*/*.hpp src/*/*.hpp src/*/*/*.hpp src/*/*/*/*.hpp)
$(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(GENERATE_BINARY) $(BENCHMARK_BINARY): $(ENGINE_HEADERS)
$(TEST_BINARY): $(wildcard tests/*/*.hpp)

test: $(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(GENERATE_BINARY) $(BENCHMARK_BINARY)
	./$(TEST_BINARY)

release-test:
	$(MAKE) MODE=release test

gguf-inspect: $(GGUF_INSPECT_BINARY)

generate: $(GENERATE_BINARY)

benchmark: $(GENERATE_BINARY) $(BENCHMARK_BINARY)

reference-benchmark:
	/opt/homebrew/bin/llama-cli --version 2>&1 | grep -q d48a56eff

$(TEST_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(TEST_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_BUILD_FLAGS) $(TEST_INCLUDE_FLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(TEST_SOURCES) $(METAL_LDFLAGS) -o $(TEST_BINARY)

$(GGUF_INSPECT_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GGUF_INSPECT_SOURCES) $(METAL_LDFLAGS) -o $(GGUF_INSPECT_BINARY)

$(GENERATE_BINARY): $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GENERATE_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GENERATE_SOURCES) $(METAL_LDFLAGS) -o $(GENERATE_BINARY)

$(BENCHMARK_BINARY): $(ENGINE_SOURCES) $(BENCHMARK_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ENGINE_SOURCES) $(BENCHMARK_SOURCES) $(METAL_LDFLAGS) -o $(BENCHMARK_BINARY)

clean:
	rm -rf build
