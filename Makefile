.DEFAULT_GOAL := guffin

CXX ?= c++
MODE ?= release
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
ENGINE_BUILD := $(if $(GIT_COMMIT),$(GIT_COMMIT)$(shell test -z "$$(git status --porcelain 2>/dev/null)" || printf '%s' '-dirty'),unknown)

BUILD_DIR := build/$(MODE)
TEST_BINARY := $(BUILD_DIR)/local_gguf_engine_tests
GGUF_INSPECT_BINARY := $(BUILD_DIR)/gguf-inspect
GUFFIN_BINARY := $(BUILD_DIR)/guffin
BENCHMARK_BINARY := $(BUILD_DIR)/guffin-bench

# Inspection depends only on the file format and tensor value types.
# Keep device execution and inference out of the standalone inspection tool.
GGUF_VALUE_SOURCES := \
	src/model_format/gguf_reader.cpp \
	src/model_format/gguf_tensor_bridge.cpp \
	src/tensor/dtype.cpp \
	src/tensor/shape.cpp \
	src/tensor/tensor.cpp \
	src/tensor/tensor_error.cpp

BENCHMARK_LIBRARY_SOURCES := \
	src/benchmarks/target_verifier_episode.cpp \
	src/benchmarks/immutable_weight_benchmark.mm \
	src/benchmarks/output_head_benchmark.cpp \
	src/benchmarks/quick_benchmark.cpp \
	src/benchmarks/target_verifier_benchmark.cpp \
	src/benchmarks/width_two_verifier_benchmark.cpp

ENGINE_SOURCES := $(GGUF_VALUE_SOURCES) \
	src/engine/gemma_chat_session.cpp \
	src/engine/gemma_model.cpp \
	src/engine/gemma2_metal_plan.cpp \
	src/engine/gemma2_runtime.cpp \
	src/engine/gemma_target_runtime.cpp \
	src/engine/chain_chat.mm \
	src/engine/standalone_chat.cpp \
	src/engine/standalone_draft_artifact.cpp \
	src/engine/gemma3_draft.mm \
	src/engine/model.cpp \
	src/engine/prepared_gemma_plan.cpp \
	src/engine/retrieval_proposer.cpp \
	src/engine/sampling.cpp \
	src/engine/speculative_controller.cpp \
	src/engine/target_verifier.cpp \
	src/engine/target_verifier_runtime.cpp \
	src/runtime/backend_error.cpp \
	src/runtime/sha256.cpp \
	src/tokenizer/tokenizer.cpp \
	src/tensor/backend.cpp \
	src/tensor/backend_buffer.mm \
	src/tensor/backends/cpu_backend.cpp \
	src/tensor/backends/metal/metal_buffer.mm \
	src/tensor/backends/metal/metal_command_context.mm \
	src/tensor/backends/metal/metal_context.mm \
	src/tensor/backends/metal/metal_kernels.mm \
	src/tensor/backends/metal/metal_tensor_ops.mm \
	src/tensor/backends/metal_backend.mm \
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
	src/tensor/typed_access.cpp

# Numerical references and fault implementations are not production code.
TEST_ONLY_SOURCES := \
	tests/reference/gemma2_reference.cpp \
	tests/support/target_verifier_test_hooks.cpp

FRONTEND_CLI_SOURCES := \
	src/frontends/cli/cli.cpp \
	src/frontends/cli/options.cpp \
	src/frontends/cli/metrics.cpp \
	src/frontends/cli/gguf_inspect.cpp

GGUF_INSPECT_SOURCES := \
	src/frontends/cli/gguf_inspect_main.cpp

GUFFIN_SOURCES := \
	src/frontends/cli/main.cpp

BENCHMARK_SOURCES := \
	src/frontends/cli/benchmark_main.cpp

TEST_INCLUDE_FLAGS := \
	-Itests/benchmarks \
	-Itests/engine \
	-Itests/frontends/cli \
	-Itests/model_format \
	-Itests/reference \
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
	tests/engine/gemma2_reference_tests.cpp \
	tests/engine/retrieval_proposer_tests.cpp \
	tests/engine/sampling_tests.cpp \
	tests/engine/lossless_sampling_tests.cpp \
	tests/engine/speculative_controller_tests.cpp \
	tests/frontends/cli/cli_tests.cpp \
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

COMMON_FLAGS := -std=c++20 -Iinclude -Wall -Wextra -Wpedantic -Wconversion -Wshadow \
	-DLOCAL_GGUF_ENGINE_BUILD=\"$(ENGINE_BUILD)\"
OBJCXXFLAGS := -fobjc-arc
METAL_LDFLAGS := -framework Foundation -framework Metal
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS ?= -O3 -DNDEBUG -mcpu=apple-m4

ifeq ($(MODE),release)
	CXXFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
else
	CXXFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
endif

# Homebrew's Apple Silicon prefix is already on PATH on the development Mac.
# Macs without that prefix use the conventional /usr/local. Override PREFIX for
# another location; DESTDIR stages an installation without touching that prefix.
PREFIX ?= $(if $(wildcard /opt/homebrew/bin),/opt/homebrew,/usr/local)
DESTDIR ?=

.PHONY: guffin install uninstall test release-test gguf-inspect generate benchmark reference-benchmark clean FORCE
.DELETE_ON_ERROR:

# Tests compile their own engine objects: fault seams must never leak into
# production, and production objects must not disable those seams in tests.
OBJECT_DIR := $(BUILD_DIR)/obj
TEST_OBJECT_DIR := $(BUILD_DIR)/test-obj
objects = $(addprefix $(1)/,$(addsuffix .o,$(basename $(2))))
INSPECT_OBJECTS := $(call objects,$(OBJECT_DIR),$(GGUF_VALUE_SOURCES) src/frontends/cli/gguf_inspect.cpp $(GGUF_INSPECT_SOURCES))
GUFFIN_OBJECTS := $(call objects,$(OBJECT_DIR),$(ENGINE_SOURCES) $(FRONTEND_CLI_SOURCES) $(GUFFIN_SOURCES))
BENCHMARK_OBJECTS := $(call objects,$(OBJECT_DIR),$(ENGINE_SOURCES) $(BENCHMARK_LIBRARY_SOURCES) $(BENCHMARK_SOURCES))
TEST_OBJECTS := $(call objects,$(TEST_OBJECT_DIR),$(ENGINE_SOURCES) $(BENCHMARK_LIBRARY_SOURCES) $(FRONTEND_CLI_SOURCES) $(TEST_ONLY_SOURCES) $(TEST_SOURCES))
ALL_OBJECTS := $(sort $(INSPECT_OBJECTS) $(GUFFIN_OBJECTS) $(BENCHMARK_OBJECTS) $(TEST_OBJECTS))

# Preserve incremental correctness when a compiler, flag, or compiled build
# identity changes. An unchanged configuration leaves the stamp untouched.
BUILD_CONFIG := $(BUILD_DIR)/build-config
$(BUILD_CONFIG): FORCE
	@mkdir -p "$(BUILD_DIR)"
	@printf '%s\n' "$(CXX)" "$(CPPFLAGS)" "$(CXXFLAGS)" "$(OBJCXXFLAGS)" "$(TEST_BUILD_FLAGS)" "$(TEST_INCLUDE_FLAGS)" "$(LDFLAGS)" "$(METAL_LDFLAGS)" > "$@.tmp"
	@cmp -s "$@.tmp" "$@" || mv "$@.tmp" "$@"
	@rm -f "$@.tmp"

FORCE:

# A changed source list must relink even when it only removes an object.
$(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(GUFFIN_BINARY) $(BENCHMARK_BINARY): Makefile $(BUILD_CONFIG)

$(OBJECT_DIR)/%.o: %.cpp $(BUILD_CONFIG)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJECT_DIR)/%.o: %.mm $(BUILD_CONFIG)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJCXXFLAGS) -MMD -MP -c $< -o $@

$(TEST_OBJECT_DIR)/%.o: %.cpp $(BUILD_CONFIG)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TEST_BUILD_FLAGS) $(TEST_INCLUDE_FLAGS) -MMD -MP -c $< -o $@

$(TEST_OBJECT_DIR)/%.o: %.mm $(BUILD_CONFIG)
	@mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJCXXFLAGS) $(TEST_BUILD_FLAGS) $(TEST_INCLUDE_FLAGS) -MMD -MP -c $< -o $@

# Compiler-generated dependencies cover all nested public/internal headers,
# including the embedded Metal source, without maintaining a header glob.
-include $(ALL_OBJECTS:.o=.d)

test: $(TEST_BINARY) $(GGUF_INSPECT_BINARY) $(GUFFIN_BINARY) $(BENCHMARK_BINARY)
	./$(TEST_BINARY)

release-test:
	$(MAKE) MODE=release test

gguf-inspect: $(GGUF_INSPECT_BINARY)

guffin generate: $(GUFFIN_BINARY)

install:
	$(MAKE) MODE=release guffin
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 755 build/release/guffin "$(DESTDIR)$(PREFIX)/bin/guffin"

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/bin/guffin"

benchmark: $(GUFFIN_BINARY) $(BENCHMARK_BINARY)

reference-benchmark:
	/opt/homebrew/bin/llama-cli --version 2>&1 | grep -q d48a56eff

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TEST_OBJECTS) $(METAL_LDFLAGS) -o $@

$(GGUF_INSPECT_BINARY): $(INSPECT_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INSPECT_OBJECTS) -o $@

$(GUFFIN_BINARY): $(GUFFIN_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(GUFFIN_OBJECTS) $(METAL_LDFLAGS) -o $@

$(BENCHMARK_BINARY): $(BENCHMARK_OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(BENCHMARK_OBJECTS) $(METAL_LDFLAGS) -o $@

clean:
	rm -rf build
