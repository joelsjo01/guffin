#include "generate_cli_tests.hpp"

#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/gemma2_reference.hpp"
#include "local_gguf_engine/frontends/cli/generate.hpp"
#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_kind.hpp"
#include "local_gguf_engine/tensor/backend.hpp"
#include "local_gguf_engine/tensor/shape.hpp"
#include "local_gguf_engine/tensor/tensor.hpp"
#include "local_gguf_engine/tensor/metal_trace_capture.hpp"

#include "tensor_test_support.hpp"
#include "../../../src/engine/target_verifier_test_hooks.hpp"
#include "../../../src/engine/prepared_plan_test_hooks.hpp"
#include "../../../src/tensor/backends/metal/metal_test_hooks.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using local_gguf_engine::frontends::cli::run_generate;
using local_gguf_engine::frontends::cli::format_chat_timing_line;
using local_gguf_engine::frontends::cli::format_chat_timing_detail_line;
using local_gguf_engine::frontends::cli::format_chat_decode_attribution_line;
using local_gguf_engine::engine::Gemma2ChatSession;
using local_gguf_engine::engine::Gemma2ChatContextError;
using local_gguf_engine::engine::Gemma2ChatSessionOptions;
using local_gguf_engine::engine::Gemma2ChatTurnOptions;
using local_gguf_engine::engine::Gemma2GenerationStopReason;
using local_gguf_engine::engine::Gemma2TokenSelectionMode;
using local_gguf_engine::engine::TokenSamplerOptions;
using local_gguf_engine::model_format::read_file_backed_gguf_model;
using local_gguf_engine::model_format::GGUFMetadataArray;
using local_gguf_engine::model_format::GGUFMetadataEntry;
using local_gguf_engine::model_format::GGUFMetadataValue;
using local_gguf_engine::model_format::GGUFMetadataValueType;
using local_gguf_engine::runtime::BackendKind;
using local_gguf_engine::tensor::DType;
using local_gguf_engine::tensor::Shape;
using local_gguf_engine::tensor::TensorDescriptor;
using local_gguf_engine::tensor::tensor_backend_is_available;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::gemma2_chat_template_fixture;
using local_gguf_engine::tests::metadata_bool;
using local_gguf_engine::tests::metadata_f32;
using local_gguf_engine::tests::metadata_i32_array;
using local_gguf_engine::tests::metadata_string;
using local_gguf_engine::tests::metadata_string_array;
using local_gguf_engine::tests::metadata_u32;
using local_gguf_engine::tests::tensor_f16_gguf_matrix;
using local_gguf_engine::tests::tensor_f32_vector;

class TempBinaryFile final {
public:
    TempBinaryFile(std::string_view name, std::span<const std::uint8_t> bytes)
        : path_(std::filesystem::temp_directory_path() / ("local-gguf-engine-" + std::string(name) + ".gguf")) {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw TestFailure("could not create temp binary file: " + path_.string());
        }

        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw TestFailure("could not write temp binary file: " + path_.string());
        }
    }

    ~TempBinaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::size_t chat_timing_line_count(std::string_view text) {
    std::size_t count = 0;
    for (std::size_t position = 0; (position = text.find("[ Prompt: ", position)) != std::string_view::npos;) {
        ++count;
        position += 10U;
    }
    return count;
}

[[nodiscard]] std::string without_chat_timing_lines(std::string_view text) {
    std::string filtered;
    std::size_t line_start = 0;
    while (line_start < text.size()) {
        const auto newline = text.find('\n', line_start);
        const auto line_end = newline == std::string_view::npos ? text.size() : newline + 1U;
        const auto line = text.substr(line_start, line_end - line_start);
        if (!line.starts_with("[ Prompt: ") && !line.starts_with("[ Detail: ")) {
            filtered.append(line);
        }
        line_start = line_end;
    }
    return filtered;
}

void test_formats_chat_timing_line_from_runtime_metrics() {
    local_gguf_engine::engine::Gemma2ChatTurnMetrics metrics;
    metrics.generation.newly_evaluated_prompt_token_count = 19U;
    metrics.generation.prompt_evaluation_duration_ns = 1'000'000'000U;
    metrics.prompt_processing_duration_ns = 1'250'000'000U;
    metrics.generation.selected_token_count = 78U;
    metrics.generation_duration_ns = 5'000'000'000U;
    metrics.generation.decode_token_count = 78U;
    metrics.generation.decode_duration_ns = 5'000'000'000U;
    metrics.generation.runtime_first_token_selected_ns = 1'250'000'000U;
    expect(
        format_chat_timing_line(metrics)
            == "[ Prompt: 15.2 t/s | Generation: 15.6 t/s | TTFT: 1.250000 s ]",
        "chat timing normal formatting mismatch"
    );
    metrics.reused_cached_token_count = 17U;
    expect(
        format_chat_timing_detail_line(metrics)
            == "[ Detail: Evaluated: 19 | Cached: 17 | Decoded: 78 | Prompt: 1.250000 s | Generation: 5.000000 s ]",
        "chat timing detail formatting mismatch"
    );

    metrics.generation.newly_evaluated_prompt_token_count = 2U;
    metrics.generation.prompt_evaluation_duration_ns = 3'000'000'000U;
    metrics.prompt_processing_duration_ns = 3'000'000'000U;
    metrics.generation.selected_token_count = 0U;
    metrics.generation_duration_ns = 0U;
    metrics.generation.decode_token_count = 0U;
    metrics.generation.decode_duration_ns = 0U;
    metrics.generation.runtime_first_token_selected_ns = 0U;
    expect(
        format_chat_timing_line(metrics)
            == "[ Prompt: 0.7 t/s | Generation: n/a | TTFT: 3.000000 s ]",
        "chat timing rounding/unavailable formatting mismatch"
    );

    metrics.generation.newly_evaluated_prompt_token_count = 0U;
    metrics.generation.prompt_evaluation_duration_ns = 1U;
    metrics.prompt_processing_duration_ns = 0U;
    metrics.generation.selected_token_count = 1U;
    metrics.generation.decode_token_count = 1U;
    metrics.generation.decode_duration_ns = 0U;
    expect(
        format_chat_timing_line(metrics)
            == "[ Prompt: n/a | Generation: n/a | TTFT: n/a ]",
        "chat timing zero input formatting mismatch"
    );
}

void test_formats_and_validates_chat_decode_attribution() {
    local_gguf_engine::engine::Gemma2ChatTurnMetrics metrics;
    metrics.initial_cache_position = 10U;
    metrics.reused_cached_token_count = 10U;
    metrics.generation.newly_evaluated_prompt_token_count = 3U;
    metrics.generation.selected_token_count = 2U;
    metrics.generation.decode_token_count = 2U;
    metrics.generation.decode_duration_ns = 200U;
    metrics.generation.selection_duration_ns = 7U;
    metrics.generation.selected_token_hash = 0x1234U;
    metrics.generation.speculative_cycle_count = 1U;
    metrics.generation.speculative_second_token_match_count = 1U;
    metrics.generation.speculative_proposal_buckets[0] = {
        .cycle_count = 1U, .match_count = 1U,
    };
    metrics.decode_execution.command_buffer_count = 2U;
    metrics.decode_execution.commit_count = 2U;
    metrics.decode_execution.synchronous_wait_count = 2U;
    metrics.decode_execution.compute_encoder_count = 2U;
    metrics.decode_execution.compute_dispatch_count = 20U;
    metrics.decode_execution.memory_barrier_count = 8U;
    metrics.decode_execution.gpu_duration_ns = 150U;
    metrics.decode_execution.decode_attention_dispatch_count = 4U;
    metrics.decode_execution.decode_attention_partition_count = 5U;
    metrics.decode_execution.decode_attention_token_count = 123U;
    metrics.decode_execution.output_head_elision_count = 1U;
    metrics.decode_execution.output_elided_weight_bytes = 5U;
    metrics.decode_scheduled_weight_bytes = {
        .embedding = 1U,
        .attention_projections = 2U,
        .feed_forward_gate_up = 3U,
        .feed_forward_down = 4U,
        .output_head = 5U,
        .norms = 6U,
        .total = 21U,
    };
    const auto line = format_chat_decode_attribution_line(metrics);
    expect(line.starts_with("[ Decode: {\"schema_version\":5"),
        "decode attribution prefix mismatch");
    expect(line.find("\"standalone_effective_sampler\":null") != std::string::npos,
        "target-only diagnostics invent a draft sampler");
    expect(line.find("\"selected_token_hash\":\"0000000000001234\"") != std::string::npos,
        "decode attribution selected-token hash mismatch");
    expect(line.find("\"proposal_buckets\":[{\"match_length\":\"2\",\"support\":\"2\","
                     "\"confidence\":\"0.5-2/3\",\"cycles\":1,\"matches\":1}]")
            != std::string::npos,
        "decode attribution proposal bucket mismatch");
    expect(line.find("\"final_cache_position\":15") != std::string::npos,
        "decode attribution final cache mismatch");
    expect(line.find("\"wall_minus_gpu_ns\":50") != std::string::npos,
        "decode attribution wall residual mismatch");
    expect(line.find("\"attention_tokens\":123") != std::string::npos,
        "decode attribution attention token mismatch");
    expect(line.find("\"output_head_elisions\":1") != std::string::npos,
        "decode attribution output-head elision mismatch");
    expect(line.find("\"output_elided_weight_bytes\":5") != std::string::npos,
        "decode attribution output elision bytes mismatch");
    expect(line.find("\"scheduled_bytes_total\":37") != std::string::npos,
        "decode attribution scheduled total mismatch");

    metrics.standalone_draft = true;
    metrics.standalone_kind = local_gguf_engine::engine::StandaloneDraftKind::gemma3_270m_q8;
    metrics.standalone_effective_sampler = local_gguf_engine::engine::TokenSamplerOptions {
        .temperature = 0.5F, .top_k = 0, .top_p = 0.75F, .min_p = 0.125F, .repetition_penalty = 1.5F};
    metrics.learned_horizon = 3;
    metrics.generation.learned_cycle_count = 4;
    metrics.generation.learned_accepted_edge_count = 6;
    metrics.generation.learned_accepted_prefix_histogram[0] = 1;
    metrics.generation.learned_accepted_prefix_histogram[1] = 1;
    metrics.generation.learned_accepted_prefix_histogram[2] = 1;
    metrics.generation.learned_accepted_prefix_histogram[3] = 1;
    metrics.generation.learned_rejection_depth_histogram[0] = 1;
    metrics.generation.learned_rejection_depth_histogram[1] = 1;
    metrics.generation.learned_censored_cycle_count = 1;
    metrics.generation.learned_full_accept_count = 1;
    metrics.generation.learned_horizon_limited_cycle_count = 1;
    const auto gemma3_line = format_chat_decode_attribution_line(metrics);
    expect(gemma3_line.find("\"speculative_draft_model\":\"standalone_gemma3_tli\"") != std::string::npos
            && gemma3_line.find("\"learned_draft_head\":\"q8_0\"") != std::string::npos,
        "Gemma3 draft attribution inherited Vicuna/F16 identity");
    expect(gemma3_line.find("\"standalone_effective_sampler\":{\"temperature\":0.5,\"top_k\":0,\"top_p\":0.75,\"min_p\":0.125,\"repetition_penalty\":1.5}") != std::string::npos,
        "standalone diagnostics omit actual effective draft scalars or confuse zero with inheritance");
    expect(gemma3_line.find("\"standalone_draft_sampler\":\"explicit_or_target_scalars_native_history_v1\"") != std::string::npos,
        "standalone sampler identity incorrectly claims unconditional target-scalar inheritance");
    expect(gemma3_line.find("\"learned_accepted_prefix_histogram\":[1,1,1,1,0") != std::string::npos
        && gemma3_line.find("\"learned_rejection_depth_histogram\":[1,1,0") != std::string::npos
        && gemma3_line.find("\"learned_censored_cycle_count\":1") != std::string::npos
        && gemma3_line.find("\"learned_full_accept_count\":1") != std::string::npos
        && gemma3_line.find("\"learned_horizon_limited_cycle_count\":1") != std::string::npos,
        "standalone diagnostics lost prefix/rejection/censoring attribution");
    metrics.standalone_kind = local_gguf_engine::engine::StandaloneDraftKind::vicuna_68m_f16;
    const auto vicuna_line = format_chat_decode_attribution_line(metrics);
    expect(vicuna_line.find("\"speculative_draft_model\":\"standalone_vicuna_tli\"") != std::string::npos
            && vicuna_line.find("\"learned_draft_head\":\"f16\"") != std::string::npos,
        "Vicuna draft attribution changed with Gemma3 support");
    for (const bool missing : {false, true}) {
        auto malformed = metrics;
        if (missing) malformed.standalone_effective_sampler.reset();
        else malformed.standalone_effective_sampler->temperature = std::numeric_limits<float>::quiet_NaN();
        bool refused = false;
        try { static_cast<void>(format_chat_decode_attribution_line(malformed)); }
        catch (const std::logic_error&) { refused = true; }
        expect(refused, "standalone diagnostics silently invented/misrepresented effective sampler values");
    }

    metrics.decode_execution.gpu_duration_ns = 201U;
    bool rejected = false;
    try {
        static_cast<void>(format_chat_decode_attribution_line(metrics));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    expect(rejected, "decode attribution must reject GPU time beyond wall time");
}

void test_metal_trace_capture_arm_validation() {
    namespace metal = local_gguf_engine::tensor::backends::metal;
    const auto path = std::filesystem::temp_directory_path()
        / "local-gguf-engine-stage0-arm-test.gputrace";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    const char* previous = std::getenv("MTL_CAPTURE_ENABLED");
    const auto previous_value = previous == nullptr ? std::optional<std::string> {}
                                                    : std::optional<std::string>(previous);
    unsetenv("MTL_CAPTURE_ENABLED");
    bool rejected = false;
    try {
        metal::arm_next_gemma2_decode_trace(path.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected && metal::metal_trace_capture_status() == metal::MetalTraceCaptureStatus::idle,
        "Metal trace arm must require MTL_CAPTURE_ENABLED=1");

    setenv("MTL_CAPTURE_ENABLED", "1", 1);
    {
        std::ofstream existing(path);
        existing << "existing";
    }
    rejected = false;
    try {
        metal::arm_next_gemma2_decode_trace(path.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    expect(rejected && metal::metal_trace_capture_status() == metal::MetalTraceCaptureStatus::idle,
        "Metal trace arm must refuse overwrite");
    std::filesystem::remove(path, ignored);

    metal::arm_next_gemma2_decode_trace(path.string());
    expect(metal::metal_trace_capture_status() == metal::MetalTraceCaptureStatus::armed,
        "Metal trace must enter armed state");
    metal::cancel_armed_metal_trace_capture();
    expect(metal::metal_trace_capture_status() == metal::MetalTraceCaptureStatus::idle,
        "Metal trace cancellation must restore idle state");
    if (previous_value.has_value()) setenv("MTL_CAPTURE_ENABLED", previous_value->c_str(), 1);
    else unsetenv("MTL_CAPTURE_ENABLED");
}

int run_generate_for_path(
    const std::filesystem::path& path,
    std::string_view prompt,
    std::string_view max_new_tokens,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string prompt_string(prompt);
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        prompt_string.c_str(),
        max_new_tokens_string.c_str(),
    };
    return run_generate(4, argv, output, error);
}

int run_generate_for_path_with_greedy(
    const std::filesystem::path& path,
    std::string_view prompt,
    std::string_view max_new_tokens,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string prompt_string(prompt);
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        prompt_string.c_str(),
        max_new_tokens_string.c_str(),
        "--greedy",
    };
    return run_generate(5, argv, output, error);
}

int run_generate_chat_for_path(
    const std::filesystem::path& path,
    std::string_view max_new_tokens,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "--chat",
        "--max-new-tokens",
        max_new_tokens_string.c_str(),
        "--greedy",
    };
    return run_generate(6, argv, input, output, error);
}

int run_generate_chat_for_path_with_reverse_stop(
    const std::filesystem::path& path,
    std::string_view max_new_tokens,
    std::string_view reverse_stop,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string max_new_tokens_string(max_new_tokens);
    const std::string reverse_stop_string(reverse_stop);
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "--chat",
        "--max-new-tokens",
        max_new_tokens_string.c_str(),
        "--reverse-prompt",
        reverse_stop_string.c_str(),
        "--greedy",
    };
    return run_generate(8, argv, input, output, error);
}

int run_generate_chat_for_path_with_sampling(
    const std::filesystem::path& path,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(), path_string.c_str(), "--chat", "--max-new-tokens", "3",
        "--temperature", "1", "--top-k", "2", "--top-p", "1",
        "--repeat-penalty", "1.1", "--seed", "42",
    };
    return run_generate(15, argv, input, output, error);
}

int run_generate_chat_for_path_with_context_shift(
    const std::filesystem::path& path,
    std::string_view max_new_tokens,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const std::string max_new_tokens_string(max_new_tokens);
    const char* argv[] = {
        program.c_str(), path_string.c_str(), "--chat", "--max-new-tokens",
        max_new_tokens_string.c_str(), "--context-shift", "--greedy",
    };
    return run_generate(7, argv, input, output, error);
}

int run_generate_chat_with_default_options(
    const std::filesystem::path& path,
    std::istream& input,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {program.c_str(), path_string.c_str(), "--chat"};
    return run_generate(3, argv, input, output, error);
}

int run_generate_chat_with_extra_arguments(
    const std::filesystem::path& path,
    const std::vector<std::string>& extra_arguments,
    std::ostringstream& output,
    std::ostringstream& error
) {
    std::vector<std::string> arguments {
        "local-gguf-generate", path.string(), "--chat", "--max-new-tokens", "1",
    };
    arguments.insert(arguments.end(), extra_arguments.begin(), extra_arguments.end());
    std::vector<const char*> argv;
    argv.reserve(arguments.size());
    for (const auto& argument : arguments) {
        argv.push_back(argument.c_str());
    }
    std::istringstream input("/exit\n");
    return run_generate(static_cast<int>(argv.size()), argv.data(), input, output, error);
}

void append_u8(std::vector<std::uint8_t>& bytes, std::uint8_t value) {
    bytes.push_back(value);
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (std::uint32_t shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_i32(std::vector<std::uint8_t>& bytes, std::int32_t value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_f32(std::vector<std::uint8_t>& bytes, float value) {
    append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void append_gguf_string(std::vector<std::uint8_t>& bytes, std::string_view value) {
    append_u64(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t gguf_metadata_type_id(GGUFMetadataValueType type) {
    return static_cast<std::uint32_t>(type);
}

void append_metadata_array(std::vector<std::uint8_t>& bytes, const GGUFMetadataArray& array) {
    append_u32(bytes, gguf_metadata_type_id(array.element_type));

    switch (array.element_type) {
        case GGUFMetadataValueType::string: {
            const auto& values = std::get<std::vector<std::string>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto& value : values) {
                append_gguf_string(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::int32: {
            const auto& values = std::get<std::vector<std::int32_t>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_i32(bytes, value);
            }
            return;
        }
        case GGUFMetadataValueType::float32: {
            const auto& values = std::get<std::vector<float>>(array.values);
            append_u64(bytes, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                append_f32(bytes, value);
            }
            return;
        }
        default:
            throw TestFailure("test GGUF writer unsupported metadata array type");
    }
}

void append_metadata_value(std::vector<std::uint8_t>& bytes, const GGUFMetadataValue& value) {
    append_u32(bytes, gguf_metadata_type_id(value.type));

    switch (value.type) {
        case GGUFMetadataValueType::uint32:
            append_u32(bytes, std::get<std::uint32_t>(value.data));
            return;
        case GGUFMetadataValueType::float32:
            append_f32(bytes, std::get<float>(value.data));
            return;
        case GGUFMetadataValueType::boolean:
            append_u8(bytes, static_cast<std::uint8_t>(std::get<bool>(value.data) ? 1 : 0));
            return;
        case GGUFMetadataValueType::string:
            append_gguf_string(bytes, std::get<std::string>(value.data));
            return;
        case GGUFMetadataValueType::array:
            append_metadata_array(bytes, std::get<GGUFMetadataArray>(value.data));
            return;
        default:
            throw TestFailure("test GGUF writer unsupported metadata value type");
    }
}

void append_metadata_entry(std::vector<std::uint8_t>& bytes, const GGUFMetadataEntry& entry) {
    append_gguf_string(bytes, entry.key);
    append_metadata_value(bytes, entry.value);
}

std::uint32_t gguf_type_id_for_dtype(DType dtype) {
    switch (dtype) {
        case DType::f32:
            return 0;
        case DType::f16:
            return 1;
        case DType::q8_0:
            return 8;
        case DType::q4_k:
            return 12;
        case DType::q6_k:
            return 14;
    }

    throw TestFailure("test GGUF writer unsupported tensor dtype");
}

struct TinyGGUFTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type_id = 0;
    std::vector<std::uint8_t> bytes;
};

std::size_t align_size(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }

    return value + alignment - remainder;
}

std::vector<std::uint8_t> write_tiny_gguf_bytes(
    const std::vector<GGUFMetadataEntry>& metadata,
    const std::vector<TinyGGUFTensor>& tensors
) {
    constexpr std::size_t alignment = 32;
    std::vector<std::uint64_t> tensor_offsets;
    tensor_offsets.reserve(tensors.size());

    std::size_t payload_offset = 0;
    for (const auto& tensor : tensors) {
        tensor_offsets.push_back(static_cast<std::uint64_t>(payload_offset));
        payload_offset = align_size(payload_offset + tensor.bytes.size(), alignment);
    }

    std::vector<std::uint8_t> bytes;
    bytes.push_back(0x47);
    bytes.push_back(0x47);
    bytes.push_back(0x55);
    bytes.push_back(0x46);
    append_u32(bytes, 3);
    append_u64(bytes, static_cast<std::uint64_t>(tensors.size()));
    append_u64(bytes, static_cast<std::uint64_t>(metadata.size()));

    for (const auto& entry : metadata) {
        append_metadata_entry(bytes, entry);
    }

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        append_gguf_string(bytes, tensor.name);
        append_u32(bytes, static_cast<std::uint32_t>(tensor.dimensions.size()));
        for (const auto dimension : tensor.dimensions) {
            append_u64(bytes, dimension);
        }
        append_u32(bytes, tensor.type_id);
        append_u64(bytes, tensor_offsets[tensor_index]);
    }

    const auto tensor_data_start = align_size(bytes.size(), alignment);
    bytes.resize(tensor_data_start, 0);

    for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto& tensor = tensors[tensor_index];
        const auto write_offset = tensor_data_start + static_cast<std::size_t>(tensor_offsets[tensor_index]);
        if (bytes.size() < write_offset) {
            bytes.resize(write_offset, 0);
        }
        bytes.insert(bytes.end(), tensor.bytes.begin(), tensor.bytes.end());
    }

    return bytes;
}

std::vector<std::uint8_t> zero_tensor_bytes(std::span<const std::uint64_t> dimensions, DType dtype) {
    return std::vector<std::uint8_t>(
        TensorDescriptor::from(Shape::from(dimensions), dtype).byte_count
    );
}

TinyGGUFTensor tiny_zero_tensor(std::string name, std::vector<std::uint64_t> dimensions, DType dtype) {
    return TinyGGUFTensor {
        .name = std::move(name),
        .dimensions = dimensions,
        .type_id = gguf_type_id_for_dtype(dtype),
        .bytes = zero_tensor_bytes(std::span<const std::uint64_t>(dimensions), dtype),
    };
}

[[nodiscard]] std::string gemma2_block_tensor_name(std::uint32_t block_index, std::string_view suffix) {
    return "blk." + std::to_string(block_index) + "." + std::string(suffix);
}

std::vector<std::string> tiny_gemma2_generate_cli_token_texts() {
    std::vector<std::string> tokens;
    tokens.reserve(110);
    for (std::uint32_t token_id = 0; token_id < 110; ++token_id) {
        tokens.push_back("<unused" + std::to_string(token_id) + ">");
    }

    tokens[0] = "<pad>";
    tokens[1] = "<eos>";
    tokens[2] = "<bos>";
    tokens[3] = "<unk>";
    tokens[4] = "<mask>";
    tokens[5] = "Hello";
    tokens[6] = "▁Hello";
    tokens[9] = "user\n";
    tokens[10] = "model\n";
    tokens[11] = "\n";
    tokens[12] = "H";
    tokens[13] = "e";
    tokens[14] = "l";
    tokens[15] = "o";
    tokens[16] = "He";
    tokens[17] = "Hel";
    tokens[18] = "Hell";
    tokens[19] = "u";
    tokens[20] = "s";
    tokens[21] = "r";
    tokens[22] = "us";
    tokens[23] = "use";
    tokens[24] = "user";
    tokens[25] = "m";
    tokens[26] = "mo";
    tokens[27] = "mod";
    tokens[28] = "mode";
    tokens[29] = "model";
    tokens[106] = "<start_of_turn>";
    tokens[107] = "<end_of_turn>";
    return tokens;
}

std::vector<std::int32_t> tiny_gemma2_generate_cli_token_types() {
    std::vector<std::int32_t> token_types(110, 1);
    token_types[0] = 3;
    token_types[1] = 3;
    token_types[2] = 3;
    token_types[3] = 2;
    token_types[4] = 3;
    token_types[106] = 3;
    token_types[107] = 3;
    return token_types;
}

std::vector<GGUFMetadataEntry> tiny_gemma2_generate_cli_metadata(std::uint32_t context_length = 16) {
    return {
        metadata_string("general.architecture", "gemma2"),
        metadata_string("general.name", "Tiny Gemma 2 Generate Fixture"),
        metadata_u32("gemma2.context_length", context_length),
        metadata_u32("gemma2.embedding_length", 256),
        metadata_u32("gemma2.block_count", 1),
        metadata_u32("gemma2.feed_forward_length", 256),
        metadata_u32("gemma2.attention.head_count", 2),
        metadata_u32("gemma2.attention.head_count_kv", 1),
        metadata_u32("gemma2.attention.key_length", 256),
        metadata_u32("gemma2.attention.value_length", 256),
        metadata_u32("gemma2.attention.sliding_window", 8),
        metadata_f32("gemma2.attention.layer_norm_rms_epsilon", 0.000001F),
        metadata_f32("gemma2.attn_logit_softcapping", 50.0F),
        metadata_f32("gemma2.final_logit_softcapping", 30.0F),
        metadata_string("tokenizer.ggml.model", "llama"),
        metadata_string_array("tokenizer.ggml.tokens", tiny_gemma2_generate_cli_token_texts()),
        metadata_i32_array("tokenizer.ggml.token_type", tiny_gemma2_generate_cli_token_types()),
        metadata_u32("tokenizer.ggml.bos_token_id", 2),
        metadata_u32("tokenizer.ggml.eos_token_id", 1),
        metadata_u32("tokenizer.ggml.unknown_token_id", 3),
        metadata_bool("tokenizer.ggml.add_bos_token", true),
        metadata_bool("tokenizer.ggml.add_eos_token", false),
        metadata_bool("tokenizer.ggml.add_space_prefix", false),
        metadata_string("tokenizer.chat_template", gemma2_chat_template_fixture()),
    };
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors() {
    std::vector<TinyGGUFTensor> tensors;
    tensors.reserve(13);
    tensors.push_back(tiny_zero_tensor("token_embd.weight", {256, 110}, DType::f16));
    tensors.push_back(tiny_zero_tensor("output_norm.weight", {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "post_attention_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_q.weight"), {256, 512}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_k.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_v.weight"), {256, 256}, DType::q6_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "attn_output.weight"), {512, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "post_ffw_norm.weight"), {256}, DType::f32));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_gate.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_up.weight"), {256, 256}, DType::q4_k));
    tensors.push_back(tiny_zero_tensor(gemma2_block_tensor_name(0, "ffn_down.weight"), {256, 256}, DType::q6_k));
    return tensors;
}

void replace_tiny_tensor(
    std::vector<TinyGGUFTensor>& tensors,
    std::string_view name,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    for (auto& tensor : tensors) {
        if (tensor.name != name) {
            continue;
        }

        tensor.bytes.assign(storage.bytes().begin(), storage.bytes().end());
        return;
    }

    throw TestFailure(std::string("missing tiny tensor fixture ") + std::string(name));
}

void insert_output_weight_tensor(
    std::vector<TinyGGUFTensor>& tensors,
    const local_gguf_engine::tensor::TensorStorage& storage
) {
    tensors.insert(
        tensors.begin() + 2,
        TinyGGUFTensor {
            .name = "output.weight",
            .dimensions = {256, 110},
            .type_id = gguf_type_id_for_dtype(DType::f16),
            .bytes = std::vector<std::uint8_t>(storage.bytes().begin(), storage.bytes().end()),
        }
    );
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors_for_forced_token(
    std::uint32_t token_id
) {
    auto tensors = tiny_gemma2_generate_cli_tensors();

    std::vector<float> embedding_values(256U * 110U, 0.0F);
    for (std::uint32_t token = 0; token < 110U; ++token) {
        embedding_values[token] = 1.0F;
    }
    const auto embedding = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(embedding_values)
    );
    replace_tiny_tensor(tensors, "token_embd.weight", embedding);

    std::vector<float> output_norm_values(256U, 0.0F);
    output_norm_values[0] = 1.0F;
    const auto output_norm = tensor_f32_vector(std::span<const float>(output_norm_values));
    replace_tiny_tensor(tensors, "output_norm.weight", output_norm);

    std::vector<float> output_weight_values(256U * 110U, 0.0F);
    output_weight_values[token_id] = 1.0F;
    const auto output_weight = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(output_weight_values)
    );
    insert_output_weight_tensor(tensors, output_weight);

    return tensors;
}

std::vector<TinyGGUFTensor> tiny_gemma2_generate_cli_tensors_for_sampled_tokens() {
    auto tensors = tiny_gemma2_generate_cli_tensors_for_forced_token(5);
    std::vector<float> output_weight_values(256U * 110U, 0.0F);
    output_weight_values[5] = 0.25F;
    output_weight_values[11] = 0.20F;
    const auto output_weight = tensor_f16_gguf_matrix(
        256,
        110,
        std::span<const float>(output_weight_values)
    );
    replace_tiny_tensor(tensors, "output.weight", output_weight);
    return tensors;
}

std::vector<std::uint8_t> tiny_gemma2_generate_cli_gguf_bytes() {
    return write_tiny_gguf_bytes(tiny_gemma2_generate_cli_metadata(), tiny_gemma2_generate_cli_tensors());
}

void test_runs_generate_gemma2_default_greedy_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-default-greedy", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 default greedy exit code mismatch");
    expect(output.str() == "\n", "generate Gemma2 default greedy output mismatch");
    expect(error.str().empty(), "generate Gemma2 default greedy should not write stderr");
}

void test_runs_generate_gemma2_greedy_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-greedy", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 greedy exit code mismatch");
    expect(output.str() == "\n", "generate Gemma2 greedy output mismatch");
    expect(error.str().empty(), "generate Gemma2 greedy should not write stderr");
}

void test_gemma2_chat_session_appends_second_turn_at_nonzero_cache_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    expect(session.warmup_stats().has_value(), "Gemma2 default constructor warm-up stats missing");
    expect(session.warmup_stats()->prefill_token_count == 16U,
        "Gemma2 constructor warm-up shape mismatch");
    expect(session.message_count() == 0U && session.cached_token_count() == 0U,
        "Gemma2 constructor warm-up mutated visible session state");

    const auto first_turn = session.generate_user_turn("Hello");
    const auto second_turn = session.generate_user_turn("Hello");

    expect(first_turn.cache_position == 0, "Gemma2 chat first turn cache position mismatch");
    expect(second_turn.cache_position > 0, "Gemma2 chat second turn should append at nonzero cache position");
    expect(second_turn.appended_token_count > 0, "Gemma2 chat second turn should append prompt tokens");
    expect(session.message_count() == 4, "Gemma2 chat session message count mismatch after two turns");
}

void test_gemma2_chat_session_appends_after_truncated_output_retokenizes_differently_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(6)
    );
    const TempBinaryFile model("tiny-gemma2-truncated-retokenize", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1U}
    );
    const auto first = session.generate_user_turn("Hello");
    const auto second = session.generate_user_turn("Hello");
    expect(
        first.text == " Hello",
        "truncated retokenization fixture text mismatch: actual='" + first.text + "'"
    );
    expect(first.generated_token_ids == std::vector<std::uint32_t> {6U},
        "truncated retokenization should retain exact generated ID");
    expect(second.cache_position > first.cache_position,
        "truncated retokenization second turn should reuse resident KV");
    expect(second.appended_token_count == 9U,
        "truncated retokenization should append end-of-turn, newline, and canonical next-turn tokens");
    expect(second.metrics.reused_cached_token_count == second.cache_position,
        "truncated retokenization cache metric mismatch");
}

void test_gemma2_chat_session_tracks_nonvisible_generated_tokens_in_cache_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(106)
    );
    const TempBinaryFile model("tiny-gemma2-hidden-cache-token", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1U}
    );

    const auto first = session.generate_user_turn("Hello");
    const auto second = session.generate_user_turn("Hello");

    expect(first.generated_token_ids.empty(), "control token should remain hidden from chat output");
    expect(first.cached_token_count == first.appended_token_count + 1U,
        "exact cache identity should retain generated control token");
    expect(second.cache_position == first.cached_token_count,
        "second turn should append after the exact control-token KV position");
}

void test_gemma2_chat_session_records_deterministic_turn_metrics_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-metrics", std::span<const std::uint8_t>(bytes));
    std::uint64_t now_ns = 100U;
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {
            .max_new_tokens = 1,
            .clock = [&now_ns]() {
                const auto current = now_ns;
                now_ns += 10U;
                return current;
            },
        }
    );

    expect(now_ns == 100U, "Gemma2 constructor warm-up must remain outside the turn clock");

    std::string streamed;
    const auto first = session.generate_user_turn("Hello", [&](std::string_view delta) {
        streamed += delta;
        now_ns += 100U;
    });
    const auto second = session.generate_user_turn("Hello");

    expect(streamed == "Hello", "Gemma2 measured turn stream mismatch");
    expect(first.metrics.generation.prefill_started_ns == 120U, "Gemma2 measured prefill start mismatch");
    expect(first.metrics.generation.prefill_completed_ns == 130U, "Gemma2 measured prefill completion mismatch");
    expect(first.metrics.generation.first_token_selected_ns == 150U, "Gemma2 measured selection boundary mismatch");
    expect(first.metrics.generation.last_token_selected_ns == 150U, "Gemma2 measured final selection boundary mismatch");
    expect(first.metrics.generation.decode_started_ns == 160U, "Gemma2 measured decode start mismatch");
    expect(first.metrics.generation.decode_completed_ns == 170U, "Gemma2 measured decode completion mismatch");
    expect(
        first.metrics.generation.generation_completed_ns == 290U,
        "Gemma2 measured runtime completion mismatch: actual="
            + std::to_string(first.metrics.generation.generation_completed_ns)
    );
    expect(first.metrics.generation.prompt_evaluation_duration_ns == 10U,
        "Gemma2 measured prompt evaluation duration mismatch");
    expect(first.metrics.generation.runtime_first_token_selected_ns == 30U,
        "Gemma2 measured runtime first-selected mismatch");
    expect(first.metrics.prompt_processing_duration_ns == 40U,
        "Gemma2 measured llama-compatible prompt duration mismatch");
    expect(first.metrics.generation_duration_ns == 1'000U,
        "Gemma2 measured llama-compatible generation duration mismatch");
    expect(first.metrics.generation.decode_duration_ns == 10U, "Gemma2 measured decode duration mismatch");
    expect(first.metrics.generation_completion_duration_ns == 170U,
        "complete generation clock must include terminal KV and callback time");
    expect(first.metrics.full_turn_duration_ns == 200U,
        "full-turn clock must include preparation through text finalization");
    expect(first.metrics.first_text_duration_ns == 80U,
        "first-text clock must include target K/V work before visible delivery");
    expect(first.metrics.generation.newly_evaluated_prompt_token_count == first.appended_token_count,
        "Gemma2 measured prompt token count mismatch");
    expect(first.metrics.generation.generated_token_count == 1U, "Gemma2 measured generated count mismatch");
    expect(first.metrics.generation.selected_token_count == 1U, "Gemma2 measured selected count mismatch");
    expect(first.metrics.generation.decode_token_count == 1U, "Gemma2 measured decode count mismatch");
    expect(first.metrics.generation.stop_reason == Gemma2GenerationStopReason::token_limit,
        "Gemma2 measured token-limit reason mismatch");
    expect(first.metrics.selection_mode == Gemma2TokenSelectionMode::greedy,
        "Gemma2 measured greedy mode mismatch");
    expect(first.metrics.initial_cache_position == 0U && first.metrics.reused_cached_token_count == 0U,
        "Gemma2 measured first-turn cache facts mismatch");
    expect(first.metrics.resident_weight_byte_count > 0U, "Gemma2 measured resident weight bytes missing");
    expect(first.metrics.resident_kv_cache_byte_count > 0U, "Gemma2 measured resident KV bytes missing");
    expect(first.metrics.resident_scratch_byte_count > 0U, "Gemma2 measured resident scratch bytes missing");
    expect(second.metrics.initial_cache_position == first.cached_token_count,
        "Gemma2 measured cached-turn initial position mismatch");
    expect(second.metrics.reused_cached_token_count == second.cache_position,
        "Gemma2 measured reused cache count mismatch");
    expect(second.metrics.generation.newly_evaluated_prompt_token_count == second.appended_token_count,
        "Gemma2 measured cached turn should count only appended prompt tokens");
}

void test_gemma2_chat_session_streams_visible_text_before_return_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-stream-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 2}
    );

    bool returned = false;
    std::vector<std::string> chunks;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        expect(!returned, "Gemma2 chat stream callback should run before generate_user_turn returns");
        chunks.push_back(std::string(text_delta));
    });
    returned = true;

    expect(chunks.size() == 2, "Gemma2 chat stream chunk count mismatch");
    expect(chunks[0] == "Hello", "Gemma2 chat first stream chunk mismatch");
    expect(chunks[1] == "Hello", "Gemma2 chat second stream chunk mismatch");
    expect(turn.text == "HelloHello", "Gemma2 chat streamed final text mismatch");
    expect(turn.generated_token_ids.size() == 2, "Gemma2 chat streamed token count mismatch");
    expect(turn.metrics.decode_execution.output_head_elision_count == 1U,
        "known token-limit termination must elide exactly one output head");
}

void test_gemma2_chat_session_can_cancel_generation_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cancel-session", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    bool should_stop = false;
    std::string streamed;
    const auto turn = session.generate_user_turn(
        "Hello",
        Gemma2ChatTurnOptions {
            .on_text_delta = [&](std::string_view text_delta) {
                streamed += text_delta;
                should_stop = true;
            },
            .should_stop = [&]() {
                return should_stop;
            },
        }
    );

    expect(turn.interrupted, "Gemma2 chat cancelled turn should report interrupted");
    expect(streamed == "Hello", "Gemma2 chat cancelled turn stream mismatch");
    expect(turn.text == streamed, "Gemma2 chat cancelled turn final text mismatch");
    expect(turn.generated_token_ids.size() == 1, "Gemma2 chat cancelled turn token count mismatch");
    expect(turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::interrupted,
        "Gemma2 chat cancelled turn measurement reason mismatch");
    expect(turn.metrics.decode_execution.output_head_elision_count == 0U,
        "callback-discovered cancellation must preserve full-output execution ordering");
}

void test_gemma2_chat_session_stops_on_eos_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(1)
    );
    const TempBinaryFile model("tiny-gemma2-chat-eos-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    std::string streamed;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });

    expect(streamed.empty(), "Gemma2 chat EOS stop should not stream visible text");
    expect(turn.text.empty(), "Gemma2 chat EOS stop text should be empty");
    expect(turn.stop_token_id.has_value() && *turn.stop_token_id == 1U, "Gemma2 chat EOS stop token mismatch");
    expect(turn.generated_token_ids.empty(), "Gemma2 chat EOS stop should not store visible tokens");
    expect(turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::stop_token,
        "Gemma2 chat EOS measurement reason mismatch");
    expect(turn.metrics.decode_execution.output_head_elision_count == 1U,
        "known EOS termination must elide the unused output head");
}

void test_gemma2_chat_session_stops_on_end_of_turn_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(107)
    );
    const TempBinaryFile model("tiny-gemma2-chat-eot-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    std::string streamed;
    const auto turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });

    expect(streamed.empty(), "Gemma2 chat EOT stop should not stream visible text");
    expect(turn.text.empty(), "Gemma2 chat EOT stop text should be empty");
    expect(turn.stop_token_id.has_value() && *turn.stop_token_id == 107U, "Gemma2 chat EOT stop token mismatch");
    expect(turn.generated_token_ids.empty(), "Gemma2 chat EOT stop should not store visible tokens");
}

void test_gemma2_chat_session_applies_reverse_stop_and_cleans_history_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-reverse-stop", std::span<const std::uint8_t>(bytes));
    auto file_model = read_file_backed_gguf_model(model.path());
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {
            .max_new_tokens = 3,
            .reverse_stop_strings = {"HelloHello"},
        }
    );

    std::string streamed;
    const auto first_turn = session.generate_user_turn("Hello", [&](std::string_view text_delta) {
        streamed += text_delta;
    });
    const auto second_turn = session.generate_user_turn("Hello");

    expect(streamed.empty(), "Gemma2 chat reverse stop should not stream the stop string");
    expect(first_turn.text.empty(), "Gemma2 chat reverse stop should trim final text");
    expect(
        first_turn.reverse_stop_string.has_value() && *first_turn.reverse_stop_string == "HelloHello",
        "Gemma2 chat reverse stop marker mismatch"
    );
    expect(first_turn.generated_token_ids.empty(), "Gemma2 chat reverse stop should not store stop token ids");
    expect(first_turn.metrics.generation.stop_reason == Gemma2GenerationStopReason::reverse_stop,
        "Gemma2 chat reverse-stop measurement reason mismatch");
    expect(first_turn.metrics.decode_execution.output_head_elision_count == 0U,
        "callback-discovered reverse stop must preserve full-output execution ordering");
    expect(second_turn.cache_position > first_turn.cache_position, "Gemma2 chat reverse stop history should continue");
    expect(session.message_count() == 4, "Gemma2 chat reverse stop message count mismatch");
}

void test_gemma2_chat_session_clear_and_reset_restore_empty_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-clear-reset", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    const auto first_turn = session.generate_user_turn("Hello");
    expect(session.cached_token_count() > 0U, "Gemma2 chat clear setup should cache tokens");
    session.clear();
    expect(session.cached_token_count() == 0U, "Gemma2 chat clear should reset cached tokens");
    expect(session.message_count() == 0U, "Gemma2 chat clear should reset messages");
    const auto after_clear = session.generate_user_turn("Hello");
    expect(after_clear.cache_position == 0U, "Gemma2 chat after clear should restart at cache zero");
    expect(after_clear.generated_token_ids == first_turn.generated_token_ids, "Gemma2 chat clear generation mismatch");

    session.reset();
    expect(session.cached_token_count() == 0U, "Gemma2 chat reset should reset cached tokens");
    expect(session.message_count() == 0U, "Gemma2 chat reset should reset messages");
    const auto after_reset = session.generate_user_turn("Hello");
    expect(after_reset.cache_position == 0U, "Gemma2 chat after reset should restart at cache zero");
    expect(after_reset.generated_token_ids == first_turn.generated_token_ids, "Gemma2 chat reset generation mismatch");
}

void test_gemma2_chat_session_regenerates_last_turn_without_full_replay_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-regen", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    const auto first_turn = session.generate_user_turn("Hello");
    const auto second_turn = session.generate_user_turn("Hello");
    const auto regenerated = session.regenerate_last_turn();

    expect(regenerated.cache_position == second_turn.cache_position, "Gemma2 chat regen cache rollback mismatch");
    expect(regenerated.cache_position > first_turn.cache_position, "Gemma2 chat regen should retain earlier KV prefix");
    expect(regenerated.generated_token_ids == second_turn.generated_token_ids, "Gemma2 chat greedy regen mismatch");
    expect(session.message_count() == 4U, "Gemma2 chat regen should replace, not append, the last turn");
}

void test_gemma2_chat_session_reports_real_token_budget_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-token-budget", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 3}
    );

    expect(session.context_length() == 64U, "Gemma2 chat context length mismatch");
    expect(session.remaining_context_tokens() == 64U, "Gemma2 chat initial remaining tokens mismatch");
    expect(session.max_new_tokens() == 3U, "Gemma2 chat configured max-new-tokens mismatch");
    const auto turn = session.generate_user_turn("Hello");
    expect(
        session.remaining_context_tokens() == session.context_length() - turn.cached_token_count,
        "Gemma2 chat remaining tokens should reflect cached context"
    );
}

void test_gemma2_chat_session_trims_oldest_turns_and_rebuilds_retained_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-trim", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1, .context_shift = true}
    );

    bool rebuilt = false;
    for (std::size_t turn_index = 0; turn_index < 8U; ++turn_index) {
        const auto turn = session.generate_user_turn("Hello");
        if (!turn.rebuilt_context) {
            continue;
        }

        rebuilt = true;
        expect(turn.trimmed_turn_count > 0U, "Gemma2 chat overflow should trim complete turns");
        expect(turn.cache_position == 0U, "Gemma2 chat rebuilt context should prefill from cache zero");
        expect(turn.generated_token_ids == std::vector<std::uint32_t> {5U}, "Gemma2 chat rebuilt output mismatch");
        expect(turn.cached_token_count <= session.context_length(), "Gemma2 chat rebuilt cache exceeds context");
        expect(turn.metrics.rebuilt_context, "Gemma2 chat measured rebuild flag mismatch");
        expect(turn.metrics.trimmed_turn_count == turn.trimmed_turn_count,
            "Gemma2 chat measured trimmed-turn count mismatch");
        expect(turn.metrics.initial_cache_position == 0U && turn.metrics.reused_cached_token_count == 0U,
            "Gemma2 rebuilt measurement should not report reused stale KV");
        break;
    }

    expect(rebuilt, "Gemma2 chat long conversation should trigger context rebuild");
}

void test_gemma2_chat_session_does_not_shift_context_by_default_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-no-context-shift", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );

    std::size_t completed_turns = 0;
    bool rejected = false;
    for (; completed_turns < 8U; ++completed_turns) {
        try {
            const auto turn = session.generate_user_turn("Hello");
            expect(!turn.rebuilt_context, "default chat should not rebuild context");
        } catch (const Gemma2ChatContextError&) {
            rejected = true;
            break;
        }
    }

    expect(rejected, "default chat should stop when conversation history fills the context");
    expect(completed_turns > 0U, "default no-shift fixture should complete an initial turn");
    expect(session.message_count() == completed_turns * 2U, "context rejection should preserve full history");
}

void test_gemma2_chat_session_rejects_impossible_turn_without_mutating_context_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-reject", std::span<const std::uint8_t>(bytes));
    auto session = Gemma2ChatSession(
        read_file_backed_gguf_model(model.path()),
        Gemma2ChatSessionOptions {.max_new_tokens = 1}
    );
    const auto first_turn = session.generate_user_turn("Hello");

    bool rejected = false;
    try {
        static_cast<void>(session.generate_user_turn(std::string(128U, 'x')));
    } catch (const Gemma2ChatContextError& error) {
        rejected = std::string_view(error.what()).find("use a shorter prompt or lower --max-new-tokens")
            != std::string_view::npos;
    }

    expect(rejected, "Gemma2 chat impossible turn should provide recovery guidance");
    expect(session.cached_token_count() == first_turn.cached_token_count, "rejected turn should preserve cached context");
    expect(session.message_count() == 2U, "rejected turn should preserve message history");
    const auto next_turn = session.generate_user_turn("Hello");
    expect(next_turn.cache_position == first_turn.cached_token_count, "turn after rejection should keep append position");
}

void test_gemma2_chat_session_seeded_sampling_and_regen_are_reproducible_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile first_model("tiny-gemma2-chat-sampled-a", std::span<const std::uint8_t>(bytes));
    const TempBinaryFile second_model("tiny-gemma2-chat-sampled-b", std::span<const std::uint8_t>(bytes));
    std::uint64_t now_ns = 1000U;
    const auto options = Gemma2ChatSessionOptions {
        .max_new_tokens = 3,
        .sampler = TokenSamplerOptions {
            .temperature = 1.0F,
            .top_k = 2,
            .top_p = 1.0F,
            .repetition_penalty = 1.1F,
            .seed = 42,
        },
        .clock = [&now_ns]() {
            const auto current = now_ns;
            now_ns += 10U;
            return current;
        },
    };
    auto first_session = Gemma2ChatSession(read_file_backed_gguf_model(first_model.path()), options);
    auto second_session = Gemma2ChatSession(read_file_backed_gguf_model(second_model.path()), options);

    const auto first = first_session.generate_user_turn("Hello");
    const auto same_seed = second_session.generate_user_turn("Hello");
    const auto regenerated = first_session.regenerate_last_turn();

    expect(first.generated_token_ids == same_seed.generated_token_ids, "seeded sampled sessions mismatch");
    expect(first.generated_token_ids == regenerated.generated_token_ids, "sampled /regen should replay sampler state");
    expect(first.metrics.selection_mode == Gemma2TokenSelectionMode::sampled,
        "sampled turn measurement mode mismatch");
    expect(first.metrics.generation.prompt_evaluation_duration_ns == 10U,
        "sampled turn deterministic prompt-evaluation timing mismatch");
    expect(first.metrics.generation.runtime_first_token_selected_ns == 30U,
        "sampled turn deterministic runtime first-selected mismatch");
    expect(first.metrics.prompt_processing_duration_ns == 40U,
        "sampled turn deterministic llama-compatible prompt timing mismatch");
    expect(first.metrics.generation_duration_ns == 1'000U,
        "sampled turn deterministic llama-compatible generation timing mismatch");
    expect(first.metrics.generation.decode_duration_ns == 30U,
        "sampled turn deterministic decode timing mismatch");
    expect(first.metrics.generation.selection_duration_ns == 30U,
        "sampled turn deterministic selection timing mismatch");
    expect(first.metrics.generation.decode_token_count == 3U,
        "sampled turn decode count mismatch");
    expect(first.metrics.generation.selected_token_count == 3U,
        "sampled turn selected count mismatch");
    expect(first_session.message_count() == 2U, "sampled /regen should replace the previous response");
}

void test_runs_generate_gemma2_chat_loop_until_exit_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\nHello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "1", input, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> \n> \n> ",
        "generate Gemma2 chat CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 2U, "generate Gemma2 chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat CLI should not write stderr");
}

void test_runs_generate_chat_decode_diagnostics_only_when_requested_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32), tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-chat-diagnostics", std::span<const std::uint8_t>(bytes));
    const std::vector<std::string> arguments {
        "local-gguf-generate", model.path().string(), "--chat", "--max-new-tokens", "1",
        "--greedy", "--decode-diagnostics",
    };
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    std::istringstream input("Hello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;
    const auto exit_code = run_generate(
        static_cast<int>(argv.size()), argv.data(), input, output, error
    );
    expect(exit_code == EXIT_SUCCESS && error.str().empty(),
        "decode-diagnostic chat invocation failed");
    expect(output.str().find("[ Decode: {\"schema_version\":5") != std::string::npos,
        "requested decode attribution record is missing");
}

void test_rejects_metal_capture_without_diagnostics() {
    std::ostringstream output;
    std::ostringstream error;
    const char* argv[] = {
        "local-gguf-generate", "missing.gguf", "--chat", "--metal-capture", "/tmp/test.gputrace",
    };
    std::istringstream input("/exit\n");
    const auto exit_code = run_generate(5, argv, input, output, error);
    expect(exit_code == EXIT_FAILURE, "Metal capture without diagnostics must fail");
    expect(error.str().find("requires --decode-diagnostics and --backend metal") != std::string::npos,
        "Metal capture validation error mismatch");
}

void test_runs_generate_gemma2_chat_streams_visible_tokens_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli-stream", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "2", input, output, error);

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat stream CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> HelloHello\n> ",
        "generate Gemma2 chat stream CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 1U, "streamed chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat stream CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_reverse_stop_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-cli-reverse-stop", std::span<const std::uint8_t>(bytes));
    std::istringstream input("Hello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path_with_reverse_stop(
        model.path(),
        "3",
        "Hello",
        input,
        output,
        error
    );

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 chat reverse-stop CLI exit code mismatch");
    expect(
        without_chat_timing_lines(output.str()) == "local-gguf-engine chat\ncommands: /exit /quit /clear /reset /regen /tokens /read <path>\nmultiline: end a line with \\\n> \n> ",
        "generate Gemma2 chat reverse-stop CLI output mismatch"
    );
    expect(chat_timing_line_count(output.str()) == 1U, "reverse-stop chat timing line count mismatch");
    expect(error.str().empty(), "generate Gemma2 chat reverse-stop CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_gate3_commands_and_input_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(128),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-gate3-cli", std::span<const std::uint8_t>(bytes));
    const std::vector<std::uint8_t> prompt_bytes {'H', 'e', 'l', 'l', 'o'};
    const TempBinaryFile prompt_file("tiny-gemma2-chat-read-prompt", std::span<const std::uint8_t>(prompt_bytes));
    std::istringstream input(
        "/tokens\n/regen\n/bogus\n/read /definitely/missing/local-gguf-prompt\n"
        "Hello\\\nHello\n/clear\nHello\n/reset\n/read " + prompt_file.path().string()
        + "\n/regen\n/quit\n"
    );
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path(model.path(), "1", input, output, error);
    const auto text = output.str();

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 Gate 3 CLI exit code mismatch");
    expect(text.find("tokens: cached=0 remaining=128 max-new-tokens=1") != std::string::npos, "/tokens output mismatch");
    expect(text.find("error: nothing to regenerate") != std::string::npos, "empty /regen output mismatch");
    expect(text.find("error: unknown command '/bogus'") != std::string::npos, "invalid command output mismatch");
    expect(text.find("error: could not read '/definitely/missing/local-gguf-prompt'") != std::string::npos, "missing /read output mismatch");
    expect(text.find("... Hello") != std::string::npos, "multiline continuation/output mismatch");
    expect(text.find("chat cleared") != std::string::npos, "/clear output mismatch");
    expect(text.find("session reset") != std::string::npos, "/reset output mismatch");
    std::size_t generated_hello_count = 0;
    for (std::size_t position = 0; (position = text.find("Hello", position)) != std::string::npos; position += 5U) {
        ++generated_hello_count;
    }
    expect(generated_hello_count == 4U, "Gate 3 multiline, /read, and /regen generation count mismatch");
    expect(chat_timing_line_count(text) == 4U, "Gate 3 successful turns should each print one timing line");
    expect(error.str().empty(), "generate Gemma2 Gate 3 CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_context_trim_and_recovers_from_impossible_turn_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(32),
        tiny_gemma2_generate_cli_tensors_for_forced_token(5)
    );
    const TempBinaryFile model("tiny-gemma2-chat-context-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream input(
        "Hello\nHello\nHello\nHello\n"
        + std::string(128U, 'x') + "\n/tokens\n/exit\n"
    );
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_for_path_with_context_shift(
        model.path(), "1", input, output, error
    );
    const auto text = output.str();

    expect(exit_code == EXIT_SUCCESS, "generate Gemma2 context CLI should survive turn rejection");
    expect(text.find("context: dropped ") != std::string::npos, "context trim notice missing");
    expect(text.find("oldest turn(s) and rebuilt retained KV") != std::string::npos, "context rebuild notice missing");
    expect(text.find("error: context: turn requires ") != std::string::npos, "context rejection prefix missing");
    expect(text.find("use a shorter prompt or lower --max-new-tokens") != std::string::npos, "context recovery guidance missing");
    expect(text.find("tokens: cached=") != std::string::npos, "REPL should continue after context rejection");
    expect(chat_timing_line_count(text) == 4U, "context rejection should not print an extra timing line");
    const auto timing_before_context = text.rfind("[ Prompt: ", text.find("context: dropped "));
    expect(timing_before_context != std::string::npos, "context rebuild timing should precede rebuild notice");
    expect(error.str().empty(), "generate Gemma2 context CLI should not write stderr");
}

void test_runs_generate_gemma2_chat_with_llama_like_defaults_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_forced_token(107)
    );
    const TempBinaryFile model("tiny-gemma2-chat-defaults", std::span<const std::uint8_t>(bytes));
    std::istringstream input("/tokens\nHello\n/exit\n");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_chat_with_default_options(model.path(), input, output, error);
    expect(exit_code == EXIT_SUCCESS, "default sampled chat CLI exit mismatch");
    expect(
        output.str().find("tokens: cached=0 remaining=64 max-new-tokens=unlimited") != std::string::npos,
        "default chat should report an unlimited response budget"
    );
    expect(chat_timing_line_count(output.str()) == 1U, "default sampled chat should complete one turn");
    expect(error.str().empty(), "default sampled chat should not write stderr");
}

void test_runs_generate_gemma2_chat_seeded_sampling_flags_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(64),
        tiny_gemma2_generate_cli_tensors_for_sampled_tokens()
    );
    const TempBinaryFile model("tiny-gemma2-chat-sampled-cli", std::span<const std::uint8_t>(bytes));
    std::istringstream first_input("Hello\n/regen\n/exit\n");
    std::istringstream second_input("Hello\n/regen\n/exit\n");
    std::ostringstream first_output;
    std::ostringstream second_output;
    std::ostringstream first_error;
    std::ostringstream second_error;

    const auto first_exit = run_generate_chat_for_path_with_sampling(
        model.path(), first_input, first_output, first_error
    );
    const auto second_exit = run_generate_chat_for_path_with_sampling(
        model.path(), second_input, second_output, second_error
    );

    expect(first_exit == EXIT_SUCCESS && second_exit == EXIT_SUCCESS, "sampled chat CLI exit mismatch");
    expect(
        without_chat_timing_lines(first_output.str()) == without_chat_timing_lines(second_output.str()),
        "seeded sampled CLI transcript mismatch"
    );
    expect(chat_timing_line_count(first_output.str()) == 2U, "sampled turn and /regen timing line count mismatch");
    expect(first_error.str().empty() && second_error.str().empty(), "sampled chat CLI should not write stderr");
}

void test_rejects_generate_gemma2_chat_invalid_sampling_flags() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-sampling.gguf";
    struct InvalidCase {
        std::vector<std::string> arguments;
        std::string_view expected_error;
    };
    const std::vector<InvalidCase> cases {
        {{"--temperature", "bad"}, "--temperature requires a finite number"},
        {{"--temperature", "1", "--top-k", "0"}, "--top-k must be a positive integer"},
        {{"--temperature", "1", "--top-p", "2"}, "--top-p must be in (0, 1]"},
        {{"--temperature", "1", "--min-p", "2"}, "--min-p must be in [0, 1]"},
        {{"--temperature", "1", "--repeat-penalty", "0"}, "--repeat-penalty must be positive"},
        {{"--temperature", "1", "--seed", "invalid"}, "--seed must be -1 or an unsigned integer"},
        {{"--greedy", "--temperature", "1"}, "--greedy cannot be combined"},
        {{"--greedy", "--seed", "42"}, "sampling modifiers require --temperature greater than zero"},
    };

    for (const auto& test_case : cases) {
        std::ostringstream output;
        std::ostringstream error;
        const auto exit_code = run_generate_chat_with_extra_arguments(
            path, test_case.arguments, output, error
        );
        expect(exit_code == EXIT_FAILURE, "invalid sampling flags should fail");
        expect(output.str().empty(), "invalid sampling flags should not write stdout");
        expect(error.str().find(test_case.expected_error) != std::string::npos, "invalid sampling error mismatch");
    }
}

void test_rejects_generate_gemma2_chat_invalid_output_head_flags() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-output-head.gguf";
    const std::vector<std::vector<std::string>> cases {
        {"--output-head"},
        {"--output-head", "q4_0"},
    };
    for (const auto& arguments : cases) {
        std::ostringstream output;
        std::ostringstream error;
        const auto exit_code = run_generate_chat_with_extra_arguments(path, arguments, output, error);
        expect(exit_code == EXIT_FAILURE, "invalid output-head flag should fail");
        expect(output.str().empty(), "invalid output-head flag should not write stdout");
        expect(error.str().find("--output-head") != std::string::npos,
            "invalid output-head error should identify the option");
    }
}

void test_reports_generate_gemma2_metal_runtime_shell_unavailable_when_missing() {
    if (tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-metal-missing", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 unavailable runtime shell exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 unavailable runtime shell should not write stdout");
    expect(
        error.str().find("error: backend:") != std::string::npos,
        "generate Gemma2 unavailable runtime shell error prefix missing"
    );
    expect(
        error.str().find("Metal backend unavailable") != std::string::npos,
        "generate Gemma2 unavailable runtime shell message missing"
    );
}

void test_rejects_generate_gemma2_cpu_backend() {
    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    const TempBinaryFile model("tiny-gemma2-generate-cpu", std::span<const std::uint8_t>(bytes));
    const std::string program = "local-gguf-generate";
    const std::string path_string = model.path().string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "Hello",
        "1",
        "--greedy",
        "--backend",
        "cpu",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(7, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 CPU backend exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 CPU backend should not write stdout");
    expect(
        error.str().find("Gemma 2 generation requires Metal backend") != std::string::npos,
        "generate Gemma2 CPU backend error missing"
    );
}

void test_rejects_generate_gemma2_after_decode_schedule_context_full_when_available() {
    if (!tensor_backend_is_available(BackendKind::metal)) {
        return;
    }

    const auto bytes = write_tiny_gguf_bytes(
        tiny_gemma2_generate_cli_metadata(8),
        tiny_gemma2_generate_cli_tensors()
    );
    const TempBinaryFile model("tiny-gemma2-decode-context-full", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path_with_greedy(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 decode schedule context-full exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 decode schedule context-full should not write stdout");
    expect(
        error.str().find("Gemma2 Metal generation token count exceeds context length") != std::string::npos,
        "generate Gemma2 decode schedule context-full error missing"
    );
}

void test_rejects_generate_gemma2_unsupported_chat_template_before_runtime() {
    auto metadata = tiny_gemma2_generate_cli_metadata();
    for (auto& entry : metadata) {
        if (entry.key == "tokenizer.chat_template") {
            entry = metadata_string("tokenizer.chat_template", "{{ messages }}");
            break;
        }
    }

    const auto bytes = write_tiny_gguf_bytes(metadata, tiny_gemma2_generate_cli_tensors());
    const TempBinaryFile model("tiny-gemma2-bad-template", std::span<const std::uint8_t>(bytes));
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(model.path(), "Hello", "1", output, error);

    expect(exit_code == EXIT_FAILURE, "generate Gemma2 unsupported template exit code mismatch");
    expect(output.str().empty(), "generate Gemma2 unsupported template should not write stdout");
    expect(
        error.str().find("unsupported chat template: tokenizer.chat_template") != std::string::npos,
        "generate Gemma2 unsupported template error missing"
    );
}

void test_rejects_generate_missing_arguments() {
    const char* argv[] = {"local-gguf-generate"};
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(1, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate missing-args exit code mismatch");
    expect(output.str().empty(), "generate missing-args should not write stdout");
    expect(
        error.str().find("usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>") != std::string::npos,
        "generate missing-args usage missing"
    );
    expect(error.str().find("missing required argument") != std::string::npos, "generate missing-args error missing");
}

void test_rejects_generate_invalid_max_new_tokens() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate_for_path(path, "", "0", output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-max exit code mismatch");
    expect(output.str().empty(), "generate invalid-max should not write stdout");
    expect(
        error.str().find("max-new-tokens must be a positive integer") != std::string::npos,
        "generate invalid-max error missing"
    );
}

void test_rejects_generate_invalid_backend_option() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-unused-generate.gguf";
    const std::string program = "local-gguf-generate";
    const std::string path_string = path.string();
    const char* argv[] = {
        program.c_str(),
        path_string.c_str(),
        "",
        "1",
        "--backend",
        "gpu",
    };
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_generate(6, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "generate invalid-backend exit code mismatch");
    expect(output.str().empty(), "generate invalid-backend should not write stdout");
    expect(
        error.str().find("--backend must be cpu or metal") != std::string::npos,
        "generate invalid-backend error missing"
    );
}

// Native weights-only fixture: the target and draft share a tied F16 head.
// Positive/negative target projections give deterministic full-match/mismatch
// controls. These tiny weights are correctness fixtures, never speed evidence.
std::vector<std::uint8_t> chat_recurrent_fixture_weights(
    const std::filesystem::path& model_path, bool mismatch, std::uint32_t horizon = 3U,
    std::uint32_t top_k = 1U
) {
    namespace engine = local_gguf_engine::engine;
    auto file = read_file_backed_gguf_model(model_path);
    auto model = engine::load_verified_engine_model_from_gguf(file.view(), file.bytes());
    engine::ResidentTargetPayloadReceipt resident;
    auto target = engine::create_gemma2_metal_model_plan(model, engine::Gemma2OutputHeadPolicy::q8_0, resident);
    engine::SpeculativeArtifactDescriptor descriptor;
    descriptor.schema_version = 3;
    descriptor.kind = engine::SpeculativeArtifactKind::trained_candidate;
    descriptor.artifact_id = 1;
    descriptor.target.model = *model.payload_receipt();
    descriptor.target.resident = resident;
    descriptor.target.config = model.gemma2_config();
    descriptor.target.output_head_policy = static_cast<std::uint32_t>(target.output_head_policy());
    auto& spec = descriptor.recurrent;
    spec.version = 1; spec.width = 16; spec.horizon = horizon; spec.top_k = top_k;
    spec.temperature = 0.8F; spec.top_p = 0.95F; spec.min_p = 0.05F;
    spec.provenance.fill(local_gguf_engine::runtime::sha256(std::array<std::uint8_t,1>{1}));
    const std::array<std::array<unsigned,2>,10> shapes {{{16,256},{1,16},{16,256},
        {16,32},{1,16},{16,32},{1,16},{256,16},{1,16},{1,1}}};
    std::vector<std::uint8_t> payload;
    for (std::size_t t = 0; t < shapes.size(); ++t) {
        auto& weight = spec.tensors[t];
        weight.rows = shapes[t][0]; weight.columns = shapes[t][1]; weight.offset = payload.size();
        weight.bytes = std::uint64_t(weight.rows) * weight.columns * 4;
        for (unsigned i = 0; i < weight.rows * weight.columns; ++i) {
            const auto value = t == 1 || t == 6 ? 1.0F : t == 7 && i == 0 ? (mismatch ? -1.0F : 1.0F) : 0.0F;
            append_f32(payload, value);
        }
        weight.digest = local_gguf_engine::runtime::sha256(
            std::span<const std::uint8_t>(payload).subspan(weight.offset, weight.bytes));
    }
    return engine::encode_recurrent_weights_artifact(descriptor, payload);
}

std::vector<std::uint8_t> chat_recurrent_target_fixture_bytes(bool sampled = false,
    bool secondary_is_stop = false, bool dense = false, std::uint32_t context = 128) {
    auto tensors = tiny_gemma2_generate_cli_tensors_for_forced_token(5);
    tensors.erase(std::find_if(tensors.begin(), tensors.end(), [](const auto& t) {return t.name == "output.weight";}));
    std::vector<float> embedding(256U * 110U, 0.0F);
    for (unsigned token = 0; token < 110; ++token) embedding[token] = token == 5 ? 1.0F : 0.1F;
    if (sampled) { embedding[5] = 0.25F; embedding[6] = 0.20F; }
    if (dense) {
        for (unsigned feature = 1; feature < 256; ++feature) {
            for (unsigned token = 0; token < 110; ++token) {
                embedding[feature * 110 + token] = embedding[token] * 0.03F
                    * static_cast<float>(static_cast<int>((feature * 13) % 17) - 8);
            }
        }
    }
    replace_tiny_tensor(tensors, "token_embd.weight", tensor_f16_gguf_matrix(256, 110, embedding));
    // Nonzero, position-sensitive K/V makes rollback parity observable even
    // though the zero attention output keeps token 5 authoritative.
    replace_tiny_tensor(tensors, "blk.0.attn_norm.weight", tensor_f32_vector(std::vector<float>(256, 1.0F)));
    replace_tiny_tensor(tensors, "blk.0.attn_k.weight",
        local_gguf_engine::tests::tensor_q4_k_gguf_weight_matrix(256, 256, 3));
    replace_tiny_tensor(tensors, "blk.0.attn_v.weight",
        local_gguf_engine::tests::tensor_q6_k_gguf_weight_matrix(256, 256, 7));
    auto metadata = tiny_gemma2_generate_cli_metadata(context);
    if (secondary_is_stop) {
        for (auto& entry : metadata) {
            if (entry.key == "tokenizer.ggml.eos_token_id") entry = metadata_u32(entry.key, 6);
        }
    }
    return write_tiny_gguf_bytes(metadata, tensors);
}

void expect_chat_cached_bytes_equal(const Gemma2ChatSession& actual,
    const Gemma2ChatSession& expected, const std::string& boundary) {
    expect(actual.cached_token_count() == expected.cached_token_count(), boundary + " logical cache mismatch");
    const auto& a = actual.plan_for_testing().kv_cache();
    const auto& b = expected.plan_for_testing().kv_cache();
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (bool keys : {false, true}) {
            const auto& ab = keys ? a[i].key : a[i].value;
            const auto& bb = keys ? b[i].key : b[i].value;
            const auto av = local_gguf_engine::tensor::copy_backend_buffer_bytes(ab, 0, ab.descriptor().byte_count);
            const auto bv = local_gguf_engine::tensor::copy_backend_buffer_bytes(bb, 0, bb.descriptor().byte_count);
            const auto active = std::min<std::size_t>(actual.cached_token_count(), a[i].row_capacity)
                * (av.size() / a[i].row_capacity);
            const auto different = std::mismatch(av.begin(), av.begin() + static_cast<std::ptrdiff_t>(active), bv.begin());
            expect(different.first == av.begin() + static_cast<std::ptrdiff_t>(active),
                boundary + (keys ? " key-cache bytes differ at " : " value-cache bytes differ at ")
                + std::to_string(different.first - av.begin()));
        }
    }
}

void check_live_recurrent_chat_transactions(local_gguf_engine::engine::SpeculativeSamplingMode sampling_mode,
    local_gguf_engine::engine::RecurrentExecutionOptions execution = {}) {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace engine = local_gguf_engine::engine;
    const auto bytes = chat_recurrent_target_fixture_bytes();
    TempBinaryFile model("recurrent-chat-target", bytes);
    std::string state_boundary = "initial generation";
    const auto exact_state = [&](const Gemma2ChatSession& actual, const Gemma2ChatSession& expected) {
        if (sampling_mode == engine::SpeculativeSamplingMode::exact_match) {
            expect(actual.sampler_for_testing().checkpoint().same_state(expected.sampler_for_testing().checkpoint()),
                "learned chat changed exact RNG/history state");
        } else {
            // The point-mass target fixes the output sequence in this fixture,
            // but rejection sampling has its own legitimate random schedule.
            expect(actual.sampler_for_testing().history_snapshot()
                == expected.sampler_for_testing().history_snapshot(),
                "lossless chat retained rejected or unobserved tokens in repetition history");
        }
        expect_chat_cached_bytes_equal(actual, expected, state_boundary);
    };
    for (bool mismatch : {false, true}) {
        state_boundary = mismatch ? "initial mismatch" : "initial match";
        TempBinaryFile weights("recurrent-chat-weights", chat_recurrent_fixture_weights(model.path(), mismatch));
        Gemma2ChatSessionOptions options {.max_new_tokens = 9,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .context_shift = true, .retrieval_decode = false,
            .development_recurrent_weights = weights.path(),
            .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
        options.speculative_sampling = sampling_mode;
        options.recurrent_execution = execution;
        auto ordinary_options = options;
        ordinary_options.development_recurrent_weights.reset();
        ordinary_options.speculative_decode = false;
        ordinary_options.speculative_sampling = engine::SpeculativeSamplingMode::exact_match;
        ordinary_options.recurrent_execution = {};
        Gemma2ChatSession baseline(read_file_backed_gguf_model(model.path()), ordinary_options);
        Gemma2ChatSession learned(read_file_backed_gguf_model(model.path()), options);
        const auto reference = baseline.generate_user_turn("Hello");
        const auto actual = learned.generate_user_turn("Hello");
        expect(actual.selected_token_ids == reference.selected_token_ids,
            "live learned chat changed target-defined tokens");
        expect(actual.metrics.generation.learned_cycle_count > 0, "live learned route was not executed");
        expect((actual.metrics.generation.learned_accepted_edge_count > 0) != mismatch,
            "live fixture did not exercise intended match/mismatch path");
        expect(actual.cached_token_count == reference.cached_token_count, "live learned cache position mismatch");
        exact_state(learned, baseline);
        const auto before_regeneration = learned.sampler_for_testing().checkpoint();
        expect(learned.regenerate_last_turn().selected_token_ids == reference.selected_token_ids,
            "live learned regeneration changed target RNG/tokens");
        expect(learned.sampler_for_testing().checkpoint().same_state(before_regeneration),
            "learned regeneration did not restore its own complete random state");
        state_boundary = "regeneration";
        exact_state(learned, baseline);
        for (unsigned depth = 1; depth <= 4; ++depth) {
            state_boundary = "throwing callback " + std::to_string(depth);
            learned.clear(); baseline.clear();
            baseline.set_max_new_tokens(depth);
            const auto prefix = baseline.generate_user_turn("Hello");
            unsigned callbacks = 0;
            bool threw = false;
            try {
                static_cast<void>(learned.generate_user_turn("Hello", [&](std::string_view) {
                    if (++callbacks == depth) throw std::runtime_error("user callback fault");
                }));
            } catch (const std::runtime_error&) { threw = true; }
            expect(threw && callbacks == depth && !learned.poisoned(), "callback fault was lost or poisoned a known completion");
            expect(learned.message_count() == 2 && learned.cached_token_count() == prefix.cached_token_count,
                "throwing callback did not retain exactly its durable prefix");
            exact_state(learned, baseline);
            baseline.set_max_new_tokens(9);
            expect(learned.generate_user_turn("Hello").selected_token_ids
                == baseline.generate_user_turn("Hello").selected_token_ids,
                "continuation after callback fault changed target tokens");
            exact_state(learned, baseline);
        }
        // Repeated turns cross both the sliding ring and the context-shift
        // boundary. The same native route must also honor cancellation.
        learned.clear(); baseline.clear();
        for (unsigned turn = 0; turn < 8; ++turn) {
            state_boundary = "context turn " + std::to_string(turn);
            expect(learned.generate_user_turn("Hello").selected_token_ids
                == baseline.generate_user_turn("Hello").selected_token_ids,
                "learned context shifting changed target tokens");
            exact_state(learned, baseline);
            state_boundary = "wrapped regeneration " + std::to_string(turn);
            static_cast<void>(learned.regenerate_last_turn());
            exact_state(learned, baseline);
        }
        for (unsigned depth = 1; depth <= 4; ++depth) {
            state_boundary = "stopping callback " + std::to_string(depth);
            learned.clear(); baseline.clear();
            baseline.set_max_new_tokens(depth);
            static_cast<void>(baseline.generate_user_turn("Hello"));
            unsigned callbacks = 0;
            static_cast<void>(learned.generate_user_turn("Hello", Gemma2ChatTurnOptions {
                .on_text_delta = [&](std::string_view) { ++callbacks; },
                .should_stop = [&] { return callbacks >= depth; },
            }));
            expect(callbacks == depth, "learned callback cancellation exceeded its prefix");
            exact_state(learned, baseline);
        }
        baseline.set_max_new_tokens(9);
        learned.clear();
        auto bad_sampler = options.sampler; bad_sampler.temperature = 0.7F;
        bool refused = false;
        try { learned.set_sampler_options(bad_sampler); } catch (const std::invalid_argument&) { refused = true; }
        expect(refused && !learned.poisoned(), "incompatible sampler was silently accepted");
        learned.reset();
        expect(learned.generate_user_turn("Hello").selected_token_ids == reference.selected_token_ids,
            "authenticated reset changed recurrent route behavior");
        // Before any generation, regeneration validation failure must leave
        // the previous assistant turn and exact cache/sampler available.
        const auto prior_cache = learned.cached_token_count();
        learned.set_max_new_tokens(1000);
        refused = false;
        try { static_cast<void>(learned.regenerate_last_turn()); }
        catch (const Gemma2ChatContextError&) { refused = true; }
        expect(refused && learned.message_count() == 2 && learned.cached_token_count() == prior_cache,
            "failed regeneration destroyed the previous turn");
        learned.set_max_new_tokens(9);
        if (!mismatch) {
            for (auto point : {engine::TargetVerifierFailurePoint::encode,
                               engine::TargetVerifierFailurePoint::wait,
                               engine::TargetVerifierFailurePoint::finalize}) {
                learned.reset(); baseline.clear();
                static_cast<void>(learned.generate_user_turn("Hello"));
                static_cast<void>(baseline.generate_user_turn("Hello"));
                engine::set_target_verifier_failure_for_testing(point);
                bool failed = false;
                try { static_cast<void>(learned.generate_user_turn("Hello")); }
                catch (const std::exception&) { failed = true; }
                engine::clear_target_verifier_failure_for_testing();
                expect(failed, "live verifier failure was swallowed");
                if (point == engine::TargetVerifierFailurePoint::encode) {
                    state_boundary = "pre-submit recovery";
                    expect(!learned.poisoned(), "known pre-submit failure poisoned chat");
                    exact_state(learned, baseline);
                } else {
                    expect(learned.poisoned(), "uncertain verifier completion remained usable");
                    bool clear_refused = false;
                    try { learned.clear(); } catch (const std::exception&) { clear_refused = true; }
                    expect(clear_refused, "clear bypassed poisoned residency");
                }
            }
            learned.reset();
            expect(!learned.poisoned(), "authenticated reset did not replace poisoned residency");
            baseline.clear();
            static_cast<void>(baseline.generate_user_turn("Hello"));
            static_cast<void>(learned.generate_user_turn("Hello"));
            learned.set_max_new_tokens(113); // forces eviction of the prior turn
            engine::set_prepared_plan_failure_for_testing(engine::PreparedPlanFailurePoint::prefill_encode, 0);
            bool failed = false;
            try { static_cast<void>(learned.generate_user_turn("Hello")); }
            catch (const std::exception&) { failed = true; }
            engine::clear_prepared_plan_failure_for_testing();
            expect(failed && !learned.poisoned() && learned.message_count() == baseline.message_count(),
                "failed context-shift prefill lost the pre-turn state");
            state_boundary = "context-shift prefill failure";
            exact_state(learned, baseline);
        }
    }
}

void test_live_recurrent_chat_transactions() {
    check_live_recurrent_chat_transactions(local_gguf_engine::engine::SpeculativeSamplingMode::exact_match);
}

void test_lossless_recurrent_chat_transactions() {
    check_live_recurrent_chat_transactions(local_gguf_engine::engine::SpeculativeSamplingMode::lossless);
}

void test_conditional_lossless_recurrent_chat_transactions() {
    check_live_recurrent_chat_transactions(local_gguf_engine::engine::SpeculativeSamplingMode::lossless,
        {.conditional_verification = true});
}

void check_lossless_recurrent_sampled_path_and_regeneration(bool retrieval = false) {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace engine = local_gguf_engine::engine;
    TempBinaryFile model("lossless-chat-sampled-target", chat_recurrent_target_fixture_bytes(true));
    for (bool mismatch : {false, true}) {
        TempBinaryFile weights("lossless-chat-sampled-weights",
            chat_recurrent_fixture_weights(model.path(), mismatch, 3, 2));
        Gemma2ChatSessionOptions options {.max_new_tokens = retrieval ? 65U : 17U,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .context_shift = true, .retrieval_decode = retrieval,
            .development_recurrent_weights = weights.path(),
            .sampler = {.temperature = 0.8F, .top_k = 2, .seed = 42}};
        options.speculative_sampling = engine::SpeculativeSamplingMode::lossless;
        options.recurrent_execution.conditional_verification = retrieval;
        Gemma2ChatSession learned(read_file_backed_gguf_model(model.path()), options);
        const auto actual = learned.generate_user_turn("Hello");
        expect(actual.selected_token_ids.size() == options.max_new_tokens,
            "lossless sampled route failed to reach its output budget");
        expect(actual.metrics.generation.learned_cycle_count > 0
            && actual.metrics.generation.learned_numeric_refusal_count == 0,
            "lossless sampled fixture did not execute native proposal/verification");
        if (retrieval) expect(actual.metrics.generation.speculative_cycle_count > 0,
            "mixed sampled fixture did not transition through retrieval and learned execution");
        expect((actual.metrics.generation.learned_accepted_edge_count > 0) != mismatch,
            "lossless sampled fixture missed its accept/reject control");
        expect(std::find(actual.selected_token_ids.begin(), actual.selected_token_ids.end(), 5U)
                != actual.selected_token_ids.end()
            && std::find(actual.selected_token_ids.begin(), actual.selected_token_ids.end(), 6U)
                != actual.selected_token_ids.end(),
            "lossless sampled fixture became a deterministic top-one proxy");
        expect(std::all_of(actual.selected_token_ids.begin(), actual.selected_token_ids.end(),
                [](std::uint32_t token) { return token == 5 || token == 6; }),
            "lossless residual selected a token outside the target support");

        // Replay the sequence actually emitted, not the unrelated baseline
        // seed sequence. Every K/V row must describe this accepted history.
        auto file = read_file_backed_gguf_model(model.path());
        auto target = engine::load_verified_engine_model_from_gguf(file.view(), file.bytes());
        auto replay = engine::create_gemma2_metal_model_plan(target, engine::Gemma2OutputHeadPolicy::q8_0);
        const auto scale = engine::gemma2_input_embedding_scale(target.gemma2_config().embedding_length);
        static_cast<void>(engine::run_gemma2_metal_resident_prefill(
            replay, actual.rendered_prompt_token_ids, 0, scale));
        auto position = static_cast<std::uint32_t>(actual.rendered_prompt_token_ids.size());
        for (auto token : actual.selected_token_ids) {
            static_cast<void>(engine::run_gemma2_metal_resident_decode_step(replay,
                engine::create_gemma2_metal_decode_schedule(replay, position++), token, scale,
                std::nullopt, engine::Gemma2MetalDecodeOutputMode::kv_only));
        }
        expect(position == actual.cached_token_count, "lossless selected-path replay length mismatch");
        const auto compare_replay_cache = [&] {
            const auto& cached = learned.plan_for_testing().kv_cache();
            for (std::size_t layer = 0; layer < cached.size(); ++layer) {
                for (bool key : {false, true}) {
                    const auto& a = key ? cached[layer].key : cached[layer].value;
                    const auto& b = key ? replay.kv_cache()[layer].key : replay.kv_cache()[layer].value;
                    const auto av = local_gguf_engine::tensor::copy_backend_buffer_bytes(a, 0, a.descriptor().byte_count);
                    const auto bv = local_gguf_engine::tensor::copy_backend_buffer_bytes(b, 0, b.descriptor().byte_count);
                    const auto active = std::min<std::size_t>(position, cached[layer].row_capacity)
                        * (av.size() / cached[layer].row_capacity);
                    expect(std::equal(av.begin(), av.begin() + static_cast<std::ptrdiff_t>(active), bv.begin()),
                        "lossless K/V differs from ordinary target execution along its emitted tokens");
                }
            }
        };
        compare_replay_cache();
        const auto checkpoint = learned.sampler_for_testing().checkpoint();
        const auto regenerated = learned.regenerate_last_turn();
        expect(regenerated.selected_token_ids == actual.selected_token_ids
                && checkpoint.same_state(learned.sampler_for_testing().checkpoint()),
            "lossless regeneration changed its own proposal/acceptance random stream");
        compare_replay_cache();
        learned.clear();
        expect(learned.generate_user_turn("Hello").selected_token_ids == actual.selected_token_ids,
            "lossless clear did not reset independent proposal state");
        learned.reset();
        expect(learned.generate_user_turn("Hello").selected_token_ids == actual.selected_token_ids,
            "lossless authenticated reset did not restore its sampling mode");
    }
}

void test_lossless_recurrent_sampled_path_and_regeneration() {
    check_lossless_recurrent_sampled_path_and_regeneration();
}

void test_lossless_recurrent_mixed_retrieval_transactions() {
    check_lossless_recurrent_sampled_path_and_regeneration(true);
}

void test_q8_conditional_recurrent_cache_recovery() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace engine = local_gguf_engine::engine;
    TempBinaryFile model("q8-conditional-chat-target", chat_recurrent_target_fixture_bytes(false, false, true));
    for (const bool mismatch : {false, true}) {
        TempBinaryFile weights("q8-conditional-chat-weights", chat_recurrent_fixture_weights(model.path(), mismatch));
        Gemma2ChatSessionOptions options {.max_new_tokens = 9,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .context_shift = true, .retrieval_decode = false,
            .development_recurrent_weights = weights.path(),
            .speculative_sampling = engine::SpeculativeSamplingMode::lossless,
            .recurrent_execution = {.draft_head = engine::RecurrentDraftHead::q8_0,
                .conditional_verification = true},
            .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
        Gemma2ChatSession learned(read_file_backed_gguf_model(model.path()), options);
        auto full_options = options;
        full_options.recurrent_execution.conditional_verification = false;
        Gemma2ChatSession control(read_file_backed_gguf_model(model.path()), full_options);
        const auto compare = [&] {
            expect_chat_cached_bytes_equal(learned, control, "conditional/full Q8 parity");
            expect(learned.sampler_for_testing().checkpoint().same_state(control.sampler_for_testing().checkpoint()),
                "conditional Q8 recovery changed sampler state");
        };
        bool shifted = false;
        for (unsigned turn = 0; turn < 8; ++turn) {
            const auto expected = control.generate_user_turn("Hello");
            const auto actual = learned.generate_user_turn("Hello");
            expect(actual.selected_token_ids == expected.selected_token_ids
                    && actual.metrics.generation.learned_cycle_count > 0,
                "conditional Q8 route did not execute or changed its seeded sequence");
            shifted = shifted || actual.metrics.trimmed_turn_count != 0;
            compare();
            expect(learned.regenerate_last_turn().selected_token_ids == actual.selected_token_ids,
                "conditional Q8 regeneration changed the selected sequence");
            compare();
        }
        expect(shifted, "conditional Q8 recovery fixture did not cross its context-shift boundary");
        for (unsigned depth = 1; depth <= 4; ++depth) {
            learned.clear(); control.clear();
            unsigned reference_callbacks = 0;
            static_cast<void>(control.generate_user_turn("Hello", Gemma2ChatTurnOptions {
                .on_text_delta = [&](std::string_view) { ++reference_callbacks; },
                .should_stop = [&] { return reference_callbacks == depth; },
            }));
            unsigned callbacks = 0;
            bool threw = false;
            try {
                static_cast<void>(learned.generate_user_turn("Hello", [&](std::string_view) {
                    if (++callbacks == depth) throw std::runtime_error("conditional Q8 callback fault");
                }));
            } catch (const std::runtime_error&) { threw = true; }
            expect(threw && callbacks == depth && !learned.poisoned(),
                "conditional Q8 throwing callback lost its durable prefix or poisoned a known completion");
            compare();
            expect(learned.generate_user_turn("Hello").selected_token_ids
                    == control.generate_user_turn("Hello").selected_token_ids,
                "conditional Q8 callback recovery changed continuation tokens");
            compare();
        }
        learned.reset(); control.reset();
        expect(learned.generate_user_turn("Hello").selected_token_ids
                == control.generate_user_turn("Hello").selected_token_ids,
            "conditional Q8 authenticated reset changed its execution contract");
        compare();
    }
}

void test_conditional_recurrent_full_horizon_feature_leases() {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace engine = local_gguf_engine::engine;
    TempBinaryFile model("conditional-horizon-chat-target", chat_recurrent_target_fixture_bytes());
    for (const auto horizon : {3U, 4U, 7U, 8U, 15U, 16U, 23U}) {
        TempBinaryFile weights("conditional-horizon-chat-weights",
            chat_recurrent_fixture_weights(model.path(), false, horizon));
        Gemma2ChatSessionOptions options {.max_new_tokens = horizon + 3,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .retrieval_decode = false, .development_recurrent_weights = weights.path(),
            .speculative_sampling = engine::SpeculativeSamplingMode::lossless,
            .recurrent_execution = {.conditional_verification = true},
            .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
        Gemma2ChatSession session(read_file_backed_gguf_model(model.path()), options);
        const auto actual = session.generate_user_turn("Hello");
        expect(actual.selected_token_ids.size() == horizon + 3
                && std::all_of(actual.selected_token_ids.begin(), actual.selected_token_ids.end(),
                    [](auto token) { return token == 5U; })
                && actual.metrics.generation.learned_accepted_edge_count >= horizon
                && actual.metrics.generation.learned_cycle_count == 2,
            "conditional schedule truncated its horizon or lost the final tile feature lease");
        const auto checkpoint = session.sampler_for_testing().checkpoint();
        expect(session.regenerate_last_turn().selected_token_ids == actual.selected_token_ids
                && checkpoint.same_state(session.sampler_for_testing().checkpoint()),
            "conditional full-horizon regeneration changed causal RNG state");
    }
}

void test_recurrent_execution_configuration_is_explicit() {
    namespace engine = local_gguf_engine::engine;
    namespace metal = local_gguf_engine::tensor::backends::metal;
    TempBinaryFile model("recurrent-execution-options-target", chat_recurrent_target_fixture_bytes());
    for (const auto& args : std::vector<std::vector<std::string>> {
        {"--speculative-verifier", "batched"},
        {"--speculative-schedule"}, {"--speculative-schedule", "short"},
        {"--speculative-schedule", "conditional"},
        {"--speculative-schedule", "full", "--speculative-schedule", "conditional"},
        {"--draft-head"}, {"--draft-head", "q4"}, {"--draft-head", "q8_0"},
        {"--draft-head", "f16", "--draft-head", "q8_0"},
        {"--development-recurrent-weights", "/unused", "--speculative-schedule", "conditional"},
        {"--development-recurrent-weights", "/unused", "--speculative-sampling", "lossless",
            "--speculative-schedule", "conditional", "--no-speculative"},
        {"--development-recurrent-weights", "/unused", "--speculative-sampling", "lossless",
            "--draft-head", "q8_0", "--output-head", "f16"},
    }) {
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0
                && !error.str().empty(), "invalid execution option silently fell back to another route");
    }
    for (const auto bad : {
        engine::RecurrentExecutionOptions {.draft_head = static_cast<engine::RecurrentDraftHead>(255)},
    }) {
        bool refused = false;
        try { engine::validate_recurrent_execution_options(engine::SpeculativeSamplingMode::lossless, bad); }
        catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "invalid typed execution enum was accepted");
    }
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    TempBinaryFile weights("recurrent-execution-options-weights", chat_recurrent_fixture_weights(model.path(), false));
    const std::vector<std::string> base {"--speculative-sampling", "lossless",
        "--development-recurrent-weights", weights.path().string(), "--temperature", "0.8", "--top-k", "1",
        "--output-head", "q8_0"};
    for (const auto& extra : std::vector<std::vector<std::string>> {
        {"--speculative-schedule", "full", "--draft-head", "f16"},
        {"--speculative-schedule", "conditional", "--draft-head", "q8_0"},
    }) {
        auto args = base; args.insert(args.end(), extra.begin(), extra.end());
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) == 0,
            "valid explicit execution options failed: " + error.str());
    }
    Gemma2ChatSessionOptions options {.max_new_tokens = 3,
        .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
        .development_recurrent_weights = weights.path(),
        .speculative_sampling = engine::SpeculativeSamplingMode::lossless,
        .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
    for (const auto alternative : {
        engine::RecurrentExecutionOptions {.draft_head = engine::RecurrentDraftHead::q8_0},
        engine::RecurrentExecutionOptions {.conditional_verification = true},
    }) {
        for (bool reversed : {false, true}) {
            options.recurrent_execution = reversed ? alternative : engine::RecurrentExecutionOptions {};
            auto prepared = engine::Gemma2ChatSessionState::prepare(read_file_backed_gguf_model(model.path()), options);
            options.recurrent_execution = reversed ? engine::RecurrentExecutionOptions {} : alternative;
            bool refused = false;
            try { Gemma2ChatSession incompatible(std::move(prepared), options); }
            catch (const std::invalid_argument&) { refused = true; }
            expect(refused, "prepared session silently replaced its head/tile contract");
        }
    }
    options.recurrent_execution = {.draft_head = engine::RecurrentDraftHead::q8_0,
        .conditional_verification = true};
    metal::set_metal_device_unavailable_for_testing(true);
    bool refused = false;
    try {
        try { Gemma2ChatSession unavailable(read_file_backed_gguf_model(model.path()), options); }
        catch (const std::exception&) { refused = true; }
    } catch (...) { metal::set_metal_device_unavailable_for_testing(false); throw; }
    metal::set_metal_device_unavailable_for_testing(false);
    expect(refused, "conditional Q8 execution silently fell back without Metal");
}

void test_lossless_recurrent_configuration_and_unavailable_device() {
    namespace engine = local_gguf_engine::engine;
    namespace metal = local_gguf_engine::tensor::backends::metal;
    TempBinaryFile model("lossless-chat-cli-invalid", chat_recurrent_target_fixture_bytes());
    for (const auto& args : std::vector<std::vector<std::string>> {
        {"--speculative-sampling"},
        {"--speculative-sampling", "approximate"},
        {"--speculative-sampling", "lossless"},
        {"--speculative-sampling", "lossless", "--no-speculative"},
        {"--speculative-sampling", "lossless", "--development-recurrent-weights", "/missing-lossless-weights"},
        {"--speculative-sampling", "lossless", "--speculative-sampling", "exact-match"},
    }) {
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0
                && !error.str().empty(),
            "invalid lossless request silently selected another decoding mode");
    }
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    TempBinaryFile weights("lossless-chat-cli-weights", chat_recurrent_fixture_weights(model.path(), false));
    const std::vector<std::string> args {"--speculative-sampling", "lossless",
        "--development-recurrent-weights", weights.path().string(), "--temperature", "0.8", "--top-k", "1",
        "--output-head", "q8_0"};
    std::ostringstream output, error;
    expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) == 0,
        "valid explicit lossless CLI configuration failed: " + error.str());
    for (auto prepared_mode : {engine::SpeculativeSamplingMode::exact_match, engine::SpeculativeSamplingMode::lossless}) {
        Gemma2ChatSessionOptions options {.max_new_tokens = 3,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .development_recurrent_weights = weights.path(),
            .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
        options.speculative_sampling = prepared_mode;
        auto prepared = engine::Gemma2ChatSessionState::prepare(read_file_backed_gguf_model(model.path()), options);
        options.speculative_sampling = prepared_mode == engine::SpeculativeSamplingMode::exact_match
            ? engine::SpeculativeSamplingMode::lossless : engine::SpeculativeSamplingMode::exact_match;
        bool refused = false;
        try { Gemma2ChatSession incompatible(std::move(prepared), options); }
        catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "prepared session silently switched its speculative sampling contract");
    }
    metal::set_metal_device_unavailable_for_testing(true);
    bool refused = false;
    try {
        Gemma2ChatSessionOptions options {.max_new_tokens = 3,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .development_recurrent_weights = weights.path(),
            .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
        options.speculative_sampling = engine::SpeculativeSamplingMode::lossless;
        try { Gemma2ChatSession unavailable(read_file_backed_gguf_model(model.path()), options); }
        catch (const std::exception&) { refused = true; }
    } catch (...) { metal::set_metal_device_unavailable_for_testing(false); throw; }
    metal::set_metal_device_unavailable_for_testing(false);
    expect(refused, "lossless session silently fell back when Metal was unavailable");
}

void check_lossless_recurrent_stop_tokens_and_reverse_stops(
    local_gguf_engine::engine::RecurrentExecutionOptions execution = {}) {
    if (!tensor_backend_is_available(BackendKind::metal)) return;
    namespace engine = local_gguf_engine::engine;
    TempBinaryFile model("lossless-chat-stop-target", chat_recurrent_target_fixture_bytes(true, true));
    TempBinaryFile weights("lossless-chat-stop-weights", chat_recurrent_fixture_weights(model.path(), false, 3, 2));
    unsigned learned_stops = 0;
    for (std::uint64_t seed = 0; seed < 8; ++seed) {
        Gemma2ChatSessionOptions options {.max_new_tokens = 17,
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .retrieval_decode = false, .development_recurrent_weights = weights.path(),
            .sampler = {.temperature = 0.8F, .top_k = 2, .seed = seed}};
        options.speculative_sampling = engine::SpeculativeSamplingMode::lossless;
        options.recurrent_execution = execution;
        Gemma2ChatSession session(read_file_backed_gguf_model(model.path()), options);
        unsigned callbacks = 0;
        const auto actual = session.generate_user_turn("Hello", [&](std::string_view) { ++callbacks; });
        if (actual.stop_token_id) {
            expect(actual.stop_token_id == 6U && actual.selected_token_ids.back() == 6U
                    && actual.metrics.generation.stop_reason == Gemma2GenerationStopReason::stop_token
                    && callbacks + 1 == actual.selected_token_ids.size(),
                "lossless native EOS escaped its stop or callback boundary");
            learned_stops += actual.metrics.generation.learned_cycle_count > 0 ? 1U : 0U;
        } else {
            expect(actual.selected_token_ids.size() == options.max_new_tokens
                    && std::find(actual.selected_token_ids.begin(), actual.selected_token_ids.end(), 6U)
                        == actual.selected_token_ids.end(),
                "lossless sampled route ignored an EOS or stopped before its token budget");
        }
        expect(actual.generated_token_ids.size() == callbacks
                && actual.cached_token_count == actual.rendered_prompt_token_ids.size() + actual.selected_token_ids.size(),
            "lossless EOS escaped callbacks or failed to retain its K/V row");
        const auto checkpoint = session.sampler_for_testing().checkpoint();
        expect(session.regenerate_last_turn().selected_token_ids == actual.selected_token_ids
                && checkpoint.same_state(session.sampler_for_testing().checkpoint()),
            "lossless EOS regeneration changed a completed random transaction");
    }
    expect(learned_stops > 0, "EOS fixture never reached a lossless speculative cycle");

    TempBinaryFile fixed_model("lossless-chat-reverse-target", chat_recurrent_target_fixture_bytes());
    for (bool mismatch : {false, true}) {
        TempBinaryFile fixed_weights("lossless-chat-reverse-weights",
            chat_recurrent_fixture_weights(fixed_model.path(), mismatch));
        for (unsigned depth = 2; depth <= 4; ++depth) {
            std::string stop;
            for (unsigned token = 0; token < depth; ++token) stop += "Hello";
            Gemma2ChatSessionOptions options {.max_new_tokens = 9,
                .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
                .retrieval_decode = false, .development_recurrent_weights = fixed_weights.path(),
                .reverse_stop_strings = {stop},
                .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 19}};
            options.speculative_sampling = engine::SpeculativeSamplingMode::lossless;
            options.recurrent_execution = execution;
            Gemma2ChatSession session(read_file_backed_gguf_model(fixed_model.path()), options);
            const auto actual = session.generate_user_turn("Hello");
            expect(actual.reverse_stop_string == stop && actual.text.empty()
                    && actual.selected_token_ids.size() == depth
                    && actual.cached_token_count == actual.rendered_prompt_token_ids.size() + depth
                    && actual.metrics.generation.stop_reason == Gemma2GenerationStopReason::reverse_stop,
                "lossless reverse-stop discarded hidden selected-token cache or overran its prefix");
        }
    }
}

void test_lossless_recurrent_stop_tokens_and_reverse_stops() {
    check_lossless_recurrent_stop_tokens_and_reverse_stops();
}

void test_conditional_lossless_recurrent_stop_tokens_and_reverse_stops() {
    check_lossless_recurrent_stop_tokens_and_reverse_stops({.conditional_verification = true});
}

void test_recurrent_cli_rejects_missing_and_conflicting_configuration() {
    const auto bytes = tiny_gemma2_generate_cli_gguf_bytes();
    TempBinaryFile model("recurrent-chat-cli-invalid", bytes);
    for (const auto& args : std::vector<std::vector<std::string>> {
        {"--development-recurrent-weights"},
        {"--development-recurrent-weights", "/missing-recurrent-weights"},
        {"--development-recurrent-weights", "/missing-recurrent-weights", "--no-speculative"},
        {"--development-recurrent-weights", "one", "--development-recurrent-weights", "two"},
    }) {
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0,
            "malformed learned CLI request silently fell back to Stage 4");
        expect(!error.str().empty(), "malformed learned CLI omitted explanation");
    }
}

void test_standalone_tli_cli_rejects_missing_and_conflicting_configuration() {
    TempBinaryFile model("standalone-tli-cli-invalid", chat_recurrent_target_fixture_bytes());
    for (const auto& args : std::vector<std::vector<std::string>> {
        {"--draft-model"},
        {"--draft-model", ""},
        {"--draft-model", "--speculative-sampling", "lossless"},
        {"--draft-model", "one", "--draft-model", "two", "--speculative-sampling", "lossless"},
        {"--draft-model", "/missing-vicuna-draft.gguf"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "exact-match"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless", "--no-speculative"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless", "--backend", "cpu"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless",
            "--development-recurrent-weights", "/unused"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless", "--draft-head", "f16"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless", "--draft-head", "q8_0"},
        {"--draft-horizon"}, {"--draft-horizon", "3"}, {"--draft-horizon", "0"},
        {"--draft-horizon", "-1"}, {"--draft-horizon", "24"}, {"--draft-horizon", "3.0"},
        {"--draft-horizon", "18446744073709551616"},
        {"--draft-model", "/missing-vicuna-draft.gguf", "--speculative-sampling", "lossless",
            "--draft-horizon", "1", "--draft-horizon", "23"},
    }) {
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0
                && !error.str().empty(), "invalid standalone TLI configuration silently selected another route");
        expect(output.str().find("local-gguf-engine chat") == std::string::npos,
            "invalid standalone TLI configuration started chat using a fallback route");
        expect(error.str().find("unknown chat option") == std::string::npos,
            "standalone TLI flags were not recognized by the real CLI parser");
    }
}

void test_standalone_draft_sampling_cli_rejects_invalid_and_conflicting_flags() {
    TempBinaryFile model("standalone-draft-sampling-cli-invalid", chat_recurrent_target_fixture_bytes());
    const std::vector<std::string> base {"--draft-model", "/missing-native-draft.gguf",
        "--speculative-sampling", "lossless"};
    for (const auto& extra : std::vector<std::vector<std::string>> {
        {"--draft-temperature"}, {"--draft-temperature", "nan"}, {"--draft-temperature", "inf"},
        {"--draft-temperature", "-0.1"}, {"--draft-temperature", "0", "--draft-temperature", "1"},
        {"--draft-top-k"}, {"--draft-top-k", "-1"}, {"--draft-top-k", "1.5"},
        {"--draft-top-k", "4294967296"}, {"--draft-top-k", "18446744073709551616"},
        {"--draft-top-k", "0", "--draft-top-k", "40"},
        {"--draft-top-p", "0"}, {"--draft-top-p", "1.01"}, {"--draft-top-p", "nan"},
        {"--draft-top-p", "0.5", "--draft-top-p", "1"},
        {"--draft-min-p", "-0.1"}, {"--draft-min-p", "1.01"}, {"--draft-min-p", "inf"},
        {"--draft-min-p", "0", "--draft-min-p", "0.1"},
        {"--draft-repeat-penalty", "0"}, {"--draft-repeat-penalty", "-1"},
        {"--draft-repeat-penalty", "nan"}, {"--draft-repeat-penalty", "1", "--draft-repeat-penalty", "2"},
    }) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        std::ostringstream output, error;
        expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0,
            "invalid draft sampler flag was accepted");
        expect(error.str().find("error: --draft-") != std::string::npos
            && error.str().find("unknown chat option") == std::string::npos,
            "invalid draft sampler must fail in explicit CLI validation before artifact loading");
    }
    for (const auto* flag : {"--draft-temperature", "--draft-top-k", "--draft-top-p", "--draft-min-p", "--draft-repeat-penalty"}) {
        for (const auto& context : std::vector<std::vector<std::string>> {
            {}, {"--no-speculative"},
            {"--development-recurrent-weights", "/unused", "--speculative-sampling", "lossless"},
            {"--draft-model", "/unused", "--speculative-sampling", "exact-match"},
            {"--draft-model", "/unused", "--speculative-sampling", "lossless", "--no-speculative"},
            {"--draft-model", "/unused", "--speculative-sampling", "lossless", "--backend", "cpu"},
        }) {
            auto args = context;
            args.insert(args.end(), {flag, "1"});
            std::ostringstream output, error;
            expect(run_generate_chat_with_extra_arguments(model.path(), args, output, error) != 0
                && output.str().find("local-gguf-engine chat") == std::string::npos,
                "draft scalar overrides leaked into target-only/recurrent/non-lossless mode");
            expect(error.str().find("requires") != std::string::npos || error.str().find("require") != std::string::npos,
                "conflicting draft sampler configuration omitted its dependency error");
        }
    }
}

void test_standalone_draft_sampling_cli_propagates_effective_values_when_available() {
    const std::filesystem::path draft("models/gemma-3-270m-it-Q8_0.gguf");
    if (!tensor_backend_is_available(BackendKind::metal) || !std::filesystem::exists(draft)) return;
    TempBinaryFile model("standalone-draft-sampling-cli-effective", chat_recurrent_target_fixture_bytes());
    std::string inherited_hash;
    for (const bool override : {false, true}) {
        std::vector<std::string> arguments {"local-gguf-generate", model.path().string(), "--chat", "--max-new-tokens", "1",
            "--draft-model", draft.string(), "--draft-horizon", "1", "--speculative-sampling", "lossless",
            "--output-head", "q8_0",
            "--temperature", "0.5", "--top-k", "2", "--top-p", "0.75", "--min-p", "0.125",
            "--repeat-penalty", "1.5", "--seed", "42", "--decode-diagnostics"};
        if (override) arguments.insert(arguments.end(), {"--draft-temperature", "0.25", "--draft-top-k", "0",
            "--draft-top-p", "1", "--draft-min-p", "0", "--draft-repeat-penalty", "1"});
        std::vector<const char*> argv;
        for (const auto& argument : arguments) argv.push_back(argument.c_str());
        std::istringstream input("Hello\n/exit\n");
        std::ostringstream output, error;
        expect(run_generate(static_cast<int>(argv.size()), argv.data(), input, output, error) == 0,
            "valid explicit standalone draft sampler CLI failed: " + error.str());
        const auto expected = override
            ? "\"standalone_effective_sampler\":{\"temperature\":0.25,\"top_k\":0,\"top_p\":1,\"min_p\":0,\"repetition_penalty\":1}"
            : "\"standalone_effective_sampler\":{\"temperature\":0.5,\"top_k\":2,\"top_p\":0.75,\"min_p\":0.125,\"repetition_penalty\":1.5}";
        expect(output.str().find(expected) != std::string::npos, "CLI overrides/inheritance did not reach actual prepared session diagnostics");
        const std::string_view marker = "\"selected_token_hash\":\"";
        const auto begin = output.str().find(marker);
        expect(begin != std::string::npos, "CLI sampler test missing target token provenance");
        const auto hash = output.str().substr(begin + marker.size(), 16);
        if (override) expect(hash == inherited_hash, "draft-only flags changed the authoritative ordinary target selection");
        else inherited_hash = hash;
    }
}

void check_standalone_real_artifact_failed_regeneration(const std::filesystem::path& draft_path) {
    namespace engine = local_gguf_engine::engine;
    if (!tensor_backend_is_available(BackendKind::metal) || !std::filesystem::exists(draft_path)) return;
    TempBinaryFile model("standalone-real-regeneration", chat_recurrent_target_fixture_bytes(false, false, false, 64));
    Gemma2ChatSessionOptions options {.max_new_tokens = 8,
        .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
        .context_shift = true, .retrieval_decode = false, .standalone_draft_model = draft_path,
        .standalone_execution = {.horizon = 3, .conditional_verification = true},
        .speculative_sampling = engine::SpeculativeSamplingMode::lossless,
        .sampler = {.temperature = 0.8F, .top_k = 1, .seed = 92, .repetition_window = 2}};
    Gemma2ChatSession session(read_file_backed_gguf_model(model.path()), options);
    const auto original = session.generate_user_turn("Hello");
    expect(original.metrics.generation.learned_cycle_count > 0,
        "real standalone regeneration test did not execute its native draft");
    expect(original.metrics.generation.standalone_draft_gpu_ns >= original.metrics.generation.learned_draft_gpu_ns
            && original.metrics.generation.standalone_draft_command_buffer_count > 0
            && original.metrics.generation.standalone_draft_compute_dispatch_count > 0
            && original.metrics.decode_execution.gpu_duration_ns <= original.metrics.generation.decode_duration_ns,
        "standalone complete-path metrics omit work or exceed their enclosing decode clock");
    const auto checkpoint = session.sampler_for_testing().checkpoint();
    const auto cached = session.cached_token_count();
    engine::set_target_verifier_failure_for_testing(engine::TargetVerifierFailurePoint::encode);
    bool failed = false;
    try { static_cast<void>(session.regenerate_last_turn()); }
    catch (const std::exception&) { failed = true; }
    engine::clear_target_verifier_failure_for_testing();
    expect(failed && !session.poisoned() && session.cached_token_count() == cached
            && session.sampler_for_testing().checkpoint().same_state(checkpoint),
        "unpublished failed standalone regeneration lost its durable target/sampler state");
    const auto regenerated = session.regenerate_last_turn();
    expect(regenerated.selected_token_ids == original.selected_token_ids
            && regenerated.metrics.generation.learned_accepted_edge_count
                == original.metrics.generation.learned_accepted_edge_count
            && session.sampler_for_testing().checkpoint().same_state(checkpoint),
        "failed standalone regeneration changed its original draft context or future probability draws");
    bool shifted = false;
    for (unsigned turn = 0; turn < 8; ++turn) {
        const auto continued = session.generate_user_turn("Hello");
        if (!continued.rebuilt_context) continue;
        shifted = true;
        expect(continued.cache_position == 0 && continued.trimmed_turn_count > 0
                && continued.cached_token_count <= session.context_length()
                && continued.metrics.generation.standalone_prefill_ns > 0
                && continued.metrics.generation.learned_cycle_count > 0
                && continued.selected_token_ids == original.selected_token_ids,
            "standalone context shift failed to rebuild and execute its native draft at target position zero");
        const auto shifted_state = session.sampler_for_testing().checkpoint();
        const auto replayed = session.regenerate_last_turn();
        expect(replayed.selected_token_ids == continued.selected_token_ids
                && replayed.metrics.generation.learned_accepted_edge_count
                    == continued.metrics.generation.learned_accepted_edge_count
                && session.sampler_for_testing().checkpoint().same_state(shifted_state),
            "regeneration after standalone context shift did not reproduce its own draft law and random state");
        break;
    }
    expect(shifted, "standalone real-artifact session did not exercise context shifting");
}

void test_standalone_real_artifact_failed_regeneration_when_available() {
    check_standalone_real_artifact_failed_regeneration("models/vicuna_68m/vicuna-68m-f16.gguf");
}
void test_gemma3_real_artifact_failed_regeneration_when_available() {
    check_standalone_real_artifact_failed_regeneration("models/gemma-3-270m-it-Q8_0.gguf");
}

const local_gguf_engine::tests::TestCase kGenerateCliTests[] = {
    {"real Gemma3 failed regeneration and context shifts preserve own sampling state when available", test_gemma3_real_artifact_failed_regeneration_when_available},
    {"real standalone failed regeneration restores original append policy when available", test_standalone_real_artifact_failed_regeneration_when_available},
    {"standalone TLI CLI rejects missing malformed and conflicting configuration", test_standalone_tli_cli_rejects_missing_and_conflicting_configuration},
    {"standalone draft sampling CLI rejects invalid and conflicting scalars", test_standalone_draft_sampling_cli_rejects_invalid_and_conflicting_flags},
    {"standalone draft sampling CLI preserves target selection and reports effective overrides when available", test_standalone_draft_sampling_cli_propagates_effective_values_when_available},
    {"recurrent execution CLI and prepared contracts are explicit", test_recurrent_execution_configuration_is_explicit},
    {"conditional lossless chat preserves full live transaction boundaries", test_conditional_lossless_recurrent_chat_transactions},
    {"conditional Q8 chat preserves full-chain tokens cache and recovery", test_q8_conditional_recurrent_cache_recovery},
    {"conditional chat preserves every full horizon and final feature lease", test_conditional_recurrent_full_horizon_feature_leases},
    {"mixed retrieval and learned lossless chat preserves emitted-path K/V", test_lossless_recurrent_mixed_retrieval_transactions},
    {"conditional lossless chat honors EOS and reverse stops", test_conditional_lossless_recurrent_stop_tokens_and_reverse_stops},
    {"live learned chat preserves target tokens and callback transaction boundaries", test_live_recurrent_chat_transactions},
    {"lossless learned chat preserves committed cache and callback transaction boundaries", test_lossless_recurrent_chat_transactions},
    {"lossless learned sampled chat replays target K/V and its own random state", test_lossless_recurrent_sampled_path_and_regeneration},
    {"lossless learned CLI and unavailable-device requests fail closed", test_lossless_recurrent_configuration_and_unavailable_device},
    {"lossless learned chat honors EOS and reverse-stop transaction boundaries", test_lossless_recurrent_stop_tokens_and_reverse_stops},
    {"learned chat CLI fails closed on missing and conflicting configuration", test_recurrent_cli_rejects_missing_and_conflicting_configuration},
    {
        "formats chat timing line from runtime metrics",
        test_formats_chat_timing_line_from_runtime_metrics,
    },
    {
        "formats and validates chat decode attribution",
        test_formats_and_validates_chat_decode_attribution,
    },
    {"Metal trace capture arm validation", test_metal_trace_capture_arm_validation},
    {
        "runs generate Gemma2 default greedy when available",
        test_runs_generate_gemma2_default_greedy_when_available,
    },
    {
        "runs generate Gemma2 greedy when available",
        test_runs_generate_gemma2_greedy_when_available,
    },
    {
        "Gemma2 chat session appends second turn at nonzero cache when available",
        test_gemma2_chat_session_appends_second_turn_at_nonzero_cache_when_available,
    },
    {
        "Gemma2 chat session appends after truncated output retokenizes differently when available",
        test_gemma2_chat_session_appends_after_truncated_output_retokenizes_differently_when_available,
    },
    {
        "Gemma2 chat session tracks nonvisible generated tokens in cache when available",
        test_gemma2_chat_session_tracks_nonvisible_generated_tokens_in_cache_when_available,
    },
    {
        "Gemma2 chat session records deterministic turn metrics when available",
        test_gemma2_chat_session_records_deterministic_turn_metrics_when_available,
    },
    {
        "Gemma2 chat session streams visible text before return when available",
        test_gemma2_chat_session_streams_visible_text_before_return_when_available,
    },
    {
        "Gemma2 chat session can cancel generation when available",
        test_gemma2_chat_session_can_cancel_generation_when_available,
    },
    {
        "Gemma2 chat session stops on EOS when available",
        test_gemma2_chat_session_stops_on_eos_when_available,
    },
    {
        "Gemma2 chat session stops on end-of-turn when available",
        test_gemma2_chat_session_stops_on_end_of_turn_when_available,
    },
    {
        "Gemma2 chat session applies reverse stop and cleans history when available",
        test_gemma2_chat_session_applies_reverse_stop_and_cleans_history_when_available,
    },
    {
        "Gemma2 chat session clear and reset restore empty context when available",
        test_gemma2_chat_session_clear_and_reset_restore_empty_context_when_available,
    },
    {
        "Gemma2 chat session regenerates last turn without full replay when available",
        test_gemma2_chat_session_regenerates_last_turn_without_full_replay_when_available,
    },
    {
        "Gemma2 chat session reports real token budget when available",
        test_gemma2_chat_session_reports_real_token_budget_when_available,
    },
    {
        "Gemma2 chat session trims oldest turns and rebuilds retained context when available",
        test_gemma2_chat_session_trims_oldest_turns_and_rebuilds_retained_context_when_available,
    },
    {
        "Gemma2 chat session does not shift context by default when available",
        test_gemma2_chat_session_does_not_shift_context_by_default_when_available,
    },
    {
        "Gemma2 chat session rejects impossible turn without mutating context when available",
        test_gemma2_chat_session_rejects_impossible_turn_without_mutating_context_when_available,
    },
    {
        "Gemma2 chat session seeded sampling and regen are reproducible when available",
        test_gemma2_chat_session_seeded_sampling_and_regen_are_reproducible_when_available,
    },
    {
        "runs generate Gemma2 chat loop until exit when available",
        test_runs_generate_gemma2_chat_loop_until_exit_when_available,
    },
    {
        "runs generate chat decode diagnostics only when requested when available",
        test_runs_generate_chat_decode_diagnostics_only_when_requested_when_available,
    },
    {"rejects Metal capture without diagnostics", test_rejects_metal_capture_without_diagnostics},
    {
        "runs generate Gemma2 chat streams visible tokens when available",
        test_runs_generate_gemma2_chat_streams_visible_tokens_when_available,
    },
    {
        "runs generate Gemma2 chat reverse stop when available",
        test_runs_generate_gemma2_chat_reverse_stop_when_available,
    },
    {
        "runs generate Gemma2 chat Gate 3 commands and input when available",
        test_runs_generate_gemma2_chat_gate3_commands_and_input_when_available,
    },
    {
        "runs generate Gemma2 chat context trim and recovers from impossible turn when available",
        test_runs_generate_gemma2_chat_context_trim_and_recovers_from_impossible_turn_when_available,
    },
    {
        "runs generate Gemma2 chat with llama-like defaults when available",
        test_runs_generate_gemma2_chat_with_llama_like_defaults_when_available,
    },
    {
        "runs generate Gemma2 chat seeded sampling flags when available",
        test_runs_generate_gemma2_chat_seeded_sampling_flags_when_available,
    },
    {
        "rejects generate Gemma2 chat invalid sampling flags",
        test_rejects_generate_gemma2_chat_invalid_sampling_flags,
    },
    {
        "rejects generate Gemma2 chat invalid output-head flags",
        test_rejects_generate_gemma2_chat_invalid_output_head_flags,
    },
    {
        "reports generate Gemma2 Metal runtime shell unavailable when missing",
        test_reports_generate_gemma2_metal_runtime_shell_unavailable_when_missing,
    },
    {
        "rejects generate Gemma2 CPU backend",
        test_rejects_generate_gemma2_cpu_backend,
    },
    {
        "rejects generate Gemma2 after decode schedule context full when available",
        test_rejects_generate_gemma2_after_decode_schedule_context_full_when_available,
    },
    {
        "rejects generate Gemma2 unsupported chat template before runtime",
        test_rejects_generate_gemma2_unsupported_chat_template_before_runtime,
    },
    {"rejects generate missing arguments", test_rejects_generate_missing_arguments},
    {"rejects generate invalid max new tokens", test_rejects_generate_invalid_max_new_tokens},
    {"rejects generate invalid backend option", test_rejects_generate_invalid_backend_option},
};

}  // namespace

namespace local_gguf_engine::tests {

std::vector<std::uint8_t> standalone_chat_target_fixture_bytes(std::uint32_t context) {
    return chat_recurrent_target_fixture_bytes(false, false, false, context);
}

std::span<const TestCase> generate_cli_test_cases() {
    return kGenerateCliTests;
}

}  // namespace local_gguf_engine::tests
