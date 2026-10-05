#include "local_gguf_engine/frontends/cli/generate.hpp"

#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tensor/metal_trace_capture.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <charconv>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

namespace local_gguf_engine::frontends::cli {

namespace {

using engine::EngineModelError;
using engine::Gemma2ChatSession;
using engine::Gemma2ChatContextError;
using engine::Gemma2ChatSessionOptions;
using engine::Gemma2ChatTurnOptions;
using engine::TokenSamplerOptions;
using engine::generate_gemma2_metal_argmax_tokens;
using engine::load_engine_model_from_gguf;
using engine::ModelArchitecture;
using engine::model_architecture_name;
using model_format::GGUFReadError;
using model_format::read_file_backed_gguf_model;
using model_format::read_gguf_model_view;
using runtime::BackendError;
using runtime::BackendKind;
using tensor::TensorError;
using tokenizer::format_generate_prompt_from_gguf_metadata;
using tokenizer::load_tokenizer_from_gguf_metadata;
using tokenizer::PromptFormatResult;
using tokenizer::TokenizerDecodeOptions;
using tokenizer::TokenizerEncodeOptions;
using tokenizer::TokenizerError;

struct EncodedGeneratePrompt {
    PromptFormatResult formatted_prompt;
    std::vector<std::uint32_t> token_ids;
};

[[nodiscard]] std::string format_tokens_per_second(
    std::uint32_t token_count,
    std::uint64_t duration_ns
) {
    if (token_count == 0U || duration_ns == 0U) {
        return "n/a";
    }

    const auto rate = static_cast<double>(token_count) * 1.0e9
        / static_cast<double>(duration_ns);
    if (!std::isfinite(rate)) {
        return "n/a";
    }

    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(1) << rate << " t/s";
    return formatted.str();
}

[[nodiscard]] std::string format_seconds(std::uint64_t duration_ns) {
    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(6)
              << static_cast<double>(duration_ns) / 1.0e9 << " s";
    return formatted.str();
}

volatile std::sig_atomic_t g_chat_interrupt_count = 0;

void handle_chat_interrupt(int) noexcept {
    g_chat_interrupt_count = static_cast<std::sig_atomic_t>(g_chat_interrupt_count + 1);
}

[[nodiscard]] std::sig_atomic_t chat_interrupt_count() noexcept {
    return g_chat_interrupt_count;
}

class ScopedSignalHandler final {
public:
    using SignalHandler = void (*)(int);

    ScopedSignalHandler(int signal_number, SignalHandler handler)
        : signal_number_(signal_number),
          previous_(std::signal(signal_number, handler)) {}

    ScopedSignalHandler(const ScopedSignalHandler&) = delete;
    ScopedSignalHandler& operator=(const ScopedSignalHandler&) = delete;

    ~ScopedSignalHandler() {
        std::signal(signal_number_, previous_);
    }

private:
    int signal_number_ = 0;
    SignalHandler previous_ = SIG_DFL;
};

void write_generate_usage(std::ostream& error) {
    error << "usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>"
          << " [--greedy] [--backend cpu|metal]"
          << '\n'
          << "       local-gguf-generate <model.gguf> --chat [--max-new-tokens N]"
          << " [--greedy | --temperature T [--top-k K] [--top-p P] [--min-p P]"
          << " [--repeat-penalty R] [--seed S]] [--backend cpu|metal]"
          << " [--reverse-prompt TEXT] [--context-shift] [--context-size TOKENS]"
          << " [--kv-cache auto|f32|f16]"
          << " [--output-head f16|q8_0|q6_k]"
          << " [--no-speculative]"
          << " [--retrieval | --no-retrieval]"
          << " [--draft-model <native.gguf>] [--draft-horizon <1..23>]"
          << " [--draft-temperature T] [--draft-top-k K] [--draft-top-p P]"
          << " [--draft-min-p P] [--draft-repeat-penalty R]"
          << " [--speculative-sampling exact-match|lossless]"
          << " [--speculative-schedule full|conditional]"
          << " [--decode-diagnostics] [--metal-capture PATH]"
          << '\n';
}

[[nodiscard]] bool starts_with_command_prefix(std::string_view line) noexcept {
    return !line.empty() && line.front() == '/';
}

[[nodiscard]] std::string_view trim_left(std::string_view text) noexcept {
    const auto first = text.find_first_not_of(" \t");
    return first == std::string_view::npos ? std::string_view {} : text.substr(first);
}

[[nodiscard]] bool read_multiline_input(
    std::istream& input,
    std::ostream& output,
    std::string& line
) {
    while (!line.empty() && line.back() == '\\') {
        line.pop_back();
        output << "... ";
        std::string continuation;
        if (!std::getline(input, continuation)) {
            output << '\n';
            return false;
        }
        line.push_back('\n');
        line += continuation;
    }
    return true;
}

[[nodiscard]] std::string read_prompt_file(std::string_view path_text) {
    if (path_text.empty()) {
        throw std::runtime_error("usage: /read <path>");
    }

    const std::filesystem::path path(path_text);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("could not read '" + path.string() + "'");
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

[[nodiscard]] bool parse_positive_size(std::string_view text, std::size_t& value) noexcept {
    if (text.empty()) {
        return false;
    }

    std::size_t parsed = 0;
    for (const auto character : text) {
        if (character < '0' || character > '9') {
            return false;
        }

        const auto digit = static_cast<std::size_t>(character - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }

    if (parsed == 0) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_u64(std::string_view text, std::uint64_t& value) noexcept {
    if (text.empty()) {
        return false;
    }
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_float(std::string_view text, float& value) noexcept {
    if (text.empty()) {
        return false;
    }
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size()
        && std::isfinite(value);
}

[[nodiscard]] bool parse_backend_kind(std::string_view text, BackendKind& backend) noexcept {
    if (text == "cpu") {
        backend = BackendKind::cpu;
        return true;
    }

    if (text == "metal") {
        backend = BackendKind::metal;
        return true;
    }

    return false;
}

struct ParsedGenerateCommand {
    GenerateOptions options;
    TokenSamplerOptions sampled_sampler {
        .temperature = 0.8F,
        .top_k = 40U,
        .top_p = 0.95F,
        .min_p = 0.05F,
        .repetition_penalty = 1.0F,
        .seed = 42U,
    };
};

[[nodiscard]] bool parse_generate_options(
    int argc,
    const char* const* argv,
    int start_index,
    ParsedGenerateCommand& command,
    std::ostream& error
) {
    for (int index = start_index; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--greedy") {
            continue;
        } else if (option == "--backend") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --backend requires a value\n";
                return false;
            }

            BackendKind backend = BackendKind::cpu;
            if (!parse_backend_kind(argv[index + 1], backend)) {
                write_generate_usage(error);
                error << "error: --backend must be cpu or metal\n";
                return false;
            }

            command.options.backend = backend;
            ++index;
        } else if (
            option == "--temperature" || option == "--top-p" || option == "--min-p"
            || option == "--repeat-penalty"
        ) {
            if (index + 1 >= argc) {
                error << "error: " << option << " requires a value\n";
                return false;
            }
            float value = 0.0F;
            if (!parse_float(argv[++index], value)) {
                error << "error: " << option << " requires a finite number\n";
                return false;
            }
            if (option == "--temperature") command.sampled_sampler.temperature = value;
            else if (option == "--top-p") command.sampled_sampler.top_p = value;
            else if (option == "--min-p") command.sampled_sampler.min_p = value;
            else command.sampled_sampler.repetition_penalty = value;
        } else if (option == "--top-k") {
            if (index + 1 >= argc || !parse_positive_size(argv[index + 1], command.sampled_sampler.top_k)) {
                error << "error: --top-k requires a positive integer\n";
                return false;
            }
            ++index;
        } else if (option == "--seed") {
            if (index + 1 >= argc || !parse_u64(argv[index + 1], command.sampled_sampler.seed)) {
                error << "error: --seed requires an unsigned integer\n";
                return false;
            }
            ++index;
        } else {
            write_generate_usage(error);
            error << "error: unknown option '" << option << "'\n";
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool is_chat_option(std::string_view option) noexcept {
    return option == "--chat" || option == "--conversation" || option == "-cnv";
}

[[nodiscard]] bool argv_requests_chat(int argc, const char* const* argv) noexcept {
    for (int index = 2; index < argc; ++index) {
        if (is_chat_option(argv[index])) {
            return true;
        }
    }

    return false;
}

struct ParsedChatCommand {
    std::size_t max_new_tokens = 0;
    GenerateOptions options;
    std::vector<std::string> reverse_stop_strings;
    TokenSamplerOptions sampler {
        .temperature = 0.8F,
        .top_k = 40U,
        .top_p = 0.95F,
        .min_p = 0.05F,
        .repetition_penalty = 1.0F,
        .seed = TokenSamplerOptions::random_seed,
        .repetition_window = 64U,
    };
    std::uint32_t context_length = 0;
    engine::GemmaKVPrecision kv_precision = engine::GemmaKVPrecision::automatic;
    bool kv_precision_requested = false;
    bool context_shift = false;
    bool greedy_requested = false;
    bool temperature_requested = false;
    bool sampled_modifier_requested = false;
    bool decode_diagnostics = false;
    bool speculative_decode = true;
    std::optional<bool> retrieval_decode;
    std::optional<std::filesystem::path> standalone_draft_model;
    engine::StandaloneExecutionOptions standalone_execution;
    bool draft_horizon_requested = false;
    std::optional<engine::SpeculativeSamplingMode> speculative_sampling;
    bool schedule_requested = false;
    engine::Gemma2OutputHeadPolicy output_head_policy = engine::Gemma2OutputHeadPolicy::automatic;
    std::optional<std::string> metal_capture_path;
};

class ScopedMetalTraceArm final {
public:
    explicit ScopedMetalTraceArm(const std::optional<std::string>& path)
        : path_(path), pending_(path.has_value()) {
        if (path_.has_value()) {
            tensor::backends::metal::arm_next_gemma2_decode_trace(*path_);
        }
    }

    ~ScopedMetalTraceArm() {
        if (pending_ && tensor::backends::metal::metal_trace_capture_status()
                == tensor::backends::metal::MetalTraceCaptureStatus::armed) {
            tensor::backends::metal::cancel_armed_metal_trace_capture();
        }
    }

    [[nodiscard]] bool pending() const noexcept { return pending_; }

    void finish(std::ostream& output) {
        if (!pending_) return;
        if (tensor::backends::metal::metal_trace_capture_status()
            != tensor::backends::metal::MetalTraceCaptureStatus::completed) {
            throw std::runtime_error("Metal GPU trace capture did not capture a production decode");
        }
        tensor::backends::metal::reset_completed_metal_trace_capture();
        output << "[ Metal capture: " << *path_ << " ]\n";
        pending_ = false;
    }

private:
    std::optional<std::string> path_;
    bool pending_ = false;
};

[[nodiscard]] bool parse_chat_command(
    int argc,
    const char* const* argv,
    ParsedChatCommand& command,
    std::ostream& error
) {
    for (int index = 2; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (is_chat_option(option)) {
            continue;
        }
        if (option == "--greedy") {
            command.greedy_requested = true;
            continue;
        }
        if (option == "--context-shift") {
            command.context_shift = true;
            continue;
        }
        if (option == "--decode-diagnostics") {
            command.decode_diagnostics = true;
            continue;
        }
        if (option == "--no-speculative") {
            command.speculative_decode = false;
            continue;
        }
        if (option == "--retrieval") {
            command.retrieval_decode = true;
            continue;
        }
        if (option == "--no-retrieval") {
            command.retrieval_decode = false;
            continue;
        }
        if (option == "--speculative-sampling") {
            if (command.speculative_sampling || index + 1 >= argc) {
                error << "error: --speculative-sampling requires one exact-match or lossless value\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value != "exact-match" && value != "lossless") {
                error << "error: --speculative-sampling requires exact-match or lossless\n";
                return false;
            }
            command.speculative_sampling = value == "lossless"
                ? engine::SpeculativeSamplingMode::lossless : engine::SpeculativeSamplingMode::exact_match;
            continue;
        }
        if (option == "--speculative-schedule") {
            if (command.schedule_requested || index + 1 >= argc) {
                error << "error: --speculative-schedule requires exactly one value\n";
                return false;
            }
            command.schedule_requested = true;
            const std::string_view value(argv[++index]);
            if (value != "full" && value != "conditional") {
                error << "error: invalid value for --speculative-schedule\n";
                return false;
            }
            command.standalone_execution.conditional_verification = value == "conditional";
            continue;
        }
        if (option == "--draft-model") {
            if (command.standalone_draft_model || index + 1 >= argc
                || std::string_view(argv[index + 1]).empty()
                || std::string_view(argv[index + 1]).starts_with("--")) {
                error << "error: --draft-model requires one native draft GGUF file\n";
                return false;
            }
            command.standalone_draft_model = argv[++index];
            continue;
        }
        if (option == "--draft-horizon") {
            std::size_t horizon = 0;
            if (command.draft_horizon_requested || index + 1 >= argc
                || !parse_positive_size(argv[index + 1], horizon)
                || horizon > engine::external_speculation_max_horizon) {
                error << "error: --draft-horizon requires one integer in [1, 23]\n";
                return false;
            }
            command.draft_horizon_requested = true;
            command.standalone_execution.horizon = static_cast<std::uint32_t>(horizon);
            ++index;
            continue;
        }
        if (option == "--draft-temperature" || option == "--draft-top-p"
            || option == "--draft-min-p" || option == "--draft-repeat-penalty") {
            auto& sampling = command.standalone_execution.draft_sampling;
            auto& destination = option == "--draft-temperature" ? sampling.temperature
                : option == "--draft-top-p" ? sampling.top_p
                : option == "--draft-min-p" ? sampling.min_p : sampling.repetition_penalty;
            float value = 0.0F;
            if (destination || index + 1 >= argc || !parse_float(argv[index + 1], value)) {
                error << "error: " << option << " requires one finite number\n";
                return false;
            }
            const bool valid = option == "--draft-temperature" ? value >= 0.0F
                : option == "--draft-top-p" ? value > 0.0F && value <= 1.0F
                : option == "--draft-min-p" ? value >= 0.0F && value <= 1.0F : value > 0.0F;
            if (!valid) {
                error << "error: " << option << " is outside its valid range\n";
                return false;
            }
            destination = value;
            ++index;
            continue;
        }
        if (option == "--draft-top-k") {
            std::uint64_t value = 0;
            auto& destination = command.standalone_execution.draft_sampling.top_k;
            if (destination || index + 1 >= argc || !parse_u64(argv[index + 1], value)
                || value > std::numeric_limits<std::uint32_t>::max()) {
                error << "error: --draft-top-k requires one unsigned 32-bit integer (0 disables top-k)\n";
                return false;
            }
            destination = static_cast<std::uint32_t>(value);
            ++index;
            continue;
        }
        if (option == "--context-size") {
            std::uint64_t value = 0;
            if (command.context_length != 0U || index + 1 >= argc
                || !parse_u64(argv[index + 1], value) || value == 0U
                || value > std::numeric_limits<std::uint32_t>::max()) {
                error << "error: --context-size requires one positive 32-bit token count\n";
                return false;
            }
            command.context_length = static_cast<std::uint32_t>(value);
            ++index;
            continue;
        }
        if (option == "--kv-cache") {
            if (command.kv_precision_requested || index + 1 >= argc) {
                error << "error: --kv-cache requires one of auto, f32, or f16, exactly once\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value == "auto") command.kv_precision = engine::GemmaKVPrecision::automatic;
            else if (value == "f32") command.kv_precision = engine::GemmaKVPrecision::f32;
            else if (value == "f16") command.kv_precision = engine::GemmaKVPrecision::f16;
            else {
                error << "error: --kv-cache must be auto, f32, or f16\n";
                return false;
            }
            command.kv_precision_requested = true;
            continue;
        }
        if (option == "--output-head") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --output-head requires f16, q8_0, or q6_k\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value == "f16") command.output_head_policy = engine::Gemma2OutputHeadPolicy::f16;
            else if (value == "q8_0") command.output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0;
            else if (value == "q6_k") command.output_head_policy = engine::Gemma2OutputHeadPolicy::q6_k;
            else {
                write_generate_usage(error);
                error << "error: --output-head must be f16, q8_0, or q6_k\n";
                return false;
            }
            continue;
        }
        if (option == "--metal-capture") {
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty()) {
                write_generate_usage(error);
                error << "error: --metal-capture requires a path\n";
                return false;
            }
            command.metal_capture_path = argv[++index];
            continue;
        }
        if (option == "--no-context-shift") {
            command.context_shift = false;
            continue;
        }
        if (option == "--max-new-tokens") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --max-new-tokens requires a value\n";
                return false;
            }
            if (!parse_positive_size(argv[index + 1], command.max_new_tokens)) {
                write_generate_usage(error);
                error << "error: max-new-tokens must be a positive integer\n";
                return false;
            }
            ++index;
            continue;
        }

        if (option == "--backend") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --backend requires a value\n";
                return false;
            }

            BackendKind backend = BackendKind::cpu;
            if (!parse_backend_kind(argv[index + 1], backend)) {
                write_generate_usage(error);
                error << "error: --backend must be cpu or metal\n";
                return false;
            }

            command.options.backend = backend;
            ++index;
            continue;
        }

        if (
            option == "--temperature" || option == "--top-p" || option == "--min-p"
            || option == "--repeat-penalty"
        ) {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: " << option << " requires a value\n";
                return false;
            }
            float value = 0.0F;
            if (!parse_float(argv[index + 1], value)) {
                write_generate_usage(error);
                error << "error: " << option << " requires a finite number\n";
                return false;
            }
            if (option == "--temperature") {
                command.sampler.temperature = value;
                command.temperature_requested = true;
            } else if (option == "--top-p") {
                command.sampler.top_p = value;
                command.sampled_modifier_requested = true;
            } else if (option == "--min-p") {
                command.sampler.min_p = value;
                command.sampled_modifier_requested = true;
            } else {
                command.sampler.repetition_penalty = value;
                command.sampled_modifier_requested = true;
            }
            ++index;
            continue;
        }

        if (option == "--top-k") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --top-k requires a value\n";
                return false;
            }
            std::size_t value = 0;
            if (!parse_positive_size(argv[index + 1], value)) {
                write_generate_usage(error);
                error << "error: --top-k must be a positive integer\n";
                return false;
            }
            command.sampler.top_k = value;
            command.sampled_modifier_requested = true;
            ++index;
            continue;
        }

        if (option == "--seed") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --seed requires a value\n";
                return false;
            }
            const auto seed_text = std::string_view(argv[index + 1]);
            if (seed_text == "-1") {
                command.sampler.seed = TokenSamplerOptions::random_seed;
            } else if (!parse_u64(seed_text, command.sampler.seed)) {
                write_generate_usage(error);
                error << "error: --seed must be -1 or an unsigned integer\n";
                return false;
            }
            command.sampled_modifier_requested = true;
            ++index;
            continue;
        }

        if (option == "--reverse-prompt" || option == "--reverse-stop" || option == "--stop" || option == "-r") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: " << option << " requires a value\n";
                return false;
            }

            command.reverse_stop_strings.push_back(argv[index + 1]);
            ++index;
            continue;
        }

        write_generate_usage(error);
        error << "error: unknown chat option '" << option << "'\n";
        return false;
    }

    if (command.standalone_draft_model
        && (!command.speculative_decode
            || (command.speculative_sampling
                && command.speculative_sampling != engine::SpeculativeSamplingMode::lossless)
            || command.options.backend != BackendKind::metal)) {
        error << "error: --draft-model requires Metal and lossless speculation,"
              << " and conflicts with --no-speculative\n";
        return false;
    }
    if (command.draft_horizon_requested && !command.standalone_draft_model) {
        error << "error: --draft-horizon requires --draft-model\n";
        return false;
    }
    if (command.standalone_execution.draft_sampling != engine::StandaloneDraftSamplingOptions{}
        && !command.standalone_draft_model) {
        error << "error: draft sampling overrides require --draft-model and lossless speculation\n";
        return false;
    }
    const bool has_draft = command.standalone_draft_model.has_value();
    if (command.speculative_sampling && (!has_draft || !command.speculative_decode)) {
        error << "error: --speculative-sampling requires a draft configuration and enabled speculation\n";
        return false;
    }
    if (command.schedule_requested
        && (!has_draft || !command.speculative_decode
            || (command.speculative_sampling
                && command.speculative_sampling != engine::SpeculativeSamplingMode::lossless))) {
        error << "error: speculative execution options require a draft configuration and lossless sampling\n";
        return false;
    }
    if (command.greedy_requested) {
        if (command.temperature_requested && command.sampler.temperature > 0.0F) {
            error << "error: --greedy cannot be combined with positive --temperature\n";
            return false;
        }
        command.sampler.temperature = 0.0F;
    }
    if (command.sampler.temperature < 0.0F) {
        error << "error: --temperature must be nonnegative\n";
        return false;
    }
    if (command.sampler.temperature == 0.0F && command.sampled_modifier_requested) {
        error << "error: sampling modifiers require --temperature greater than zero\n";
        return false;
    }
    if (command.sampler.top_p <= 0.0F || command.sampler.top_p > 1.0F) {
        error << "error: --top-p must be in (0, 1]\n";
        return false;
    }
    if (command.sampler.min_p < 0.0F || command.sampler.min_p > 1.0F) {
        error << "error: --min-p must be in [0, 1]\n";
        return false;
    }
    if (command.sampler.repetition_penalty <= 0.0F) {
        error << "error: --repeat-penalty must be positive\n";
        return false;
    }
    if (command.metal_capture_path.has_value()
        && (!command.decode_diagnostics || command.options.backend != BackendKind::metal)) {
        error << "error: --metal-capture requires --decode-diagnostics and --backend metal\n";
        return false;
    }

    return true;
}

[[nodiscard]] EncodedGeneratePrompt encode_generate_prompt(
    const model_format::GGUFMetadataSection& metadata,
    const tokenizer::Tokenizer& tokenizer,
    std::string_view prompt
) {
    const auto formatted_prompt = format_generate_prompt_from_gguf_metadata(metadata, prompt);
    const auto default_encode_options = tokenizer.default_encode_options();
    const auto prompt_token_ids = tokenizer.encode(
        formatted_prompt.text,
        TokenizerEncodeOptions {
            .add_bos = formatted_prompt.add_bos.value_or(default_encode_options.add_bos),
            .add_eos = formatted_prompt.add_eos.value_or(default_encode_options.add_eos),
            .encode_special_tokens = formatted_prompt.encode_special_tokens,
        }
    );

    return EncodedGeneratePrompt {
        .formatted_prompt = formatted_prompt,
        .token_ids = prompt_token_ids,
    };
}

[[nodiscard]] std::string unsupported_generation_architecture_message(ModelArchitecture architecture) {
    return "local-gguf-generate supports Gemma 2 and Gemma 3 generation; architecture '"
        + std::string(model_architecture_name(architecture))
        + "' is unsupported";
}

[[nodiscard]] std::string generate_gemma2_greedy_text(
    const engine::EngineModel& model,
    const tokenizer::Tokenizer& tokenizer,
    std::span<const std::uint32_t> prompt_token_ids,
    std::size_t max_new_tokens,
    const GenerateOptions& options
) {
    if (options.backend != BackendKind::metal) {
        throw std::runtime_error("Gemma 2 generation requires Metal backend");
    }

    const auto output_token_ids = generate_gemma2_metal_argmax_tokens(
        model,
        prompt_token_ids,
        max_new_tokens,
        tokenizer.special_tokens().eos_token_id
    );

    return tokenizer.decode(
        std::span<const std::uint32_t>(output_token_ids).subspan(prompt_token_ids.size()),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );
}

[[nodiscard]] std::string generate_text_from_gguf_model_view(
    const model_format::GGUFModelView& model_view,
    std::string_view prompt,
    std::size_t max_new_tokens,
    const GenerateOptions& options
) {
    const auto model = load_engine_model_from_gguf(model_view);
    if (model.architecture() != ModelArchitecture::gemma2
        && model.architecture() != ModelArchitecture::gemma3) {
        throw std::runtime_error(unsupported_generation_architecture_message(model.architecture()));
    }

    const auto tokenizer = load_tokenizer_from_gguf_metadata(model_view.directory.metadata);
    const auto encoded_prompt = encode_generate_prompt(model_view.directory.metadata, tokenizer, prompt);
    return generate_gemma2_greedy_text(
        model,
        tokenizer,
        std::span<const std::uint32_t>(encoded_prompt.token_ids),
        max_new_tokens,
        options
    );
}

int run_generate_chat(
    const char* model_path,
    const ParsedChatCommand& command,
    std::istream& input,
    std::ostream& output
) {
    const ScopedSignalHandler chat_interrupt_handler(SIGINT, handle_chat_interrupt);
    auto file_model = read_file_backed_gguf_model(model_path);
    auto session = Gemma2ChatSession(
        std::move(file_model),
        Gemma2ChatSessionOptions {
            .max_new_tokens = command.max_new_tokens,
            .backend = command.options.backend,
            .output_head_policy = command.output_head_policy,
            .context_length = command.context_length,
            .kv_precision = command.kv_precision,
            .context_shift = command.context_shift,
            .speculative_decode = command.speculative_decode,
            .retrieval_decode = command.retrieval_decode,
            .standalone_draft_model = command.standalone_draft_model,
            .standalone_execution = command.standalone_execution,
            .speculative_sampling = command.speculative_sampling,
            .reverse_stop_strings = command.reverse_stop_strings,
            .sampler = command.sampler,
        }
    );
    ScopedMetalTraceArm metal_trace(command.metal_capture_path);

    output << "local-gguf-engine chat\n";
    if (command.standalone_draft_model) {
        output << "standalone TLI draft: authenticated native Metal model; no admitted routing policy\n"
               << "speculative sampling: lossless (target distribution; different seeded text)\n"
               << "standalone execution: horizon=" << command.standalone_execution.horizon
               << " schedule=" << (command.standalone_execution.conditional_verification
                   ? "conditional" : "full") << " draft-head=artifact-defined\n";
    }
    output << "commands: /exit /quit /clear /reset /regen /tokens /read <path>\n";
    output << "multiline: end a line with \\\n";

    std::string line;
    while (true) {
        output << "> ";
        const auto prompt_interrupt_count = chat_interrupt_count();
        if (!std::getline(input, line)) {
            output << '\n';
            return 0;
        }
        if (chat_interrupt_count() != prompt_interrupt_count) {
            output << '\n';
            continue;
        }

        if (line == "/exit" || line == "/quit") {
            return 0;
        }

        if (line == "/clear") {
            session.clear();
            output << "chat cleared\n";
            continue;
        }
        if (line == "/reset") {
            session.reset();
            output << "session reset\n";
            continue;
        }
        if (line == "/tokens") {
            output << "tokens: cached=" << session.cached_token_count()
                   << " remaining=" << session.remaining_context_tokens()
                   << " max-new-tokens=";
            if (session.max_new_tokens() == 0U) output << "unlimited";
            else output << session.max_new_tokens();
            output << '\n';
            continue;
        }

        bool regenerate = false;
        if (line == "/regen") {
            if (session.message_count() < 2U) {
                output << "error: nothing to regenerate\n";
                continue;
            }
            regenerate = true;
        } else if (line == "/read" || line.starts_with("/read ") || line.starts_with("/read\t")) {
            try {
                line = read_prompt_file(trim_left(std::string_view(line).substr(5U)));
            } catch (const std::exception& read_error) {
                output << "error: " << read_error.what() << '\n';
                continue;
            }
        } else if (starts_with_command_prefix(line)) {
            output << "error: unknown command '" << line << "'\n";
            continue;
        } else if (!read_multiline_input(input, output, line)) {
            return 0;
        }

        const auto turn_interrupt_count = chat_interrupt_count();
        const auto turn_options = Gemma2ChatTurnOptions {
                .on_text_delta = [&](std::string_view text_delta) {
                    output << text_delta;
                    output.flush();
                },
                .should_stop = [&]() {
                    return chat_interrupt_count() != turn_interrupt_count;
                },
            };
        std::optional<engine::Gemma2ChatTurnResult> turn;
        try {
            turn = regenerate
                ? session.regenerate_last_turn(turn_options)
                : session.generate_user_turn(line, turn_options);
        } catch (const Gemma2ChatContextError& context_error) {
            output << "error: context: " << context_error.what() << '\n';
            continue;
        }
        output << '\n';
        output << format_chat_timing_line(turn->metrics) << '\n';
        output << format_chat_timing_detail_line(turn->metrics) << '\n';
        if (command.decode_diagnostics) {
            output << format_chat_decode_attribution_line(turn->metrics) << '\n';
        }
        if (metal_trace.pending()) metal_trace.finish(output);
        if (turn->rebuilt_context) {
            output << "context: dropped " << turn->trimmed_turn_count
                   << " oldest turn(s) and rebuilt retained KV\n";
        }
        if (turn->interrupted && chat_interrupt_count() > turn_interrupt_count + 1) {
            return 130;
        }
    }
}

}  // namespace

std::string format_chat_timing_line(const engine::Gemma2ChatTurnMetrics& metrics) {
    return "[ Prompt: "
        + format_tokens_per_second(
            metrics.generation.newly_evaluated_prompt_token_count,
            metrics.prompt_processing_duration_ns
        )
        + " | Generation: "
        + format_tokens_per_second(
            metrics.generation.selected_token_count,
            metrics.generation_duration_ns
        )
        + " | TTFT: "
        + (metrics.prompt_processing_duration_ns != 0U
            ? format_seconds(metrics.prompt_processing_duration_ns)
            : std::string("n/a"))
        + " ]";
}

std::string format_chat_timing_detail_line(const engine::Gemma2ChatTurnMetrics& metrics) {
    return "[ Detail: Evaluated: "
        + std::to_string(metrics.generation.newly_evaluated_prompt_token_count)
        + " | Cached: " + std::to_string(metrics.reused_cached_token_count)
        + " | Decoded: " + std::to_string(metrics.generation.selected_token_count)
        + " | Prompt: " + format_seconds(metrics.prompt_processing_duration_ns)
        + " | Generation: " + format_seconds(metrics.generation_duration_ns) + " ]";
}

std::string format_chat_decode_attribution_line(const engine::Gemma2ChatTurnMetrics& metrics) {
    constexpr std::uint32_t schema_version = 6U;
    if (metrics.standalone_draft != metrics.standalone_effective_sampler.has_value()) {
        throw std::logic_error("standalone diagnostics require the actual effective draft sampler");
    }
    if (metrics.standalone_effective_sampler) {
        const auto& sampler = *metrics.standalone_effective_sampler;
        if (!std::isfinite(sampler.temperature) || sampler.temperature < 0.0F
            || !std::isfinite(sampler.top_p) || sampler.top_p <= 0.0F || sampler.top_p > 1.0F
            || !std::isfinite(sampler.min_p) || sampler.min_p < 0.0F || sampler.min_p > 1.0F
            || !std::isfinite(sampler.repetition_penalty) || sampler.repetition_penalty <= 0.0F) {
            throw std::logic_error("standalone diagnostics contain invalid effective draft scalars");
        }
    }
    const auto decode_tokens = static_cast<std::uint64_t>(metrics.generation.decode_token_count);
    const auto checked_multiply = [](std::uint64_t left, std::uint64_t right, const char* name) {
        if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
            throw std::overflow_error(std::string("decode attribution overflow: ") + name);
        }
        return left * right;
    };
    const auto checked_add = [](std::uint64_t left, std::uint64_t right, const char* name) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            throw std::overflow_error(std::string("decode attribution overflow: ") + name);
        }
        return left + right;
    };
    const auto& bytes = metrics.decode_scheduled_weight_bytes;
    std::uint64_t byte_sum = 0U;
    for (const auto value : {
             bytes.embedding, bytes.attention_projections, bytes.feed_forward_gate_up,
             bytes.feed_forward_down, bytes.output_head, bytes.norms,
         }) {
        byte_sum = checked_add(byte_sum, value, "scheduled byte sum");
    }
    if (byte_sum != bytes.total) {
        throw std::logic_error("decode attribution scheduled byte categories do not sum to total");
    }
    const auto& execution = metrics.decode_execution;
    if (decode_tokens > 0U && (execution.command_buffer_count == 0U || execution.gpu_duration_ns == 0U)) {
        throw std::logic_error("decode attribution has decoded tokens without GPU execution");
    }
    if (execution.gpu_duration_ns > metrics.generation.decode_duration_ns) {
        throw std::logic_error("decode attribution GPU duration exceeds decode wall duration");
    }
    if (execution.output_head_elision_count > decode_tokens) {
        throw std::logic_error("decode attribution output-head elisions exceed decoded tokens");
    }
    const auto final_cache_position = checked_add(
        checked_add(
            metrics.initial_cache_position,
            metrics.generation.newly_evaluated_prompt_token_count,
            "prompt cache position"
        ),
        decode_tokens,
        "final cache position"
    );
    const auto full_scheduled_bytes = checked_multiply(bytes.total, decode_tokens, "scheduled bytes");
    const auto minimum_elided_bytes = checked_multiply(
        bytes.output_head, execution.output_head_elision_count, "elided output-head bytes"
    );
    if (execution.output_elided_weight_bytes < minimum_elided_bytes
        || execution.output_elided_weight_bytes > full_scheduled_bytes) {
        throw std::logic_error("decode attribution elided byte count is invalid");
    }
    const auto total_scheduled_bytes = full_scheduled_bytes - execution.output_elided_weight_bytes;
    const auto wall_residual_ns = metrics.generation.decode_duration_ns - execution.gpu_duration_ns;

    std::ostringstream token_hash;
    token_hash << std::hex << std::setw(16) << std::setfill('0')
               << metrics.generation.selected_token_hash;

    std::ostringstream json;
    json << "[ Decode: {\"schema_version\":" << schema_version
         << ",\"engine_build\":\"" << LOCAL_GGUF_ENGINE_BUILD << "\""
         << ",\"model_architecture\":\"" << model_architecture_name(metrics.model_architecture) << "\""
         << ",\"retrieval_decode\":" << (metrics.retrieval_decode ? "true" : "false")
         << ",\"context_capacity\":" << metrics.context_capacity
         << ",\"kv_cache_dtype\":\""
         << engine::gemma_kv_precision_name(metrics.kv_precision) << "\""
         << ",\"selected_tokens\":" << metrics.generation.selected_token_count
         << ",\"selected_token_hash\":\"" << token_hash.str() << "\""
         << ",\"decoded_tokens\":" << metrics.generation.decode_token_count
         << ",\"initial_cache_position\":" << metrics.initial_cache_position
         << ",\"final_cache_position\":" << final_cache_position
         << ",\"evaluated_prompt_tokens\":" << metrics.generation.newly_evaluated_prompt_token_count
         << ",\"reused_cache_tokens\":" << metrics.reused_cached_token_count
         << ",\"decode_wall_ns\":" << metrics.generation.decode_duration_ns
         << ",\"gpu_ns\":" << execution.gpu_duration_ns
         << ",\"selection_ns\":" << metrics.generation.selection_duration_ns
         << ",\"proposer_ns\":" << metrics.generation.proposer_duration_ns
         << ",\"generation_completion_ns\":" << metrics.generation_completion_duration_ns
         << ",\"full_turn_ns\":" << metrics.full_turn_duration_ns
         << ",\"first_text_ns\":" << metrics.first_text_duration_ns
         << ",\"learned_owned_bytes\":" << metrics.learned_owned_byte_count
         << ",\"learned_horizon\":" << metrics.learned_horizon
         << ",\"speculative_draft_model\":\""
         << (metrics.standalone_draft ? "standalone_gemma3_tli" : "none") << "\""
         << ",\"speculative_sampling\":\""
         << (metrics.speculative_sampling == engine::SpeculativeSamplingMode::lossless
             ? "lossless" : "exact-match") << "\""
         << ",\"learned_verification_schedule\":\""
         << (metrics.standalone_execution.conditional_verification ? "conditional" : "full") << "\""
         << ",\"learned_draft_head\":\""
         << (metrics.standalone_draft ? "q8_0" : "f16") << "\""
         << ",\"scheduled_bytes_basis\":\"one_row_equivalent_not_physical_bandwidth\""
         << ",\"learned_cycles\":" << metrics.generation.learned_cycle_count
         << ",\"learned_accepted_edges\":" << metrics.generation.learned_accepted_edge_count
         << ",\"learned_proposed_edges\":" << metrics.generation.learned_proposed_edge_count
         << ",\"learned_numeric_refusals\":" << metrics.generation.learned_numeric_refusal_count
         << ",\"learned_draft_ns\":" << metrics.generation.learned_draft_duration_ns
         << ",\"learned_draft_gpu_ns\":" << metrics.generation.learned_draft_gpu_ns
         << ",\"learned_target_gpu_ns\":" << metrics.generation.learned_target_gpu_ns
         << ",\"learned_cycle_ns\":" << metrics.generation.learned_cycle_duration_ns
         << ",\"standalone_prefill_ns\":" << metrics.generation.standalone_prefill_ns
         << ",\"standalone_rebuild_ns\":" << metrics.generation.standalone_rebuild_ns
         << ",\"standalone_draft_uploaded_bytes\":" << metrics.generation.standalone_draft_uploaded_bytes
         << ",\"standalone_draft_readback_bytes\":" << metrics.generation.standalone_draft_readback_bytes
         << ",\"standalone_draft_wall_ns\":" << metrics.generation.standalone_draft_wall_ns
         << ",\"standalone_draft_gpu_ns\":" << metrics.generation.standalone_draft_gpu_ns
         << ",\"standalone_draft_command_buffer_count\":" << metrics.generation.standalone_draft_command_buffer_count
         << ",\"standalone_draft_compute_dispatch_count\":" << metrics.generation.standalone_draft_compute_dispatch_count
         << ",\"standalone_draft_sampler\":\""
         << (metrics.standalone_draft
             ? (metrics.standalone_bounded_repetition_history
                 ? "explicit_or_target_scalars_target_window_v2"
                 : "explicit_or_target_scalars_native_history_v2")
             : "none") << "\""
         << ",\"standalone_repetition_window\":";
    if (metrics.standalone_bounded_repetition_history && metrics.standalone_effective_sampler)
        json << metrics.standalone_effective_sampler->repetition_window;
    else json << "null";
    json << ",\"standalone_effective_sampler\":";
    if (metrics.standalone_effective_sampler) {
        const auto& sampler = *metrics.standalone_effective_sampler;
        // Emit round-trippable F32 values actually consumed by the draft, not
        // the original CLI strings or unchanged target configuration.
        const auto precision = json.precision();
        json << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "{\"temperature\":" << sampler.temperature
             << ",\"top_k\":" << sampler.top_k
             << ",\"top_p\":" << sampler.top_p
             << ",\"min_p\":" << sampler.min_p
             << ",\"repetition_penalty\":" << sampler.repetition_penalty << '}';
        json.precision(precision);
    } else json << "null";
    json << ",\"learned_horizon_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_horizon_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_horizon_histogram[index];
    }
    json << "],\"learned_reached_depth_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_reached_depth_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_reached_depth_histogram[index];
    }
    json << "],\"learned_accepted_prefix_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_accepted_prefix_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_accepted_prefix_histogram[index];
    }
    json << "],\"learned_rejection_depth_histogram\":[";
    for (std::size_t index = 0; index < metrics.generation.learned_rejection_depth_histogram.size(); ++index) {
        if (index) json << ',';
        json << metrics.generation.learned_rejection_depth_histogram[index];
    }
    json << ']'
         << ",\"learned_censored_cycle_count\":" << metrics.generation.learned_censored_cycle_count
         << ",\"learned_full_accept_count\":" << metrics.generation.learned_full_accept_count
         << ",\"learned_horizon_limited_cycle_count\":" << metrics.generation.learned_horizon_limited_cycle_count
         << ",\"speculative_cycles\":" << metrics.generation.speculative_cycle_count
         << ",\"speculative_matches\":" << metrics.generation.speculative_second_token_match_count
         << ",\"speculative_misses\":" << metrics.generation.speculative_second_token_miss_count
         << ",\"speculative_fallbacks\":" << metrics.generation.speculative_fallback_count
         << ",\"speculative_rollbacks\":" << metrics.generation.speculative_rollback_count
         << ",\"proposal_buckets\":[";
    bool first_bucket = true;
    constexpr std::array<const char*, 4> length_labels {"2", "3-4", "5-8", "9-16"};
    constexpr std::array<const char*, 3> support_labels {"2", "3", "4+"};
    constexpr std::array<const char*, 3> confidence_labels {"0.5-2/3", "2/3-1", "1"};
    for (std::size_t length = 0; length < length_labels.size(); ++length) {
        for (std::size_t support = 0; support < support_labels.size(); ++support) {
            for (std::size_t confidence = 0; confidence < confidence_labels.size(); ++confidence) {
                const auto index = (length * support_labels.size() + support)
                    * confidence_labels.size() + confidence;
                const auto& bucket = metrics.generation.speculative_proposal_buckets[index];
                if (bucket.cycle_count == 0U) continue;
                if (!first_bucket) json << ',';
                first_bucket = false;
                json << "{\"match_length\":\"" << length_labels[length]
                     << "\",\"support\":\"" << support_labels[support]
                     << "\",\"confidence\":\"" << confidence_labels[confidence]
                     << "\",\"cycles\":" << bucket.cycle_count
                     << ",\"matches\":" << bucket.match_count << '}';
            }
        }
    }
    json << ']'
         << ",\"wall_minus_gpu_ns\":" << wall_residual_ns
         << ",\"resident_weights_bytes\":" << metrics.resident_weight_byte_count
         << ",\"output_head_policy\":\""
         << engine::gemma2_output_head_policy_name(metrics.output_head_policy) << "\""
         << ",\"resident_kv_bytes\":" << metrics.resident_kv_cache_byte_count
         << ",\"resident_scratch_bytes\":" << metrics.resident_scratch_byte_count
         << ",\"warmup_kv_clear_passes\":" << metrics.warmup_kv_clear_pass_count
         << ",\"warmup_host_cleared_kv_bytes\":" << metrics.warmup_host_cleared_kv_bytes
         << ",\"command_buffers\":" << execution.command_buffer_count
         << ",\"commits\":" << execution.commit_count
         << ",\"waits\":" << execution.synchronous_wait_count
         << ",\"compute_encoders\":" << execution.compute_encoder_count
         << ",\"compute_dispatches\":" << execution.compute_dispatch_count
         << ",\"blit_encoders\":" << execution.blit_encoder_count
         << ",\"barriers\":" << execution.memory_barrier_count
         << ",\"transient_allocations\":" << execution.transient_buffer_allocation_count
         << ",\"selector_dispatches\":" << execution.selector_dispatch_count
         << ",\"max_candidates\":" << execution.candidate_count
         << ",\"compact_readback_bytes\":" << execution.compact_readback_bytes
         << ",\"q4_decode\":" << execution.q4_decode_projection_count
         << ",\"q8_decode\":" << execution.q8_decode_projection_count
         << ",\"q8_width_two\":" << execution.q8_width_two_projection_count
         << ",\"q4_small_batch\":" << execution.q4_small_batch_projection_count
         << ",\"q4_width_two\":" << execution.q4_width_two_projection_count
         << ",\"q4_exact_16\":" << execution.q4_exact_16_projection_count
         << ",\"q4_generic_tiled\":" << execution.q4_generic_tiled_projection_count
         << ",\"q6_decode\":" << execution.q6_decode_projection_count
         << ",\"q6_small_batch\":" << execution.q6_small_batch_projection_count
         << ",\"q6_width_two\":" << execution.q6_width_two_projection_count
         << ",\"q6_exact_16\":" << execution.q6_exact_16_projection_count
         << ",\"q6_generic_tiled\":" << execution.q6_generic_tiled_projection_count
         << ",\"f16\":" << execution.f16_projection_count
         << ",\"output_head_elisions\":" << execution.output_head_elision_count
         << ",\"output_elided_weight_bytes\":" << execution.output_elided_weight_bytes
         << ",\"attention_dispatches\":" << execution.decode_attention_dispatch_count
         << ",\"attention_partitions\":" << execution.decode_attention_partition_count
         << ",\"attention_tokens\":" << execution.decode_attention_token_count
         << ",\"fused_embedding_norm\":" << execution.fused_embedding_norm_count
         << ",\"direct_qkv_cache\":" << execution.direct_qkv_cache_count
         << ",\"dual_norm_bridge\":" << execution.dual_norm_bridge_count
         << ",\"fused_gate_up_gelu\":" << execution.fused_gate_up_gelu_count
         << ",\"scheduled_bytes_per_token\":{\"embedding\":" << bytes.embedding
         << ",\"attention_projections\":" << bytes.attention_projections
         << ",\"ffn_gate_up\":" << bytes.feed_forward_gate_up
         << ",\"ffn_down\":" << bytes.feed_forward_down
         << ",\"output_head\":" << bytes.output_head
         << ",\"norms\":" << bytes.norms
         << ",\"total\":" << bytes.total
         << "},\"scheduled_bytes_total\":" << total_scheduled_bytes << "} ]";
    return json.str();
}

std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens
) {
    return generate_text_from_gguf_bytes(bytes, prompt, max_new_tokens, GenerateOptions {});
}

std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens,
    const GenerateOptions& options
) {
    const auto model_view = read_gguf_model_view(bytes);
    return generate_text_from_gguf_model_view(model_view, prompt, max_new_tokens, options);
}

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    return run_generate(argc, argv, std::cin, output, error);
}

int run_generate_impl(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error,
    std::optional<std::uint64_t> process_started_ns
) {
    static_cast<void>(process_started_ns);
    if (argc < 2) {
        write_generate_usage(error);
        error << "error: missing required argument\n";
        return 1;
    }

    if (argv_requests_chat(argc, argv)) {
        ParsedChatCommand command;
        if (!parse_chat_command(argc, argv, command, error)) {
            return 1;
        }

        try {
            return run_generate_chat(argv[1], command, input, output);
        } catch (const GGUFReadError& read_error) {
            error << "error: invalid GGUF: " << read_error.what() << '\n';
        } catch (const EngineModelError& model_error) {
            error << "error: invalid model: " << model_error.what() << '\n';
        } catch (const TokenizerError& tokenizer_error) {
            error << "error: tokenizer: " << tokenizer_error.what() << '\n';
        } catch (const BackendError& backend_error) {
            error << "error: backend: " << backend_error.what() << '\n';
        } catch (const TensorError& tensor_error) {
            error << "error: tensor: " << tensor_error.what() << '\n';
        } catch (const std::exception& unexpected_error) {
            error << "error: " << unexpected_error.what() << '\n';
        }

        return 1;
    }

    if (argc < 4) {
        write_generate_usage(error);
        error << "error: missing required argument\n";
        return 1;
    }

    std::size_t max_new_tokens = 0;
    if (!parse_positive_size(argv[3], max_new_tokens)) {
        write_generate_usage(error);
        error << "error: max-new-tokens must be a positive integer\n";
        return 1;
    }

    ParsedGenerateCommand command;
    if (!parse_generate_options(argc, argv, 4, command, error)) {
        return 1;
    }

    try {
        const auto file_model = read_file_backed_gguf_model(argv[1]);
        output << generate_text_from_gguf_model_view(file_model.view(), argv[2], max_new_tokens, command.options)
               << '\n';
        return 0;
    } catch (const GGUFReadError& read_error) {
        error << "error: invalid GGUF: " << read_error.what() << '\n';
    } catch (const EngineModelError& model_error) {
        error << "error: invalid model: " << model_error.what() << '\n';
    } catch (const TokenizerError& tokenizer_error) {
        error << "error: tokenizer: " << tokenizer_error.what() << '\n';
    } catch (const BackendError& backend_error) {
        error << "error: backend: " << backend_error.what() << '\n';
    } catch (const TensorError& tensor_error) {
        error << "error: tensor: " << tensor_error.what() << '\n';
    } catch (const std::exception& unexpected_error) {
        error << "error: " << unexpected_error.what() << '\n';
    }

    return 1;
}

int run_generate(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error
) {
    return run_generate_impl(argc, argv, input, output, error, std::nullopt);
}

int run_generate(
    int argc,
    const char* const* argv,
    std::ostream& output,
    std::ostream& error,
    std::uint64_t process_started_ns
) {
    return run_generate_impl(argc, argv, std::cin, output, error, process_started_ns);
}

}  // namespace local_gguf_engine::frontends::cli
