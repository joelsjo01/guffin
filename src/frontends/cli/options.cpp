#include "options.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string_view>

namespace local_gguf_engine::frontends::cli {
namespace {
using runtime::BackendKind;

void write_usage(std::ostream& error) {
    error << "usage: guffin -m MODEL [options]\n"
          << "Run 'guffin --help' for examples and options.\n";
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

} // namespace

void write_help(std::ostream& output, bool advanced) {
    output << "Guffin - native GGUF chat on Metal\n\n"
           << "Usage: guffin -m MODEL [options]\n"
           << "       guffin inspect MODEL\n\n"
           << "Chat is the default. Give a prompt for one answer and exit.\n"
           << "Chat shows prompt/generation rates and internal TTFT after each reply.\n\n"
           << "  -m, --model FILE          Model GGUF (or use positional MODEL)\n"
           << "  -p, --prompt TEXT         Answer TEXT once, then exit\n"
           << "  -n, --max-new-tokens N    Optional reply token limit\n"
           << "  -c, --context-size N      Context capacity in tokens\n"
           << "      --draft-model FILE    Enable lossless speculative decoding\n"
           << "      --temperature T       Sampling temperature (default: 0.8)\n"
           << "      --greedy              Select the most likely token\n"
           << "      --seed N              Sampling seed (default: random)\n"
           << "      --stats               Also show timing for one-shot/piped output\n"
           << "  -h, --help                Show this help\n"
           << "      --help-all            Include advanced options\n"
           << "      --version             Show build identity\n\n"
           << "Examples:\n"
           << "  guffin -m models/gemma-3-12b-it-Q4_K_M.gguf\n"
           << "  guffin -m MODEL --draft-model DRAFT\n"
           << "  guffin -m MODEL -p \"Explain unified memory\" -n 128\n";
    if (!advanced) return;
    output << "\nSampling:\n"
           << "  --top-k N                 Candidate limit (default: 40)\n"
           << "  --top-p P                 Nucleus threshold (default: 0.95)\n"
           << "  --min-p P                 Relative threshold (default: 0.05)\n"
           << "  --repeat-penalty R        Repetition penalty (default: 1.0)\n"
           << "  --stop TEXT               Stop at TEXT; may be repeated\n"
           << "\nContext and execution:\n"
           << "  --context-shift           Drop oldest complete turns when full\n"
           << "  --no-context-shift        Stop when full (default)\n"
           << "  --kv-cache auto|f16|f32   Target KV storage (auto selects F16)\n"
           << "  --output-head TYPE        Override the model's head with f16|q8_0|q6_k\n"
           << "  --no-speculative          Disable all speculation\n"
           << "  --retrieval               Enable experimental suffix retrieval\n"
           << "  --no-retrieval            Disable suffix retrieval\n"
           << "\nDraft tuning:\n"
           << "  --draft-horizon N         Proposed tokens (default: 1, maximum: 23)\n"
           << "  --draft-temperature T     Override draft temperature\n"
           << "  --draft-top-k N           Override draft top-k (0 disables it)\n"
           << "  --draft-top-p P           Override draft nucleus threshold\n"
           << "  --draft-min-p P           Override draft relative threshold\n"
           << "  --draft-repeat-penalty R  Override draft repetition penalty\n"
           << "  --speculative-sampling lossless\n"
           << "  --speculative-schedule full|conditional  Default: full\n"
           << "\nDiagnostics:\n"
           << "  --decode-diagnostics      Timings, raw counts and decode attribution\n"
           << "  --metal-capture PATH      Capture a decode with diagnostics enabled\n"
           << "  --interactive             Show chat UI with redirected streams\n";
}

[[nodiscard]] bool parse_command(
    int argc,
    const char* const* argv,
    CommandOptions& command,
    std::ostream& error
) {
    bool kv_precision_requested = false;
    bool greedy_requested = false;
    bool temperature_requested = false;
    bool sampled_modifier_requested = false;
    bool draft_horizon_requested = false;
    bool schedule_requested = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help" || option == "-h" || option == "--help-all") {
            command.help = true;
            command.help_all = option == "--help-all";
            return true;
        }
        if (option == "--version") {
            command.version = true;
            return true;
        }
        if (option == "--model" || option == "-m") {
            if (command.model_path || index + 1 >= argc
                || std::string_view(argv[index + 1]).empty()
                || std::string_view(argv[index + 1]).starts_with("-")) {
                error << "error: " << option << " requires one model GGUF file\n";
                return false;
            }
            command.model_path = argv[++index];
            continue;
        }
        if (option == "--prompt" || option == "-p") {
            if (command.prompt || index + 1 >= argc) {
                error << "error: " << option << " requires one prompt\n";
                return false;
            }
            command.prompt = argv[++index];
            continue;
        }
        if (option == "--interactive") {
            command.interactive = true;
            continue;
        }
        if (option == "--stats") {
            command.stats = true;
            continue;
        }
        if (!option.starts_with("-")) {
            if (option.empty() || command.model_path) {
                error << "error: unexpected argument '" << option << "'\n";
                return false;
            }
            command.model_path = argv[index];
            continue;
        }
        if (option == "--greedy") {
            greedy_requested = true;
            continue;
        }
        if (option == "--context-shift") {
            command.session.context_shift = true;
            continue;
        }
        if (option == "--decode-diagnostics") {
            command.decode_diagnostics = true;
            continue;
        }
        if (option == "--no-speculative") {
            command.session.speculative_decode = false;
            continue;
        }
        if (option == "--retrieval") {
            command.session.retrieval_decode = true;
            continue;
        }
        if (option == "--no-retrieval") {
            command.session.retrieval_decode = false;
            continue;
        }
        if (option == "--speculative-sampling") {
            if (command.session.speculative_sampling || index + 1 >= argc) {
                error << "error: --speculative-sampling requires one exact-match or lossless value\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value != "exact-match" && value != "lossless") {
                error << "error: --speculative-sampling requires exact-match or lossless\n";
                return false;
            }
            command.session.speculative_sampling = value == "lossless"
                ? engine::SpeculativeSamplingMode::lossless : engine::SpeculativeSamplingMode::exact_match;
            continue;
        }
        if (option == "--speculative-schedule") {
            if (schedule_requested || index + 1 >= argc) {
                error << "error: --speculative-schedule requires exactly one value\n";
                return false;
            }
            schedule_requested = true;
            const std::string_view value(argv[++index]);
            if (value != "full" && value != "conditional") {
                error << "error: invalid value for --speculative-schedule\n";
                return false;
            }
            command.session.standalone_execution.conditional_verification = value == "conditional";
            continue;
        }
        if (option == "--draft-model") {
            if (command.session.standalone_draft_model || index + 1 >= argc
                || std::string_view(argv[index + 1]).empty()
                || std::string_view(argv[index + 1]).starts_with("--")) {
                error << "error: --draft-model requires one native draft GGUF file\n";
                return false;
            }
            command.session.standalone_draft_model = argv[++index];
            continue;
        }
        if (option == "--draft-horizon") {
            std::size_t horizon = 0;
            if (draft_horizon_requested || index + 1 >= argc
                || !parse_positive_size(argv[index + 1], horizon)
                || horizon > engine::external_speculation_max_horizon) {
                error << "error: --draft-horizon requires one integer in [1, 23]\n";
                return false;
            }
            draft_horizon_requested = true;
            command.session.standalone_execution.horizon = static_cast<std::uint32_t>(horizon);
            ++index;
            continue;
        }
        if (option == "--draft-temperature" || option == "--draft-top-p"
            || option == "--draft-min-p" || option == "--draft-repeat-penalty") {
            auto& sampling = command.session.standalone_execution.draft_sampling;
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
            auto& destination = command.session.standalone_execution.draft_sampling.top_k;
            if (destination || index + 1 >= argc || !parse_u64(argv[index + 1], value)
                || value > std::numeric_limits<std::uint32_t>::max()) {
                error << "error: --draft-top-k requires one unsigned 32-bit integer (0 disables top-k)\n";
                return false;
            }
            destination = static_cast<std::uint32_t>(value);
            ++index;
            continue;
        }
        if (option == "--context-size" || option == "-c") {
            std::uint64_t value = 0;
            if (command.session.context_length != 0U || index + 1 >= argc
                || !parse_u64(argv[index + 1], value) || value == 0U
                || value > std::numeric_limits<std::uint32_t>::max()) {
                error << "error: --context-size requires one positive 32-bit token count\n";
                return false;
            }
            command.session.context_length = static_cast<std::uint32_t>(value);
            ++index;
            continue;
        }
        if (option == "--kv-cache") {
            if (kv_precision_requested || index + 1 >= argc) {
                error << "error: --kv-cache requires one of auto, f32, or f16, exactly once\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value == "auto") command.session.kv_precision = engine::GemmaKVPrecision::automatic;
            else if (value == "f32") command.session.kv_precision = engine::GemmaKVPrecision::f32;
            else if (value == "f16") command.session.kv_precision = engine::GemmaKVPrecision::f16;
            else {
                error << "error: --kv-cache must be auto, f32, or f16\n";
                return false;
            }
            kv_precision_requested = true;
            continue;
        }
        if (option == "--output-head") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: --output-head requires f16, q8_0, or q6_k\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value == "f16") command.session.output_head_policy = engine::GemmaOutputHeadPolicy::f16;
            else if (value == "q8_0") command.session.output_head_policy = engine::GemmaOutputHeadPolicy::q8_0;
            else if (value == "q6_k") command.session.output_head_policy = engine::GemmaOutputHeadPolicy::q6_k;
            else {
                write_usage(error);
                error << "error: --output-head must be f16, q8_0, or q6_k\n";
                return false;
            }
            continue;
        }
        if (option == "--metal-capture") {
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty()) {
                write_usage(error);
                error << "error: --metal-capture requires a path\n";
                return false;
            }
            command.metal_capture_path = argv[++index];
            continue;
        }
        if (option == "--no-context-shift") {
            command.session.context_shift = false;
            continue;
        }
        if (option == "--max-new-tokens" || option == "-n") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: --max-new-tokens requires a value\n";
                return false;
            }
            if (!parse_positive_size(argv[index + 1], command.session.max_new_tokens)) {
                write_usage(error);
                error << "error: max-new-tokens must be a positive integer\n";
                return false;
            }
            ++index;
            continue;
        }

        if (option == "--backend") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: --backend requires a value\n";
                return false;
            }

            BackendKind backend = BackendKind::cpu;
            if (!parse_backend_kind(argv[index + 1], backend)) {
                write_usage(error);
                error << "error: --backend must be cpu or metal\n";
                return false;
            }

            command.session.backend = backend;
            ++index;
            continue;
        }

        if (
            option == "--temperature" || option == "--top-p" || option == "--min-p"
            || option == "--repeat-penalty"
        ) {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: " << option << " requires a value\n";
                return false;
            }
            float value = 0.0F;
            if (!parse_float(argv[index + 1], value)) {
                write_usage(error);
                error << "error: " << option << " requires a finite number\n";
                return false;
            }
            if (option == "--temperature") {
                command.session.sampler.temperature = value;
                temperature_requested = true;
            } else if (option == "--top-p") {
                command.session.sampler.top_p = value;
                sampled_modifier_requested = true;
            } else if (option == "--min-p") {
                command.session.sampler.min_p = value;
                sampled_modifier_requested = true;
            } else {
                command.session.sampler.repetition_penalty = value;
                sampled_modifier_requested = true;
            }
            ++index;
            continue;
        }

        if (option == "--top-k") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: --top-k requires a value\n";
                return false;
            }
            std::size_t value = 0;
            if (!parse_positive_size(argv[index + 1], value)) {
                write_usage(error);
                error << "error: --top-k must be a positive integer\n";
                return false;
            }
            command.session.sampler.top_k = value;
            sampled_modifier_requested = true;
            ++index;
            continue;
        }

        if (option == "--seed") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: --seed requires a value\n";
                return false;
            }
            const auto seed_text = std::string_view(argv[index + 1]);
            if (seed_text == "-1") {
                command.session.sampler.seed = engine::TokenSamplerOptions::random_seed;
            } else if (!parse_u64(seed_text, command.session.sampler.seed)) {
                write_usage(error);
                error << "error: --seed must be -1 or an unsigned integer\n";
                return false;
            }
            sampled_modifier_requested = true;
            ++index;
            continue;
        }

        if (option == "--reverse-prompt" || option == "--reverse-stop" || option == "--stop" || option == "-r") {
            if (index + 1 >= argc) {
                write_usage(error);
                error << "error: " << option << " requires a value\n";
                return false;
            }

            command.session.reverse_stop_strings.push_back(argv[index + 1]);
            ++index;
            continue;
        }

        write_usage(error);
        error << "error: unknown option '" << option << "'\n";
        return false;
    }

    if (!command.model_path) {
        write_usage(error);
        error << "error: missing model GGUF file\n";
        return false;
    }
    if (command.prompt && command.interactive) {
        error << "error: --prompt and --interactive cannot be combined\n";
        return false;
    }
    if (command.session.standalone_draft_model
        && (!command.session.speculative_decode
            || (command.session.speculative_sampling
                && command.session.speculative_sampling != engine::SpeculativeSamplingMode::lossless)
            || command.session.backend != BackendKind::metal)) {
        error << "error: --draft-model requires Metal and lossless speculation,"
              << " and conflicts with --no-speculative\n";
        return false;
    }
    if (draft_horizon_requested && !command.session.standalone_draft_model) {
        error << "error: --draft-horizon requires --draft-model\n";
        return false;
    }
    if (command.session.standalone_execution.draft_sampling != engine::StandaloneDraftSamplingOptions{}
        && !command.session.standalone_draft_model) {
        error << "error: draft sampling overrides require --draft-model and lossless speculation\n";
        return false;
    }
    const bool has_draft = command.session.standalone_draft_model.has_value();
    if (command.session.speculative_sampling && (!has_draft || !command.session.speculative_decode)) {
        error << "error: --speculative-sampling requires a draft configuration and enabled speculation\n";
        return false;
    }
    if (schedule_requested
        && (!has_draft || !command.session.speculative_decode
            || (command.session.speculative_sampling
                && command.session.speculative_sampling != engine::SpeculativeSamplingMode::lossless))) {
        error << "error: speculative execution options require a draft configuration and lossless sampling\n";
        return false;
    }
    if (greedy_requested) {
        if (temperature_requested && command.session.sampler.temperature > 0.0F) {
            error << "error: --greedy cannot be combined with positive --temperature\n";
            return false;
        }
        command.session.sampler.temperature = 0.0F;
    }
    if (command.session.sampler.temperature < 0.0F) {
        error << "error: --temperature must be nonnegative\n";
        return false;
    }
    if (command.session.sampler.temperature == 0.0F && sampled_modifier_requested) {
        error << "error: sampling modifiers require --temperature greater than zero\n";
        return false;
    }
    if (command.session.sampler.top_p <= 0.0F || command.session.sampler.top_p > 1.0F) {
        error << "error: --top-p must be in (0, 1]\n";
        return false;
    }
    if (command.session.sampler.min_p < 0.0F || command.session.sampler.min_p > 1.0F) {
        error << "error: --min-p must be in [0, 1]\n";
        return false;
    }
    if (command.session.sampler.repetition_penalty <= 0.0F) {
        error << "error: --repeat-penalty must be positive\n";
        return false;
    }
    if (command.metal_capture_path.has_value()
        && (!command.decode_diagnostics || command.session.backend != BackendKind::metal)) {
        error << "error: --metal-capture requires --decode-diagnostics and --backend metal\n";
        return false;
    }

    return true;
}

} // namespace local_gguf_engine::frontends::cli
