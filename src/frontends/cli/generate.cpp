#include "local_gguf_engine/frontends/cli/generate.hpp"

#include "local_gguf_engine/benchmarks/conversation_benchmark.hpp"
#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
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
using benchmarks::ConversationBenchmarkRunResult;
using benchmarks::ConversationBenchmarkRunnerOptions;
using benchmarks::ConversationBenchmarkStatistics;
using benchmarks::benchmark_rate;
using benchmarks::run_conversation_benchmark;
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
          << "       local-gguf-generate <model.gguf> --bench"
          << " [--warmup N] [--repetitions N] [--output human|jsonl] [--backend metal]"
          << '\n'
          << "       local-gguf-generate <model.gguf> --chat [--max-new-tokens N]"
          << " [--greedy | --temperature T [--top-k K] [--top-p P] [--min-p P]"
          << " [--repeat-penalty R] [--seed S]] [--backend cpu|metal]"
          << " [--reverse-prompt TEXT] [--context-shift]"
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

[[nodiscard]] bool parse_nonnegative_size(std::string_view text, std::size_t& value) noexcept {
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

enum class BenchmarkOutputFormat { human, jsonl };

struct ParsedGenerateCommand {
    GenerateOptions options;
    bool benchmark = false;
    bool benchmark_option_seen = false;
    bool benchmark_sampling_option_seen = false;
    std::size_t warmup_count = 1U;
    std::size_t repetition_count = 5U;
    BenchmarkOutputFormat output_format = BenchmarkOutputFormat::human;
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
        } else if (option == "--bench") {
            command.benchmark = true;
        } else if (option == "--warmup" || option == "--repetitions") {
            command.benchmark_option_seen = true;
            if (index + 1 >= argc) {
                error << "error: " << option << " requires a value\n";
                return false;
            }
            std::size_t value = 0;
            const auto valid = option == "--warmup"
                ? parse_nonnegative_size(argv[index + 1], value)
                : parse_positive_size(argv[index + 1], value);
            if (!valid) {
                error << "error: " << option << " requires "
                      << (option == "--warmup" ? "a nonnegative" : "a positive") << " integer\n";
                return false;
            }
            if (option == "--warmup") command.warmup_count = value;
            else command.repetition_count = value;
            ++index;
        } else if (option == "--output") {
            command.benchmark_option_seen = true;
            if (index + 1 >= argc) {
                error << "error: --output requires a value\n";
                return false;
            }
            const std::string_view value(argv[++index]);
            if (value == "human") command.output_format = BenchmarkOutputFormat::human;
            else if (value == "jsonl") command.output_format = BenchmarkOutputFormat::jsonl;
            else {
                error << "error: --output must be human or jsonl\n";
                return false;
            }
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
            command.benchmark_option_seen = true;
            command.benchmark_sampling_option_seen = true;
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
            command.benchmark_option_seen = true;
            command.benchmark_sampling_option_seen = true;
            if (index + 1 >= argc || !parse_positive_size(argv[index + 1], command.sampled_sampler.top_k)) {
                error << "error: --top-k requires a positive integer\n";
                return false;
            }
            ++index;
        } else if (option == "--seed") {
            command.benchmark_option_seen = true;
            command.benchmark_sampling_option_seen = true;
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
    if (!command.benchmark && command.benchmark_option_seen) {
        error << "error: benchmark options require --bench\n";
        return false;
    }
    if (command.benchmark && command.benchmark_sampling_option_seen) {
        error << "error: primary conversation benchmark is greedy; sampling options are unsupported\n";
        return false;
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

[[nodiscard]] bool argv_has_option(int argc, const char* const* argv, std::string_view wanted) noexcept {
    for (int index = 2; index < argc; ++index) {
        if (std::string_view(argv[index]) == wanted) return true;
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
    bool context_shift = false;
    bool greedy_requested = false;
    bool temperature_requested = false;
    bool sampled_modifier_requested = false;
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
    return "local-gguf-generate supports Gemma 2 generation only; architecture '"
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
    if (model.architecture() != ModelArchitecture::gemma2) {
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
            .context_shift = command.context_shift,
            .reverse_stop_strings = command.reverse_stop_strings,
            .sampler = command.sampler,
        }
    );

    output << "local-gguf-engine chat\n";
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
        if (turn->rebuilt_context) {
            output << "context: dropped " << turn->trimmed_turn_count
                   << " oldest turn(s) and rebuilt retained KV\n";
        }
        if (turn->interrupted && chat_interrupt_count() > turn_interrupt_count + 1) {
            return 130;
        }
    }
}

[[nodiscard]] std::string_view backend_name(BackendKind backend) noexcept {
    return backend == BackendKind::metal ? "metal" : "cpu";
}

[[nodiscard]] std::string_view selection_mode_name(engine::Gemma2TokenSelectionMode mode) noexcept {
    return mode == engine::Gemma2TokenSelectionMode::greedy ? "greedy" : "sampled";
}

[[nodiscard]] std::string_view stop_reason_name(engine::Gemma2GenerationStopReason reason) noexcept {
    switch (reason) {
        case engine::Gemma2GenerationStopReason::token_limit: return "token_limit";
        case engine::Gemma2GenerationStopReason::stop_token: return "stop_token";
        case engine::Gemma2GenerationStopReason::callback_stop: return "callback_stop";
        case engine::Gemma2GenerationStopReason::reverse_stop: return "reverse_stop";
        case engine::Gemma2GenerationStopReason::interrupted: return "interrupted";
    }
    return "unknown";
}

[[nodiscard]] std::string json_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        switch (character) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (byte < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(byte) << std::dec << std::setfill(' ');
                } else {
                    out << character;
                }
        }
    }
    out << '"';
    return out.str();
}

[[nodiscard]] std::string json_optional_u64(std::optional<std::uint64_t> value) {
    return value.has_value() ? std::to_string(*value) : "null";
}

[[nodiscard]] std::uint64_t benchmark_duration_between(
    std::uint64_t start,
    std::uint64_t finish
) {
    if (finish < start) throw std::runtime_error("benchmark event times are not monotonic");
    return finish - start;
}

[[nodiscard]] std::string json_optional_double(std::optional<double> value) {
    if (!value.has_value() || !std::isfinite(*value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << *value;
    return out.str();
}

void write_json_statistics(std::ostream& output, const ConversationBenchmarkStatistics& stats) {
    output << "{\"count\":" << stats.sample_count
           << ",\"median\":" << json_optional_double(stats.median)
           << ",\"min\":" << json_optional_double(stats.minimum)
           << ",\"max\":" << json_optional_double(stats.maximum)
           << ",\"stddev\":" << json_optional_double(stats.sample_standard_deviation)
           << ",\"mad\":" << json_optional_double(stats.median_absolute_deviation) << '}';
}

void write_benchmark_human(std::ostream& output, const ConversationBenchmarkRunResult& result) {
    output << "conversation benchmark schema=" << result.setup.identity.schema_version
           << " scenario=" << result.setup.identity.scenario_version
           << " model=" << result.setup.identity.model_label
           << " backend=" << backend_name(result.setup.identity.backend) << '\n';
    output << "setup load_ns=" << json_optional_u64(result.setup.model_load_duration_ns)
           << " plan_ns=" << json_optional_u64(result.setup.plan_creation_duration_ns)
           << " process_start_to_ready_ns="
           << json_optional_u64(result.setup.process_start_to_ready_duration_ns)
           << " weights_bytes=" << result.setup.resident_weight_byte_count
           << " kv_bytes=" << result.setup.resident_kv_cache_byte_count
           << " scratch_bytes=" << result.setup.resident_scratch_byte_count
           << " metal_allocated_bytes=" << json_optional_u64(result.setup.metal_current_allocated_bytes)
           << " footprint_bytes=" << json_optional_u64(result.setup.process_physical_footprint_bytes) << '\n';
    for (const auto& record : result.turns) {
        const auto& turn = record.result;
        output << (record.warmup ? "warmup" : "repetition")
               << " index=" << record.repetition_index
               << " turn=" << record.turn_index
               << " mode=" << selection_mode_name(turn.metrics.selection_mode)
               << " seed=" << record.sampler.seed
               << " prompt_tokens=" << turn.metrics.generation.newly_evaluated_prompt_token_count
               << " prompt_tps=" << json_optional_double(benchmark_rate(
                    turn.metrics.generation.newly_evaluated_prompt_token_count,
                    turn.metrics.generation.prompt_evaluation_duration_ns))
               << " decode_tokens=" << turn.metrics.generation.decode_token_count
               << " decode_tps=" << json_optional_double(benchmark_rate(
                    turn.metrics.generation.decode_token_count, turn.metrics.generation.decode_duration_ns))
               << " frontend_ttft_ns=" << json_optional_u64(turn.metrics.frontend_complete_ttft_ns)
               << " total_ns=" << turn.metrics.total_turn_duration_ns
               << " stop=" << stop_reason_name(turn.metrics.generation.stop_reason)
               << " reused=" << turn.metrics.reused_cached_token_count
               << " dispatches=" << turn.metrics.prefill_execution.compute_dispatch_count
                    + turn.metrics.decode_execution.compute_dispatch_count << '\n';
    }
    const auto write_stat = [&](std::string_view scope, std::string_view metric, const auto& stats) {
        output << "summary scope=" << scope << " metric=" << metric
               << " count=" << stats.sample_count
               << " median=" << json_optional_double(stats.median)
               << " min=" << json_optional_double(stats.minimum)
               << " max=" << json_optional_double(stats.maximum)
               << " stddev=" << json_optional_double(stats.sample_standard_deviation)
               << " mad=" << json_optional_double(stats.median_absolute_deviation) << '\n';
    };
    for (const auto& summary : result.summaries) {
        const auto scope = summary.turn_index.has_value()
            ? "turn_" + std::to_string(*summary.turn_index)
            : std::string("conversation");
        output << "summary scope=" << scope << " exact_selected_tokens_stable="
               << (summary.exact_selected_tokens_stable ? "true" : "false") << '\n';
        write_stat(scope, "frontend_complete_ttft_ns", summary.frontend_complete_ttft_ns);
        write_stat(scope, "complete_duration_ns", summary.complete_duration_ns);
        write_stat(scope, "prompt_tps", summary.prompt_tokens_per_second);
        write_stat(scope, "decode_tps", summary.decode_tokens_per_second);
        write_stat(scope, "visible_tps", summary.visible_tokens_per_second);
    }
}

void write_benchmark_jsonl(std::ostream& output, const ConversationBenchmarkRunResult& result) {
    const auto units = "\"units\":{\"duration\":\"ns\",\"throughput\":\"tokens_per_second\",\"memory\":\"bytes\"}";
    const auto& identity = result.setup.identity;
    const auto write_identity = [&](const auto& value) {
        output << ",\"engine\":" << json_string(value.engine)
               << ",\"engine_build\":" << json_string(value.engine_build)
               << ",\"model\":" << json_string(value.model_label)
               << ",\"model_identity\":" << json_string(value.model_identity)
               << ",\"backend\":" << json_string(backend_name(value.backend))
               << ",\"os\":" << json_string(value.operating_system)
               << ",\"hardware\":" << json_string(value.hardware)
               << ",\"frontend\":" << json_string(value.frontend);
    };
    const auto write_sampler = [&](const auto& sampler) {
        output << ",\"mode\":" << json_string(engine::TokenSampler(sampler).is_greedy() ? "greedy" : "sampled")
               << ",\"seed\":" << sampler.seed
               << ",\"temperature\":" << sampler.temperature
               << ",\"top_k\":" << sampler.top_k
               << ",\"top_p\":" << sampler.top_p
               << ",\"min_p\":" << sampler.min_p
               << ",\"repeat_penalty\":" << sampler.repetition_penalty;
    };
    output << "{\"schema_version\":" << identity.schema_version
           << ",\"scenario_version\":" << identity.scenario_version
           << ",\"record_kind\":\"setup\"";
    write_identity(identity);
    write_sampler(result.turns.front().sampler);
    output << ",\"model_load_ns\":" << json_optional_u64(result.setup.model_load_duration_ns)
           << ",\"plan_creation_ns\":" << json_optional_u64(result.setup.plan_creation_duration_ns)
           << ",\"process_start_to_ready_ns\":"
           << json_optional_u64(result.setup.process_start_to_ready_duration_ns)
           << ",\"resident_weight_bytes\":" << result.setup.resident_weight_byte_count
           << ",\"resident_kv_bytes\":" << result.setup.resident_kv_cache_byte_count
           << ",\"resident_scratch_bytes\":" << result.setup.resident_scratch_byte_count
           << ",\"metal_current_allocated_bytes\":" << json_optional_u64(result.setup.metal_current_allocated_bytes)
           << ",\"process_physical_footprint_bytes\":" << json_optional_u64(result.setup.process_physical_footprint_bytes)
           << ",\"process_physical_footprint_peak_bytes\":" << json_optional_u64(result.setup.process_physical_footprint_peak_bytes)
           << ',' << units << "}\n";
    const auto write_ids = [&](std::span<const std::uint32_t> ids) {
        output << '[';
        for (std::size_t index = 0; index < ids.size(); ++index) {
            if (index != 0U) output << ',';
            output << ids[index];
        }
        output << ']';
    };
    const auto write_execution = [&](const auto& stats) {
        output << "{\"command_buffers\":" << stats.command_buffer_count
               << ",\"commits\":" << stats.commit_count
               << ",\"waits\":" << stats.synchronous_wait_count
               << ",\"compute_encoders\":" << stats.compute_encoder_count
               << ",\"compute_dispatches\":" << stats.compute_dispatch_count
               << ",\"blit_encoders\":" << stats.blit_encoder_count
               << ",\"transient_allocations\":" << stats.transient_buffer_allocation_count
               << ",\"selector_dispatches\":" << stats.selector_dispatch_count
               << ",\"compact_readback_bytes\":" << stats.compact_readback_bytes
               << ",\"gpu_duration_ns\":" << stats.gpu_duration_ns
               << ",\"q4_decode\":" << stats.q4_decode_projection_count
               << ",\"q4_small_batch\":" << stats.q4_small_batch_projection_count
               << ",\"q4_exact_16\":" << stats.q4_exact_16_projection_count
               << ",\"q4_generic_tiled\":" << stats.q4_generic_tiled_projection_count
               << ",\"q6_decode\":" << stats.q6_decode_projection_count
               << ",\"q6_small_batch\":" << stats.q6_small_batch_projection_count
               << ",\"q6_exact_16\":" << stats.q6_exact_16_projection_count
               << ",\"q6_generic_tiled\":" << stats.q6_generic_tiled_projection_count
               << ",\"f16_projection\":" << stats.f16_projection_count
               << ",\"decode_attention_dispatches\":" << stats.decode_attention_dispatch_count
               << ",\"decode_attention_partitions\":" << stats.decode_attention_partition_count
               << ",\"fused_embedding_norm\":" << stats.fused_embedding_norm_count
               << ",\"direct_qkv_cache\":" << stats.direct_qkv_cache_count
               << ",\"dual_norm_bridge\":" << stats.dual_norm_bridge_count
               << ",\"fused_gate_up_gelu\":" << stats.fused_gate_up_gelu_count << '}';
    };
    for (const auto& record : result.turns) {
        const auto& turn = record.result;
        const auto& metrics = turn.metrics;
        output << "{\"schema_version\":" << record.identity.schema_version
               << ",\"scenario_version\":" << record.identity.scenario_version
               << ",\"record_kind\":\"turn\"";
        write_identity(record.identity);
        output
               << ",\"warmup\":" << (record.warmup ? "true" : "false")
               << ",\"repetition_index\":" << record.repetition_index
               << ",\"turn_index\":" << record.turn_index
               << ",\"timing_eligible\":" << (record.timing_eligible ? "true" : "false")
               << ",\"memory_audit\":" << (record.memory_audit ? "true" : "false")
               << ",\"user_text\":" << json_string(record.user_text)
               << ",\"generated_token_limit\":" << record.generated_token_limit
               << ",\"selection_mode\":" << json_string(selection_mode_name(metrics.selection_mode));
        write_sampler(record.sampler);
        output << ",\"initial_context_length\":" << metrics.initial_cache_position
               << ",\"final_context_length\":" << turn.cached_token_count
               << ",\"reused_tokens\":" << metrics.reused_cached_token_count
               << ",\"trimmed_turns\":" << metrics.trimmed_turn_count
               << ",\"rebuilt_context\":" << (metrics.rebuilt_context ? "true" : "false")
               << ",\"session_preparation_ns\":" << benchmark_duration_between(
                    metrics.turn_accepted_ns, metrics.generation.prefill_started_ns)
               << ",\"prompt_evaluation_ns\":" << metrics.generation.prompt_evaluation_duration_ns
               << ",\"decode_duration_ns\":" << metrics.generation.decode_duration_ns
               << ",\"selection_ns\":" << metrics.generation.selection_duration_ns
               << ",\"first_token_selection_ns\":"
               << benchmark_duration_between(
                    metrics.generation.prefill_completed_ns, metrics.generation.first_token_selected_ns)
               << ",\"runtime_first_token_selected_ns\":" << metrics.generation.runtime_first_token_selected_ns
               << ",\"text_ready_ttft_ns\":" << json_optional_u64(metrics.text_ready_ttft_ns)
               << ",\"first_selected_to_visible_ready_ns\":";
        if (metrics.first_visible_ready_ns.has_value()) {
            output << benchmark_duration_between(
                metrics.generation.first_token_selected_ns, *metrics.first_visible_ready_ns);
        } else {
            output << "null";
        }
        output
               << ",\"frontend_complete_ttft_ns\":" << json_optional_u64(metrics.frontend_complete_ttft_ns)
               << ",\"visible_stream_interval_ns\":" << json_optional_u64(metrics.visible_stream_interval_ns)
               << ",\"total_ns\":" << metrics.total_turn_duration_ns
               << ",\"prompt_tps\":" << json_optional_double(benchmark_rate(
                    metrics.generation.newly_evaluated_prompt_token_count,
                    metrics.generation.prompt_evaluation_duration_ns))
               << ",\"decode_tps\":" << json_optional_double(benchmark_rate(
                    metrics.generation.decode_token_count, metrics.generation.decode_duration_ns))
               << ",\"visible_emissions\":" << metrics.visible_emission_count
               << ",\"stop_reason\":" << json_string(stop_reason_name(metrics.generation.stop_reason))
               << ",\"stop_token_id\":" << json_optional_u64(turn.stop_token_id)
               << ",\"interrupted\":" << (turn.interrupted ? "true" : "false")
               << ",\"final_text\":" << json_string(turn.text)
               << ",\"plan_revision\":" << record.plan_revision
               << ",\"plan_allocation_identities\":[";
        for (std::size_t index = 0; index < record.plan_allocation_identities.size(); ++index) {
            if (index != 0U) output << ',';
            output << record.plan_allocation_identities[index];
        }
        output << ']'
               << ",\"metal_current_allocated_bytes\":" << json_optional_u64(record.metal_current_allocated_bytes)
               << ",\"process_physical_footprint_bytes\":" << json_optional_u64(record.process_physical_footprint_bytes)
               << ",\"process_physical_footprint_peak_bytes\":" << json_optional_u64(record.process_physical_footprint_peak_bytes)
               << ",\"resident_weight_bytes\":" << metrics.resident_weight_byte_count
               << ",\"resident_kv_bytes\":" << metrics.resident_kv_cache_byte_count
               << ",\"resident_scratch_bytes\":" << metrics.resident_scratch_byte_count
               << ",\"configured_stop_token_ids\":"; write_ids(record.configured_stop_token_ids);
        output << ",\"configured_reverse_stop_strings\":[";
        for (std::size_t index = 0; index < record.configured_reverse_stop_strings.size(); ++index) {
            if (index != 0U) output << ',';
            output << json_string(record.configured_reverse_stop_strings[index]);
        }
        output << "]";
        output << ",\"prefill_execution\":"; write_execution(metrics.prefill_execution);
        output << ",\"decode_execution\":"; write_execution(metrics.decode_execution);
        output << ",\"rendered_prompt_token_ids\":"; write_ids(turn.rendered_prompt_token_ids);
        output << ",\"evaluated_suffix_token_ids\":"; write_ids(turn.evaluated_suffix_token_ids);
        output << ",\"selected_token_ids\":"; write_ids(turn.selected_token_ids);
        output << ",\"visible_token_ids\":"; write_ids(turn.visible_token_ids);
        output
               << ',' << units << "}\n";
    }
    for (const auto& conversation : result.conversations) {
        output << "{\"schema_version\":" << conversation.identity.schema_version
               << ",\"scenario_version\":" << conversation.identity.scenario_version
               << ",\"record_kind\":\"conversation\"";
        write_identity(conversation.identity);
        write_sampler(conversation.sampler);
        output << ",\"warmup\":"
               << (conversation.warmup ? "true" : "false")
               << ",\"repetition_index\":" << conversation.repetition_index
               << ",\"turn_index\":null"
               << ",\"timing_eligible\":" << (conversation.timing_eligible ? "true" : "false")
               << ",\"memory_audit\":" << (conversation.memory_audit ? "true" : "false")
               << ",\"complete_conversation_ns\":" << conversation.complete_conversation_duration_ns
               << ",\"per_turn_frontend_complete_ttft_ns\":[";
        for (std::size_t index = 0; index < conversation.per_turn_frontend_complete_ttft_ns.size(); ++index) {
            if (index != 0U) output << ',';
            output << json_optional_u64(conversation.per_turn_frontend_complete_ttft_ns[index]);
        }
        output << ']'
               << ',' << units << "}\n";
    }
    for (const auto& summary : result.summaries) {
        output << "{\"schema_version\":" << summary.identity.schema_version
               << ",\"scenario_version\":" << summary.identity.scenario_version
               << ",\"record_kind\":\"summary\"";
        write_identity(summary.identity);
        write_sampler(summary.sampler);
        output << ",\"warmup\":false,\"repetition_index\":null,\"turn_index\":"
               << (summary.turn_index.has_value() ? std::to_string(*summary.turn_index) : "null")
               << ",\"exact_selected_tokens_stable\":"
               << (summary.exact_selected_tokens_stable ? "true" : "false")
               << ",\"exact_rendered_prompts_stable\":"
               << (summary.exact_rendered_prompts_stable ? "true" : "false")
               << ",\"workload_stable\":" << (summary.workload_stable ? "true" : "false")
               << ",\"frontend_complete_ttft_ns\":"; write_json_statistics(output, summary.frontend_complete_ttft_ns);
        output << ",\"prompt_tps\":"; write_json_statistics(output, summary.prompt_tokens_per_second);
        output << ",\"decode_tps\":"; write_json_statistics(output, summary.decode_tokens_per_second);
        output << ",\"visible_tps\":"; write_json_statistics(output, summary.visible_tokens_per_second);
        output << ",\"complete_duration_ns\":"; write_json_statistics(output, summary.complete_duration_ns);
        output << ',' << units << "}\n";
    }
    output << "{\"schema_version\":" << identity.schema_version
           << ",\"scenario_version\":" << identity.scenario_version
           << ",\"record_kind\":\"validation\"";
    write_identity(identity);
    output << ",\"valid\":" << (result.valid ? "true" : "false") << ",\"errors\":[";
    for (std::size_t index = 0; index < result.validation_errors.size(); ++index) {
        if (index != 0U) output << ',';
        output << json_string(result.validation_errors[index]);
    }
    output << "]," << units << "}\n";
}

}  // namespace

std::string format_chat_timing_line(const engine::Gemma2ChatTurnMetrics& metrics) {
    return "[ Prompt: "
        + format_tokens_per_second(
            metrics.generation.newly_evaluated_prompt_token_count,
            metrics.generation.prompt_evaluation_duration_ns
        )
        + " | Generation: "
        + format_tokens_per_second(
            metrics.generation.decode_token_count,
            metrics.generation.decode_duration_ns
        )
        + " | Frontend TTFT: "
        + (metrics.frontend_complete_ttft_ns.has_value()
            ? std::to_string(*metrics.frontend_complete_ttft_ns) + " ns"
            : std::string("n/a"))
        + " ]";
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
    if (argc < 2) {
        write_generate_usage(error);
        error << "error: missing required argument\n";
        return 1;
    }

    if (argv_requests_chat(argc, argv)) {
        if (argv_has_option(argc, argv, "--bench")) {
            error << "error: --bench cannot be combined with --chat\n";
            return 1;
        }
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

    if (argv_has_option(argc, argv, "--bench")) {
        ParsedGenerateCommand command;
        if (!parse_generate_options(argc, argv, 2, command, error)) return 1;
        try {
            // Benchmark JSONL owns stdout. Use a real synchronous OS stream as
            // the frontend sink so callback-complete TTFT includes write/flush
            // work without mixing generated text into the record stream.
            std::ofstream frontend_sink("/dev/null", std::ios::binary);
            if (!frontend_sink) throw std::runtime_error("could not open benchmark frontend sink");
            const auto result = run_conversation_benchmark(ConversationBenchmarkRunnerOptions {
                .model_path = argv[1],
                .warmup_count = command.warmup_count,
                .repetition_count = command.repetition_count,
                .backend = command.options.backend,
                .sampler = TokenSamplerOptions {.seed = 42U},
                .frontend_stream_callback = [&](std::string_view delta) {
                    frontend_sink.write(delta.data(), static_cast<std::streamsize>(delta.size()));
                    frontend_sink.flush();
                    if (!frontend_sink) throw std::runtime_error("benchmark frontend write failed");
                },
                .frontend_name = "synchronous:/dev/null",
                .process_started_ns = process_started_ns,
            });
            if (command.output_format == BenchmarkOutputFormat::human) write_benchmark_human(output, result);
            else write_benchmark_jsonl(output, result);
            if (!result.valid) {
                error << "error: benchmark workload is not stable across measured repetitions\n";
                return 1;
            }
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
