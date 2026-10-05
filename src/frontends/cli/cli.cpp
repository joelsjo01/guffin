#include "local_gguf_engine/frontends/cli/cli.hpp"
#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"
#include "local_gguf_engine/frontends/cli/metrics.hpp"
#include "options.hpp"
#include "build_info.hpp"

#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/metal_trace_capture.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>

namespace local_gguf_engine::frontends::cli {
namespace {
using engine::EngineModelError;
using engine::GemmaChatSession;
using engine::GemmaChatContextError;
using engine::GemmaChatTurnOptions;
using model_format::GGUFReadError;
using model_format::read_file_backed_gguf_model;
using runtime::BackendError;
using tensor::TensorError;
using tokenizer::TokenizerError;
using tokenizer::format_generate_prompt_from_gguf_metadata;

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
    std::string& line,
    bool interactive
) {
    while (!line.empty() && line.back() == '\\') {
        line.pop_back();
        if (interactive) output << "... " << std::flush;
        std::string continuation;
        if (!std::getline(input, continuation)) {
            if (interactive) output << '\n';
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

int run_chat(
    const CommandOptions& command,
    std::istream& input,
    std::ostream& output,
    std::ostream& error,
    bool terminal
) {
    const bool interactive = !command.prompt && (terminal || command.interactive);
    // A terminal UI can contain status lines. Redirected stdout contains answers
    // only, so scripts can consume the same engine without filtering a transcript.
    auto& status = interactive ? output : error;
    const ScopedSignalHandler chat_interrupt_handler(SIGINT, handle_chat_interrupt);
    auto file_model = read_file_backed_gguf_model(*command.model_path);
    // Validate supported prompt formatting before allocating a Metal plan.
    static_cast<void>(format_generate_prompt_from_gguf_metadata(
        file_model.view().directory.metadata, command.prompt.value_or("")));
    auto session = GemmaChatSession(
        std::move(file_model),
        command.session
    );
    ScopedMetalTraceArm metal_trace(command.metal_capture_path);
    if (interactive) {
        output << "Guffin\n";
        if (command.session.standalone_draft_model) {
            output << "Speculative decoding: "
                   << command.session.standalone_draft_model->filename().string() << '\n';
        }
        output << "/help for commands, /exit to quit.\n\n";
    }

    const auto reply = [&](std::string_view text, bool regenerate) {
        const auto turn_interrupt_count = chat_interrupt_count();
        const auto turn_options = GemmaChatTurnOptions {
            .on_text_delta = [&](std::string_view delta) {
                output << delta;
                output.flush();
            },
            .should_stop = [&]() {
                return chat_interrupt_count() != turn_interrupt_count;
            },
        };
        const auto turn = regenerate
            ? session.regenerate_last_turn(turn_options)
            : session.generate_user_turn(text, turn_options);
        output << '\n';
        if (interactive || command.stats || command.decode_diagnostics) {
            status << format_chat_timing_line(turn.metrics) << '\n';
        }
        if (command.decode_diagnostics) {
            status << format_chat_timing_detail_line(turn.metrics) << '\n'
                   << format_chat_decode_attribution_line(turn.metrics) << '\n';
        }
        if (metal_trace.pending()) metal_trace.finish(status);
        if (turn.rebuilt_context && interactive) {
            status << "context: dropped " << turn.trimmed_turn_count << " oldest turn(s)\n";
        }
        // A stopped one-shot reply must be distinguishable from a completed
        // answer to scripts. In chat, one interrupt returns to the prompt.
        return turn.interrupted && (command.prompt || chat_interrupt_count() > turn_interrupt_count + 1)
            ? 130 : 0;
    };

    if (command.prompt) return reply(*command.prompt, false);

    std::string line;
    while (true) {
        if (interactive) {
            output << "> ";
            output.flush();
        }
        const auto prompt_interrupt_count = chat_interrupt_count();
        if (!std::getline(input, line)) {
            if (interactive) output << '\n';
            return 0;
        }
        if (chat_interrupt_count() != prompt_interrupt_count) {
            if (interactive) output << '\n';
            continue;
        }
        if (line == "/exit" || line == "/quit") return 0;
        if (line == "/help") {
            status << "/exit, /quit     Leave chat\n"
                   << "/clear, /reset   Clear conversation, keep the model loaded\n"
                   << "/regen           Regenerate the last reply\n"
                   << "/tokens          Show context usage\n"
                   << "/read PATH       Send a text file\n"
                   << "End a line with \\ to continue a multiline prompt.\n";
            continue;
        }
        if (line == "/clear" || line == "/reset") {
            if (line == "/clear") session.clear();
            else session.reset();
            if (interactive) output << (line == "/clear" ? "chat cleared\n" : "session reset\n");
            continue;
        }
        if (line == "/tokens") {
            status << "tokens: cached=" << session.cached_token_count()
                   << " remaining=" << session.remaining_context_tokens()
                   << " max-new-tokens=";
            if (session.max_new_tokens() == 0U) status << "unlimited";
            else status << session.max_new_tokens();
            status << '\n';
            continue;
        }
        bool regenerate = false;
        if (line == "/regen") {
            if (session.message_count() < 2U) {
                status << "error: nothing to regenerate\n";
                if (!interactive) return 1;
                continue;
            }
            regenerate = true;
        } else if (line == "/read" || line.starts_with("/read ") || line.starts_with("/read\t")) {
            try {
                line = read_prompt_file(trim_left(std::string_view(line).substr(5U)));
            } catch (const std::exception& read_error) {
                status << "error: " << read_error.what() << '\n';
                if (!interactive) return 1;
                continue;
            }
        } else if (starts_with_command_prefix(line)) {
            status << "error: unknown command '" << line << "'\n";
            if (!interactive) return 1;
            continue;
        } else {
            if (!read_multiline_input(input, output, line, interactive)) return 0;
        }
        if (line.empty() && !regenerate) continue;
        try {
            const int result = reply(line, regenerate);
            if (result != 0) return result;
        } catch (const GemmaChatContextError& context_error) {
            status << "error: context: " << context_error.what() << '\n';
            if (!interactive) return 1;
        }
    }
}

} // namespace

namespace {

int run_cli_impl(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error,
    bool interactive
) {
    if (argc < 2) {
        write_help(output, false);
        return 0;
    }
    if (std::string_view(argv[1]) == "inspect") {
        return run_gguf_inspect(argc - 1, argv + 1, output, error);
    }
    CommandOptions command;
    if (!parse_command(argc, argv, command, error)) return 1;
    if (command.help) {
        write_help(output, command.help_all);
        return 0;
    }
    if (command.version) {
        output << "guffin " << build_identity << '\n';
        return 0;
    }
    try {
        return run_chat(command, input, output, error, interactive);
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

} // namespace

int run_cli(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    return run_cli_impl(argc, argv, std::cin, output, error,
        ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO));
}

int run_cli(
    int argc,
    const char* const* argv,
    std::istream& input,
    std::ostream& output,
    std::ostream& error,
    bool interactive
) {
    return run_cli_impl(argc, argv, input, output, error, interactive);
}

}  // namespace local_gguf_engine::frontends::cli
