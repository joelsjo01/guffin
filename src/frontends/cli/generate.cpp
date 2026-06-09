#include "local_gguf_engine/frontends/cli/generate.hpp"

#include "local_gguf_engine/engine/kv_cache.hpp"
#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace local_gguf_engine::frontends::cli {

namespace {

using engine::bind_llama_model_weights;
using engine::create_llama_kv_cache;
using engine::EngineModelError;
using engine::generate_llama_sampled_tokens_with_kv_cache_f32;
using engine::load_engine_model_from_gguf;
using engine::run_llama_model_logits_with_kv_cache_f32;
using engine::top_logit_tokens_f32;
using model_format::GGUFReadError;
using model_format::read_gguf_model_view;
using tensor::TensorError;
using tokenizer::format_generate_prompt_from_gguf_metadata;
using tokenizer::load_tokenizer_from_gguf_metadata;
using tokenizer::PromptFormatResult;
using tokenizer::TokenizerDecodeOptions;
using tokenizer::TokenizerEncodeOptions;
using tokenizer::TokenizerError;

constexpr std::size_t default_debug_top_token_count = 16;

class GenerateFileReadError final : public std::runtime_error {
public:
    explicit GenerateFileReadError(std::string message) : std::runtime_error(std::move(message)) {}
};

struct EncodedGeneratePrompt {
    PromptFormatResult formatted_prompt;
    std::vector<std::uint32_t> token_ids;
};

void write_generate_usage(std::ostream& error) {
    error << "usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>"
          << " [--temperature <value>] [--top-k <count>] [--seed <integer>] [--debug-first-token]\n";
}

[[nodiscard]] std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto file_size = std::filesystem::file_size(path, error_code);
    if (error_code) {
        throw GenerateFileReadError("could not read '" + path.string() + "': " + error_code.message());
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw GenerateFileReadError("could not read '" + path.string() + "': file is too large");
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw GenerateFileReadError("could not read '" + path.string() + "': file is too large");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw GenerateFileReadError("could not open '" + path.string() + "' for reading");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!input) {
            throw GenerateFileReadError("could not read entire file '" + path.string() + "'");
        }
    }

    return bytes;
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

    std::uint64_t parsed = 0;
    for (const auto character : text) {
        if (character < '0' || character > '9') {
            return false;
        }

        const auto digit = static_cast<std::uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL) {
            return false;
        }
        parsed = parsed * 10ULL + digit;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_positive_u32(std::string_view text, std::uint32_t& value) noexcept {
    std::uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    value = static_cast<std::uint32_t>(parsed);
    return true;
}

[[nodiscard]] bool parse_positive_finite_float(std::string_view text, float& value) {
    if (text.empty()) {
        return false;
    }

    const std::string text_copy(text);
    char* parsed_end = nullptr;
    errno = 0;
    const auto parsed = std::strtof(text_copy.c_str(), &parsed_end);
    if (
        errno == ERANGE
        || parsed_end != text_copy.c_str() + text_copy.size()
        || parsed <= 0.0F
        || !std::isfinite(parsed)
    ) {
        return false;
    }

    value = parsed;
    return true;
}

[[nodiscard]] bool parse_sampling_options(
    int argc,
    const char* const* argv,
    GenerateOptions& options,
    std::ostream& error
) {
    bool temperature_set = false;
    bool top_k_set = false;
    bool seed_set = false;
    for (int index = 4; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--temperature") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --temperature requires a value\n";
                return false;
            }

            float temperature = 0.0F;
            if (!parse_positive_finite_float(argv[index + 1], temperature)) {
                write_generate_usage(error);
                error << "error: --temperature must be a positive finite number\n";
                return false;
            }

            options.sampling.temperature = temperature;
            temperature_set = true;
            ++index;
        } else if (option == "--top-k") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --top-k requires a value\n";
                return false;
            }

            std::uint32_t top_k = 0;
            if (!parse_positive_u32(argv[index + 1], top_k)) {
                write_generate_usage(error);
                error << "error: --top-k must be a positive uint32 integer\n";
                return false;
            }

            options.sampling.top_k = top_k;
            top_k_set = true;
            ++index;
        } else if (option == "--seed") {
            if (index + 1 >= argc) {
                write_generate_usage(error);
                error << "error: --seed requires a value\n";
                return false;
            }

            std::uint64_t seed = 0;
            if (!parse_u64(argv[index + 1], seed)) {
                write_generate_usage(error);
                error << "error: --seed must be an unsigned 64-bit integer\n";
                return false;
            }

            options.sampling.seed = seed;
            seed_set = true;
            ++index;
        } else if (option == "--debug-first-token") {
            options.debug_first_token = true;
        } else {
            write_generate_usage(error);
            error << "error: unknown option '" << option << "'\n";
            return false;
        }
    }

    if (!temperature_set && (top_k_set || seed_set)) {
        write_generate_usage(error);
        error << "error: --temperature must be set to enable sampling options\n";
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
            .add_bos = default_encode_options.add_bos,
            .add_eos = default_encode_options.add_eos,
            .encode_special_tokens = formatted_prompt.encode_special_tokens,
        }
    );

    return EncodedGeneratePrompt {
        .formatted_prompt = formatted_prompt,
        .token_ids = prompt_token_ids,
    };
}

void append_escaped_text(std::string& output, std::string_view text) {
    constexpr char hex_digits[] = "0123456789abcdef";
    output.push_back('"');
    for (const auto character : text) {
        const auto byte = static_cast<unsigned char>(character);
        switch (character) {
            case '\\':
                output += "\\\\";
                break;
            case '"':
                output += "\\\"";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (byte < 0x20U || byte == 0x7FU) {
                    output += "\\x";
                    output.push_back(hex_digits[byte >> 4U]);
                    output.push_back(hex_digits[byte & 0x0FU]);
                } else {
                    output.push_back(character);
                }
                break;
        }
    }
    output.push_back('"');
}

[[nodiscard]] std::string escaped_text(std::string_view text) {
    std::string output;
    output.reserve(text.size() + 2U);
    append_escaped_text(output, text);
    return output;
}

void write_token_id_list(std::ostream& output, std::span<const std::uint32_t> token_ids) {
    output << '[';
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index > 0) {
            output << ", ";
        }
        output << token_ids[index];
    }
    output << ']';
}

}  // namespace

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
    const auto model = load_engine_model_from_gguf(model_view);
    const auto tokenizer = load_tokenizer_from_gguf_metadata(model_view.directory.metadata);
    const auto encoded_prompt = encode_generate_prompt(model_view.directory.metadata, tokenizer, prompt);
    const auto weights = bind_llama_model_weights(model);
    const auto output_token_ids = generate_llama_sampled_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(encoded_prompt.token_ids),
        max_new_tokens,
        weights,
        model.llama_config(),
        options.sampling,
        tokenizer.special_tokens().eos_token_id
    );

    return tokenizer.decode(
        std::span<const std::uint32_t>(output_token_ids).subspan(encoded_prompt.token_ids.size()),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );
}

GenerateFirstTokenDebugInfo inspect_generate_first_token_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t top_token_count
) {
    const auto model_view = read_gguf_model_view(bytes);
    const auto model = load_engine_model_from_gguf(model_view);
    const auto tokenizer = load_tokenizer_from_gguf_metadata(model_view.directory.metadata);
    auto encoded_prompt = encode_generate_prompt(model_view.directory.metadata, tokenizer, prompt);
    if (encoded_prompt.token_ids.empty()) {
        throw std::invalid_argument("Llama token id span must not be empty");
    }

    const auto weights = bind_llama_model_weights(model);
    auto cache = create_llama_kv_cache(model.llama_config());
    const auto prompt_final_position = static_cast<std::uint64_t>(encoded_prompt.token_ids.size() - 1U);
    const auto logits = run_llama_model_logits_with_kv_cache_f32(
        std::span<const std::uint32_t>(encoded_prompt.token_ids),
        weights,
        model.llama_config(),
        prompt_final_position,
        cache
    );

    std::vector<GenerateFirstTokenLogit> top_logits;
    for (const auto& token_logit : top_logit_tokens_f32(logits, top_token_count)) {
        const auto token_id = token_logit.token;
        top_logits.push_back(GenerateFirstTokenLogit {
            .token = token_id,
            .logit = token_logit.logit,
            .text = tokenizer.decode(
                std::span<const std::uint32_t>(&token_id, 1),
                TokenizerDecodeOptions {.skip_special_tokens = false}
            ),
        });
    }

    return GenerateFirstTokenDebugInfo {
        .formatted_prompt = std::move(encoded_prompt.formatted_prompt.text),
        .encode_special_tokens = encoded_prompt.formatted_prompt.encode_special_tokens,
        .prompt_token_ids = std::move(encoded_prompt.token_ids),
        .top_logits = std::move(top_logits),
    };
}

void write_generate_first_token_debug(
    const GenerateFirstTokenDebugInfo& info,
    std::ostream& output
) {
    output << "debug: formatted_prompt=" << escaped_text(info.formatted_prompt) << '\n';
    output << "debug: encode_special_tokens=" << (info.encode_special_tokens ? "true" : "false") << '\n';
    output << "debug: prompt_token_ids=";
    write_token_id_list(output, std::span<const std::uint32_t>(info.prompt_token_ids));
    output << '\n';
    output << "debug: first_token_top_logits\n";
    for (const auto& token_logit : info.top_logits) {
        output << "debug:   token=" << token_logit.token
               << " logit=" << token_logit.logit
               << " text=" << escaped_text(token_logit.text)
               << '\n';
    }
}

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
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

    GenerateOptions options;
    if (!parse_sampling_options(argc, argv, options, error)) {
        return 1;
    }

    try {
        const auto bytes = read_file_bytes(argv[1]);
        if (options.debug_first_token) {
            write_generate_first_token_debug(
                inspect_generate_first_token_from_gguf_bytes(
                    std::span<const std::uint8_t>(bytes),
                    argv[2],
                    default_debug_top_token_count
                ),
                error
            );
        }

        output << generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), argv[2], max_new_tokens, options)
               << '\n';
        return 0;
    } catch (const GenerateFileReadError& read_error) {
        error << "error: " << read_error.what() << '\n';
    } catch (const GGUFReadError& read_error) {
        error << "error: invalid GGUF: " << read_error.what() << '\n';
    } catch (const EngineModelError& model_error) {
        error << "error: invalid model: " << model_error.what() << '\n';
    } catch (const TokenizerError& tokenizer_error) {
        error << "error: tokenizer: " << tokenizer_error.what() << '\n';
    } catch (const TensorError& tensor_error) {
        error << "error: tensor: " << tensor_error.what() << '\n';
    } catch (const std::exception& unexpected_error) {
        error << "error: " << unexpected_error.what() << '\n';
    }

    return 1;
}

}  // namespace local_gguf_engine::frontends::cli
