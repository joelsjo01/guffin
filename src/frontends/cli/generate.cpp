#include "local_gguf_engine/frontends/cli/generate.hpp"

#include "local_gguf_engine/engine/llama_model.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <cstddef>
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
using engine::EngineModelError;
using engine::generate_llama_argmax_tokens_with_kv_cache_f32;
using engine::load_engine_model_from_gguf;
using model_format::GGUFReadError;
using model_format::read_gguf_model_view;
using tensor::TensorError;
using tokenizer::format_generate_prompt_from_gguf_metadata;
using tokenizer::load_tokenizer_from_gguf_metadata;
using tokenizer::TokenizerDecodeOptions;
using tokenizer::TokenizerEncodeOptions;
using tokenizer::TokenizerError;

class GenerateFileReadError final : public std::runtime_error {
public:
    explicit GenerateFileReadError(std::string message) : std::runtime_error(std::move(message)) {}
};

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

}  // namespace

std::string generate_text_from_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::string_view prompt,
    std::size_t max_new_tokens
) {
    const auto model_view = read_gguf_model_view(bytes);
    const auto model = load_engine_model_from_gguf(model_view);
    const auto tokenizer = load_tokenizer_from_gguf_metadata(model_view.directory.metadata);
    const auto formatted_prompt = format_generate_prompt_from_gguf_metadata(model_view.directory.metadata, prompt);
    const auto default_encode_options = tokenizer.default_encode_options();
    const auto prompt_token_ids = tokenizer.encode(
        formatted_prompt.text,
        TokenizerEncodeOptions {
            .add_bos = default_encode_options.add_bos,
            .add_eos = default_encode_options.add_eos,
            .encode_special_tokens = formatted_prompt.encode_special_tokens,
        }
    );
    const auto weights = bind_llama_model_weights(model);
    const auto output_token_ids = generate_llama_argmax_tokens_with_kv_cache_f32(
        std::span<const std::uint32_t>(prompt_token_ids),
        max_new_tokens,
        weights,
        model.llama_config(),
        tokenizer.special_tokens().eos_token_id
    );

    return tokenizer.decode(
        std::span<const std::uint32_t>(output_token_ids).subspan(prompt_token_ids.size()),
        TokenizerDecodeOptions {.skip_special_tokens = true}
    );
}

int run_generate(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    if (argc != 4) {
        error << "usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>\n";
        if (argc < 4) {
            error << "error: missing required argument\n";
        } else {
            error << "error: expected exactly three arguments\n";
        }
        return 1;
    }

    std::size_t max_new_tokens = 0;
    if (!parse_positive_size(argv[3], max_new_tokens)) {
        error << "usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>\n";
        error << "error: max-new-tokens must be a positive integer\n";
        return 1;
    }

    try {
        const auto bytes = read_file_bytes(argv[1]);
        output << generate_text_from_gguf_bytes(std::span<const std::uint8_t>(bytes), argv[2], max_new_tokens) << '\n';
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
