#include "local_gguf_engine/frontends/cli/generate.hpp"

#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/runtime/backend_error.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::frontends::cli {

namespace {

using engine::EngineModelError;
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

void write_generate_usage(std::ostream& error) {
    error << "usage: local-gguf-generate <model.gguf> <prompt> <max-new-tokens>"
          << " [--greedy] [--backend cpu|metal]"
          << '\n';
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

[[nodiscard]] bool parse_generate_options(
    int argc,
    const char* const* argv,
    GenerateOptions& options,
    std::ostream& error
) {
    for (int index = 4; index < argc; ++index) {
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

            options.backend = backend;
            ++index;
        } else {
            write_generate_usage(error);
            error << "error: unknown option '" << option << "'\n";
            return false;
        }
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
    return generate_text_from_gguf_model_view(model_view, prompt, max_new_tokens, options);
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

    GenerateOptions options {};
    if (!parse_generate_options(argc, argv, options, error)) {
        return 1;
    }

    try {
        const auto file_model = read_file_backed_gguf_model(argv[1]);
        output << generate_text_from_gguf_model_view(file_model.view(), argv[2], max_new_tokens, options)
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

}  // namespace local_gguf_engine::frontends::cli
