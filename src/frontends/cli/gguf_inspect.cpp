#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"

#include "local_gguf_engine/model_format/gguf_metadata.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"
#include "local_gguf_engine/tensor/tensor_error.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <variant>

namespace local_gguf_engine::frontends::cli {

namespace {

using model_format::GGUFMetadataSection;
using model_format::GGUFMetadataValueType;
using model_format::GGUFReadError;
using model_format::GGUFReadErrorCode;
using model_format::GGUFTensorDirectory;
using model_format::GGUFTensorInfo;
using model_format::load_gguf_tensor_payload;
using model_format::read_gguf_model_view;
using model_format::read_gguf_tensor_directory;
using tensor::TensorError;

class GGUFFileReadError final : public std::runtime_error {
public:
    explicit GGUFFileReadError(std::string message) : std::runtime_error(std::move(message)) {}
};

[[nodiscard]] bool gguf_type_is_supported(std::uint32_t type_id) noexcept {
    return type_id == 0 || type_id == 1 || type_id == 8 || type_id == 12 || type_id == 14;
}

void add_dtype_count(GGUFTensorDTypeCounts& counts, std::uint32_t type_id) noexcept {
    switch (type_id) {
        case 0:
            ++counts.f32;
            return;
        case 1:
            ++counts.f16;
            return;
        case 8:
            ++counts.q8_0;
            return;
        case 12:
            ++counts.q4_k;
            return;
        case 14:
            ++counts.q6_k;
            return;
        default:
            ++counts.unsupported;
            return;
    }
}

[[nodiscard]] std::optional<std::string> architecture_from_metadata(const GGUFMetadataSection& metadata) {
    for (const auto& entry : metadata.entries) {
        if (entry.key != "general.architecture") {
            continue;
        }

        if (entry.value.type != GGUFMetadataValueType::string) {
            return std::nullopt;
        }

        if (!std::holds_alternative<std::string>(entry.value.data)) {
            return std::nullopt;
        }

        return std::get<std::string>(entry.value.data);
    }

    return std::nullopt;
}

[[nodiscard]] std::string shape_to_string(std::span<const std::uint64_t> dimensions) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << dimensions[index];
    }
    output << ']';
    return output.str();
}

[[nodiscard]] std::vector<GGUFTensorSummaryLine> first_tensor_lines(
    const GGUFTensorDirectory& directory,
    std::size_t limit
) {
    std::vector<GGUFTensorSummaryLine> lines;
    const auto line_count = std::min(limit, directory.tensors.size());
    lines.reserve(line_count);

    for (std::size_t index = 0; index < line_count; ++index) {
        const auto& tensor = directory.tensors[index];
        lines.push_back(GGUFTensorSummaryLine {
            .name = tensor.name,
            .shape = tensor.dimensions,
            .dtype = dtype_name_for_gguf_type(tensor.type_id),
        });
    }

    return lines;
}

[[nodiscard]] GGUFInspectionSummary summarize_directory(
    const GGUFTensorDirectory& directory,
    std::size_t first_tensor_limit
) {
    GGUFInspectionSummary summary {
        .version = directory.metadata.header.version,
        .tensor_count = directory.metadata.header.tensor_count,
        .metadata_kv_count = directory.metadata.header.metadata_kv_count,
        .architecture = architecture_from_metadata(directory.metadata),
        .alignment = directory.alignment,
        .tensor_data_start_offset = directory.tensor_data_start_offset,
        .dtype_counts = {},
        .total_supported_payload_bytes = 0,
        .first_tensors = first_tensor_lines(directory, first_tensor_limit),
    };

    for (const auto& tensor : directory.tensors) {
        add_dtype_count(summary.dtype_counts, tensor.type_id);
    }

    return summary;
}

void add_supported_payload_bytes(GGUFInspectionSummary& summary, std::size_t byte_count) {
    if (byte_count > std::numeric_limits<std::size_t>::max() - summary.total_supported_payload_bytes) {
        throw GGUFReadError(GGUFReadErrorCode::invalid_tensor_size, 0, 0);
    }

    summary.total_supported_payload_bytes += byte_count;
}

[[nodiscard]] std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto file_size = std::filesystem::file_size(path, error_code);
    if (error_code) {
        throw GGUFFileReadError("could not read '" + path.string() + "': " + error_code.message());
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw GGUFFileReadError("could not read '" + path.string() + "': file is too large");
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw GGUFFileReadError("could not read '" + path.string() + "': file is too large");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw GGUFFileReadError("could not open '" + path.string() + "' for reading");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!input) {
            throw GGUFFileReadError("could not read entire file '" + path.string() + "'");
        }
    }

    return bytes;
}

[[nodiscard]] bool is_payload_malformed_error(GGUFReadErrorCode code) noexcept {
    return code == GGUFReadErrorCode::truncated_tensor_payload
        || code == GGUFReadErrorCode::invalid_tensor_size;
}

[[nodiscard]] std::string unsupported_tensor_message(const GGUFReadError& error) {
    std::ostringstream message;
    message << "unsupported tensor type during payload conversion: GGUF tensor type "
            << error.actual_byte_count()
            << " is not supported by Tensor Core";
    return message.str();
}

}  // namespace

std::string dtype_name_for_gguf_type(std::uint32_t type_id) {
    switch (type_id) {
        case 0:
            return "F32";
        case 1:
            return "F16";
        case 8:
            return "Q8_0";
        case 12:
            return "Q4_K";
        case 14:
            return "Q6_K";
        default: {
            std::ostringstream output;
            output << "unsupported(" << type_id << ')';
            return output.str();
        }
    }
}

GGUFInspectionSummary inspect_gguf_bytes(std::span<const std::uint8_t> bytes, std::size_t first_tensor_limit) {
    const auto directory = read_gguf_tensor_directory(bytes);
    auto summary = summarize_directory(directory, first_tensor_limit);

    for (const auto& tensor : directory.tensors) {
        if (!gguf_type_is_supported(tensor.type_id)) {
            throw GGUFReadError(GGUFReadErrorCode::unsupported_tensor_type, 0, tensor.type_id);
        }
    }

    const auto model = read_gguf_model_view(bytes);
    for (const auto& payload : model.tensor_payloads) {
        const auto loaded = load_gguf_tensor_payload(payload);
        add_supported_payload_bytes(summary, loaded.storage.bytes().size());
    }

    return summary;
}

void write_gguf_inspection_summary(
    const GGUFInspectionSummary& summary,
    std::string_view path,
    std::ostream& output
) {
    output << "GGUF summary\n";
    output << "path: " << path << '\n';
    output << "version: " << summary.version << '\n';
    output << "tensor count: " << summary.tensor_count << '\n';
    output << "metadata kv count: " << summary.metadata_kv_count << '\n';
    output << "architecture: " << (summary.architecture.has_value() ? *summary.architecture : "<absent>") << '\n';
    output << "alignment: " << summary.alignment << '\n';
    output << "tensor data start offset: " << summary.tensor_data_start_offset << '\n';
    output << "tensor dtype counts:\n";
    output << "  F32: " << summary.dtype_counts.f32 << '\n';
    output << "  F16: " << summary.dtype_counts.f16 << '\n';
    output << "  Q8_0: " << summary.dtype_counts.q8_0 << '\n';
    output << "  Q4_K: " << summary.dtype_counts.q4_k << '\n';
    output << "  Q6_K: " << summary.dtype_counts.q6_k << '\n';
    output << "  unsupported: " << summary.dtype_counts.unsupported << '\n';
    output << "total supported payload bytes: " << summary.total_supported_payload_bytes << '\n';
    output << "first tensors:\n";

    if (summary.first_tensors.empty()) {
        output << "  <none>\n";
        return;
    }

    for (const auto& tensor : summary.first_tensors) {
        output << "  - " << tensor.name
               << " shape=" << shape_to_string(std::span<const std::uint64_t>(tensor.shape))
               << " dtype=" << tensor.dtype << '\n';
    }
}

int run_gguf_inspect(int argc, const char* const* argv, std::ostream& output, std::ostream& error) {
    if (argc != 2) {
        error << "usage: gguf-inspect <model.gguf>\n";
        if (argc < 2) {
            error << "error: missing GGUF path\n";
        } else {
            error << "error: expected exactly one GGUF path\n";
        }
        return 1;
    }

    const std::string path = argv[1];
    try {
        const auto bytes = read_file_bytes(path);
        const auto summary = inspect_gguf_bytes(std::span<const std::uint8_t>(bytes));
        write_gguf_inspection_summary(summary, path, output);
        return 0;
    } catch (const GGUFFileReadError& read_error) {
        error << "error: " << read_error.what() << '\n';
    } catch (const GGUFReadError& read_error) {
        if (read_error.code() == GGUFReadErrorCode::unsupported_tensor_type) {
            error << "error: " << unsupported_tensor_message(read_error) << '\n';
        } else if (is_payload_malformed_error(read_error.code())) {
            error << "error: malformed tensor payload: " << read_error.what() << '\n';
        } else {
            error << "error: invalid GGUF: " << read_error.what() << '\n';
        }
    } catch (const TensorError& tensor_error) {
        error << "error: malformed tensor payload: " << tensor_error.what() << '\n';
    } catch (const std::exception& unexpected_error) {
        error << "error: " << unexpected_error.what() << '\n';
    }

    return 1;
}

}  // namespace local_gguf_engine::frontends::cli
