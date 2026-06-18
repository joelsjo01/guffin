#include "benchmark_model_load.hpp"

#include "benchmark_support.hpp"

#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace local_gguf_engine::benchmarks {

namespace {

using engine::load_engine_model_from_gguf;
using model_format::load_gguf_tensor_payloads;
using model_format::read_gguf_model_view;

[[nodiscard]] std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::error_code error_code;
    const auto file_size = std::filesystem::file_size(path, error_code);
    if (error_code) {
        throw std::runtime_error("could not read '" + path.string() + "': " + error_code.message());
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("could not read '" + path.string() + "': file is too large");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open '" + path.string() + "' for reading");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            throw std::runtime_error("could not read entire file '" + path.string() + "'");
        }
    }

    return bytes;
}

}  // namespace

void add_model_load_benchmarks(BenchmarkReport& report, const std::filesystem::path& model_path) {
    report.measurements.push_back(measure(
        "gguf.read_file",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model_path]() {
            const auto bytes = read_binary_file(model_path);
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = checksum_byte_span(bytes),
            };
        }
    ));

    const auto bytes = read_binary_file(model_path);
    const auto model = read_gguf_model_view(std::span<const std::uint8_t>(bytes));

    report.measurements.push_back(measure(
        "gguf.parse_model_view",
        report.config.iterations,
        "model=" + model_path.string(),
        [&bytes]() {
            const auto parsed = read_gguf_model_view(std::span<const std::uint8_t>(bytes));
            return IterationMeasurement {
                .bytes_processed = bytes.size(),
                .checksum = parsed.directory.metadata.header.tensor_count
                    ^ parsed.directory.metadata.header.metadata_kv_count,
            };
        }
    ));

    report.measurements.push_back(measure(
        "gguf.load_tensor_payloads",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model]() {
            const auto loaded = load_gguf_tensor_payloads(model);
            std::size_t bytes_processed = 0;
            std::uint64_t checksum = 0;
            for (const auto& tensor : loaded) {
                bytes_processed += tensor.storage.bytes().size();
                checksum ^= checksum_byte_span(tensor.storage.bytes());
            }

            return IterationMeasurement {
                .bytes_processed = bytes_processed,
                .checksum = checksum,
            };
        }
    ));

    report.measurements.push_back(measure(
        "engine.load_model",
        report.config.iterations,
        "model=" + model_path.string(),
        [&model]() {
            const auto loaded = load_engine_model_from_gguf(model);
            std::size_t bytes_processed = 0;
            std::uint64_t checksum = loaded.llama_config().block_count
                ^ loaded.llama_config().embedding_length
                ^ static_cast<std::uint64_t>(loaded.tensors().size());
            for (const auto& tensor : loaded.tensors()) {
                bytes_processed += tensor.storage.bytes().size();
                checksum ^= checksum_byte_span(tensor.storage.bytes());
            }

            return IterationMeasurement {
                .bytes_processed = bytes_processed,
                .checksum = checksum,
            };
        }
    ));
}

}  // namespace local_gguf_engine::benchmarks
