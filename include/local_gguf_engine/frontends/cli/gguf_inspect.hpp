#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace local_gguf_engine::frontends::cli {

struct GGUFTensorDTypeCounts {
    std::size_t f32 = 0;
    std::size_t f16 = 0;
    std::size_t q8_0 = 0;
    std::size_t unsupported = 0;
};

struct GGUFTensorSummaryLine {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::string dtype;
};

struct GGUFInspectionSummary {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_kv_count = 0;
    std::optional<std::string> architecture;
    std::uint32_t alignment = 0;
    std::size_t tensor_data_start_offset = 0;
    GGUFTensorDTypeCounts dtype_counts;
    std::size_t total_supported_payload_bytes = 0;
    std::vector<GGUFTensorSummaryLine> first_tensors;
};

[[nodiscard]] std::string dtype_name_for_gguf_type(std::uint32_t type_id);
[[nodiscard]] GGUFInspectionSummary inspect_gguf_bytes(
    std::span<const std::uint8_t> bytes,
    std::size_t first_tensor_limit = 5
);
void write_gguf_inspection_summary(
    const GGUFInspectionSummary& summary,
    std::string_view path,
    std::ostream& output
);
int run_gguf_inspect(int argc, const char* const* argv, std::ostream& output, std::ostream& error);

}  // namespace local_gguf_engine::frontends::cli
