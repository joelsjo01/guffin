#include "gguf_inspect_tests.hpp"

#include "local_gguf_engine/frontends/cli/gguf_inspect.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using local_gguf_engine::frontends::cli::inspect_gguf_bytes;
using local_gguf_engine::frontends::cli::run_gguf_inspect;
using local_gguf_engine::frontends::cli::write_gguf_inspection_summary;
using local_gguf_engine::tests::TestFailure;
using local_gguf_engine::tests::expect;
using local_gguf_engine::tests::fixture;

class TempBinaryFixture final {
public:
    explicit TempBinaryFixture(std::string_view name)
        : path_(std::filesystem::temp_directory_path() / ("local-gguf-engine-" + std::string(name) + ".gguf")) {
        const auto bytes = fixture(name);
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw TestFailure("could not create temp binary fixture: " + path_.string());
        }

        output.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw TestFailure("could not write temp binary fixture: " + path_.string());
        }
    }

    ~TempBinaryFixture() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

int run_gguf_inspect_for_path(
    const std::filesystem::path& path,
    std::ostringstream& output,
    std::ostringstream& error
) {
    const std::string program = "gguf-inspect";
    const std::string path_string = path.string();
    const char* argv[] = {program.c_str(), path_string.c_str()};
    return run_gguf_inspect(2, argv, output, error);
}

void test_inspects_gguf_summary_from_f16_payload() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto summary = inspect_gguf_bytes(std::span<const std::uint8_t>(bytes), 3);

    expect(summary.version == 3, "GGUF inspect version mismatch");
    expect(summary.tensor_count == 1, "GGUF inspect tensor count mismatch");
    expect(summary.metadata_kv_count == 0, "GGUF inspect metadata count mismatch");
    expect(!summary.architecture.has_value(), "GGUF inspect architecture should be absent");
    expect(summary.alignment == 32, "GGUF inspect alignment mismatch");
    expect(summary.tensor_data_start_offset == 64, "GGUF inspect tensor data start mismatch");
    expect(summary.dtype_counts.f32 == 0, "GGUF inspect F32 count mismatch");
    expect(summary.dtype_counts.f16 == 1, "GGUF inspect F16 count mismatch");
    expect(summary.dtype_counts.q8_0 == 0, "GGUF inspect Q8_0 count mismatch");
    expect(summary.dtype_counts.unsupported == 0, "GGUF inspect unsupported count mismatch");
    expect(summary.total_supported_payload_bytes == 4, "GGUF inspect supported payload byte mismatch");
    expect(summary.first_tensors.size() == 1, "GGUF inspect first tensor count mismatch");
    expect(summary.first_tensors[0].name == "h2", "GGUF inspect first tensor name mismatch");
    expect(summary.first_tensors[0].shape == std::vector<std::uint64_t> {2}, "GGUF inspect shape mismatch");
    expect(summary.first_tensors[0].dtype == "F16", "GGUF inspect first tensor dtype mismatch");
}

void test_formats_gguf_inspection_summary() {
    const auto bytes = fixture("valid-f16-tensor-payload-view");
    const auto summary = inspect_gguf_bytes(std::span<const std::uint8_t>(bytes), 1);

    std::ostringstream output;
    write_gguf_inspection_summary(summary, "fixture.gguf", output);
    const auto text = output.str();

    expect(text.find("GGUF summary\n") == 0, "GGUF inspect formatted heading mismatch");
    expect(text.find("version: 3\n") != std::string::npos, "GGUF inspect formatted version missing");
    expect(text.find("tensor count: 1\n") != std::string::npos, "GGUF inspect formatted tensor count missing");
    expect(text.find("metadata kv count: 0\n") != std::string::npos, "GGUF inspect formatted metadata count missing");
    expect(text.find("architecture: <absent>\n") != std::string::npos, "GGUF inspect formatted architecture missing");
    expect(text.find("alignment: 32\n") != std::string::npos, "GGUF inspect formatted alignment missing");
    expect(
        text.find("tensor data start offset: 64\n") != std::string::npos,
        "GGUF inspect formatted tensor data start missing"
    );
    expect(text.find("  F16: 1\n") != std::string::npos, "GGUF inspect formatted F16 count missing");
    expect(
        text.find("total supported payload bytes: 4\n") != std::string::npos,
        "GGUF inspect formatted payload byte count missing"
    );
    expect(
        text.find("  - h2 shape=[2] dtype=F16\n") != std::string::npos,
        "GGUF inspect formatted tensor line missing"
    );
}

void test_runs_gguf_inspect_success() {
    const TempBinaryFixture model("valid-f16-tensor-payload-view");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_SUCCESS, "gguf-inspect success exit code mismatch");
    expect(error.str().empty(), "gguf-inspect success should not write stderr");
    expect(output.str().find("GGUF summary\n") == 0, "gguf-inspect success output missing summary");
    expect(
        output.str().find("total supported payload bytes: 4\n") != std::string::npos,
        "gguf-inspect success output missing payload byte count"
    );
}

void test_rejects_gguf_inspect_missing_path() {
    const char* argv[] = {"gguf-inspect"};
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect(1, argv, output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect missing path exit code mismatch");
    expect(output.str().empty(), "gguf-inspect missing path should not write stdout");
    expect(error.str().find("usage: gguf-inspect <model.gguf>") != std::string::npos, "gguf-inspect usage missing");
    expect(error.str().find("missing GGUF path") != std::string::npos, "gguf-inspect missing path error missing");
}

void test_rejects_gguf_inspect_unreadable_path() {
    const auto path = std::filesystem::temp_directory_path() / "local-gguf-engine-missing-inspect.gguf";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(path, output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect unreadable path exit code mismatch");
    expect(output.str().empty(), "gguf-inspect unreadable path should not write stdout");
    expect(error.str().find("could not read") != std::string::npos, "gguf-inspect unreadable path error missing");
}

void test_reports_invalid_gguf_in_gguf_inspect() {
    const TempBinaryFixture model("invalid-magic");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect invalid GGUF exit code mismatch");
    expect(output.str().empty(), "gguf-inspect invalid GGUF should not write stdout");
    expect(
        error.str().find("invalid GGUF: invalid GGUF magic") != std::string::npos,
        "gguf-inspect invalid GGUF error missing"
    );
}

void test_reports_unsupported_tensor_type_in_gguf_inspect() {
    const TempBinaryFixture model("unsupported-tensor-payload-type");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect unsupported tensor exit code mismatch");
    expect(output.str().empty(), "gguf-inspect unsupported tensor should not write stdout");
    expect(
        error.str().find("unsupported tensor type during payload conversion") != std::string::npos,
        "gguf-inspect unsupported tensor error missing"
    );
    expect(error.str().find("GGUF tensor type 39") != std::string::npos, "gguf-inspect unsupported type id missing");
}

void test_reports_malformed_tensor_payload_in_gguf_inspect() {
    const TempBinaryFixture model("truncated-tensor-payload");
    std::ostringstream output;
    std::ostringstream error;

    const auto exit_code = run_gguf_inspect_for_path(model.path(), output, error);

    expect(exit_code == EXIT_FAILURE, "gguf-inspect malformed payload exit code mismatch");
    expect(output.str().empty(), "gguf-inspect malformed payload should not write stdout");
    expect(
        error.str().find("malformed tensor payload: truncated GGUF tensor payload") != std::string::npos,
        "gguf-inspect malformed payload error missing"
    );
}

}  // namespace

namespace local_gguf_engine::tests {

std::span<const TestCase> gguf_inspect_test_cases() {
    static constexpr TestCase tests[] = {
        {"inspects GGUF summary from F16 payload", test_inspects_gguf_summary_from_f16_payload},
        {"formats GGUF inspection summary", test_formats_gguf_inspection_summary},
        {"runs gguf-inspect success path", test_runs_gguf_inspect_success},
        {"rejects gguf-inspect missing path", test_rejects_gguf_inspect_missing_path},
        {"rejects gguf-inspect unreadable path", test_rejects_gguf_inspect_unreadable_path},
        {"reports invalid GGUF in gguf-inspect", test_reports_invalid_gguf_in_gguf_inspect},
        {
            "reports unsupported tensor type in gguf-inspect",
            test_reports_unsupported_tensor_type_in_gguf_inspect,
        },
        {"reports malformed tensor payload in gguf-inspect", test_reports_malformed_tensor_payload_in_gguf_inspect},
    };

    return tests;
}

}  // namespace local_gguf_engine::tests
