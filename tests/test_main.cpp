#include "engine_model_tests.hpp"
#include "quick_benchmark_tests.hpp"
#include "gemma2_metal_plan_tests.hpp"
#include "cli_tests.hpp"
#include "gemma2_reference_tests.hpp"
#include "gguf_inspect_tests.hpp"
#include "gguf_reader_tests.hpp"
#include "gguf_tensor_bridge_tests.hpp"
#include "sampling_tests.hpp"
#include "speculative_controller_tests.hpp"
#include "retrieval_proposer_tests.hpp"
#include "tensor_backend_tests.hpp"
#include "tensor_tests.hpp"
#include "test_support.hpp"
#include "tokenizer_tests.hpp"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>

namespace local_gguf_engine::tests {
std::span<const TestCase> lossless_sampling_test_cases();
std::span<const TestCase> standalone_draft_artifact_test_cases();
std::span<const TestCase> standalone_chat_test_cases();
std::span<const TestCase> gemma3_draft_test_cases();
std::span<const TestCase> gemma3_model_plan_test_cases();
std::span<const TestCase> gemma3_target_runtime_test_cases();
std::span<const TestCase> gemma3_metal_test_cases();
}

int main() {
    int failures = 0;
    std::size_t test_count = 0;
    std::cout << std::unitbuf;
    const auto run_test = [&](const local_gguf_engine::tests::TestCase& test) {
        ++test_count;
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    };

    for (const auto& test : local_gguf_engine::tests::tensor_core_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::tensor_backend_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::tensor_post_backend_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::sampling_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::lossless_sampling_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::standalone_draft_artifact_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::standalone_chat_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::gemma3_draft_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::gemma3_model_plan_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::gemma3_target_runtime_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::gemma3_metal_test_cases()) run_test(test);
    for (const auto& test : local_gguf_engine::tests::speculative_controller_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::retrieval_proposer_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::quick_benchmark_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::cli_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_tensor_bridge_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_reader_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gguf_inspect_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::engine_model_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gemma2_metal_plan_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::gemma2_reference_test_cases()) {
        run_test(test);
    }
    for (const auto& test : local_gguf_engine::tests::tokenizer_test_cases()) {
        run_test(test);
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }

    std::cout << test_count << " test(s) passed\n";
    return EXIT_SUCCESS;
}
