#pragma once

#include "local_gguf_engine/engine/speculative_controller.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace local_gguf_engine::engine {

inline constexpr std::size_t target_verifier_source_shape_count = 1'246U;

enum class TargetVerifierSourceFamily : std::uint8_t {
    bridging,
    ddtree,
    caddtree,
    treeflash,
    pctree,
    dominotree,
    dartree,
    jetspec,
    full_path,
    dblast,
    eagle3,
    redrafter,
};

// These identities deliberately keep paper descriptions, executable code,
// transfers, and diagnostics separate even when their target row count is the
// same. They describe target-side shape contracts, not completed drafters.
enum class TargetVerifierSourceMechanism : std::uint8_t {
    bridging_independent_marginal,
    ddtree_fixed,
    caddtree_adaptive,
    treeflash_paper_h16_full_v_lazy,
    treeflash_executable_h15_top_k,
    treeflash_executable_h15_full_v_transfer,
    pctree_parent_conditioned_markov,
    pctree_parent_independent_control,
    pctree_dspark_chain_control,
    dominotree_depth_synchronous_with_sequential_oracle,
    dartree_deferred_pruned,
    dartree_fixed_balanced,
    jetspec_executable_h15_latent,
    jetspec_h16_latent_extension,
    jetspec_h16_argmax_diagnostic,
    full_path_jetspec_executable,
    full_path_jetspec_h16_latent,
    full_path_jetspec_h16_argmax,
    full_path_ddtree,
    dblast_exact_seed_category_trie,
    eagle3_final_tree,
    redrafter_mlx_repeated_root,
    redrafter_article_prefix_deduplicated,
    redrafter_pytorch_l4_compressed,
    redrafter_pytorch_l5_w45_compressed,
};

enum class TargetVerifierCandidateDomain : std::uint8_t {
    full_vocabulary,
    explicit_top_k,
    explicit_top_m,
};

struct TargetVerifierBuilderBounds {
    std::uint32_t raw_row_count = 0;
    std::uint32_t packed_row_count = 0;
    std::uint32_t kv_row_count = 0;
    std::uint32_t score_event_count = 0;
};

struct TargetVerifierSourceShape {
    std::uint32_t registry_index = 0;
    TargetVerifierSourceFamily family = TargetVerifierSourceFamily::bridging;
    TargetVerifierSourceMechanism mechanism =
        TargetVerifierSourceMechanism::bridging_independent_marginal;
    ProposalGraphKind topology = ProposalGraphKind::forest;
    SourceTreeBudget source_budget;
    ResolvedTreeBudget resolved_budget;
    std::uint16_t maximum_depth = 0;
    std::uint16_t child_branch_width = 0;
    TargetVerifierCandidateDomain candidate_domain =
        TargetVerifierCandidateDomain::full_vocabulary;
    // K/top-k candidate menu. Zero means the declared domain is streamed full-V.
    std::uint16_t candidate_width = 0;
    // M predecessor/correction width, distinct from child branch k and active W.
    std::uint16_t conditioning_width = 0;
    std::uint16_t active_width = 0;
    std::uint32_t physical_target_rows = 0;
    TargetVerifierBuilderBounds builder;
};

[[nodiscard]] std::span<const TargetVerifierSourceShape>
target_verifier_source_shapes();

[[nodiscard]] std::string_view target_verifier_source_mechanism_name(
    TargetVerifierSourceMechanism mechanism
) noexcept;

}  // namespace local_gguf_engine::engine
