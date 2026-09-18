#include "local_gguf_engine/engine/target_verifier_registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace local_gguf_engine::engine {
namespace {

using ShapeArray = std::array<TargetVerifierSourceShape, target_verifier_source_shape_count>;

struct RegistryBuilder {
    ShapeArray shapes {};
    std::size_t count = 0U;

    void add(TargetVerifierSourceShape shape) {
        if (count >= shapes.size()) {
            throw std::logic_error("target verifier source-shape registry overflow");
        }
        if (shape.physical_target_rows == 0U
            || shape.physical_target_rows > VerifierRowMax
            || shape.resolved_budget.input_row_cap != shape.physical_target_rows
            || shape.resolved_budget.draft_node_cap > ExternalTreeDraftNodeMax
            || shape.maximum_depth > external_speculation_max_horizon
            || shape.child_branch_width > MaterializedDegreeMax
            || (shape.candidate_domain == TargetVerifierCandidateDomain::full_vocabulary
                && shape.candidate_width != 0U)
            || (shape.candidate_domain != TargetVerifierCandidateDomain::full_vocabulary
                && shape.candidate_width == 0U)) {
            throw std::logic_error("target verifier source-shape registry entry is invalid");
        }
        shape.registry_index = static_cast<std::uint32_t>(count);
        shapes[count++] = shape;
    }
};

[[nodiscard]] TargetVerifierSourceShape rooted_shape(
    TargetVerifierSourceFamily family,
    TargetVerifierSourceMechanism mechanism,
    SourceTreeBudget source_budget,
    std::uint16_t maximum_depth,
    std::uint16_t child_branch_width,
    TargetVerifierCandidateDomain candidate_domain,
    std::uint16_t candidate_width,
    std::uint16_t conditioning_width,
    std::uint16_t active_width,
    TargetVerifierBuilderBounds builder = {}
) {
    const auto resolved = resolve_source_tree_budget(source_budget);
    return TargetVerifierSourceShape {
        .family = family,
        .mechanism = mechanism,
        .topology = ProposalGraphKind::forest,
        .source_budget = source_budget,
        .resolved_budget = resolved,
        .maximum_depth = maximum_depth,
        .child_branch_width = child_branch_width,
        .candidate_domain = candidate_domain,
        .candidate_width = candidate_width,
        .conditioning_width = conditioning_width,
        .active_width = active_width,
        .physical_target_rows = resolved.input_row_cap,
        .builder = builder,
    };
}

[[nodiscard]] std::uint32_t pctree_pool_rows(
    std::uint32_t depth,
    std::uint32_t branch
) {
    return 1U + branch + (depth - 1U) * branch * branch;
}

[[nodiscard]] ShapeArray make_registry() {
    RegistryBuilder registry;
    constexpr std::array<std::uint32_t, 8> bridging_budgets {
        4U, 8U, 16U, 32U, 64U, 128U, 192U, 256U,
    };
    constexpr std::array<std::uint16_t, 7> bridging_branches {
        1U, 2U, 3U, 4U, 7U, 8U, 12U,
    };
    for (const auto budget : bridging_budgets) {
        for (const auto branch : bridging_branches) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::bridging,
                TargetVerifierSourceMechanism::bridging_independent_marginal,
                {.value = budget, .root_convention = SourceRootConvention::excluded},
                static_cast<std::uint16_t>(std::min<std::uint32_t>(
                    budget, external_speculation_max_horizon
                )),
                branch, TargetVerifierCandidateDomain::full_vocabulary,
                0U, 0U, 0U,
                {.raw_row_count = budget + 1U, .packed_row_count = budget + 1U,
                 .kv_row_count = budget + 1U}
            ));
        }
    }

    constexpr std::array<std::uint32_t, 7> ddtree_budgets {
        16U, 32U, 64U, 128U, 256U, 512U, 1'024U,
    };
    for (const auto budget : ddtree_budgets) {
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::ddtree,
            TargetVerifierSourceMechanism::ddtree_fixed,
            {.value = budget, .root_convention = SourceRootConvention::excluded},
            16U, 0U, TargetVerifierCandidateDomain::full_vocabulary,
            0U, 0U, 0U,
            {.raw_row_count = budget + 1U, .packed_row_count = budget + 1U,
             .kv_row_count = budget + 1U, .score_event_count = 2U * budget}
        ));
    }
    for (std::uint32_t non_root = 0U; non_root <= ExternalTreeDraftNodeMax; ++non_root) {
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::caddtree,
            TargetVerifierSourceMechanism::caddtree_adaptive,
            {.value = non_root, .root_convention = SourceRootConvention::excluded},
            16U, 0U, TargetVerifierCandidateDomain::full_vocabulary,
            0U, 0U, 0U,
            {.raw_row_count = non_root + 1U, .packed_row_count = non_root + 1U,
             .kv_row_count = non_root + 1U, .score_event_count = 2U * non_root}
        ));
    }

    constexpr std::array<std::uint32_t, 6> treeflash_paper_budgets {
        16U, 32U, 64U, 128U, 256U, 512U,
    };
    constexpr std::array<std::uint32_t, 6> treeflash_code_budgets {
        16U, 32U, 64U, 128U, 256U, 512U,
    };
    constexpr std::array<std::uint16_t, 3> treeflash_m {16U, 32U, 64U};
    for (const auto budget : treeflash_paper_budgets) {
        for (const auto width : treeflash_m) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::treeflash,
                TargetVerifierSourceMechanism::treeflash_paper_h16_full_v_lazy,
                {.value = budget, .root_convention = SourceRootConvention::excluded},
                16U, 0U, TargetVerifierCandidateDomain::full_vocabulary,
                0U, width, 0U,
                {.raw_row_count = 1U + 15U * width,
                 .packed_row_count = 16U * width,
                 .kv_row_count = 16U * width}
            ));
        }
    }
    for (const auto budget : treeflash_code_budgets) {
        for (const auto width : treeflash_m) {
            const auto source = SourceTreeBudget {
                .value = budget, .root_convention = SourceRootConvention::included,
            };
            const auto builder = TargetVerifierBuilderBounds {
                .raw_row_count = 1U + 14U * width,
                .packed_row_count = 15U * width,
                .kv_row_count = 15U * width,
            };
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::treeflash,
                TargetVerifierSourceMechanism::treeflash_executable_h15_top_k,
                source, 15U, 0U, TargetVerifierCandidateDomain::explicit_top_k,
                16U, width, 0U, builder
            ));
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::treeflash,
                TargetVerifierSourceMechanism::treeflash_executable_h15_full_v_transfer,
                source, 15U, 0U, TargetVerifierCandidateDomain::full_vocabulary,
                0U, width, 0U, builder
            ));
        }
    }

    constexpr std::array<std::uint32_t, 8> pctree_budgets {
        8U, 16U, 24U, 32U, 48U, 64U, 96U, 128U,
    };
    constexpr std::array<std::uint16_t, 2> pctree_depths {7U, 16U};
    for (const auto depth : pctree_depths) {
        for (const auto budget : pctree_budgets) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::pctree,
                TargetVerifierSourceMechanism::pctree_parent_conditioned_markov,
                {.value = budget, .root_convention = SourceRootConvention::included},
                depth, 4U, TargetVerifierCandidateDomain::full_vocabulary,
                0U, 0U, 4U,
                {.raw_row_count = pctree_pool_rows(depth, 4U),
                 .packed_row_count = budget, .kv_row_count = budget}
            ));
        }
        for (const auto branch : {1U, 2U, 8U}) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::pctree,
                TargetVerifierSourceMechanism::pctree_parent_conditioned_markov,
                {.value = 32U, .root_convention = SourceRootConvention::included},
                depth, static_cast<std::uint16_t>(branch),
                TargetVerifierCandidateDomain::full_vocabulary,
                0U, 0U, static_cast<std::uint16_t>(branch),
                {.raw_row_count = pctree_pool_rows(depth, branch),
                 .packed_row_count = 32U, .kv_row_count = 32U}
            ));
        }
        for (const auto mechanism : {
            TargetVerifierSourceMechanism::pctree_parent_independent_control,
            TargetVerifierSourceMechanism::pctree_dspark_chain_control,
        }) {
            const auto branch = mechanism
                    == TargetVerifierSourceMechanism::pctree_dspark_chain_control
                ? 1U : 4U;
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::pctree, mechanism,
                {.value = 32U, .root_convention = SourceRootConvention::included},
                depth, static_cast<std::uint16_t>(branch),
                TargetVerifierCandidateDomain::full_vocabulary,
                0U, 0U, static_cast<std::uint16_t>(branch),
                {.raw_row_count = pctree_pool_rows(depth, branch),
                 .packed_row_count = 32U, .kv_row_count = 32U}
            ));
        }
    }

    constexpr std::array<std::uint32_t, 3> domino_budgets {16U, 32U, 64U};
    constexpr std::array<std::uint16_t, 5> domino_m {16U, 32U, 64U, 128U, 256U};
    for (const auto budget : domino_budgets) {
        for (const auto width : domino_m) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::dominotree,
                TargetVerifierSourceMechanism::dominotree_depth_synchronous_with_sequential_oracle,
                {.value = budget, .root_convention = SourceRootConvention::excluded},
                15U, 8U, TargetVerifierCandidateDomain::explicit_top_m,
                width, width, static_cast<std::uint16_t>(budget),
                {.raw_row_count = 15U * budget * 8U,
                 .packed_row_count = budget + 1U,
                 .kv_row_count = 15U * budget,
                 .score_event_count = 15U * budget * width}
            ));
        }
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::dominotree,
            TargetVerifierSourceMechanism::dominotree_depth_synchronous_with_sequential_oracle,
            {.value = budget, .root_convention = SourceRootConvention::excluded},
            15U, 8U, TargetVerifierCandidateDomain::full_vocabulary,
            0U, 0U, static_cast<std::uint16_t>(budget),
            {.raw_row_count = 15U * budget * 8U,
             .packed_row_count = budget + 1U, .kv_row_count = 15U * budget}
        ));
    }

    constexpr std::array<std::uint32_t, 4> dartree_budgets {32U, 64U, 128U, 192U};
    constexpr std::array<std::uint16_t, 5> dartree_widths {4U, 8U, 12U, 16U, 32U};
    for (const auto width : dartree_widths) {
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::dartree,
            TargetVerifierSourceMechanism::dartree_deferred_pruned,
            {.value = 64U, .root_convention = SourceRootConvention::excluded},
            16U, 0U, TargetVerifierCandidateDomain::explicit_top_k,
            64U, 0U, width,
            {.raw_row_count = 16U * width, .packed_row_count = 65U,
             .kv_row_count = 16U * width,
             .score_event_count = 64U + 15U * width * 64U}
        ));
    }
    for (const auto budget : dartree_budgets) {
        if (budget != 64U) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::dartree,
                TargetVerifierSourceMechanism::dartree_deferred_pruned,
                {.value = budget, .root_convention = SourceRootConvention::excluded},
                16U, 0U, TargetVerifierCandidateDomain::explicit_top_k,
                64U, 0U, 12U,
                {.raw_row_count = 16U * 12U, .packed_row_count = budget + 1U,
                 .kv_row_count = 16U * 12U,
                 .score_event_count = 64U + 15U * 12U * 64U}
            ));
        }
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::dartree,
            TargetVerifierSourceMechanism::dartree_fixed_balanced,
            {.value = budget, .root_convention = SourceRootConvention::excluded},
            16U, 0U, TargetVerifierCandidateDomain::explicit_top_k,
            64U, 0U, 0U,
            {.raw_row_count = budget, .packed_row_count = budget + 1U,
             .kv_row_count = budget, .score_event_count = budget * 64U}
        ));
    }

    constexpr std::array<std::uint32_t, 6> jetspec_budgets {
        16U, 32U, 64U, 128U, 255U, 256U,
    };
    for (const auto mechanism : {
        TargetVerifierSourceMechanism::jetspec_executable_h15_latent,
        TargetVerifierSourceMechanism::jetspec_h16_latent_extension,
        TargetVerifierSourceMechanism::jetspec_h16_argmax_diagnostic,
    }) {
        const auto depth = mechanism
                == TargetVerifierSourceMechanism::jetspec_executable_h15_latent
            ? 15U : 16U;
        for (const auto budget : jetspec_budgets) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::jetspec, mechanism,
                {.value = budget, .root_convention = SourceRootConvention::included},
                static_cast<std::uint16_t>(depth), 7U,
                TargetVerifierCandidateDomain::full_vocabulary,
                0U, 0U, 0U,
                {.raw_row_count = budget, .packed_row_count = budget,
                 .kv_row_count = budget, .score_event_count = 2U * (budget - 1U)}
            ));
        }
    }

    constexpr std::array<std::uint32_t, 3> full_path_budgets {16U, 64U, 256U};
    for (const auto mechanism : {
        TargetVerifierSourceMechanism::full_path_jetspec_executable,
        TargetVerifierSourceMechanism::full_path_jetspec_h16_latent,
        TargetVerifierSourceMechanism::full_path_jetspec_h16_argmax,
        TargetVerifierSourceMechanism::full_path_ddtree,
    }) {
        const auto depth = mechanism
                == TargetVerifierSourceMechanism::full_path_jetspec_executable
            ? 15U : 16U;
        for (const auto budget : full_path_budgets) {
            registry.add(rooted_shape(
                TargetVerifierSourceFamily::full_path, mechanism,
                {.value = budget, .root_convention = SourceRootConvention::excluded},
                static_cast<std::uint16_t>(depth),
                mechanism == TargetVerifierSourceMechanism::full_path_ddtree ? 0U : 7U,
                TargetVerifierCandidateDomain::full_vocabulary,
                0U, mechanism == TargetVerifierSourceMechanism::full_path_ddtree ? 64U : 0U,
                0U,
                {.raw_row_count = FullPathSupertreeMax,
                 .packed_row_count = budget + 1U,
                 .kv_row_count = FullPathSupertreeMax}
            ));
        }
    }

    for (const auto categories : {1U, 2U, 4U, 8U}) {
        const auto rows = 1U + 15U * categories;
        registry.add(rooted_shape(
            TargetVerifierSourceFamily::dblast,
            TargetVerifierSourceMechanism::dblast_exact_seed_category_trie,
            {.value = rows, .root_convention = SourceRootConvention::included},
            15U, static_cast<std::uint16_t>(categories),
            TargetVerifierCandidateDomain::full_vocabulary,
            0U, 0U, static_cast<std::uint16_t>(categories),
            {.raw_row_count = rows, .packed_row_count = rows, .kv_row_count = rows}
        ));
    }
    registry.add(rooted_shape(
        TargetVerifierSourceFamily::eagle3,
        TargetVerifierSourceMechanism::eagle3_final_tree,
        {.value = EAGLE3VerifierRowMax, .root_convention = SourceRootConvention::included},
        8U, 10U, TargetVerifierCandidateDomain::full_vocabulary,
        0U, 10U, 10U,
        {.raw_row_count = EAGLE3RawSupertreeRowMax,
         .packed_row_count = EAGLE3VerifierRowMax,
         .kv_row_count = EAGLE3ExpansionMaskKeyMax,
         .score_event_count = EAGLE3RawCandidateNodeMax}
    ));

    const auto add_redrafter = [&](
        TargetVerifierSourceMechanism mechanism,
        std::uint32_t source_rows_per_beam,
        std::uint32_t width,
        std::uint32_t physical_rows,
        std::uint32_t non_root_rows,
        std::uint16_t depth,
        std::uint32_t raw_rows,
        std::uint32_t kv_rows,
        ProposalGraphKind topology
    ) {
        auto shape = rooted_shape(
            TargetVerifierSourceFamily::redrafter, mechanism,
            {.value = source_rows_per_beam,
             .root_convention = SourceRootConvention::included},
            depth, static_cast<std::uint16_t>(width),
            TargetVerifierCandidateDomain::explicit_top_k,
            50U, 0U, static_cast<std::uint16_t>(width),
            {.raw_row_count = raw_rows, .packed_row_count = physical_rows,
             .kv_row_count = kv_rows}
        );
        shape.topology = topology;
        shape.resolved_budget = {
            .draft_node_cap = non_root_rows,
            .input_row_cap = physical_rows,
        };
        shape.physical_target_rows = physical_rows;
        registry.add(shape);
    };
    for (std::uint32_t width = 1U; width <= 4U; ++width) {
        add_redrafter(
            TargetVerifierSourceMechanism::redrafter_mlx_repeated_root,
            4U, width, 4U * width, 3U * width, 3U,
            4U * width, 4U * width, ProposalGraphKind::repeated_root_beam_batch
        );
        add_redrafter(
            TargetVerifierSourceMechanism::redrafter_article_prefix_deduplicated,
            5U, width, 1U + 4U * width, 4U * width, 4U,
            5U * width, 1U + 4U * width, ProposalGraphKind::forest
        );
        add_redrafter(
            TargetVerifierSourceMechanism::redrafter_pytorch_l4_compressed,
            4U, width, 1U + 3U * width, 3U * width, 3U,
            4U * width, 4U * width, ProposalGraphKind::forest
        );
    }
    add_redrafter(
        TargetVerifierSourceMechanism::redrafter_pytorch_l5_w45_compressed,
        5U, 45U, 181U, 180U, 4U,
        ReDrafterRawBeamEntryMax, ReDrafterPyUnpackedCacheRowMax,
        ProposalGraphKind::forest
    );

    if (registry.count != registry.shapes.size()) {
        throw std::logic_error("target verifier source-shape registry count drifted");
    }
    return registry.shapes;
}

}  // namespace

std::span<const TargetVerifierSourceShape> target_verifier_source_shapes() {
    static const auto shapes = make_registry();
    return shapes;
}

std::string_view target_verifier_source_mechanism_name(
    TargetVerifierSourceMechanism mechanism
) noexcept {
    switch (mechanism) {
        case TargetVerifierSourceMechanism::bridging_independent_marginal: return "bridging-independent-marginal";
        case TargetVerifierSourceMechanism::ddtree_fixed: return "ddtree-fixed";
        case TargetVerifierSourceMechanism::caddtree_adaptive: return "caddtree-adaptive";
        case TargetVerifierSourceMechanism::treeflash_paper_h16_full_v_lazy: return "treeflash-paper-h16-full-v-lazy";
        case TargetVerifierSourceMechanism::treeflash_executable_h15_top_k: return "treeflash-executable-h15-top-k";
        case TargetVerifierSourceMechanism::treeflash_executable_h15_full_v_transfer: return "treeflash-executable-h15-full-v-transfer";
        case TargetVerifierSourceMechanism::pctree_parent_conditioned_markov: return "pctree-parent-conditioned-markov";
        case TargetVerifierSourceMechanism::pctree_parent_independent_control: return "pctree-parent-independent-control";
        case TargetVerifierSourceMechanism::pctree_dspark_chain_control: return "pctree-dspark-chain-control";
        case TargetVerifierSourceMechanism::dominotree_depth_synchronous_with_sequential_oracle: return "dominotree-depth-synchronous-with-sequential-oracle";
        case TargetVerifierSourceMechanism::dartree_deferred_pruned: return "dartree-deferred-pruned";
        case TargetVerifierSourceMechanism::dartree_fixed_balanced: return "dartree-fixed-balanced";
        case TargetVerifierSourceMechanism::jetspec_executable_h15_latent: return "jetspec-executable-h15-latent";
        case TargetVerifierSourceMechanism::jetspec_h16_latent_extension: return "jetspec-h16-latent-extension";
        case TargetVerifierSourceMechanism::jetspec_h16_argmax_diagnostic: return "jetspec-h16-argmax-diagnostic";
        case TargetVerifierSourceMechanism::full_path_jetspec_executable: return "full-path-jetspec-executable";
        case TargetVerifierSourceMechanism::full_path_jetspec_h16_latent: return "full-path-jetspec-h16-latent";
        case TargetVerifierSourceMechanism::full_path_jetspec_h16_argmax: return "full-path-jetspec-h16-argmax";
        case TargetVerifierSourceMechanism::full_path_ddtree: return "full-path-ddtree";
        case TargetVerifierSourceMechanism::dblast_exact_seed_category_trie: return "dblast-exact-seed-category-trie";
        case TargetVerifierSourceMechanism::eagle3_final_tree: return "eagle3-final-tree";
        case TargetVerifierSourceMechanism::redrafter_mlx_repeated_root: return "redrafter-mlx-repeated-root";
        case TargetVerifierSourceMechanism::redrafter_article_prefix_deduplicated: return "redrafter-article-prefix-deduplicated";
        case TargetVerifierSourceMechanism::redrafter_pytorch_l4_compressed: return "redrafter-pytorch-l4-compressed";
        case TargetVerifierSourceMechanism::redrafter_pytorch_l5_w45_compressed: return "redrafter-pytorch-l5-w45-compressed";
    }
    return "unknown";
}

}  // namespace local_gguf_engine::engine
