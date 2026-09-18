#pragma once

#include "local_gguf_engine/engine/feature_bank.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>

namespace local_gguf_engine::frontends::cli {

struct SpeculationTraceResumeGroupPlan {
    std::size_t published_prefix_turn_count = 0U;
    bool fully_published = false;
};

enum class SpeculationTraceFeatureProfile : std::uint8_t {
    all_target_taps,
    final_normalized,
};

// Collection profiles are explicit because tap availability is part of an
// immutable trace's feature-contract identity.  A reduced profile is a valid
// corpus only for registry cells whose complete tap set it contains.
[[nodiscard]] std::vector<engine::FeatureTapSpec> plan_speculation_trace_features(
    SpeculationTraceFeatureProfile profile,
    std::uint32_t target_layer_count
);

// A resumable conversation may contain either a complete published trace set
// or one contiguous published prefix. A hole would make the cache state needed
// by a later trace unauthenticated, so it is rejected rather than guessed.
[[nodiscard]] SpeculationTraceResumeGroupPlan plan_speculation_trace_resume_group(
    std::span<const std::uint8_t> published_turns
);

int run_speculation_trace_cli(
    int argc,
    const char* const* argv,
    std::ostream& output,
    std::ostream& error
);

}  // namespace local_gguf_engine::frontends::cli
