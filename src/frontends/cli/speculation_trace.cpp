#include "local_gguf_engine/frontends/cli/speculation_trace.hpp"

#include "local_gguf_engine/engine/speculation_trace.hpp"

#include "local_gguf_engine/engine/gemma2_model.hpp"
#include "local_gguf_engine/engine/gemma2_chat_session.hpp"
#include "local_gguf_engine/engine/gemma2_runtime.hpp"
#include "local_gguf_engine/engine/model.hpp"
#include "local_gguf_engine/model_format/gguf_reader.hpp"
#include "local_gguf_engine/tokenizer/tokenizer.hpp"
#include "local_gguf_engine/tensor/backend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

namespace local_gguf_engine::frontends::cli {
namespace {

[[nodiscard]] runtime::Sha256Digest fixture_digest(std::uint8_t seed) {
    runtime::Sha256Digest result;
    for (std::size_t index = 0U; index < result.bytes.size(); ++index) {
        result.bytes[index] = static_cast<std::uint8_t>(seed + index);
    }
    return result;
}

void hash_u32(runtime::Sha256& hash, std::uint32_t value) {
    const std::array<std::uint8_t, 4U> bytes {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8U),
        static_cast<std::uint8_t>(value >> 16U),
        static_cast<std::uint8_t>(value >> 24U),
    };
    hash.update(bytes);
}

void hash_u64(runtime::Sha256& hash, std::uint64_t value) {
    std::array<std::uint8_t, 8U> bytes {};
    for (std::size_t index = 0U; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
    }
    hash.update(bytes);
}

void hash_text(runtime::Sha256& hash, std::string_view value) {
    hash_u64(hash, value.size());
    hash.update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size()
    ));
}

[[nodiscard]] std::vector<std::uint8_t> read_bounded_file(
    const std::filesystem::path& path,
    std::uint64_t maximum
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > maximum) {
        throw std::runtime_error("file exceeds bounded input size: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty() && !input.read(
        reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())
    )) throw std::runtime_error("cannot read " + path.string());
    return bytes;
}

[[nodiscard]] bool parse_u64(std::string_view text, std::uint64_t& value) noexcept {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc {} && result.ptr == text.data() + text.size();
}

[[nodiscard]] std::vector<std::string_view> split_tabs(std::string_view line) {
    std::vector<std::string_view> fields;
    std::size_t start = 0U;
    while (true) {
        const auto tab = line.find('\t', start);
        if (tab == std::string_view::npos) {
            fields.push_back(line.substr(start));
            return fields;
        }
        fields.push_back(line.substr(start, tab - start));
        start = tab + 1U;
    }
}

[[nodiscard]] std::string decode_base64(std::string_view input) {
    static constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (input.empty() || input.size() % 4U != 0U || input.size() > (1U << 26U)) {
        throw std::invalid_argument("manifest prompt base64 length is invalid");
    }
    std::string output;
    output.reserve(input.size() / 4U * 3U);
    for (std::size_t offset = 0U; offset < input.size(); offset += 4U) {
        std::array<std::uint32_t, 4U> values {};
        auto padding = std::uint32_t {0U};
        for (std::size_t lane = 0U; lane < 4U; ++lane) {
            const auto character = input[offset + lane];
            if (character == '=') {
                if (lane < 2U || offset + 4U != input.size()) {
                    throw std::invalid_argument("manifest prompt base64 padding is invalid");
                }
                ++padding;
                values[lane] = 0U;
            } else {
                if (padding != 0U) {
                    throw std::invalid_argument("manifest prompt base64 padding is not terminal");
                }
                const auto found = alphabet.find(character);
                if (found == std::string_view::npos) {
                    throw std::invalid_argument("manifest prompt base64 character is invalid");
                }
                values[lane] = static_cast<std::uint32_t>(found);
            }
        }
        if (padding > 2U) throw std::invalid_argument("manifest prompt base64 padding count");
        const auto bits = (values[0] << 18U) | (values[1] << 12U)
            | (values[2] << 6U) | values[3];
        output.push_back(static_cast<char>(bits >> 16U));
        if (padding < 2U) output.push_back(static_cast<char>(bits >> 8U));
        if (padding == 0U) output.push_back(static_cast<char>(bits));
    }
    if (output.empty() || output.find('\0') != std::string::npos) {
        throw std::invalid_argument("manifest prompt is empty or contains NUL");
    }
    return output;
}

struct CollectionTurn {
    std::uint64_t sequence_id = 0U;
    std::uint32_t turn_index = 0U;
    runtime::Sha256Digest source_group;
    runtime::Sha256Digest source_cluster;
    runtime::Sha256Digest conversation;
    runtime::Sha256Digest prompt_derivation;
    engine::SpeculationTraceSplit split =
        engine::SpeculationTraceSplit::training_development;
    engine::SpeculationTraceWorkload workload =
        engine::SpeculationTraceWorkload::instruction_chat;
    std::uint64_t generation_seed = 0U;
    std::uint64_t maximum_new_tokens = 0U;
    std::string prompt;
};

struct BalancedCollectionGoal {
    std::uint32_t horizon = 0U;
    std::uint64_t training_origins_per_workload_seed = 0U;
    std::uint32_t minimum_validation_clusters_per_workload = 0U;
};

struct BalancedCollectionCoverage {
    std::array<std::map<std::uint64_t, std::uint64_t>, 8U> training_origins;
    std::array<std::set<std::string>, 8U> validation_clusters;
};

[[nodiscard]] std::vector<CollectionTurn> parse_collection_manifest(
    std::span<const std::uint8_t> bytes,
    const runtime::Sha256Digest& registry_digest
) {
    if (bytes.empty()) throw std::invalid_argument("collection manifest is empty");
    const auto text = std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()
    );
    std::vector<CollectionTurn> turns;
    std::size_t offset = 0U;
    auto line_number = std::uint64_t {0U};
    while (offset < text.size()) {
        const auto end = text.find('\n', offset);
        auto line = text.substr(offset, end == std::string_view::npos
            ? text.size() - offset : end - offset);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1U);
        ++line_number;
        offset = end == std::string_view::npos ? text.size() : end + 1U;
        if (line_number == 1U) {
            const auto header = split_tabs(line);
            if (header.size() != 2U || header[0] != "LGSOURCE4"
                || runtime::parse_sha256_hex(header[1]) != registry_digest) {
                throw std::invalid_argument("collection manifest magic/version is invalid");
            }
            continue;
        }
        if (line.empty()) throw std::invalid_argument("collection manifest has empty row");
        const auto fields = split_tabs(line);
        if (fields.size() != 11U) {
            throw std::invalid_argument("collection manifest row must have eleven fields");
        }
        CollectionTurn turn;
        std::uint64_t turn_index = 0U;
        std::uint64_t split = 0U;
        std::uint64_t workload = 0U;
        if (!parse_u64(fields[0], turn.sequence_id) || turn.sequence_id == 0U
            || !parse_u64(fields[1], turn_index)
            || turn_index > std::numeric_limits<std::uint32_t>::max()
            || !parse_u64(fields[6], split)
            || split < static_cast<std::uint8_t>(
                engine::SpeculationTraceSplit::training_development)
            || split > static_cast<std::uint8_t>(
                engine::SpeculationTraceSplit::sealed_admission)
            || !parse_u64(fields[7], workload)
            || workload < static_cast<std::uint8_t>(
                engine::SpeculationTraceWorkload::instruction_chat)
            || workload > static_cast<std::uint8_t>(
                engine::SpeculationTraceWorkload::repeated_retrieval)
            || !parse_u64(fields[8], turn.generation_seed)
            || !parse_u64(fields[9], turn.maximum_new_tokens)) {
            throw std::invalid_argument("collection manifest numeric field is invalid");
        }
        turn.turn_index = static_cast<std::uint32_t>(turn_index);
        turn.source_group = runtime::parse_sha256_hex(fields[2]);
        turn.source_cluster = runtime::parse_sha256_hex(fields[3]);
        turn.conversation = runtime::parse_sha256_hex(fields[4]);
        turn.prompt_derivation = runtime::parse_sha256_hex(fields[5]);
        if (turn.source_group.is_zero() || turn.source_cluster.is_zero()
            || turn.conversation.is_zero() || turn.prompt_derivation.is_zero()) {
            throw std::invalid_argument("collection manifest identity digest is zero");
        }
        turn.split = static_cast<engine::SpeculationTraceSplit>(split);
        turn.workload = static_cast<engine::SpeculationTraceWorkload>(workload);
        turn.prompt = decode_base64(fields[10]);
        turns.push_back(std::move(turn));
    }
    if (turns.empty()) throw std::invalid_argument("collection manifest has no turns");
    for (std::size_t index = 0U; index < turns.size(); ++index) {
        if (index != 0U && turns[index].sequence_id <= turns[index - 1U].sequence_id) {
            throw std::invalid_argument("collection sequence IDs are not strictly increasing");
        }
        const auto starts_group = index == 0U
            || turns[index].conversation != turns[index - 1U].conversation
            || turns[index].generation_seed != turns[index - 1U].generation_seed;
        if ((starts_group && turns[index].turn_index != 0U)
            || (!starts_group && turns[index].turn_index != turns[index - 1U].turn_index + 1U)) {
            throw std::invalid_argument("collection conversation turns are not contiguous");
        }
    }
    return turns;
}

[[nodiscard]] std::vector<std::uint64_t> validate_collection_products(
    std::span<const CollectionTurn> turns
) {
    std::set<std::uint64_t> seed_set;
    for (const auto& turn : turns) seed_set.insert(turn.generation_seed);
    if (seed_set.size() != 5U) {
        throw std::invalid_argument(
            "collection manifest does not contain the frozen five-seed product"
        );
    }
    const std::vector<std::uint64_t> seeds(seed_set.begin(), seed_set.end());
    auto conversation_begin = std::size_t {0U};
    while (conversation_begin < turns.size()) {
        auto conversation_end = conversation_begin + 1U;
        while (conversation_end < turns.size()
               && turns[conversation_end].conversation
                    == turns[conversation_begin].conversation) {
            ++conversation_end;
        }
        const auto& first = turns[conversation_begin];
        std::map<std::uint64_t, std::span<const CollectionTurn>> products;
        auto seed_begin = conversation_begin;
        while (seed_begin < conversation_end) {
            auto seed_end = seed_begin + 1U;
            while (seed_end < conversation_end
                   && turns[seed_end].generation_seed
                        == turns[seed_begin].generation_seed) {
                ++seed_end;
            }
            if (!products.emplace(
                    turns[seed_begin].generation_seed,
                    turns.subspan(seed_begin, seed_end - seed_begin)
                ).second) {
                throw std::invalid_argument(
                    "collection manifest repeats a conversation seed product"
                );
            }
            seed_begin = seed_end;
        }
        if (products.size() != seeds.size()) {
            throw std::invalid_argument(
                "collection manifest conversation lacks a seed product"
            );
        }
        const auto expected = products.begin()->second;
        for (const auto seed : seeds) {
            const auto found = products.find(seed);
            if (found == products.end() || found->second.size() != expected.size()) {
                throw std::invalid_argument(
                    "collection manifest conversation seed products have different turn counts"
                );
            }
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                const auto& turn = found->second[index];
                const auto& reference = expected[index];
                if (turn.turn_index != index
                    || turn.source_group != first.source_group
                    || turn.source_cluster != first.source_cluster
                    || turn.conversation != first.conversation
                    || turn.split != first.split || turn.workload != first.workload
                    || turn.prompt_derivation != reference.prompt_derivation
                    || turn.maximum_new_tokens != reference.maximum_new_tokens
                    || turn.prompt != reference.prompt) {
                    throw std::invalid_argument(
                        "collection manifest conversation seed products are incompatible"
                    );
                }
            }
        }
        conversation_begin = conversation_end;
    }
    return seeds;
}

[[nodiscard]] bool development_validation_cluster(
    const runtime::Sha256Digest& cluster
) noexcept {
    auto value = std::uint64_t {0U};
    for (std::size_t index = 0U; index < 8U; ++index) {
        value |= static_cast<std::uint64_t>(cluster.bytes[index]) << (index * 8U);
    }
    return value % 10U == 0U;
}

[[nodiscard]] bool balanced_goal_satisfied(
    const BalancedCollectionCoverage& coverage,
    std::span<const std::uint64_t> seeds,
    const BalancedCollectionGoal& goal
) {
    for (std::size_t workload = 0U; workload < coverage.training_origins.size();
         ++workload) {
        for (const auto seed : seeds) {
            const auto found = coverage.training_origins[workload].find(seed);
            if (found == coverage.training_origins[workload].end()
                || found->second < goal.training_origins_per_workload_seed) {
                return false;
            }
        }
        if (coverage.validation_clusters[workload].size()
            < goal.minimum_validation_clusters_per_workload) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool balanced_conversation_needed(
    const BalancedCollectionCoverage& coverage,
    std::span<const std::uint64_t> seeds,
    const BalancedCollectionGoal& goal,
    const CollectionTurn& turn
) {
    const auto workload = static_cast<std::size_t>(turn.workload) - 1U;
    if (development_validation_cluster(turn.source_cluster)) {
        return coverage.validation_clusters[workload].size()
            < goal.minimum_validation_clusters_per_workload
            && !coverage.validation_clusters[workload].contains(
                turn.source_cluster.hex()
            );
    }
    return std::any_of(seeds.begin(), seeds.end(), [&](const auto seed) {
        const auto found = coverage.training_origins[workload].find(seed);
        return found == coverage.training_origins[workload].end()
            || found->second < goal.training_origins_per_workload_seed;
    });
}

void commit_balanced_conversation(
    BalancedCollectionCoverage& coverage,
    std::span<const std::uint64_t> seeds,
    const CollectionTurn& turn,
    const std::map<std::uint64_t, std::uint64_t>& origins
) {
    const auto workload = static_cast<std::size_t>(turn.workload) - 1U;
    if (development_validation_cluster(turn.source_cluster)) {
        if (std::all_of(seeds.begin(), seeds.end(), [&](const auto seed) {
                const auto found = origins.find(seed);
                return found != origins.end() && found->second != 0U;
            })) {
            coverage.validation_clusters[workload].insert(turn.source_cluster.hex());
        }
        return;
    }
    for (const auto seed : seeds) {
        const auto increment = origins.contains(seed) ? origins.at(seed) : 0U;
        auto& total = coverage.training_origins[workload][seed];
        if (increment > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error("balanced collection origin count overflows uint64");
        }
        total += increment;
    }
}

[[nodiscard]] runtime::Sha256Digest tokenizer_digest(
    const tokenizer::Tokenizer& tokenizer
) {
    runtime::Sha256 hash;
    hash_text(hash, "local-gguf-engine/tokenizer/v1");
    hash_u32(hash, static_cast<std::uint32_t>(tokenizer.model()));
    hash_u64(hash, tokenizer.vocabulary().size());
    for (const auto& token : tokenizer.vocabulary()) {
        hash_text(hash, token.text);
        hash_u32(hash, token.score.has_value() ? 1U : 0U);
        if (token.score.has_value()) hash_u32(hash, std::bit_cast<std::uint32_t>(*token.score));
        hash_u32(hash, token.type.has_value() ? 1U : 0U);
        if (token.type.has_value()) hash_u32(hash, std::bit_cast<std::uint32_t>(*token.type));
    }
    const auto& special = tokenizer.special_tokens();
    for (const auto token : {
            special.bos_token_id, special.eos_token_id, special.unknown_token_id,
        }) {
        hash_u32(hash, token.has_value() ? 1U : 0U);
        hash_u32(hash, token.value_or(0U));
    }
    hash_u32(hash, special.add_bos_token ? 1U : 0U);
    hash_u32(hash, special.add_eos_token ? 1U : 0U);
    return hash.finalize();
}

[[nodiscard]] std::string_view required_chat_template(
    const model_format::GGUFMetadataSection& metadata
) {
    for (const auto& entry : metadata.entries) {
        if (entry.key == "tokenizer.chat_template") {
            if (entry.value.type != model_format::GGUFMetadataValueType::string) {
                throw std::invalid_argument("tokenizer.chat_template is not a string");
            }
            return std::get<std::string>(entry.value.data);
        }
    }
    throw std::invalid_argument("model has no tokenizer.chat_template");
}

[[nodiscard]] runtime::Sha256Digest trace_feature_contract_digest(
    std::span<const engine::FeatureTapSpec> taps,
    std::uint32_t hidden_width,
    const runtime::Sha256Digest& collector_executable,
    const std::optional<BalancedCollectionGoal>& balanced_goal
) {
    runtime::Sha256 hash;
    hash_text(hash, "local-gguf-engine/raw-target-feature-trace/v3");
    hash_u32(hash, engine::speculation_trace_schema_version);
    hash_text(hash, LOCAL_GGUF_ENGINE_BUILD);
    hash.update(collector_executable.bytes);
    const auto metal_library = tensor::metal_backend_library_source_digest();
    hash.update(metal_library.bytes);
    hash_text(hash, "resolved-target-output-policy:q8_0");
    hash_u32(hash, hidden_width);
    hash_u32(hash, static_cast<std::uint32_t>(tensor::DType::f16));
    hash_text(hash, "Metal-float-to-half-round-to-nearest-even");
    hash_u64(hash, taps.size());
    for (const auto& tap : taps) {
        hash_u32(hash, static_cast<std::uint32_t>(tap.stage));
        hash_u32(hash, tap.layer_index);
    }
    hash_u32(hash, balanced_goal.has_value() ? 1U : 0U);
    if (balanced_goal.has_value()) {
        hash_text(hash, "balanced-workload-seed-horizon-origin-and-validation-cluster/v1");
        hash_u32(hash, balanced_goal->horizon);
        hash_u64(hash, balanced_goal->training_origins_per_workload_seed);
        hash_u32(hash, balanced_goal->minimum_validation_clusters_per_workload);
    }
    return hash.finalize();
}

[[nodiscard]] std::uint32_t context_flags(
    std::uint32_t position,
    const engine::Gemma2ModelConfig& config,
    bool rebuilt
) noexcept {
    auto flags = std::uint32_t {0U};
    if (position < config.context_length / 4U) flags |= engine::speculation_trace_context_early;
    else if (position < 3U * config.context_length / 4U) flags |= engine::speculation_trace_context_middle;
    if (position >= config.sliding_window) flags |= engine::speculation_trace_context_sliding_wrap;
    if (position + 128U >= config.context_length) flags |= engine::speculation_trace_context_near_limit;
    if (position + 1U == config.context_length) flags |= engine::speculation_trace_context_global_boundary;
    else if (position + 512U >= config.context_length) flags |= engine::speculation_trace_context_global_tail;
    if (rebuilt) flags |= engine::speculation_trace_context_rebuilt;
    return flags;
}

[[nodiscard]] engine::TokenSamplerOptions collection_sampler(std::uint64_t seed) {
    return engine::TokenSamplerOptions {
        .temperature = 0.8F,
        .top_k = 40U,
        .top_p = 0.95F,
        .min_p = 0.05F,
        .repetition_penalty = 1.0F,
        .seed = seed,
        .repetition_window = 64U,
    };
}

[[nodiscard]] std::filesystem::path trace_path_for(
    const std::filesystem::path& directory,
    std::uint64_t sequence_id
) {
    std::ostringstream name;
    name << std::setw(20) << std::setfill('0') << sequence_id << ".lgtrace";
    return directory / name.str();
}

inline constexpr std::string_view balanced_collection_receipt_name =
    "balanced_collection.lgsreceipt";

[[nodiscard]] std::vector<std::uint64_t> published_trace_sequences(
    const std::filesystem::path& directory
) {
    std::vector<std::uint64_t> result;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".lgtrace") continue;
        const auto stem = entry.path().stem().string();
        auto sequence = std::uint64_t {0U};
        if (stem.size() != 20U || !parse_u64(stem, sequence) || sequence == 0U) {
            throw std::runtime_error("trace output contains an invalid trace filename");
        }
        result.push_back(sequence);
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::runtime_error("trace output contains duplicate sequence filenames");
    }
    return result;
}

[[nodiscard]] std::string balanced_collection_receipt(
    const runtime::Sha256Digest& source_manifest,
    const runtime::Sha256Digest& registry,
    const runtime::Sha256Digest& feature_contract,
    const BalancedCollectionGoal& goal,
    std::span<const std::uint64_t> sequences
) {
    std::ostringstream prefix;
    prefix << "LGBALANCED1\n"
           << "source_manifest_sha256\t" << source_manifest.hex() << '\n'
           << "registry_sha256\t" << registry.hex() << '\n'
           << "feature_contract_sha256\t" << feature_contract.hex() << '\n'
           << "algorithm\tbalanced-workload-seed-horizon-origin-and-validation-cluster/v1\n"
           << "horizon\t" << goal.horizon << '\n'
           << "training_origins_per_workload_seed\t"
           << goal.training_origins_per_workload_seed << '\n'
           << "minimum_validation_clusters_per_workload\t"
           << goal.minimum_validation_clusters_per_workload << '\n'
           << "sequence_count\t" << sequences.size() << '\n';
    for (const auto sequence : sequences) {
        prefix << "sequence\t" << sequence << '\n';
    }
    const auto payload = prefix.str();
    const auto digest = runtime::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()
    ));
    return payload + "sha256\t" + digest.hex() + "\n";
}

void publish_or_validate_balanced_collection_receipt(
    const std::filesystem::path& directory,
    std::string_view payload
) {
    const auto path = directory / balanced_collection_receipt_name;
    if (std::filesystem::exists(path)) {
        const auto bytes = read_bounded_file(path, 1ULL << 26U);
        if (bytes.size() != payload.size()
            || !std::equal(bytes.begin(), bytes.end(), payload.begin())) {
            throw std::runtime_error(
                "balanced collection receipt differs from authenticated output"
            );
        }
        return;
    }
    auto partial = path;
    partial += ".partial";
    if (std::filesystem::exists(partial)) {
        throw std::runtime_error("balanced collection receipt has an unpublished partial file");
    }
    {
        std::ofstream output(partial, std::ios::binary | std::ios::trunc);
        if (!output || !output.write(
                payload.data(), static_cast<std::streamsize>(payload.size())
            ) || !output.flush()) {
            throw std::runtime_error("could not write balanced collection receipt");
        }
    }
    std::error_code error;
    std::filesystem::rename(partial, path, error);
    if (error) {
        std::filesystem::remove(partial);
        throw std::runtime_error(
            "could not publish balanced collection receipt: " + error.message()
        );
    }
}

[[nodiscard]] engine::SpeculationTraceIdentity make_trace_identity(
    const engine::EngineModelPayloadReceipt& model,
    const engine::ResidentTargetPayloadReceipt& resident,
    const runtime::Sha256Digest& feature_contract,
    const runtime::Sha256Digest& tokenizer,
    const runtime::Sha256Digest& chat_template,
    const runtime::Sha256Digest& source_manifest,
    const runtime::Sha256Digest& registry,
    std::uint64_t seed
) {
    return engine::SpeculationTraceIdentity {
        .model = model,
        .resident = resident,
        .artifact = feature_contract,
        .tokenizer = tokenizer,
        .chat_template = chat_template,
        .source_manifest = source_manifest,
        .registry = registry,
        .sampler = collection_sampler(seed),
    };
}

class TurnTraceCollector final {
public:
    TurnTraceCollector(
        const CollectionTurn& turn,
        const engine::Gemma2ModelConfig& config,
        engine::RetrievalProposer& retrieval,
        engine::SpeculationTraceSequenceWriter& writer,
        std::uint32_t initial_context_generation,
        std::uint32_t previous_cache_position
    ) : turn_(turn), config_(config), retrieval_(retrieval), writer_(writer),
        context_generation_(initial_context_generation),
        previous_cache_position_(previous_cache_position) {}

    void consume_features(engine::TargetFeatureTraceBatch&& batch) {
        if (finished_ || batch.absolute_positions.empty()
            || batch.absolute_positions.size() != batch.input_token_ids.size()) {
            throw std::logic_error("trace collector received invalid or late feature batch");
        }
        for (std::size_t row = 0U; row < batch.absolute_positions.size(); ++row) {
            const auto position = batch.absolute_positions[row];
            const auto token = batch.input_token_ids[row];
            if (!saw_feature_) {
                rebuilt_ = previous_cache_position_ != 0U && position == 0U;
                if (rebuilt_) {
                    if (context_generation_ == std::numeric_limits<std::uint32_t>::max()) {
                        throw std::overflow_error("trace context generation overflows");
                    }
                    ++context_generation_;
                }
                saw_feature_ = true;
            }
            if (pending_.has_value()) {
                append_pending(token, std::nullopt, false, 0U);
            }
            if (position >= config_.context_length) {
                throw std::invalid_argument("trace feature position exceeds model context");
            }
            if (input_already_observed_) {
                const auto history = retrieval_.history();
                if (history.empty() || history.back() != token) {
                    throw std::logic_error("trace input diverged from selected token");
                }
                input_already_observed_ = false;
            } else {
                retrieval_.observe(token);
            }
            const auto proposal = retrieval_.propose();
            Pending pending;
            pending.position = position;
            pending.token = token;
            pending.kind = saw_selection_
                ? engine::SpeculationTraceRowKind::generated
                : engine::SpeculationTraceRowKind::prompt;
            pending.features = engine::make_speculation_trace_features(batch, row);
            pending.retrieval_eligible = engine::retrieval_proposal_is_profitable(proposal);
            if (pending.retrieval_eligible) {
                pending.retrieval_proposal = proposal.token_id;
            }
            pending_ = std::move(pending);
        }
    }

    void consume_selection(const engine::Gemma2TargetSelectionTraceEvent& event) {
        if (finished_ || !pending_.has_value()
            || pending_->position != event.target_position
            || pending_->token != event.input_token_id
            || event.sampler_options != collection_sampler(turn_.generation_seed)) {
            throw std::logic_error("trace selection does not match its target feature row");
        }
        auto selection = engine::make_speculation_trace_selection(
            event.compact_candidates,
            event.sampler_options,
            event.request,
            event.sampler_pre,
            event.draw_ticket,
            event.sampler_post,
            event.selected_token_id
        );
        append_pending(
            event.selected_token_id,
            std::move(selection),
            false,
            event.remaining_token_budget
        );
        retrieval_.observe(event.selected_token_id);
        input_already_observed_ = true;
        saw_selection_ = true;
        expects_terminal_leaf_ = event.terminal;
    }

    [[nodiscard]] runtime::Sha256Digest finish() {
        if (expects_terminal_leaf_ && pending_.has_value() && !input_already_observed_) {
            append_terminal_leaf();
        }
        if (!finished_ || pending_.has_value() || input_already_observed_) {
            throw std::logic_error("trace collection ended without an exact terminal selection");
        }
        return writer_.finish();
    }

    [[nodiscard]] std::uint32_t context_generation() const noexcept {
        return context_generation_;
    }

    [[nodiscard]] bool rebuilt() const noexcept { return rebuilt_; }

private:
    struct Pending {
        std::uint32_t position = 0U;
        std::uint32_t token = 0U;
        engine::SpeculationTraceRowKind kind = engine::SpeculationTraceRowKind::prompt;
        bool retrieval_eligible = false;
        std::optional<std::uint32_t> retrieval_proposal;
        std::vector<engine::SpeculationTraceFeature> features;
    };

    void append_pending(
        std::uint32_t label,
        std::optional<engine::SpeculationTraceSelection> selection,
        bool terminal,
        std::uint32_t remaining_budget
    ) {
        if (!pending_.has_value()) throw std::logic_error("trace row is absent");
        engine::SpeculationTraceRecord record;
        record.sequence_id = turn_.sequence_id;
        record.turn_index = turn_.turn_index;
        record.kind = pending_->kind;
        if (writer_.record_count() != 0U) {
            record.parent_record_index = writer_.record_count() - 1U;
        }
        record.target_position = pending_->position;
        record.draft_position = pending_->position + 1U;
        record.token_id = pending_->token;
        record.target_label_token_id = label;
        record.sampler_draw_count = selection.has_value()
            ? selection->sampler_pre.draw_count : 0U;
        record.terminal = terminal;
        record.source = {
            .source_group = turn_.source_group,
            .source_cluster = turn_.source_cluster,
            .conversation = turn_.conversation,
            .prompt_derivation = turn_.prompt_derivation,
            .split = turn_.split,
            .workload = turn_.workload,
            .context_generation = context_generation_,
            .context_flags = context_flags(pending_->position, config_, rebuilt_),
            .generation_seed = turn_.generation_seed,
        };
        record.remaining_token_budget = selection.has_value()
            ? remaining_budget : config_.context_length - record.draft_position;
        record.retrieval_eligible = pending_->retrieval_eligible;
        record.retrieval_proposal_token_id = pending_->retrieval_proposal;
        record.selection = std::move(selection);
        record.features = std::move(pending_->features);
        writer_.append(record);
        pending_.reset();
    }

    void append_terminal_leaf() {
        if (!pending_.has_value() || !expects_terminal_leaf_
            || pending_->kind != engine::SpeculationTraceRowKind::generated) {
            throw std::logic_error("trace terminal context leaf is invalid");
        }
        engine::SpeculationTraceRecord record;
        record.sequence_id = turn_.sequence_id;
        record.turn_index = turn_.turn_index;
        record.kind = pending_->kind;
        if (writer_.record_count() != 0U) {
            record.parent_record_index = writer_.record_count() - 1U;
        }
        record.target_position = pending_->position;
        record.draft_position = pending_->position + 1U;
        record.token_id = pending_->token;
        record.terminal = true;
        record.source = {
            .source_group = turn_.source_group,
            .source_cluster = turn_.source_cluster,
            .conversation = turn_.conversation,
            .prompt_derivation = turn_.prompt_derivation,
            .split = turn_.split,
            .workload = turn_.workload,
            .context_generation = context_generation_,
            .context_flags = context_flags(pending_->position, config_, rebuilt_),
            .generation_seed = turn_.generation_seed,
        };
        record.features = std::move(pending_->features);
        record.retrieval_eligible = pending_->retrieval_eligible;
        record.retrieval_proposal_token_id = pending_->retrieval_proposal;
        writer_.append(record);
        pending_.reset();
        expects_terminal_leaf_ = false;
        finished_ = true;
    }

    const CollectionTurn& turn_;
    const engine::Gemma2ModelConfig& config_;
    engine::RetrievalProposer& retrieval_;
    engine::SpeculationTraceSequenceWriter& writer_;
    std::optional<Pending> pending_;
    bool saw_selection_ = false;
    bool input_already_observed_ = false;
    bool expects_terminal_leaf_ = false;
    bool finished_ = false;
    bool saw_feature_ = false;
    bool rebuilt_ = false;
    std::uint32_t context_generation_ = 0U;
    std::uint32_t previous_cache_position_ = 0U;
};

void validate_existing_trace(
    const engine::AuthenticatedSpeculationTrace& trace,
    const engine::SpeculationTraceIdentity& identity,
    const CollectionTurn& turn
) {
    if (trace.identity() != identity || trace.records().empty()) {
        throw std::runtime_error("existing trace identity does not match this collection run");
    }
    for (const auto& record : trace.records()) {
        if (record.sequence_id != turn.sequence_id || record.turn_index != turn.turn_index
            || record.source.source_group != turn.source_group
            || record.source.source_cluster != turn.source_cluster
            || record.source.conversation != turn.conversation
            || record.source.prompt_derivation != turn.prompt_derivation
            || record.source.split != turn.split || record.source.workload != turn.workload
            || record.source.generation_seed != turn.generation_seed) {
            throw std::runtime_error("existing trace source provenance does not match manifest");
        }
    }
}

[[nodiscard]] std::vector<std::uint32_t> selected_trace_tokens(
    const engine::AuthenticatedSpeculationTrace& trace
) {
    std::vector<std::uint32_t> result;
    for (const auto& record : trace.records()) {
        if (record.selection.has_value()) result.push_back(*record.target_label_token_id);
    }
    return result;
}

[[nodiscard]] std::filesystem::path trace_partial_path_for(
    const std::filesystem::path& path
) {
    auto partial = path;
    partial += ".partial";
    return partial;
}

[[nodiscard]] std::filesystem::path trace_body_path_for(
    const std::filesystem::path& path
) {
    auto body = path;
    body += ".body.partial";
    return body;
}

int discard_unpublished_trace_fragments(
    const std::filesystem::path& directory,
    std::uint64_t sequence_id,
    std::ostream& output
) {
    if (sequence_id == 0U) {
        throw std::invalid_argument("trace sequence ID must be positive");
    }
    if (!std::filesystem::exists(directory)
        || !std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("trace output path is not an existing directory");
    }
    const auto path = trace_path_for(directory, sequence_id);
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to discard fragments for an already published trace"
        );
    }
    const auto partial = trace_partial_path_for(path);
    const auto body = trace_body_path_for(path);
    const auto had_partial = std::filesystem::exists(partial);
    const auto had_body = std::filesystem::exists(body);
    if (!had_partial && !had_body) {
        throw std::runtime_error("trace has no unpublished fragments to discard");
    }
    for (const auto& fragment : {partial, body}) {
        if (!std::filesystem::exists(fragment)) continue;
        std::error_code error;
        if (!std::filesystem::remove(fragment, error) || error) {
            throw std::runtime_error(
                "could not discard unpublished trace fragment " + fragment.string()
                + (error ? ": " + error.message() : std::string {})
            );
        }
    }
    output << "{\"sequence_id\":" << sequence_id
           << ",\"discarded_envelope_partial\":"
           << (had_partial ? "true" : "false")
           << ",\"discarded_body_partial\":"
           << (had_body ? "true" : "false") << "}\n";
    return 0;
}

void reject_unpublished_trace_fragments(const std::filesystem::path& path) {
    if (std::filesystem::exists(trace_partial_path_for(path))
        || std::filesystem::exists(trace_body_path_for(path))) {
        throw std::runtime_error(
            "trace output contains an unpublished partial file for " + path.string()
            + "; run `local-gguf-speculation-trace discard-partial <output-directory> "
            + path.stem().string() + "` before resuming"
        );
    }
}

void add_collected_origins(
    std::uint64_t& total,
    std::uint64_t increment
) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - total) {
        throw std::overflow_error("collected trace origin count overflows uint64");
    }
    total += increment;
}

struct ExistingTraceState {
    std::uint64_t selection_count = 0U;
    std::uint64_t eligible_origin_count = 0U;
    std::uint32_t context_generation = 0U;
    std::uint32_t final_draft_position = 0U;
    bool rebuilt_context = false;
    runtime::Sha256Digest envelope;
};

[[nodiscard]] ExistingTraceState validate_existing_trace_state(
    const engine::AuthenticatedSpeculationTrace& trace,
    const engine::SpeculationTraceIdentity& identity,
    const CollectionTurn& turn,
    const engine::Gemma2ModelConfig& config,
    std::uint32_t previous_context_generation,
    std::optional<std::uint32_t> expected_target_position,
    std::optional<std::uint32_t> origin_horizon = std::nullopt
) {
    validate_existing_trace(trace, identity, turn);
    const auto records = trace.records();
    const auto rebuilt = (records.front().source.context_flags
        & engine::speculation_trace_context_rebuilt) != 0U;
    if (previous_context_generation == std::numeric_limits<std::uint32_t>::max()
        && rebuilt) {
        throw std::overflow_error("resumed trace context generation overflows");
    }
    const auto expected_generation = previous_context_generation
        + static_cast<std::uint32_t>(rebuilt);
    if (records.front().source.context_generation != expected_generation
        || records.front().target_position
            != (rebuilt ? 0U : expected_target_position.value_or(0U))) {
        throw std::runtime_error("existing trace context sequence does not match manifest group");
    }
    for (const auto& record : records) {
        const auto record_rebuilt = (record.source.context_flags
            & engine::speculation_trace_context_rebuilt) != 0U;
        if (record_rebuilt != rebuilt
            || record.source.context_flags
                != context_flags(record.target_position, config, rebuilt)) {
            throw std::runtime_error("existing trace context provenance is inconsistent");
        }
    }
    const auto selection_count = static_cast<std::uint64_t>(std::count_if(
        records.begin(), records.end(), [](const auto& record) {
            return record.selection.has_value();
        }
    ));
    auto causal_run = std::uint64_t {0U};
    auto eligible_origin_count = std::uint64_t {0U};
    const engine::SpeculationTraceRecord* previous = nullptr;
    for (const auto& record : records) {
        const auto eligible = record.selection.has_value()
            && record.target_label_token_id.has_value();
        if (eligible && previous != nullptr
            && previous->selection.has_value()
            && previous->target_label_token_id.has_value()
            && previous->sequence_id == record.sequence_id
            && previous->turn_index == record.turn_index
            && previous->source.source_cluster == record.source.source_cluster
            && previous->source.conversation == record.source.conversation
            && previous->source.split == record.source.split
            && previous->source.workload == record.source.workload
            && previous->source.context_generation == record.source.context_generation
            && previous->target_position + 1U == record.target_position
            && *previous->target_label_token_id == record.token_id) {
            ++causal_run;
        } else {
            causal_run = eligible ? 1U : 0U;
        }
        if (origin_horizon.has_value()
            && causal_run >= static_cast<std::uint64_t>(*origin_horizon) + 1U) {
            ++eligible_origin_count;
        }
        previous = &record;
    }
    return ExistingTraceState {
        .selection_count = selection_count,
        .eligible_origin_count = eligible_origin_count,
        .context_generation = expected_generation,
        .final_draft_position = records.back().draft_position,
        .rebuilt_context = rebuilt,
        .envelope = trace.envelope_digest(),
    };
}

int collect_traces(
    const std::filesystem::path& model_path,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& registry_path,
    const std::filesystem::path& output_directory,
    std::optional<std::uint64_t> maximum_origins,
    SpeculationTraceFeatureProfile feature_profile,
    std::optional<BalancedCollectionGoal> balanced_goal,
    const runtime::Sha256Digest& collector_executable,
    std::ostream& output
) {
    const auto manifest_bytes = read_bounded_file(manifest_path, 1ULL << 30U);
    const auto registry_bytes = read_bounded_file(registry_path, 1ULL << 26U);
    const auto manifest_digest = runtime::sha256(manifest_bytes);
    const auto registry_digest = runtime::sha256(registry_bytes);
    if (registry_digest.is_zero()) throw std::runtime_error("registry digest is unexpectedly zero");
    const auto turns = parse_collection_manifest(manifest_bytes, registry_digest);
    const auto generation_seeds = validate_collection_products(turns);
    if (balanced_goal.has_value()
        && (maximum_origins.has_value() || balanced_goal->horizon == 0U
            || balanced_goal->horizon > 23U
            || balanced_goal->training_origins_per_workload_seed == 0U
            || balanced_goal->minimum_validation_clusters_per_workload == 0U)) {
        throw std::invalid_argument("balanced trace collection goal is invalid");
    }
    for (const auto& turn : turns) {
        if (turn.maximum_new_tokens > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("manifest maximum_new_tokens exceeds size_t");
        }
    }

    if (std::filesystem::exists(output_directory)) {
        if (!std::filesystem::is_directory(output_directory)) {
            throw std::invalid_argument("trace output path is not a directory");
        }
    } else if (!std::filesystem::create_directories(output_directory)) {
        throw std::runtime_error("could not create trace output directory");
    }

    auto maximum_existing_sequence = std::uint64_t {0U};
    if (balanced_goal.has_value()) {
        const auto known_sequence = [&](std::uint64_t sequence) {
            const auto found = std::lower_bound(
                turns.begin(), turns.end(), sequence,
                [](const CollectionTurn& turn, std::uint64_t value) {
                    return turn.sequence_id < value;
                }
            );
            return found != turns.end() && found->sequence_id == sequence;
        };
        for (const auto& entry : std::filesystem::directory_iterator(output_directory)) {
            if (!entry.is_regular_file()) continue;
            const auto name = entry.path().filename().string();
            if (name.ends_with(".partial")) {
                throw std::runtime_error(
                    "balanced trace output contains an unpublished partial file; "
                    "use `local-gguf-speculation-trace discard-partial "
                    "<output-directory> <sequence-id>` for an interrupted trace"
                );
            }
            if (entry.path().extension() != ".lgtrace") continue;
            const auto stem = entry.path().stem().string();
            auto sequence = std::uint64_t {0U};
            if (stem.size() != 20U || !parse_u64(stem, sequence)
                || sequence == 0U || !known_sequence(sequence)) {
                throw std::runtime_error(
                    "balanced trace output contains an unknown trace filename"
                );
            }
            maximum_existing_sequence = std::max(maximum_existing_sequence, sequence);
        }
    }

    output << "{\"event\":\"collection_manifest_authenticated\""
           << ",\"manifest_turns\":" << turns.size()
           << ",\"existing_trace_count\":"
           << published_trace_sequences(output_directory).size()
           << ",\"mode\":\""
           << (balanced_goal.has_value() ? "balanced" : "prefix")
           << "\"}\n" << std::flush;

    auto file_model = model_format::read_file_backed_gguf_model(model_path);
    const auto template_text = required_chat_template(file_model.view().directory.metadata);
    const auto template_receipt = runtime::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(template_text.data()), template_text.size()
    ));
    auto prepared_state = engine::Gemma2ChatSessionState::prepare(std::move(file_model),
        engine::Gemma2ChatSessionOptions {
            .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
            .speculative_decode = false,
        }, true);
    const auto& model = prepared_state.model();
    if (model.architecture() != engine::ModelArchitecture::gemma2
        || !model.payload_receipt().has_value()) {
        throw std::invalid_argument("trace collection requires a verified Gemma 2 model");
    }
    const auto& target_tokenizer = prepared_state.tokenizer();
    const auto config = model.gemma2_config();
    const auto taps = plan_speculation_trace_features(
        feature_profile, config.block_count
    );
    const auto tokenizer_receipt = tokenizer_digest(target_tokenizer);
    const auto feature_contract = trace_feature_contract_digest(
        taps, config.embedding_length, collector_executable, balanced_goal
    );
    const auto model_receipt = *model.payload_receipt();
    const auto resident_receipt = *prepared_state.prepared().resident_receipt();

    // Constructing a chat session performs a live target warm-up. Keep the
    // prepared model state idle until a group actually has a missing trace.
    std::unique_ptr<engine::Gemma2ChatSession> session;
    engine::RetrievalProposer retrieval;
    retrieval.rebuild({});
    auto completed = std::uint64_t {0U};
    auto collected = std::uint64_t {0U};
    auto resumed = std::uint64_t {0U};
    auto authenticated_without_replay = std::uint64_t {0U};
    auto replayed = std::uint64_t {0U};
    auto collected_origins = std::uint64_t {0U};
    BalancedCollectionCoverage balanced_coverage;
    auto active_conversation_end = std::size_t {0U};
    std::map<std::uint64_t, std::uint64_t> active_conversation_origins;
    auto group_begin = std::size_t {0U};
    while (group_begin < turns.size()
           && (balanced_goal.has_value()
               ? (!balanced_goal_satisfied(
                    balanced_coverage, generation_seeds, *balanced_goal
                  ) || turns[group_begin].sequence_id <= maximum_existing_sequence)
               : (!maximum_origins.has_value()
                  || collected_origins < *maximum_origins))) {
        if (balanced_goal.has_value() && group_begin >= active_conversation_end) {
            active_conversation_end = group_begin + 1U;
            while (active_conversation_end < turns.size()
                   && turns[active_conversation_end].conversation
                        == turns[group_begin].conversation) {
                ++active_conversation_end;
            }
            auto any_published = false;
            for (auto index = group_begin; index < active_conversation_end; ++index) {
                const auto path = trace_path_for(
                    output_directory, turns[index].sequence_id
                );
                reject_unpublished_trace_fragments(path);
                any_published = any_published || std::filesystem::exists(path);
            }
            const auto needed = balanced_conversation_needed(
                    balanced_coverage, generation_seeds, *balanced_goal,
                    turns[group_begin]
                );
            if (any_published && !needed) {
                throw std::runtime_error(
                    "balanced trace output contains a conversation outside its canonical schedule"
                );
            }
            if (!needed) {
                group_begin = active_conversation_end;
                active_conversation_origins.clear();
                continue;
            }
            active_conversation_origins.clear();
        }
        auto group_end = group_begin + 1U;
        while (group_end < turns.size()
               && turns[group_end].conversation == turns[group_begin].conversation
               && turns[group_end].generation_seed == turns[group_begin].generation_seed) {
            ++group_end;
        }

        std::vector<std::uint8_t> published(group_end - group_begin, 0U);
        for (auto index = group_begin; index < group_end; ++index) {
            const auto path = trace_path_for(output_directory, turns[index].sequence_id);
            reject_unpublished_trace_fragments(path);
            published[index - group_begin] = std::filesystem::exists(path) ? 1U : 0U;
        }
        const auto resume_plan = plan_speculation_trace_resume_group(published);
        auto context_generation = std::uint32_t {0U};
        auto expected_target_position = std::optional<std::uint32_t> {};

        // A complete group is immutable evidence. Authenticating each envelope,
        // record, identity, and cross-turn context edge is sufficient; replaying
        // the target would only reproduce work already bound into that evidence.
        if (resume_plan.fully_published) {
            for (auto index = group_begin; index < group_end; ++index) {
                if (maximum_origins.has_value() && collected_origins >= *maximum_origins) break;
                const auto& turn = turns[index];
                const auto identity = make_trace_identity(
                    model_receipt, resident_receipt, feature_contract, tokenizer_receipt,
                    template_receipt, manifest_digest, registry_digest, turn.generation_seed
                );
                const auto trace = engine::load_speculation_trace_file(
                    trace_path_for(output_directory, turn.sequence_id)
                );
                const auto state = validate_existing_trace_state(
                    trace, identity, turn, config, context_generation,
                    expected_target_position,
                    balanced_goal.has_value()
                        ? std::optional<std::uint32_t>(balanced_goal->horizon)
                        : std::nullopt
                );
                context_generation = state.context_generation;
                expected_target_position = state.final_draft_position;
                add_collected_origins(collected_origins, state.selection_count);
                if (balanced_goal.has_value()) {
                    add_collected_origins(
                        active_conversation_origins[turn.generation_seed],
                        state.eligible_origin_count
                    );
                }
                ++resumed;
                ++authenticated_without_replay;
                ++completed;
                output << "{\"sequence_id\":" << turn.sequence_id
                       << ",\"trace_sha256\":\"" << state.envelope.hex()
                       << "\",\"resume\":\"authenticated\"}\n" << std::flush;
            }
            group_begin = group_end;
            if (balanced_goal.has_value()
                && group_begin == active_conversation_end) {
                commit_balanced_conversation(
                    balanced_coverage, generation_seeds, turns[group_begin - 1U],
                    active_conversation_origins
                );
            }
            continue;
        }

        // Only an unfinished multi-turn group needs its published prefix replayed:
        // the live chat/cache state is required to collect the first missing turn.
        if (!session) {
            session = std::make_unique<engine::Gemma2ChatSession>(
                std::move(prepared_state),
                engine::Gemma2ChatSessionOptions {
                    .max_new_tokens = 0U,
                    .backend = runtime::BackendKind::metal,
                    .output_head_policy = engine::Gemma2OutputHeadPolicy::q8_0,
                    .context_shift = true,
                    .speculative_decode = false,
                    .sampler = collection_sampler(turns[group_begin].generation_seed),
                }
            );
        } else {
            session->set_sampler_options(collection_sampler(
                turns[group_begin].generation_seed
            ));
        }
        retrieval.rebuild({});
        auto& active_session = *session;
        for (auto index = group_begin; index < group_end; ++index) {
            if (maximum_origins.has_value() && collected_origins >= *maximum_origins) break;
            const auto& turn = turns[index];
            const auto identity = make_trace_identity(
                model_receipt, resident_receipt, feature_contract, tokenizer_receipt,
                template_receipt, manifest_digest, registry_digest, turn.generation_seed
            );
            const auto path = trace_path_for(output_directory, turn.sequence_id);
            active_session.set_max_new_tokens(
                static_cast<std::size_t>(turn.maximum_new_tokens)
            );
            const auto previous_cache_position = active_session.cached_token_count();
            if (index - group_begin < resume_plan.published_prefix_turn_count) {
                const auto trace = engine::load_speculation_trace_file(path);
                const auto state = validate_existing_trace_state(
                    trace, identity, turn, config, context_generation,
                    expected_target_position,
                    balanced_goal.has_value()
                        ? std::optional<std::uint32_t>(balanced_goal->horizon)
                        : std::nullopt
                );
                context_generation = state.context_generation;
                expected_target_position = state.final_draft_position;
                add_collected_origins(collected_origins, state.selection_count);
                if (balanced_goal.has_value()) {
                    add_collected_origins(
                        active_conversation_origins[turn.generation_seed],
                        state.eligible_origin_count
                    );
                }
                ++resumed;
                if (maximum_origins.has_value()
                    && collected_origins >= *maximum_origins) {
                    ++authenticated_without_replay;
                    output << "{\"sequence_id\":" << turn.sequence_id
                           << ",\"trace_sha256\":\"" << state.envelope.hex()
                           << "\",\"resume\":\"authenticated\"}\n" << std::flush;
                } else {
                    const auto result = active_session.generate_user_turn(turn.prompt);
                    if (result.selected_token_ids != selected_trace_tokens(trace)) {
                        throw std::runtime_error(
                            "resumed target trajectory differs from existing trace"
                        );
                    }
                    if (result.rebuilt_context != state.rebuilt_context) {
                        throw std::runtime_error(
                            "resumed context-shift provenance differs from trace"
                        );
                    }
                    if (result.cache_position != trace.records().front().target_position
                        || result.cached_token_count != state.final_draft_position) {
                        throw std::runtime_error(
                            "resumed chat/cache positions differ from existing trace"
                        );
                    }
                    for (const auto token : result.evaluated_suffix_token_ids) {
                        retrieval.observe(token);
                    }
                    for (const auto token : result.selected_token_ids) {
                        retrieval.observe(token);
                    }
                    ++replayed;
                    output << "{\"sequence_id\":" << turn.sequence_id
                           << ",\"trace_sha256\":\"" << state.envelope.hex()
                           << "\",\"resume\":\"replayed-prefix\"}\n" << std::flush;
                }
            } else {
                engine::SpeculationTraceSequenceWriter writer(path, identity);
                TurnTraceCollector collector(
                    turn, config, retrieval, writer, context_generation,
                    previous_cache_position
                );
                auto sink = engine::create_target_feature_trace_sink(
                    taps,
                    config.embedding_length,
                    config.block_count,
                    engine::gemma2_metal_prefill_chunk_size,
                    [&](engine::TargetFeatureTraceBatch&& batch) {
                        collector.consume_features(std::move(batch));
                    }
                );
                const engine::Gemma2GenerationTraceBinding binding {
                    .feature_sink = {
                        .trace_sink = &sink,
                        .destination = engine::FeatureCaptureDestination::canonical,
                        .execution_epoch = turn.sequence_id,
                    },
                    .on_target_selection = [&](const engine::Gemma2TargetSelectionTraceEvent& event) {
                        collector.consume_selection(event);
                    },
                };
                const auto result = active_session.generate_user_turn(
                    turn.prompt, engine::Gemma2ChatTurnOptions {.trace = &binding}
                );
                const auto digest = collector.finish();
                const auto trace = engine::load_speculation_trace_file(path);
                const auto state = validate_existing_trace_state(
                    trace, identity, turn, config, context_generation,
                    expected_target_position,
                    balanced_goal.has_value()
                        ? std::optional<std::uint32_t>(balanced_goal->horizon)
                        : std::nullopt
                );
                if (state.envelope != digest
                    || state.selection_count != result.selected_token_ids.size()) {
                    throw std::runtime_error(
                        "published trace differs from the completed target turn"
                    );
                }
                if (collector.rebuilt() != result.rebuilt_context
                    || state.rebuilt_context != result.rebuilt_context) {
                    throw std::runtime_error(
                        "trace collector context-shift inference disagrees with session"
                    );
                }
                if (state.context_generation != collector.context_generation()
                    || state.final_draft_position != active_session.cached_token_count()) {
                    throw std::runtime_error(
                        "published trace chat/cache positions differ from target turn"
                    );
                }
                context_generation = state.context_generation;
                expected_target_position = state.final_draft_position;
                add_collected_origins(collected_origins, result.selected_token_ids.size());
                if (balanced_goal.has_value()) {
                    add_collected_origins(
                        active_conversation_origins[turn.generation_seed],
                        state.eligible_origin_count
                    );
                }
                output << "{\"sequence_id\":" << turn.sequence_id
                       << ",\"trace_sha256\":\"" << digest.hex() << "\"}\n"
                       << std::flush;
                ++collected;
            }
            ++completed;
        }
        group_begin = group_end;
        if (balanced_goal.has_value() && group_begin == active_conversation_end) {
            commit_balanced_conversation(
                balanced_coverage, generation_seeds, turns[group_begin - 1U],
                active_conversation_origins
            );
        }
    }
    if (balanced_goal.has_value() && !balanced_goal_satisfied(
            balanced_coverage, generation_seeds, *balanced_goal
        )) {
        throw std::runtime_error(
            "collection manifest cannot satisfy the balanced training/validation goal"
        );
    }
    auto balanced_receipt_digest = std::optional<runtime::Sha256Digest> {};
    if (balanced_goal.has_value()) {
        const auto sequences = published_trace_sequences(output_directory);
        const auto receipt = balanced_collection_receipt(
            manifest_digest, registry_digest, feature_contract,
            *balanced_goal, sequences
        );
        publish_or_validate_balanced_collection_receipt(output_directory, receipt);
        balanced_receipt_digest = runtime::sha256(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(receipt.data()), receipt.size()
        ));
    }
    output << "{\"completed_turns\":" << completed
           << ",\"collected_turns\":" << collected
           << ",\"resumed_turns\":" << resumed
           << ",\"authenticated_turns\":" << authenticated_without_replay
           << ",\"replayed_turns\":" << replayed
           << ",\"eligible_origins\":" << collected_origins;
    if (balanced_receipt_digest.has_value()) {
        output << ",\"balanced_collection_receipt_sha256\":\""
               << balanced_receipt_digest->hex() << "\"";
    }
    output << "}\n";
    return 0;
}

void write_deterministic_fixture(const char* path) {
    engine::SpeculationTraceIdentity identity;
    identity.model = {
        .version = engine::engine_model_payload_receipt_version,
        .whole_gguf_byte_count = 1U,
        .source_tensor_count = 1U,
        .whole_gguf = fixture_digest(1U),
        .canonical_consumed_metadata = fixture_digest(2U),
        .source_tensors = fixture_digest(3U),
    };
    identity.resident = {
        .version = 1U,
        .allocation_count = 1U,
        .byte_count = 1U,
        .bytes = fixture_digest(4U),
    };
    identity.artifact = fixture_digest(5U);
    identity.tokenizer = fixture_digest(6U);
    identity.chat_template = fixture_digest(7U);
    identity.source_manifest = fixture_digest(8U);
    identity.registry = fixture_digest(9U);
    identity.sampler = {
        .temperature = 0.0F,
        .top_k = 1U,
        .top_p = 1.0F,
        .min_p = 0.0F,
        .repetition_penalty = 1.0F,
        .seed = 42U,
        .repetition_window = 0U,
    };
    engine::SpeculationTraceFeature feature;
    feature.state = {
        .execution_epoch = 1U,
        .bank_id = 1U,
        .child_id = 1U,
        .source_stage = engine::FeatureTapStage::final_normalized,
        .coordinate = {.target_position = 0U, .draft_position = 1U},
        .row_count = 1U,
        .feature_width = 1U,
    };
    feature.dtype = tensor::DType::f16;
    feature.bytes = {0U, 0U};
    engine::SpeculationTraceRecord record;
    record.sequence_id = 1U;
    record.kind = engine::SpeculationTraceRowKind::prompt;
    record.target_position = 0U;
    record.draft_position = 1U;
    record.token_id = 0U;
    record.terminal = true;
    record.source = {
        .source_group = fixture_digest(10U),
        .source_cluster = fixture_digest(11U),
        .conversation = fixture_digest(12U),
        .prompt_derivation = fixture_digest(13U),
        .split = engine::SpeculationTraceSplit::training_development,
        .workload = engine::SpeculationTraceWorkload::instruction_chat,
        .generation_seed = 42U,
    };
    record.features.push_back(std::move(feature));
    const std::array records {record};
    engine::write_speculation_trace(path, identity, records);
}

}  // namespace

std::vector<engine::FeatureTapSpec> plan_speculation_trace_features(
    SpeculationTraceFeatureProfile profile,
    std::uint32_t target_layer_count
) {
    if (target_layer_count == 0U
        || target_layer_count > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("target layer count is invalid for trace collection");
    }
    if (profile == SpeculationTraceFeatureProfile::final_normalized) {
        return {{.stage = engine::FeatureTapStage::final_normalized}};
    }
    if (profile != SpeculationTraceFeatureProfile::all_target_taps) {
        throw std::invalid_argument("trace collection feature profile is invalid");
    }
    if (target_layer_count + 2U > engine::target_feature_trace_tap_max_count) {
        throw std::invalid_argument("model layer count exceeds raw trace tap capacity");
    }
    std::vector<engine::FeatureTapSpec> taps;
    taps.reserve(target_layer_count + 2U);
    taps.push_back({.stage = engine::FeatureTapStage::embedding_residual});
    for (std::uint32_t layer = 0U; layer < target_layer_count; ++layer) {
        taps.push_back({
            .stage = engine::FeatureTapStage::layer_residual,
            .layer_index = static_cast<std::uint16_t>(layer),
        });
    }
    taps.push_back({.stage = engine::FeatureTapStage::final_normalized});
    return taps;
}

SpeculationTraceResumeGroupPlan plan_speculation_trace_resume_group(
    std::span<const std::uint8_t> published_turns
) {
    if (published_turns.empty()) {
        throw std::invalid_argument("trace resume group is empty");
    }
    auto prefix = std::size_t {0U};
    auto saw_missing = false;
    for (const auto published : published_turns) {
        if (published > 1U) {
            throw std::invalid_argument("trace resume publication flag is invalid");
        }
        if (published == 0U) {
            saw_missing = true;
        } else if (saw_missing) {
            throw std::runtime_error(
                "trace output has a published turn after an unpublished turn"
            );
        } else {
            ++prefix;
        }
    }
    return SpeculationTraceResumeGroupPlan {
        .published_prefix_turn_count = prefix,
        .fully_published = prefix == published_turns.size(),
    };
}

int run_speculation_trace_cli(
    int argc,
    const char* const* argv,
    std::ostream& output,
    std::ostream& error
) {
    if (argc < 2) {
        error << "usage: local-gguf-speculation-trace <audit|write-fixture> <path>\n"
              << "       local-gguf-speculation-trace discard-partial "
                 "<output-directory> <sequence-id>\n"
              << "       local-gguf-speculation-trace collect <model.gguf> "
                 "<source.lgsmanifest> <registry.json> <output-directory> "
                 "[maximum-origins [all-target-taps|final-normalized]]\n"
              << "       local-gguf-speculation-trace collect-balanced <model.gguf> "
                 "<source.lgsmanifest> <registry.json> <output-directory> <horizon> "
                 "<training-origins-per-workload-seed> "
                 "<minimum-validation-clusters-per-workload> "
                 "<all-target-taps|final-normalized>\n";
        return 2;
    }
    try {
        const auto command = std::string_view(argv[1]);
        if (command == "discard-partial") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "discard-partial requires output directory and positive sequence ID"
                );
            }
            auto sequence_id = std::uint64_t {0U};
            if (!parse_u64(argv[3], sequence_id) || sequence_id == 0U) {
                throw std::invalid_argument("trace sequence ID must be positive");
            }
            return discard_unpublished_trace_fragments(
                argv[2], sequence_id, output
            );
        }
        if (command == "collect") {
            if (argc < 6 || argc > 8) {
                throw std::invalid_argument(
                    "collect requires model, manifest, registry, output directory, "
                    "optional positive maximum-origins, and optional feature profile"
                );
            }
            std::optional<std::uint64_t> maximum_origins;
            if (argc >= 7) {
                std::uint64_t parsed = 0U;
                if (!parse_u64(argv[6], parsed) || parsed == 0U) {
                    throw std::invalid_argument("maximum-origins must be a positive integer");
                }
                maximum_origins = parsed;
            }
            auto feature_profile = SpeculationTraceFeatureProfile::all_target_taps;
            if (argc == 8) {
                const auto profile = std::string_view(argv[7]);
                if (profile == "final-normalized") {
                    feature_profile = SpeculationTraceFeatureProfile::final_normalized;
                } else if (profile != "all-target-taps") {
                    throw std::invalid_argument(
                        "feature profile must be all-target-taps or final-normalized"
                    );
                }
            }
            const auto executable_bytes = read_bounded_file(argv[0], 1ULL << 30U);
            const auto executable_digest = runtime::sha256(executable_bytes);
            return collect_traces(
                argv[2], argv[3], argv[4], argv[5], maximum_origins,
                feature_profile, std::nullopt, executable_digest, output
            );
        }
        if (command == "collect-balanced") {
            if (argc != 10) {
                throw std::invalid_argument(
                    "collect-balanced requires model, manifest, registry, output directory, "
                    "horizon, positive training-origin quota, positive validation-cluster "
                    "minimum, and feature profile"
                );
            }
            std::uint64_t horizon = 0U;
            std::uint64_t training_quota = 0U;
            std::uint64_t validation_minimum = 0U;
            if (!parse_u64(argv[6], horizon) || horizon == 0U || horizon > 23U
                || !parse_u64(argv[7], training_quota) || training_quota == 0U
                || !parse_u64(argv[8], validation_minimum)
                || validation_minimum == 0U
                || validation_minimum > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "balanced collection horizon/quota/minimum is invalid"
                );
            }
            SpeculationTraceFeatureProfile feature_profile;
            const auto profile = std::string_view(argv[9]);
            if (profile == "final-normalized") {
                feature_profile = SpeculationTraceFeatureProfile::final_normalized;
            } else if (profile == "all-target-taps") {
                feature_profile = SpeculationTraceFeatureProfile::all_target_taps;
            } else {
                throw std::invalid_argument(
                    "feature profile must be all-target-taps or final-normalized"
                );
            }
            const auto executable_bytes = read_bounded_file(argv[0], 1ULL << 30U);
            const auto executable_digest = runtime::sha256(executable_bytes);
            return collect_traces(
                argv[2], argv[3], argv[4], argv[5], std::nullopt,
                feature_profile,
                BalancedCollectionGoal {
                    .horizon = static_cast<std::uint32_t>(horizon),
                    .training_origins_per_workload_seed = training_quota,
                    .minimum_validation_clusters_per_workload =
                        static_cast<std::uint32_t>(validation_minimum),
                },
                executable_digest, output
            );
        }
        if (argc != 3) {
            throw std::invalid_argument("audit and write-fixture require exactly one path");
        }
        if (command == "write-fixture") {
            write_deterministic_fixture(argv[2]);
            output << "wrote deterministic non-training trace fixture: " << argv[2] << '\n';
            return 0;
        }
        if (command != "audit") {
            error << "unknown speculation-trace command: " << command << '\n';
            return 2;
        }
        const auto trace = engine::load_speculation_trace_file(argv[2]);
        const auto audit = engine::audit_speculation_trace(trace);
        output << "{\"schema_version\":" << engine::speculation_trace_schema_version
               << ",\"records\":" << audit.record_count
               << ",\"sequences\":" << audit.sequence_count
               << ",\"features\":" << audit.feature_count
               << ",\"feature_bytes\":" << audit.feature_byte_count
               << ",\"maximum_target_position\":" << audit.maximum_target_position
               << ",\"selections\":" << audit.selection_count
               << ",\"prepared_candidates\":" << audit.prepared_candidate_count
               << ",\"training_development_records\":" << audit.split_record_counts[0]
               << ",\"architecture_selection_records\":" << audit.split_record_counts[1]
               << ",\"calibration_records\":" << audit.split_record_counts[2]
               << ",\"sealed_admission_records\":" << audit.split_record_counts[3]
               << ",\"sha256\":\"" << audit.envelope_digest.hex() << "\"}\n";
        return 0;
    } catch (const std::exception& exception) {
        error << exception.what() << '\n';
        return 1;
    }
}

}  // namespace local_gguf_engine::frontends::cli
