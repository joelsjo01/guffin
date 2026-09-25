#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"

#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace local_gguf_engine::engine {
namespace {

// Bind the complete checkpoint bytes, including tokenizer semantics, tensor
// names, geometry, and payload, rather than authenticating weights alone.
constexpr std::string_view gemma3_artifact_sha256 =
    "0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3";
constexpr std::uintmax_t gemma3_bytes = 291545600U;

model_format::GGUFFileBackedModel authenticated_model(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (size != gemma3_bytes) {
        throw std::invalid_argument("standalone draft artifact size does not match a pinned complete checkpoint");
    }
    auto model = model_format::read_file_backed_gguf_model(path);
    if (runtime::sha256(model.bytes()) != runtime::parse_sha256_hex(
            gemma3_artifact_sha256)) {
        throw std::invalid_argument("standalone draft artifact authentication failed");
    }
    return model;
}

bool shared_gemma_control(std::string_view text) {
    return text == "<bos>" || text == "<eos>" || text == "<start_of_turn>" || text == "<end_of_turn>";
}

std::optional<std::string> gemma_mapping_key(const tokenizer::Tokenizer& tokenizer, std::uint32_t id) {
    const auto& token = tokenizer.vocabulary()[id];
    if (tokenizer.special_tokens().unknown_token_id == id) return std::nullopt;
    if (tokenizer.token_id_is_special(id)) {
        if (!shared_gemma_control(token.text)) return std::nullopt;
        return "c" + token.text;
    }
    if (token.type != 1 && token.type != 4 && token.type != 6) return std::nullopt;
    std::string key(1, static_cast<char>(*token.type));
    if (token.type == 4) {
        // GGUF converts only whitespace-only SP user-defined ▁ runs into
        // literal spaces. Do not conflate normal, byte and user-defined pieces.
        constexpr std::string_view marker = "▁";
        auto rest = std::string_view(token.text);
        while (rest.starts_with(marker)) rest.remove_prefix(marker.size());
        if (rest.empty()) {
            key.append(token.text.size() / marker.size(), ' ');
            return key;
        }
    }
    key += token.text;
    return key;
}

}  // namespace

StandaloneDraftArtifact::StandaloneDraftArtifact(const std::filesystem::path& path)
    : mapped_(authenticated_model(path)),
      tokenizer_(tokenizer::load_tokenizer_from_gguf_metadata(mapped_.view().directory.metadata)),
      digest_(runtime::parse_sha256_hex(gemma3_artifact_sha256)) {}

tensor::TensorView StandaloneDraftArtifact::tensor(std::string_view name) const {
    const auto& payloads = mapped_.view().tensor_payloads;
    const auto found = std::find_if(payloads.begin(), payloads.end(), [name](const auto& item) {
        return item.tensor.name == name;
    });
    if (found == payloads.end()) throw std::invalid_argument("standalone draft tensor not found: " + std::string(name));
    return model_format::view_gguf_tensor_payload(*found).view;
}

const tokenizer::Tokenizer& StandaloneDraftArtifact::tokenizer() const noexcept { return tokenizer_; }
const runtime::Sha256Digest& StandaloneDraftArtifact::digest() const noexcept { return digest_; }
StandaloneDraftKind StandaloneDraftArtifact::kind() const noexcept { return StandaloneDraftKind::gemma3_270m_q8; }
std::uint32_t StandaloneDraftArtifact::context_limit() const noexcept {
    return 32768U;
}

StandaloneTliMapping::StandaloneTliMapping(
    const tokenizer::Tokenizer& target, const tokenizer::Tokenizer& draft
) : target_(&target), draft_(&draft), draft_to_target_(draft.vocabulary().size(), unmapped),
    target_to_draft_(target.vocabulary().size(), unmapped) {
    if (draft.profile() != tokenizer::TokenizerProfile::gemma3)
        throw std::invalid_argument("standalone bridge requires a Gemma3 draft tokenizer");
    if (target.vocabulary().size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())
        || draft.vocabulary().size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TLI vocabulary exceeds signed device token capacity");
    }
    std::size_t maximum_piece_bytes = 0;
    for (const auto& token : target.vocabulary()) maximum_piece_bytes = std::max(maximum_piece_bytes, token.text.size());
    if (maximum_piece_bytes > (std::numeric_limits<std::size_t>::max() - 3U) / 3U) {
        throw std::length_error("TLI append capacity overflow");
    }
    // UTF-8 replacement and SP whitespace escaping expand each input byte by
    // at most three. Fallback emits at most one token per normalized byte.
    max_append_tokens_ = maximum_piece_bytes * 3U;
    if (target.profile() != tokenizer::TokenizerProfile::gemma2) throw std::invalid_argument("Gemma3 bridge requires a Gemma2 target tokenizer");
    std::unordered_map<std::string, std::uint32_t> target_ids;
    target_ids.reserve(target.vocabulary().size());
    for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
        const auto key = gemma_mapping_key(target, id);
        if (key && !target_ids.emplace(*key, id).second) throw std::invalid_argument("ambiguous target Gemma TLI vocabulary");
    }
    std::unordered_map<std::string, std::uint32_t> draft_ids;
    std::unordered_map<std::string_view, std::uint32_t> context_ids;
    context_ids.reserve(draft.vocabulary().size());
    draft_ids.reserve(draft.vocabulary().size());
    for (std::uint32_t id = 0; id < draft.vocabulary().size(); ++id) {
        const auto& token = draft.vocabulary()[id];
        if (token.type == 2 || token.type == 3 || token.type == 4) context_ids.emplace(token.text, id);
        const auto key = gemma_mapping_key(draft, id);
        if (!key) continue;
        if (!draft_ids.emplace(*key, id).second) throw std::invalid_argument("ambiguous draft Gemma TLI vocabulary");
        const auto found = target_ids.find(*key);
        if (found == target_ids.end()) continue;
        draft_to_target_[id] = found->second;
        target_to_draft_[found->second] = id;
        ++intersection_size_;
    }
    target_control_context_.assign(target.vocabulary().size(), unmapped);
    for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
        if (!target.token_id_is_special(id) && target.special_tokens().unknown_token_id != id) continue;
        const auto found = context_ids.find(target.vocabulary()[id].text);
        if (found != context_ids.end()) target_control_context_[id] = found->second;
    }
    if (!intersection_size_) throw std::invalid_argument("empty Gemma TLI vocabulary intersection");

}

std::span<const std::uint32_t> StandaloneTliMapping::draft_to_target() const noexcept { return draft_to_target_; }
std::span<const std::uint32_t> StandaloneTliMapping::target_to_draft() const noexcept { return target_to_draft_; }
std::size_t StandaloneTliMapping::intersection_size() const noexcept { return intersection_size_; }
std::size_t StandaloneTliMapping::max_append_tokens() const noexcept { return max_append_tokens_; }
std::size_t StandaloneTliMapping::owned_bytes() const noexcept {
    return (draft_to_target_.capacity() + target_to_draft_.capacity() + target_control_context_.capacity()) * sizeof(std::uint32_t);
}

std::size_t StandaloneTliMapping::append_target_token(
    std::uint32_t token, std::span<std::uint32_t> output,
    std::span<char> scratch, tokenizer::TokenizerEncodeWorkspace& workspace
) const {
    if (token >= target_to_draft_.size()) throw std::invalid_argument("TLI target token out of range");
    if (target_to_draft_[token] != unmapped) {
        if (output.empty()) throw std::length_error("TLI output capacity exhausted");
        output[0] = target_to_draft_[token];
        return 1;
    }
    if (target_->token_id_is_special(token) || target_->special_tokens().unknown_token_id == token) {
        const auto context_id = target_control_context_[token];
        if (context_id != unmapped) {
            if (output.empty()) throw std::length_error("TLI output capacity exhausted");
            output[0] = context_id;
            return 1;
        }
        // No draft role is invented for an opaque target control. Its literal
        // spelling conditions the draft without changing target support or
        // recognizing it as a new control during correction retokenization.
        return draft_->encode_into(target_->vocabulary()[token].text, output, workspace);
    }
    auto count = target_->decode_into(std::span<const std::uint32_t>(&token, 1), scratch, {.skip_special_tokens = true});
    return draft_->encode_into(std::string_view(scratch.data(), count), output, workspace);
}

std::size_t StandaloneTliMapping::encode_target_context(
    std::span<const std::uint32_t> tokens, std::span<std::uint32_t> output,
    std::span<char> scratch, tokenizer::TokenizerEncodeWorkspace& workspace
) const {
    // Retokenize ordinary spans together (including complete UTF-8 byte
    // runs), but never strip or re-detect semantic target control tokens.
    // Literal control-looking user text remains ordinary text.
    std::size_t written = 0;
    for (std::size_t begin = 0; begin < tokens.size();) {
        const auto token = tokens[begin];
        if (token >= target_to_draft_.size()) throw std::invalid_argument("TLI target token out of range");
        if (target_->token_id_is_special(token) || target_->special_tokens().unknown_token_id == token) {
            written += append_target_token(token, output.subspan(written), scratch, workspace);
            ++begin;
            continue;
        }
        auto end = begin + 1;
        while (end < tokens.size()) {
            if (tokens[end] >= target_to_draft_.size()) throw std::invalid_argument("TLI target token out of range");
            if (target_->token_id_is_special(tokens[end]) || target_->special_tokens().unknown_token_id == tokens[end]) break;
            ++end;
        }
        const auto count = target_->decode_into(tokens.subspan(begin, end - begin), scratch);
        written += draft_->encode_into(std::string_view(scratch.data(), count), output.subspan(written), workspace);
        begin = end;
    }
    return written;

}

}  // namespace local_gguf_engine::engine
