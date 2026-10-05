#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"

#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
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

bool gemma_proposal_eligible(const tokenizer::Tokenizer& tokenizer, std::uint32_t id) {
    const auto& token = tokenizer.vocabulary()[id];
    if (tokenizer.special_tokens().unknown_token_id == id) return false;
    if (tokenizer.token_id_is_special(id)) return shared_gemma_control(token.text);
    return token.type == 1 || token.type == 4 || token.type == 6;
}

std::optional<std::string> gemma_mapping_key(const tokenizer::Tokenizer& tokenizer, std::uint32_t id) {
    const auto& token = tokenizer.vocabulary()[id];
    if (!gemma_proposal_eligible(tokenizer, id)) return std::nullopt;
    if (tokenizer.token_id_is_special(id)) {
        return "c" + token.text;
    }
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

void validate_gemma3_direct_ids(const tokenizer::Tokenizer& target, const tokenizer::Tokenizer& draft) {
    const auto fail = [](std::string_view detail) {
        throw std::invalid_argument("Gemma3 direct-ID tokenizer mismatch: " + std::string(detail));
    };
    if (target.model() != draft.model() || target.vocabulary().size() != draft.vocabulary().size())
        fail("model or vocabulary size");
    const auto& ts = target.special_tokens();
    const auto& ds = draft.special_tokens();
    if (ts.bos_token_id != ds.bos_token_id || ts.eos_token_id != ds.eos_token_id
        || ts.unknown_token_id != ds.unknown_token_id || ts.add_bos_token != ds.add_bos_token
        || ts.add_eos_token != ds.add_eos_token)
        fail("special IDs or insertion flags");
    const auto require_role = [&](std::optional<std::uint32_t> id, std::string_view text, std::int32_t type) {
        if (!id || *id >= target.vocabulary().size() || target.vocabulary()[*id].text != text
            || target.vocabulary()[*id].type != type) fail("missing or invalid Gemma control role");
    };
    require_role(ts.bos_token_id, "<bos>", 3);
    require_role(ts.eos_token_id, "<eos>", 3);
    // The downloaded Gemma3 converter represents <unk> as CONTROL (3), not
    // SentencePiece UNKNOWN (2). Its optional explicit ID still owns the role;
    // the spelling remains special even when neither file declares that ID.
    if (ts.unknown_token_id) {
        const auto& unknown = target.vocabulary()[*ts.unknown_token_id];
        if (unknown.text != "<unk>" || (unknown.type != 2 && unknown.type != 3))
            fail("invalid unknown control role");
    }
    bool start_of_turn = false, end_of_turn = false;
    for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
        const auto& t = target.vocabulary()[id];
        const auto& d = draft.vocabulary()[id];
        if (t.text != d.text || !t.score || !d.score
            || std::bit_cast<std::uint32_t>(*t.score) != std::bit_cast<std::uint32_t>(*d.score))
            fail("piece or score at ID " + std::to_string(id));
        // The released 12B and 270M GGUFs differ only in these image-marker
        // types. Both spellings are special under the Gemma3 profile and are
        // excluded from proposals. Other control/user-defined changes fail.
        const bool image_marker = t.text == "<start_of_image>" || t.text == "<end_of_image>";
        const bool compatible_image_type = image_marker && (t.type == 3 || t.type == 4)
            && (d.type == 3 || d.type == 4);
        if ((t.type != d.type && !compatible_image_type)
            || target.token_id_is_special(id) != draft.token_id_is_special(id))
            fail("piece type or special semantics at ID " + std::to_string(id));
        if (t.text == "<start_of_turn>" || t.text == "<end_of_turn>") {
            if (t.type != 3 || d.type != 3) fail("invalid turn control type");
            if (t.text == "<start_of_turn>") start_of_turn = true;
            else end_of_turn = true;
        }
    }
    if (!start_of_turn || !end_of_turn) fail("missing turn control");
    // Constructing a Gemma3 Tokenizer already validates identity normalization,
    // no dummy prefix/collapse, and the complete byte fallback. Its remaining
    // consumed semantics are checked above. add_sep_token is not consumed by
    // this runtime. The target renders chat exactly once; no draft template is
    // executed. Neither this compatibility proof nor image-type normalization
    // changes either tokenizer's authenticated identity.
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
    if (target.profile() == tokenizer::TokenizerProfile::gemma3) {
        validate_gemma3_direct_ids(target, draft);
        for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
            if (!gemma_proposal_eligible(draft, id)) continue;
            draft_to_target_[id] = id;
            target_to_draft_[id] = id;
            ++intersection_size_;
        }
        if (!intersection_size_) throw std::invalid_argument("empty Gemma3 proposal support");
        // This bound follows from the total conditioning identity, including
        // controls outside proposal support and incomplete UTF-8 byte runs.
        direct_ids_ = true;
        max_append_tokens_ = 1;
        return;
    }
    std::size_t maximum_piece_bytes = 0;
    for (const auto& token : target.vocabulary()) maximum_piece_bytes = std::max(maximum_piece_bytes, token.text.size());
    if (maximum_piece_bytes > (std::numeric_limits<std::size_t>::max() - 3U) / 3U) {
        throw std::length_error("TLI append capacity overflow");
    }
    // UTF-8 replacement and SP whitespace escaping expand each input byte by
    // at most three. Fallback emits at most one token per normalized byte.
    max_append_tokens_ = maximum_piece_bytes * 3U;
    if (target.profile() != tokenizer::TokenizerProfile::gemma2) throw std::invalid_argument("Gemma3 bridge requires a Gemma2 or compatible Gemma3 target tokenizer");
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
bool StandaloneTliMapping::direct_ids() const noexcept { return direct_ids_; }
std::size_t StandaloneTliMapping::owned_bytes() const noexcept {
    return (draft_to_target_.capacity() + target_to_draft_.capacity() + target_control_context_.capacity()) * sizeof(std::uint32_t);
}

std::size_t StandaloneTliMapping::append_target_token(
    std::uint32_t token, std::span<std::uint32_t> output
) const {
    if (!direct_ids_) throw std::logic_error("TLI conversion requires a text workspace");
    if (token >= target_to_draft_.size()) throw std::invalid_argument("TLI target token out of range");
    if (output.empty()) throw std::length_error("TLI output capacity exhausted");
    output[0] = token;
    return 1;
}

std::size_t StandaloneTliMapping::encode_target_context(
    std::span<const std::uint32_t> tokens, std::span<std::uint32_t> output
) const {
    if (!direct_ids_) throw std::logic_error("TLI conversion requires a text workspace");
    if (tokens.size() > output.size()) throw std::length_error("TLI output capacity exhausted");
    for (const auto token : tokens)
        if (token >= target_to_draft_.size()) throw std::invalid_argument("TLI target token out of range");
    // Validation precedes copying so an invalid history leaves its destination
    // unchanged. memmove also permits in-place compaction of a native ledger.
    if (!tokens.empty()) std::memmove(output.data(), tokens.data(), tokens.size_bytes());
    return tokens.size();
}

std::size_t StandaloneTliMapping::append_target_token(
    std::uint32_t token, std::span<std::uint32_t> output,
    std::span<char> scratch, tokenizer::TokenizerEncodeWorkspace& workspace
) const {
    if (direct_ids_) return append_target_token(token, output);
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
    if (direct_ids_) return encode_target_context(tokens, output);
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
