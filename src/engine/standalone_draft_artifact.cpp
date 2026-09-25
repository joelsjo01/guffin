#include "local_gguf_engine/engine/standalone_draft_artifact.hpp"

#include "local_gguf_engine/model_format/gguf_tensor_bridge.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace local_gguf_engine::engine {
namespace {

// Reproducible output of scripts/convert_vicuna_draft.py, authenticated from
// pinned source bytes. Binding the entire artifact also protects tokenizer
// scores/types/normalization, geometry, tensor names and payload, not just weights.
constexpr std::string_view artifact_sha256 =
    "2038020ca7b9a2584252f4bf1cfd5e4cb833130b0853822e58cfa538299211b7";
constexpr std::string_view gemma3_artifact_sha256 =
    "0ef57d2c838458a1952664260dcba38e5bdda37494f3af732f06e4add24068e3";
constexpr std::uintmax_t vicuna_bytes = 136789856U;
constexpr std::uintmax_t gemma3_bytes = 291545600U;

model_format::GGUFFileBackedModel authenticated_model(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (size != vicuna_bytes && size != gemma3_bytes) {
        throw std::invalid_argument("standalone draft artifact size does not match a pinned complete checkpoint");
    }
    auto model = model_format::read_file_backed_gguf_model(path);
    if (runtime::sha256(model.bytes()) != runtime::parse_sha256_hex(
            size == vicuna_bytes ? artifact_sha256 : gemma3_artifact_sha256)) {
        throw std::invalid_argument("standalone draft artifact authentication failed");
    }
    return model;
}

bool mappable(const tokenizer::Tokenizer& tokenizer, std::uint32_t id) {
    const auto& token = tokenizer.vocabulary()[id];
    return !tokenizer.token_id_is_special(id)
        && tokenizer.special_tokens().unknown_token_id != id
        && (token.type == 1 || token.type == 6);
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

std::size_t clean_up_target_text(std::span<char> text, std::size_t count) {
    // Exact ordered replacements of Transformers 4.50.0
    // PreTrainedTokenizerBase.clean_up_tokenization, requested by the released
    // TLI context adapter. Every replacement shrinks, so scratch is in-place.
    constexpr std::pair<std::string_view, std::string_view> replacements[] {
        {" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"},
        {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"},
    };
    for (const auto& [before, after] : replacements) {
        std::size_t read = 0, write = 0;
        while (read < count) {
            if (std::string_view(text.data() + read, count - read).starts_with(before)) {
                for (const auto byte : after) text[write++] = byte;
                read += before.size();
            } else text[write++] = text[read++];
        }
        count = write;
    }
    return count;
}

}  // namespace

StandaloneDraftArtifact::StandaloneDraftArtifact(const std::filesystem::path& path)
    : mapped_(authenticated_model(path)),
      tokenizer_(tokenizer::load_tokenizer_from_gguf_metadata(mapped_.view().directory.metadata)),
      digest_(runtime::parse_sha256_hex(mapped_.bytes().size() == vicuna_bytes ? artifact_sha256 : gemma3_artifact_sha256)),
      kind_(mapped_.bytes().size() == vicuna_bytes ? StandaloneDraftKind::vicuna_68m_f16 : StandaloneDraftKind::gemma3_270m_q8) {}

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
StandaloneDraftKind StandaloneDraftArtifact::kind() const noexcept { return kind_; }
std::uint32_t StandaloneDraftArtifact::context_limit() const noexcept {
    return kind_ == StandaloneDraftKind::gemma3_270m_q8 ? 32768U : context_capacity;
}

StandaloneTliMapping::StandaloneTliMapping(
    const tokenizer::Tokenizer& target, const tokenizer::Tokenizer& draft
) : target_(&target), draft_(&draft), draft_to_target_(draft.vocabulary().size(), unmapped),
    target_to_draft_(target.vocabulary().size(), unmapped),
    preserve_gemma_controls_(draft.profile() == tokenizer::TokenizerProfile::gemma3) {
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
    max_append_tokens_ = maximum_piece_bytes * 3U + (preserve_gemma_controls_ ? 0U : 3U);
    if (preserve_gemma_controls_) {
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
        return;
    }
    std::unordered_map<std::string_view, std::uint32_t> target_ids;
    target_ids.reserve(target.vocabulary().size());
    for (std::uint32_t id = 0; id < target.vocabulary().size(); ++id) {
        if (mappable(target, id) && !target_ids.emplace(target.vocabulary()[id].text, id).second) {
            throw std::invalid_argument("ambiguous target TLI vocabulary");
        }
    }
    for (std::uint32_t id = 0; id < draft.vocabulary().size(); ++id) {
        if (!mappable(draft, id)) continue;
        const auto found = target_ids.find(draft.vocabulary()[id].text);
        if (found == target_ids.end()) continue;
        const auto target_id = found->second;
        if (target.vocabulary()[target_id].type != draft.vocabulary()[id].type) continue;
        if (target_to_draft_[target_id] != unmapped) throw std::invalid_argument("non-injective TLI mapping");
        draft_to_target_[id] = target_id;
        target_to_draft_[target_id] = id;
        ++intersection_size_;
    }
    if (intersection_size_ == 0) throw std::invalid_argument("empty TLI vocabulary intersection");
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
    if (!preserve_gemma_controls_ && (target_->token_id_is_special(token) || target_->special_tokens().unknown_token_id == token)) return 0;
    if (target_to_draft_[token] != unmapped) {
        if (output.empty()) throw std::length_error("TLI output capacity exhausted");
        output[0] = target_to_draft_[token];
        return 1;
    }
    if (preserve_gemma_controls_ && (target_->token_id_is_special(token) || target_->special_tokens().unknown_token_id == token)) {
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
    if (!preserve_gemma_controls_) count = clean_up_target_text(scratch, count);
    return draft_->encode_into(std::string_view(scratch.data(), count), output, workspace,
        {.encode_special_tokens = !preserve_gemma_controls_});
}

std::size_t StandaloneTliMapping::encode_target_context(
    std::span<const std::uint32_t> tokens, std::span<std::uint32_t> output,
    std::span<char> scratch, tokenizer::TokenizerEncodeWorkspace& workspace
) const {
    if (preserve_gemma_controls_) {
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
    const auto count = clean_up_target_text(scratch, target_->decode_into(tokens, scratch, {.skip_special_tokens = true}));
    return draft_->encode_into(std::string_view(scratch.data(), count), output, workspace, {.encode_special_tokens = true});
}

}  // namespace local_gguf_engine::engine
