#pragma once

#include <string_view>

#ifndef LOCAL_GGUF_ENGINE_BUILD
#define LOCAL_GGUF_ENGINE_BUILD "unknown"
#endif

namespace local_gguf_engine::frontends::cli {
inline constexpr std::string_view build_identity = LOCAL_GGUF_ENGINE_BUILD;
} // namespace local_gguf_engine::frontends::cli
