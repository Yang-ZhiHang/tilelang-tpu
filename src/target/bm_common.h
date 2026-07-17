#pragma once

#include <cstdint>

namespace tvm {
namespace tl {

inline int64_t DivUp(int64_t value, int64_t factor) {
  return (value + factor - 1) / factor;
}

inline int64_t AlignUp(int64_t value, int64_t align) {
  return DivUp(value, align) * align;
}

} // namespace tl
} // namespace tvm
