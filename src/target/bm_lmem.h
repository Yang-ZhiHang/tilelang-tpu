/*
 * Runtime dispatch layer for BM chip selection.
 *
 * Unlike CUDA's compile-time __CUDA_ARCH_LIST__ macro dispatch,
 * `codegen_ppl.cc` is compiled ONCE into `tilelang.so` and selects the chip at
 * runtime. This header provides a BMChip enum + inline dispatch functions that
 * delegate to the correct chip-specific namespace (bm1684x / bm1690).
 */
#pragma once

#include "bm1684x_lmem.h"
#include "bm1690_lmem.h"
#include "bm_common.h"

#include <string>
#include <vector>

#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {

/// Chip selector used at codegen time.
enum class BM16X {
  BM1690,
  BM1684X,
};

/// Convert chip string to BMChip.
inline BM16X ChipFromString(const std::string &chip) {
  if (chip == "bm1684x" || chip == "BM1684X")
    return BM16X::BM1684X;
  return BM16X::BM1690;
}

// ---------------------------------------------------------------------------
//  Chip-specific constant accessors
// ---------------------------------------------------------------------------

inline int64_t BankNum(BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::kBankNum;
  default:
    return bm1690::kBankNum;
  }
}

inline int64_t BankSize(BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::kBankSize;
  default:
    return bm1690::kBankSize;
  }
}

inline int64_t TensorAlignBytes(BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::kTensorAlignBytes;
  default:
    return bm1690::kTensorAlignBytes;
  }
}

// ---------------------------------------------------------------------------
//  Dispatch wrappers: forward to the active chip-specific namespace
// ---------------------------------------------------------------------------

inline std::vector<int64_t> NormalizeLocalShape(const Array<PrimExpr> &shape,
                                                const char *context,
                                                BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::NormalizeLocalShape(shape, context);
  default:
    return bm1690::NormalizeLocalShape(shape, context);
  }
}

inline int64_t TpuAlignSizeBytesFromShape4(const std::vector<int64_t> &shape4,
                                           DataType dtype,
                                           BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::TpuAlignSizeBytesFromShape4(shape4, dtype);
  default:
    return bm1690::TpuAlignSizeBytesFromShape4(shape4, dtype);
  }
}

inline int64_t TpuAlignSizeBytes(const Array<PrimExpr> &shape, DataType dtype,
                                 const char *context,
                                 BM16X arch = BM16X::BM1690) {
  switch (arch) {
  case BM16X::BM1684X:
    return bm1684x::TpuAlignSizeBytes(shape, dtype, context);
  default:
    return bm1690::TpuAlignSizeBytes(shape, dtype, context);
  }
}

} // namespace tl
} // namespace tvm