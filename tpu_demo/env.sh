export SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/tl_templates/tpu"

# The ld path for bm1690
export LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib:${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator/lib:${LD_LIBRARY_PATH}"
# The ld path for bm1684x, havn't tested pcie ld path yet
# export LD_LIBRARY_PATH="/opt/sophon/libsophon-0.5.2/lib:${PPL_PROJECT_ROOT}/runtime/bm1684x/libsophon/bmlib/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${PPL_PROJECT_ROOT}/runtime/bm1684x/libsophon/bmlib/lib:${LD_LIBRARY_PATH}"

export TPU_KERNEL_PATH="${SRC_DIR}"
export PPL_KERNEL_PATH="${SRC_DIR}/libkernel.so"