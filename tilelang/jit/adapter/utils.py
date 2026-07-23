# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Union, Optional, Literal
from tilelang import tvm as tvm
from tvm import IRModule, tir
from tvm.target import Target
from tilelang.engine.lower import (
    get_device_call,
    get_host_call,
    determine_target,
    canon_target_host,
    is_cpu_device_backend,
)
from tilelang.engine.phase import (
    LowerAndLegalize,
    OptimizeForTarget,
)


def match_global_kernel(source: str, annotation: str = "__global__") -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0])
    raise ValueError("No global kernel found in the source code")


def match_declare_kernel(source: str, annotation: str = "__global__") -> int:
    pattern = r"__global__\s+void\s+\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0] + "(")
    raise ValueError("No global kernel found in the source code")


def match_declare_kernel_cpu(source: str, annotation: str = "int32_t") -> int:
    pattern = r"int32_t\s+\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0] + "(")
    raise ValueError("No global kernel found in the source code")


def is_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda"


def is_hip_target(target: Target) -> bool:
    return target.kind.name == "hip"


def is_cpu_target(target: Target) -> bool:
    return target.kind.name in ["c"]


def is_tpu_target(target: Target) -> bool:
    if isinstance(target, str):
        return target == "tpu"
    return target.kind.name == "tpu"


def get_tpu_template_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../../src/tl_templates/tpu"))


@dataclass(frozen=True)
class ChipConfig:
    """Per-CHIP build configuration aggregated into a single lookup table."""

    # Emulator SDK subdirectory under `{PPL_TOP}/runtime/{chip}/`
    runtime_relative_dir: str
    # Emulator shared library path relative to `{PPL_TOP}/runtime/{chip}/`
    bmlib_cmodel_path: str
    # Host-side runtime installation sysroot (absolute path)
    runtime_sysroot: str
    # Host-side runtime link library names (passed to -l)
    runtime_link_libs: list[str]
    # Cross-compilation toolchain path, relative to PPL_TOP
    toolchain_relpath: str
    # Cross-compilation GCC prefix (e.g. "riscv64-unknown-linux-gnu-")
    toolchain_prefix: str

    def toolchain_dir(self, ppl_top: str) -> str:
        """Absolute path to the cross-compilation toolchain directory."""
        return f"{ppl_top}/{self.toolchain_relpath}"

    def cross_compile(self, ppl_top: str) -> str:
        """Full cross-compiler prefix including path, e.g. /path/to/bin/riscv64-unknown-linux-gnu-."""
        return f"{self.toolchain_dir(ppl_top)}/bin/{self.toolchain_prefix}"


_CHIP_CONFIGS: dict[str, ChipConfig] = {
    "bm1690":
        ChipConfig(
            runtime_relative_dir="tpuv7-runtime-emulator",
            bmlib_cmodel_path="tpuv7-runtime-emulator/lib/libtpuv7_emulator.so",
            runtime_sysroot="/opt/tpuv7/tpuv7-current",
            runtime_link_libs=["tpuv7_rt", "cdm_daemon_emulator", "pthread"],
            toolchain_relpath="third_party/toolchains_dir/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1",
            toolchain_prefix="riscv64-unknown-linux-gnu-",
        ),
    "bm1684x":
        ChipConfig(
            runtime_relative_dir="libsophon/bmlib",
            bmlib_cmodel_path="lib/libcmodel_firmware.so",
            runtime_sysroot="/opt/sophon/libsophon-0.5.2",
            runtime_link_libs=["bmlib", "pthread"],
            toolchain_relpath="third_party/toolchains_dir/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu",
            toolchain_prefix="aarch64-linux-gnu-",
        ),
}


def get_chip_config(chip: str) -> ChipConfig | None:
    """Return the :class:`ChipConfig` for *chip*, or ``None`` if unsupported."""
    return _CHIP_CONFIGS.get(chip)


def get_annotated_mod(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    model_type: Literal["device", "host", "all"] = "all",
) -> Union[IRModule, tuple[IRModule, IRModule]]:

    # Validate model_type early
    if model_type not in {"device", "host", "all"}:
        raise ValueError(f"Invalid model type: {model_type}")

    # Convert PrimFunc to IRModule if needed
    mod = func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        mod = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})

    # Handle target and target_host
    if isinstance(target, str):
        target = determine_target(target)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, target_host))
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Apply transformations
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

    # Define dispatch dictionary for different model types
    dispatch = {
        "device":
            lambda m: tir.transform.Filter(_is_device_call)(m),
        "host":
            lambda m: tir.transform.Filter(_is_host_call)(m),
        "all":
            lambda m: (tir.transform.Filter(_is_device_call)(m), tir.transform.Filter(_is_host_call)
                       (m)),
    }

    return dispatch[model_type](mod)
