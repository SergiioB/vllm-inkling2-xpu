#!/usr/bin/env bash
# Build the opt-in CUTLASS Xe2 expert extension (_inkling_nvfp4_xe2) for INK2_CUTLASS_MOE=1 and copy it
# into the installed vllm_inkling2_xpu/ops/. AOT-compiled for the Arc Pro B70 (bmg_g31).
#
# Needs: Intel oneAPI icpx (tested 2026.1.1; source /opt/intel/oneapi/setvars.sh), `ocloc` on PATH
# (compute-runtime `intel-ocloc` matching the installed Level Zero driver), the serving Python
# environment (torch XPU, vllm-xpu-kernels, this plugin installed), and header checkouts of
# intel/sycl-tla @7b56b936 and vllm-project/vllm-xpu-kernels @e9b6e4b2.
#
#   PYTHON=/path/to/venv/bin/python SYCL_TLA_DIR=... VLLM_XPU_KERNELS_DIR=... scripts/build-cutlass-xe2.sh
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SYCL_TLA_DIR="${SYCL_TLA_DIR:?checkout of intel/sycl-tla@7b56b936b87f799f80dde60c2fd8d11c1128cebb}"
VXK="${VLLM_XPU_KERNELS_DIR:?checkout of vllm-project/vllm-xpu-kernels@e9b6e4b24a69886a8b384edc06e7a631ad282705}"
AOT_TARGET="${AOT_TARGET:-bmg_g31}"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
command -v icpx >/dev/null || { echo "icpx not on PATH (source /opt/intel/oneapi/setvars.sh)" >&2; exit 1; }
command -v ocloc >/dev/null || { echo "ocloc not on PATH (install intel-ocloc matching your driver)" >&2; exit 1; }

py() { "$PYTHON" -c "$1"; }
PY_INC=$(py 'import sysconfig; print(sysconfig.get_path("include"))')
EXT=$(py 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
TORCH_DIR=$(py 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)')
ABI=$(py 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
# find_spec: locate the installed package without importing vLLM.
OPS_DIR=$(py 'import importlib.util, pathlib; s = importlib.util.find_spec("vllm_inkling2_xpu"); print(pathlib.Path(s.origin).parent / "ops")')
OUT="$BUILD_DIR/_inkling_nvfp4_xe2$EXT"

# -spirv-ext: the same list as vllm-xpu-kernels' CMakeLists (icpx 2026.1 rejects the Xe2 GEMM's split
# barriers without it). TORCH_EXTENSION_NAME names the pybind11 module's init symbol.
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device $AOT_TARGET -options -ze-opt-large-register-file" \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
  -O3 -std=c++20 -fPIC -shared \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED \
  -DTORCH_EXTENSION_NAME=_inkling_nvfp4_xe2 -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  -I"$PY_INC" -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -I"$SYCL_TLA_DIR/include" -I"$SYCL_TLA_DIR/tools/util/include" \
  -I"$VXK" -I"$VXK/csrc" -I"$VXK/csrc/xpu/cutlass_kernels" \
  "$here/csrc/nvfp4_xe2_ext.cpp" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
  -o "$OUT"
cp "$OUT" "$OPS_DIR/"
echo "Installed $OPS_DIR/_inkling_nvfp4_xe2$EXT; enable with INK2_CUTLASS_MOE=1"
