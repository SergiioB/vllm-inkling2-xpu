# vllm-inkling2-xpu

Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an
Intel Arc Pro B70 with [vLLM](https://github.com/vllm-project/vllm).

vLLM supports Inkling 2 Mini on NVIDIA GPUs only. This is an out-of-tree
[vLLM plugin](https://docs.vllm.ai/en/latest/design/plugin_system/) that adds
Intel XPU support: Triton kernels for the NVFP4/MXFP8 experts and attention, full
decode graphs, and DSpark speculative decoding with the bundled drafter. It
registers itself through vLLM's `vllm.general_plugins` entry point and does not
patch vLLM.

Not affiliated with vLLM, Intel, or Thinking Machines.

## Tested configuration

| | |
|---|---|
| GPU | 1x Intel Arc Pro B70 (32 GB, Xe2 / Battlemage) |
| Host | Ubuntu 26.04.1, kernel 7.0.0-34, `xe` driver |
| vLLM | main at or after `c32251b5d1` (Inkling 2 Mini, #69028), XPU Docker image |
| PyTorch / Triton | 2.15.0+xpu / 3.9.0+xpu (pinned by vLLM's `requirements/xpu.txt`) |
| Model | `thinkingmachines/Inkling-2-Mini` (Apache-2.0, ~20 GB download) |

Other Battlemage cards are untested. The model needs ~18 GB of device memory for
weights, so a 24 GB card such as the B60 may only fit a short context.

## Quick start (Docker)

The steps below are the exact ones used to validate this README on a fresh host
with the Intel GPU driver installed (`ls /dev/dri/renderD*` lists your GPUs).

**1. Build vLLM's XPU image from `main`.** No published vLLM release or
`vllm/vllm-openai-xpu` nightly built before 2026-11-22 contains Inkling 2. Once a
nightly includes [#69028](https://github.com/vllm-project/vllm/pull/69028), you can
skip this step and use `BASE=vllm/vllm-openai-xpu:nightly` in step 2.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Optional, ~45% faster decode (about 40 -> 57 tok/s): XPU fp32 lm_head fast path
# from open PR #69186. Not needed once that PR is merged.
git fetch origin pull/69186/head
git checkout FETCH_HEAD -- vllm/model_executor/layers/logits_processor.py
docker build -f docker/Dockerfile.xpu --target vllm-openai -t vllm-xpu:main .
cd ..
```

**2. Add the plugin.**

```bash
git clone https://github.com/mgaruccio/vllm-inkling2-xpu.git
docker build --build-arg BASE=vllm-xpu:main -t vllm-xpu:inkling2 vllm-inkling2-xpu
```

**3. Download the checkpoint** (optional; `vllm serve` downloads it otherwise).

```bash
mkdir -p ~/.cache/huggingface
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint hf vllm-xpu:inkling2 download thinkingmachines/Inkling-2-Mini
```

**4. Serve** on GPU 0 (`ZE_AFFINITY_MASK` selects the card).

```bash
docker run -d --name inkling2 \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host -p 8000:8000 -e ZE_AFFINITY_MASK=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu:inkling2 \
  thinkingmachines/Inkling-2-Mini \
  --language-model-only --dtype bfloat16 \
  --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.93 --block-size 64 --no-enable-prefix-caching \
  -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
docker logs -f inkling2   # ready when "Application startup complete" appears
```

Drop the `--speculative-config` line to run without speculative decoding.

**5. Send a request.**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": "Write a haiku about Intel GPUs."}],
  "max_tokens": 512
}' | python3 -m json.tool
```

## Install into an existing environment

Any XPU build of vLLM main at or after `c32251b5d1` works:

```bash
python -m pip install --no-deps git+https://github.com/mgaruccio/vllm-inkling2-xpu
```

Install it into the same Python environment that runs `vllm`. Check that it loaded:

```bash
python -c "from vllm.plugins import load_general_plugins as l; l(); \
from vllm.platforms.xpu import XPUPlatform as p; print('inkling2' in p.supported_quantization)"
```

## Serving flags

| Flag | Why |
|---|---|
| `--language-model-only` | Text only; image inputs are not supported on XPU. |
| `--dtype bfloat16` | BF16 activations (experts stay NVFP4/MXFP8). |
| `--max-model-len 8192` | The default context length does not fit on one B70. Raise it as far as your KV cache allows; the server log reports the KV cache size. |
| `-cc.mode=none` | `torch.compile` is not supported for this model on XPU. |
| `-cc.cudagraph_mode=FULL_DECODE_ONLY` | Decode graphs (the name says CUDA; it applies to XPU). |
| `--max-num-seqs 1` | The configuration that was tuned and validated; batching is untested. |
| `--reasoning-parser inkling`, `--tool-call-parser inkling` | Same parsers as on NVIDIA. |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":7}'` | DSpark with the bundled `mtp.safetensors` drafter. DSpark requires exactly 7. |

## Performance

Development-tier measurements on one Arc Pro B70, single stream, greedy, 256
output tokens, 6 coding and reasoning chat prompts. These are not standardized
benchmark results; raw results and the script are in [`results/`](results/).

| | Decode tok/s (median) | DSpark acceptance length |
|---|---:|---:|
| No speculation, vLLM main + #69186 | 57.2 | — |
| DSpark k=7, vLLM main + #69186 | 75.9 | 2.99 |

Prefill is about 3,250 tok/s for a 2,000-token prompt. Without #69186, decode is
about 7.5 ms/token slower. DSpark gains depend on the workload: on short chat
prompts with reasoning disabled, acceptance length was 2.02 and DSpark was slower
than no speculation. The first requests after startup are slower while Triton
kernels compile.

With DSpark enabled, greedy output can differ from greedy output without it.
Verification runs several tokens through the batched expert kernel instead of the
single-token kernel, so rounding differs and near-tied tokens can flip; in the
cases we checked, the top-two log-probabilities at the first divergence were 0.025
to 0.068 apart.

## Numerical parity

Compared against an independent reference: Hugging Face Transformers Inkling 2 on
CPU, experts dequantized from the same checkpoint to fp32, BF16 activations
(matching the XPU semantics). Teacher-forced next-token log-probabilities on 4
short prompts and the first 2,048 positions of 2 long prompts (4,174 positions):

| vs CPU fp32 | top-1 agreement | median / p95 / max abs Δ logprob |
|---|---|---|
| XPU | 99.4% | 0.00017 / 0.024 / 5.69 |
| CPU bf16 (noise floor) | 99.4% | 0.00017 / 0.029 / 5.63 |

Short prompts individually agree 96.2–100% on top-1 tokens. A 64-token greedy chat
continuation matched the CPU fp32 reference exactly. The largest single layer
difference found was a router near-tie (8th vs 9th expert scores 6e-5 apart).
The XPU side was an earlier in-tree build of the same kernels (vLLM branch commit
`32ecd66097`); later builds and this plugin produced token-identical greedy output
on the validation prompts. This is bounded evidence on sampled sequences, not a
bitwise or general guarantee. Details: [`results/parity.md`](results/parity.md).

## Limitations

- One device: tensor and expert parallelism raise an error at startup.
- Text only, BF16, no LoRA, no `torch.compile`.
- Expert activations stay BF16 (W4A16/W8A16), like vLLM's Hopper path; the
  checkpoint's NVFP4/MXFP8 activation quantization is not applied.
- Validated with one sequence at a time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `inkling2 quantization is currently not supported in xpu` | The plugin is not installed in the Python environment that runs `vllm`. |
| `Model architectures ['Inkling2LoopForConditionalGeneration'] are not supported` | vLLM is older than `c32251b5d1`; build from current `main`. |
| `Free memory on device ... is less than desired GPU memory utilization` | Another process is using the GPU, or lower `--gpu-memory-utilization`. |
| Out of memory or KV cache too small | Lower `--max-model-len`. |
| Using another GPU | Set `ZE_AFFINITY_MASK=<index>`; one card per server. |

## Tests

From a vLLM source checkout, after installing the plugin, on an XPU machine:

```bash
PYTHONPATH="$PWD:/path/to/vllm-inkling2-xpu/tests" \
  pytest -q -p tests.conftest /path/to/vllm-inkling2-xpu/tests
```

## Compatibility

The plugin uses vLLM's documented `vllm.general_plugins` entry point and
`ModelRegistry.register_model` API, and appends `inkling2` to
`vllm.platforms.xpu.XPUPlatform.supported_quantization`. It imports the shared
`vllm.models.inkling2.{configs,processing,quant_config,speculator_config}`
modules, `vllm.models.inkling.nvidia.sconv_swa_attn`, and vLLM internals in
`vllm.config`, `vllm.forward_context`, `vllm.model_executor`, `vllm.multimodal`,
`vllm.triton_utils`, `vllm.v1`, `vllm.distributed`, `vllm.envs`, `vllm.sequence`,
and `vllm.utils`. Changes to these may require a plugin update.

## License

Apache-2.0. Most of the code is derived from vLLM's Inkling 2 implementation and
keeps its copyright headers.
