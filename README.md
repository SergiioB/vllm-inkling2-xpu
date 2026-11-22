# vLLM Inkling 2 XPU plugin

Out-of-tree support for running Inkling 2 Mini on Intel XPU with vLLM. The plugin
registers the XPU model implementations and enables the model's `inkling2`
quantization on XPU; it does not patch or vendor vLLM itself.

## Compatibility

Requires an unmodified vLLM main checkout at or after `c32251b5d1` (Inkling 2
Mini support, #69028); no released vLLM version contains the shared model
implementation. Requires a PyTorch `2.15+xpu` environment. Tested on one Intel
Arc Pro B70.

The plugin uses vLLM's documented `vllm.general_plugins` entry point and
`ModelRegistry.register_model` API. vLLM's internal
`vllm.plugins.load_general_plugins` is called before model quantization
validation in the API server, engine core, and worker. Compatibility-sensitive
internals include `vllm.platforms.xpu.XPUPlatform.supported_quantization` and the
shared `vllm.models.inkling2.{configs,processing,quant_config,speculator_config}`
modules, `vllm.models.inkling.nvidia.sconv_swa_attn`, and interfaces in
`vllm.config`, `vllm.forward_context`, `vllm.model_executor.*`, `vllm.multimodal`,
`vllm.triton_utils`, `vllm.v1.*`, `vllm.distributed`, `vllm.envs`, `vllm.sequence`,
and `vllm.utils`. The registration test uses the private
`ModelRegistry._try_load_model_cls` helper only to verify lazy registration
targets. Changes to these internals may require a plugin update.

## Install

Install into an XPU-enabled vLLM environment, such as the
`vllm/vllm-openai-xpu` image or a source build:

```bash
python -m pip install --no-deps /path/to/vllm-inkling2-xpu
```


## Tests

Run from a vLLM source checkout after installing the plugin:

```bash
PYTHONPATH="$PWD:/path/to/plugin/tests" pytest -q -p tests.conftest /path/to/plugin/tests
```

## Usage

The checkpoint's default context length does not fit on one B70; set
`--max-model-len` to fit your KV cache. `-cc.mode=none` disables `torch.compile`;
full decode graphs still work.

```bash
vllm serve thinkingmachines/Inkling-2-Mini \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 1 \
  --block-size 64 \
  --no-enable-prefix-caching \
  -cc.mode=none \
  -cc.cudagraph_mode=FULL_DECODE_ONLY
```

Optionally enable DSpark with seven speculative tokens:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

## Scope and performance

Single-device, text-only inference. Activations are BF16; expert weights use
NVFP4 or MXFP8, following vLLM's Hopper path. Tensor/expert parallelism, LoRA,
and `torch.compile` are not supported.

Development-tier measurements on one Arc Pro B70 (single stream, greedy, 6 coding
and reasoning chat prompts, 256 output tokens), with vLLM PR #69186 applied:
57.2 tok/s decode; 75.9 tok/s with DSpark k=7 at mean acceptance length 2.99;
about 3,250 tok/s prefill for a 2,000-token prompt. Without #69186 (fp32 lm_head
fast path on XPU), decode is about 7.5 ms/token slower. DSpark gains depend on
the workload: on short chat prompts with reasoning disabled, acceptance length
was 2.02 and DSpark was slower than no speculation.

With DSpark enabled, greedy output can differ from greedy output without it.
Verification runs several tokens through the batched expert kernel instead of
the single-token kernel, so rounding differs and near-tied tokens can flip; in
the cases we checked, the top-two log-probabilities at the first divergence were
0.025 to 0.068 apart.

## Numerical parity

Compared against an independent reference: Hugging Face Transformers Inkling 2
on CPU, experts dequantized from the same checkpoint to fp32, BF16 activations
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
bitwise or general guarantee.
This project is not affiliated with vLLM, Intel, or Thinking Machines.
