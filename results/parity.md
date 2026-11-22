# Numerical parity vs an independent reference

Development-tier evidence, not a standardized evaluation.

- **Reference:** Hugging Face Transformers Inkling 2 on CPU, loaded from the same
  checkpoint with the experts dequantized to fp32 (NVFP4: E2M1 codes x per-16 E4M3
  scale x per-expert `amax / outer_scale_denominator`; MXFP8: E4M3 x 2^(UE8M0-127) per
  32), the checkpoint's gate/up row interleave converted as Transformers' loader does,
  BF16 activations in the experts (W4A16/W8A16, matching the XPU path). A second CPU
  run in BF16 is the noise floor.
- **Candidate:** vLLM XPU on one Arc Pro B70, BF16, `FULL_DECODE_ONLY` graphs,
  teacher-forced `prompt_logprobs` (top-20).
- **Metrics:** top-1 agreement and |delta log-probability| of the reference token at
  each position, against CPU fp32.
- **Gates passed first:** one full MoE block within the kernel tests' elementwise
  bound; one decoder layer (the single outlier is a router near-tie, 8th/9th expert
  scores 6e-5 apart, within BF16 rounding of the gate logits); CPU bf16 vs fp32
  sanity.

Four short prompts in full and the first 2,048 positions of two long prompts (raw-long-00 and chat-long-00).

| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |
|---|---:|---:|---:|---:|---:|
| raw-short-00 | 11 | 1.0000 | 0.0136929 / 0.138678 / 0.231039 | 0.0109509 / 0.120713 / 0.147867 | 6 |
| chat-short-00 | 26 | 0.9615 | 0.0538898 / 0.514572 / 0.652895 | 0.0825133 / 0.707399 / 2.06629 | 15 |
| raw-short-01 | 14 | 1.0000 | 0.0128835 / 0.113344 / 0.156247 | 0.0201674 / 0.128471 / 0.200066 | 6 |
| chat-short-01 | 29 | 0.9655 | 0.062067 / 0.53059 / 0.910388 | 0.0679903 / 0.630672 / 0.740498 | 13 |
| raw-long-00 | 2047 | 0.9971 | 0.000377988 / 0.00978781 / 0.198292 | 0.000372576 / 0.00944667 / 0.362145 | 1072 |
| chat-long-00 | 2047 | 0.9912 | 3.1348e-05 / 0.0322392 / 5.68923 | 3.23035e-05 / 0.0372843 / 5.63051 | 981 |

Overall (4174 scored positions): XPU top-1 agreement 0.9938; CPU bf16 top-1 agreement 0.9938.
XPU-vs-fp32 |Δlogprob| median/p95/max: 0.000167908 / 0.0235346 / 5.68923.
CPU-bf16-vs-fp32 noise floor median/p95/max: 0.000166295 / 0.0285079 / 5.63051.
XPU delta exceeded same-position bf16 noise at 2093 positions.

## Greedy decode first divergences

| Prompt | Ref tokens | Candidate tokens | First divergence index |
|---|---:|---:|---:|
| chat-short-00 | 64 | 64 | None |
