# Results

Development-tier measurements (one Arc Pro B70, single stream). Not standardized
benchmark results.

## `decode/`

`decode_bench.py`: six coding and reasoning chat prompts, greedy, `ignore_eos`, 256
output tokens, streamed; decode tok/s = (tokens - 1) / (last token time - first token
time); DSpark acceptance from the server's `/metrics` counters. The script sends
`"model": "inkling-2-mini"`, so serve with `--served-model-name inkling-2-mini`:

```bash
python decode_bench.py out-dir --port 8000 --tokens 256
```

| File | Server |
|---|---|
| `no-spec-summary.json` | Quick-start flags without `--speculative-config` |
| `dspark-summary.json` | Quick-start flags with DSpark k=7 |

Both: vLLM main `c32251b5d1` plus PR #69186, plugin `96593b9`, checkpoint
`thinkingmachines/Inkling-2-Mini@ed40e7da` (weights identical to `b85c3cfa`).

## `parity.md`

Teacher-forced log-probability comparison against a CPU Transformers reference.
