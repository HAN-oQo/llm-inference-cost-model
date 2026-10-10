# llm-inference-cost-model

Analytic FLOPs and memory-traffic models for LLM inference, derived layer by layer
from a model's published config. Prefill and decode are counted separately.

The point is the ratio:

```
Operational Intensity (OI) = FLOPs / Memory Traffic
```

which is the x-axis of a roofline plot. Having the theoretical number in hand before
measuring means the gap between theory and a real run is itself the starting point
for an investigation.

## Layout

```
deepseek_v3/
  flops.py          # DeepSeek-V3 — MLA attention + MoE, TP-aware
  memory_io.py      # DeepSeek-V3 — weights, activations, KV cache, TP all-reduce
llama3/
  flops.py          # Llama 3.1 — dense, GQA, TP-aware
  memory_traffic.py # Llama 3.1 — weight reads, activations, KV cache
  llama3_8b.yaml    # example config
```

## Usage

Llama 3.1 (config file driven):

```bash
python llama3/flops.py           --config llama3/llama3_8b.yaml --input_len 128 --output_len 128 --tp_size 1
python llama3/memory_traffic.py  --config llama3/llama3_8b.yaml --input_len 128 --output_len 128 --tp_size 1
```

DeepSeek-V3 (config inlined in `__main__`, TP_SIZE at the top of the file):

```bash
python deepseek_v3/flops.py      --batch 1 --prefill-len 128 --decode-len 128
python deepseek_v3/memory_io.py  --batch 1 --prefill-len 128 --decode-len 128
```

## What is counted

**FLOPs** — per operator: RMSNorm, QKV / output projections, attention matmuls,
softmax and scaling, SwiGLU feed-forward, LM head. Tensor parallelism divides the
per-rank work and adds the all-reduce term. For DeepSeek-V3, MLA's low-rank Q/KV
projections and the MoE gate + shared/routed experts are counted separately.

**Memory traffic** — input reads, weight reads and output writes for every operator,
plus KV cache reads during decode. On-chip (SRAM to SRAM) traffic inside an operator
is assumed negligible. Tensor parallelism adds the reduce traffic.

Decode weight reads scale with the number of decode steps, which is why decode sits
far to the left on the roofline: almost no arithmetic per byte moved.

## Caveats

- Analytic, not measured. Kernel fusion, cache hits and scheduling are not modelled,
  so these are upper bounds on traffic and lower bounds on achieved intensity.
- `batch_size` multiplies inputs and outputs; weights are read once per step
  (except where the decode path models per-step weight re-reads).
- DeepSeek-V3 numbers assume FP8 weights and FP16 KV cache; Llama assumes FP16
  throughout.
- Sanity check on Llama 3.1 8B: total weight read comes out at 14–15 GB against a
  16 GB checkpoint. The difference is the positional-embedding table, which is not
  read in full on every forward pass.
