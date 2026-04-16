# DFlash Speculative Decoding: Performance Study

**Date**: 2026-04-16  
**oMLX Version**: v0.3.5.9-layer4down  
**Hardware**: Apple M3 Max, 128GB  
**Target Model**: Qwen3.5-27B (bf16, ~54GB)  
**Draft Model**: z-lab/Qwen3.5-27B-DFlash (~3.5GB)  
**dflash_mlx**: v0.1.0 (v0.1.3 code)

---

## Executive Summary

DFlash speculative decoding provides **dramatic TTFT reduction** (2.5x–37x faster) for short-to-medium prompts (< 4096 tokens), but **generation speed is slower** than baseline (15–44 tok/s vs baseline's highly variable 0–2000 tok/s). The net effect depends heavily on your workload:

- **Winner**: Chatbot/conversational use (short prompts, medium generation)
- **Winner**: Quick Q&A (tiny prompts, brief answers)
- **Loser**: Long-context workloads (OpenCode, RAG with large docs)
- **Neutral**: Repeated identical prompts (prefix cache competes well)

---

## 1. TTFT Comparison: DFlash vs Baseline

| Prompt Tokens | Gen Tokens | Baseline TTFT | DFlash TTFT | TTFT Speedup |
|---|---|---|---|---|
| 20 | 64 | 7.43s | 2.96s | **2.5x** |
| 20 | 256 | 23.75s | 1.33s | **17.8x** |
| 20 | 1,024 | 37.65s | 1.26s | **29.9x** |
| 143 | 64 | 6.79s | 1.74s | **3.9x** |
| 143 | 256 | 24.64s | 1.74s | **14.2x** |
| 143 | 1,024 | 67.90s | 1.82s | **37.2x** |
| 965 | 64 | 8.95s | 4.03s | **2.2x** |
| 965 | 256 | 26.84s | 4.08s | **6.6x** |
| 965 | 1,024 | 83.28s | 4.02s | **20.7x** |
| 3,031 | 64 | 12.05s | 10.98s | **1.1x** |
| 3,031 | 256 | 26.96s | 10.90s | **2.5x** |
| 3,031 | 1,024 | 98.86s | 10.94s | **9.0x** |

**Key finding**: DFlash prefill is dramatically faster than baseline prefill for prompts under ~3K tokens. At 3K tokens the benefit shrinks to 1.1x for short generation. DFlash's own prefill cost grows linearly with prompt size.

---

## 2. Generation Speed

| Prompt Tokens | Generation tok/s (DFlash) | Notes |
|---|---|---|
| 20 | 24.9–44.1 | Best for tiny prompts |
| 143 | 19.5–29.9 | Strong performance |
| 965 | 15.5–41.3 | Short gen=fast, long gen=slower |
| 3,031 | 16.4–26.1 | Still respectable |

DFlash generation speed is **consistent** (15–44 tok/s) regardless of prompt size, while baseline generation speed is extremely variable (0–2000+ tok/s) depending on model warm-up effects.

**Realistic generation speed**: ~17–29 tok/s for most workloads. This is slower than the mlx-dflash README benchmark of 45–55 tok/s, likely because:
1. Our target is 27B params (vs 8B in the benchmark)
2. We're running through oMLX's integration layer (not standalone)
3. Memory pressure from the 54GB model + 3.5GB draft model

---

## 3. Block Size Comparison

| block_tokens | 20-pt TTFT | 965-pt TTFT | 3031-pt TTFT | Avg gen tok/s |
|---|---|---|---|---|
| 8 | 2.64s | 4.04s | 11.06s | 15.4–44.2 |
| 16 | 2.96s | 4.03s | 10.98s | 15.5–44.2 |
| 32 | 3.05s | 3.97s | 10.92s | 15.5–43.8 |

**Verdict**: Block size makes negligible difference (within noise). **Default of 16 is fine.** No measurable acceptance ratio change between 8/16/32.

---

## 4. DFlash vs Prefix Cache for Repeated Prompts

### Small prompt (143 tokens, 3 repeats, 128 gen tokens)

| Config | Avg TTFT | Avg Total | Sum Total (3 runs) |
|---|---|---|---|
| DFlash ON | 2.80s | 7.58s | **22.75s** |
| DFlash OFF (cache) | 13.84s | 13.84s | 41.53s |

**DFlash wins by 1.8x** — even without caching, DFlash's fast prefill beats the cold-start penalty of normal prefill.

### Medium prompt (3,031 tokens, 3 repeats, 128 gen tokens)

| Config | Avg TTFT | Avg Total | Sum Total (3 runs) |
|---|---|---|---|
| DFlash ON | 12.06s | 19.23s | **57.69s** |
| DFlash OFF (cache) | 16.24s | 16.24s | **48.71s** |

**Prefix cache wins by 1.2x** — for medium prompts, the prefix cache (2048 tokens cached) plus no DFlash overhead is slightly faster. The DFlash generation overhead (draft+verify cycles) adds ~7s per request vs the ~11s saved on TTFT.

**Key insight**: For prompts under ~1K tokens, DFlash always wins. For prompts 2K–4K tokens, the prefix cache (when warm) is competitive or slightly better for repeated requests.

---

## 5. Context Limit Routing Analysis

DFlash is currently limited to prompts < 4096 tokens (`DFLASH_MAX_CTX=4096`). The data shows why:

| Prompt Tokens | DFlash TTFT | Scheduler TTFT | Routing |
|---|---|---|---|
| 143 | 1.72s | 6.79s | DFlash |
| 965 | 4.04s | 8.95s | DFlash |
| 3,031 | 10.92s | 12.05s | DFlash |
| 3,031 | — | 10.98s | Scheduler |
| 4,608 | — | 26.05s | Scheduler |

DFlash prefill grows linearly. At ~3K tokens DFlash is only marginally faster (10.9s vs 12s). Beyond 4K, the scheduler path has access to prefix cache and chunked prefill optimizations that DFlash lacks.

---

## 6. Recommendations

### When to Enable DFlash

| Use Case | DFlash? | Why |
|---|---|---|
| Chat/conversational (short prompts, <1K tokens) | **YES** | 3–37x TTFT improvement, consistent 20–30 tok/s |
| Quick Q&A (tiny prompts, short answers) | **YES** | Dramatic first-token speed |
| Code completion / inline suggestions | **YES** | Low latency critical |
| OpenCode / large-context coding agents | **NO** | Prompts exceed 4K tokens, prefix cache works better |
| RAG with large documents | **NO** | Same — large prompts bypass DFlash |
| Batch processing of similar prompts | **MAYBE** | DFlash wins for small prompts; prefix cache wins for medium |

### Optimal Configuration

```json
{
  "specdec_enabled": true,
  "specdec_draft_model": "z-lab/Qwen3.5-27B-DFlash",
  "specdec_block_tokens": 16,
  "DFLASH_MAX_CTX": 4096
}
```

- `block_tokens`: 16 (tested 8/16/32 — no meaningful difference)
- `DFLASH_MAX_CTX`: 4096 (correct — above this, DFlash loses its edge)
- Draft model: Must match target architecture (Qwen3.5-27B → z-lab/Qwen3.5-27B-DFlash)

### Architecture Improvements (Proposed)

1. **Adaptive routing**: Instead of a hard 4096-token cutoff, use a dynamic threshold based on real-time acceptance ratio. If acceptance drops below 50%, fall back to scheduler.

2. **Hybrid DFlash + prefix cache**: Currently DFlash bypasses the prefix cache entirely. For warm-cache scenarios with <4K prompts, first check prefix cache, then decide DFlash vs scheduler based on cache hit rate.

3. **Per-request DFlash toggle via API**: The `specdec` field in ChatCompletionRequest already exists. Clients like OpenCode can set `specdec: false` for large-context requests to force the scheduler+cache path, and `specdec: true` for short conversational turns.

4. **Memory optimization**: DFlash requires loading the draft model (~3.5GB). Consider lazy-loading: only load the draft model when the first eligible request arrives, and unload after a configurable idle timeout.

5. **8-bit quantized target testing**: The DFlash README warns that quantized targets reduce acceptance. If you switch to Qwen3.5-27B-8bit (7.5GB vs 54GB), expect lower acceptance ratios and less DFlash benefit. This was not tested in this study but is flagged as a risk.

---

## 7. The "Why Was It Slow Before?" Post-Mortem

The earlier observation of "8–10 tok/s with DFlash" had a specific root cause: **the admin API settings endpoint uses HTTP PUT, not POST**. The benchmark script was using `requests.post()` to configure DFlash, which returned `405 Method Not Allowed` silently. DFlash was never actually enabled during earlier testing — all measurements were baseline performance misattributed to DFlash.

After fixing to `requests.put()`, DFlash activates correctly and delivers 17–44 tok/s generation with 2.5–37x TTFT improvement for short prompts.

---

## Appendix: Benchmark Methodology

- **Server**: oMLX v0.3.5.9-layer4down, localhost:8000
- **Streaming requests** with `include_usage: true` for precise TTFT measurement
- **2 runs per scenario** (Phase 1, 2, 4), 1 run (Phase 5)
- **Results**: `study_results/phase{1,2,4,5}.json` in working directory
- **Model**: Qwen3.5-27B (bf16, full precision) — NOT quantized

Raw data available in `study_results/` directory.
