# oMLX Architecture: DFlash Speculative Decoding & Prefix Cache

This document describes the architecture of DFlash speculative decoding
and the paged prefix cache in the layer4down/omlx fork. It serves as a
reference to prevent design drift across development sessions.

## Repository Layout

```
omlx-github/                          # Source repo (layer4down/omlx)
├── omlx/
│   ├── _version.py                   # Single source of truth for version
│   ├── engine/
│   │   ├── vlm.py                    # VLMBatchedEngine — main request router
│   │   ├── batched.py                # BatchedEngine — text-only models
│   │   └── base.py                   # BaseEngine interface
│   ├── engine_core.py                # EngineCore — scheduler + BatchGenerator
│   ├── scheduler.py                  # Scheduler — prefix cache, prefill, batching
│   ├── patches/
│   │   └── dflash_specdec.py         # DFlash integration layer
│   ├── cache/
│   │   ├── prefix_cache.py           # BlockAwarePrefixCache — cache orchestration
│   │   ├── paged_cache.py            # PagedCacheManager — block metadata + hash index
│   │   ├── paged_ssd_cache.py        # PagedSSDCacheManager — SSD storage tier
│   │   └── tiered_manager.py         # Cache stats aggregation
│   ├── server.py                     # HTTP API + request handling
│   └── admin/routes.py               # Dashboard REST API
├── packaging/
│   ├── build.py                      # Build script (venvstacks)
│   └── dist/                         # Build output (oMLX.app, .dmg)
└── docs/
    └── ARCHITECTURE.md               # This file
```

Installed app mirrors the repo layout:
```
/Applications/oMLX.app/Contents/Resources/omlx/
```

## Request Flow Overview

```
                          HTTP Request
                              │
                     server.py:chat_completion()
                              │
                    ┌─────────┴─────────┐
                    │ Read model_settings │
                    │ specdec_enabled?    │
                    └─────────┬─────────┘
                              │
                    VLMBatchedEngine.generate()
                    or VLMBatchedEngine.stream_generate()
                              │
              ┌───────────────┼───────────────┐
              │                               │
    DFlash path                        Scheduler path
    (prompt < threshold)              (prompt >= threshold)
              │                               │
    engine_core.py:                    engine_core.py:
    add_dflash_request()              add_request()
              │                               │
    dflash_specdec.py:                scheduler.py:
    run_dflash_generation()           Prefix cache lookup
              │                        → Prefill remaining
    dflash_mlx runtime                → BatchGenerator decode
    (bypasses scheduler)                      │
              │                        GenerationOutput
    RequestOutputCollector                    │
              │                        stream_outputs()
    stream_outputs()                          │
              │                        SSE to client
    SSE to client
```

## DFlash Speculative Decoding

### What It Does

DFlash uses block-diffusion draft models to generate token blocks,
then verifies them against the target model. For short prompts, this
provides 2-4x speedup on token generation (measured 22-24 tok/s vs
baseline ~10 tok/s for structured prompts).

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `dflash_specdec.py` | `omlx/patches/` | Integration layer — loads draft model, runs generation |
| `engine_core.py` | `omlx/` | `configure_dflash()`, `add_dflash_request()`, `_run_dflash_generation()` |
| `vlm.py` routing | `omlx/engine/` | DFlash vs scheduler routing with fallback threshold |
| `dflash_mlx` | External package | Block diffusion runtime (from bstnxbt/dflash-mlx) |

### Weight Sharing

The draft model (`dflash_specdec.py:run_dflash_generation`) reuses the
target model's weights via zero-copy sharing. Only the lightweight draft
model (~3.5GB) is loaded separately. The patching strips the
`language_model.` prefix from draft model keys to match the target's
parameter names. Result: 755/851 parameters shared, 96 skipped.

### Draft Model Cache

Draft models are cached at module level (`_cached_drafts` dict in
`dflash_specdec.py`) with thread-safe locking. First request loads the
draft model (~0.1s from disk), subsequent requests reuse it.

### Fallback Mechanism

**Critical design decision**: DFlash prefill is fundamentally slower than
normal prefill because it processes through BOTH target and draft models
and captures hidden states from intermediate layers. For large prompts,
this overhead is catastrophic (250-330s for 38K tokens vs ~90s normal).

The fallback threshold (`DFLASH_MAX_CTX`) controls this:

```python
# vlm.py:generate() and stream_generate()
_dflash_ctx_limit = int(os.environ.get("DFLASH_MAX_CTX", "0") or "0") or 4096
_prompt_token_count = len(self._tokenizer.encode(prompt))
_dflash_fallback = _prompt_token_count >= _dflash_ctx_limit
```

- Default threshold: **4096 tokens** (when env var unset or 0)
- Prompts < 4096: DFlash path (speculative decode speedup)
- Prompts >= 4096: Scheduler path (prefix cache, paged cache, SSD cache)
- `dflash_mlx` v0.1.3+ defaults to `sys.maxsize` internally, but our
  routing in `vlm.py` applies the 4096 threshold regardless

### DFlash Limitations

1. **No prefix cache**: DFlash bypasses the scheduler entirely. Every
   request does full prefill from scratch — no KV state reuse.
2. **No continuous batching**: DFlash handles one request at a time.
3. **No paged cache**: No memory-efficient KV cache management.
4. **Prefill overhead**: Dual-model processing makes prefill slower.
5. **Token generation speedup only**: Benefits are in tok/s, not TTFT.

## Paged Prefix Cache

### Architecture

The prefix cache stores KV cache blocks on SSD so that repeated prompts
with the same prefix can skip prefilling those tokens. This is modeled
after vLLM's paged attention design.

```
Request arrives with token IDs [t0, t1, ..., tN]
         │
    scheduler.py: step()
         │
    block_aware_cache.fetch_cache(request_id, token_ids)
         │
    paged_cache.py: get_computed_blocks(token_ids)
         │
    For each 2048-token block:
      compute_block_hash(parent_hash, block_tokens, model_name)
         │
      ┌──┴──┐
      │ Hit │ → Reuse block, advance parent_hash
      └──┬──┘
      │ Miss │ → Stop, return cached prefix
      └──────┘
         │
    prefix_cache.py: reconstruct_cache(block_table)
         │
    Load block tensors from SSD → Build KVCache objects
         │
    request.cached_tokens = N_cached
    request.remaining_tokens = token_ids[N_cached:]
         │
    Prefill only remaining_tokens through model
```

### Block Hashing

Blocks use a chain hash: each block's hash depends on its parent's hash
plus its token content. This ensures that even a single token difference
at position 0 invalidates ALL subsequent blocks.

```python
# paged_cache.py:compute_block_hash()
hash = sha256(model_name + parent_hash + tuple(token_ids) + extra_keys)
```

### Block Size

- Config default: 256 tokens
- Enlarged to **2048 tokens** for models with ArraysCache (hybrid cache
  models like Qwen3.5-27B). See `scheduler.py:_ARRAYS_CACHE_BLOCK_SIZE`.
- This enlargement is required because ArraysCache boundary snapshots
  are expensive at small block sizes.
- Side effect: prompts shorter than 2048 tokens never produce a full
  block, so they're never cached.

### Cache Tiers

```
┌─────────────────────────┐
│  Hot Cache (in-memory)   │  16 GB max, raw bytes, O(1) access
├─────────────────────────┤
│  SSD Cache (safetensors) │  64 GB max, async background writes
├─────────────────────────┤
│  SSD Index (in-memory)   │  Scanned on startup, tracks block hashes
└─────────────────────────┘
```

On server startup:
1. `PagedSSDCacheManager.__init__()` scans `~/.omlx/cache/` for
   `.safetensors` files
2. Reads metadata (block_hash, model_name, num_layers) into
   `PagedSSDCacheIndex`
3. The `PagedCacheManager.cached_block_hash_to_block` starts empty
4. Blocks are lazily registered during `get_computed_blocks()` when
   the SSD index confirms a block exists

### Cache Stats (as of 2026-04-15)

```
SSD cache: 263 files, 64.1 GB stored
Block size: 2048 tokens
indexed_blocks: 0 on fresh start, grows with requests
```

### Exact Prefix Hit Edge Case

When a request's tokens exactly match all cached blocks (no remaining
tokens), the cache must be trimmed by one token so the model can produce
its first decode logit. This follows vLLM's approach:

```python
# scheduler.py:2183-2223
if len(remaining_tokens) == 0 and cached_tokens > 0:
    if _cache_list_needs_boundary_snapshot(cache):
        # Non-sliceable caches (RotatingKVCache, ArraysCache) can't
        # be safely trimmed → fall back to full prefill
        request.prompt_cache = None
    elif _trim_prompt_cache_for_generation(cache):
        # Sliceable caches (KVCache) → trim 1 token
        cached_tokens -= 1
        remaining_tokens = token_ids[-1:]
    else:
        # Can't trim → fall back to full prefill
        request.prompt_cache = None
```

### Measured Cache Performance

| Test | Cold (first) | Warm (cached) | Reduction |
|------|-------------|---------------|-----------|
| 2858 tokens (1 block) | 9.78s | 3.83s | 61% |
| 4579 tokens (2 blocks) | 15.05s | 2.86s | 81% |
| 5543 tokens (2 blocks) | 10.09s | 4.10s | 59% |

Cache is functional but `cached_tokens` field in API response always
reports 0 (reporting bug, not a functional issue).

## Design Decisions & Tradeoffs

### 1. DFlash as Patch vs Separate Engine

**Original (jundot/omlx)**: Separate `DFlashEngine` class that wraps
`BaseEngine`, with `_evict_dflash_and_start_fallback()` for lifecycle
management. Clean separation but requires the entire engine abstraction.

**Our fork**: DFlash integrated as a patch (`dflash_specdec.py`) that
hooks into the existing `EngineCore` and `VLMBatchedEngine`. Routing
logic in `vlm.py` decides DFlash vs scheduler at request time. Simpler
but the two paths don't share infrastructure (no cache reuse).

### 2. DFlash Fallback Threshold

Set to 4096 tokens (matching original DFlashEngine default). This
ensures:
- Short interactive prompts (< 4096) get DFlash token speedup
- Long prompts (>= 4096, like OpenCode's 38K) use the scheduler's
  prefix cache and optimized prefill

The threshold is hardcoded as a fallback default but can be overridden
via `DFLASH_MAX_CTX` environment variable.

### 3. Block Size Enlargement (256 → 2048)

Required for hybrid models with ArraysCache (Qwen3.5-27B, etc.).
Boundary snapshots for non-sliceable caches are expensive at small block
sizes. The tradeoff: prompts under 2048 tokens never cache.

### 4. Weight Sharing via Zero-Copy

The draft model reuses the target model's weights by stripping the
`language_model.` prefix from parameter names. This avoids loading a
second copy of the 53GB model into memory.

## Key Configuration

| Setting | Location | Default | Notes |
|---------|----------|---------|-------|
| `DFLASH_MAX_CTX` | Environment variable | 4096 | Token threshold for DFlash fallback |
| `paged_cache_block_size` | SchedulerConfig | 256 | Enlarged to 2048 for ArraysCache models |
| `paged_ssd_cache_dir` | CLI `--ssd-cache-dir` | None | Must be set for prefix cache to work |
| `specdec_enabled` | model_settings.json | false | Per-model DFlash toggle |
| `specdec_draft_model` | model_settings.json | None | HuggingFace ID or local path |
| `specdec_block_tokens` | model_settings.json | 16 | Tokens per DFlash block |

## Known Issues

1. **`cached_tokens` not reported in API response**: The
   `prompt_tokens_details.cached_tokens` field always shows 0 even when
   the prefix cache is working. The cache IS functional (proven by TTFT
   reduction), but the reporting path doesn't propagate the value.

2. **DFlash bypasses prefix cache**: All requests routed to DFlash do
   full prefill with no KV reuse. This is inherent to the current
   architecture — DFlash has its own generation pipeline that doesn't
   interact with the scheduler's cache.

3. **Exact prefix hit with non-sliceable caches**: When a prompt exactly
   matches cached blocks and the model uses non-sliceable caches
   (ArraysCache, RotatingKVCache), the system falls back to full
   prefill because these caches can't be safely trimmed by one token.

4. **Block size prevents short prompt caching**: With 2048-token blocks,
   prompts under 2048 tokens never produce a cacheable block. The
   system stores boundary snapshots but there's nothing to cache.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.3.5.2 | 2026-04-13 | Initial DFlash integration, streaming, compatibility checks |
| v0.3.5.4 | 2026-04-14 | DFlash KV injection fix, build script dflash_mlx discovery |
| v0.3.5.5 | 2026-04-14 | DFlash fallback mechanism (DFLASH_MAX_CTX threshold) |
| v0.3.5.6 | 2026-04-14 | Remove DFLASH_MAX_CTX cap (let dflash_mlx v0.1.3 handle it) |

## Related References

- Original DFlashEngine: `/Users/hunterbr/Documents/GitHub/omlx-original/omlx/engine/dflash.py`
- dflash-mlx v0.1.3: `/Users/hunterbr/Documents/GitHub/dflash-mlx-latest/`
- Model settings: `/Users/hunterbr/.omlx/model_settings.json`
- SSD cache: `/Users/hunterbr/.omlx/cache/`
- Server logs: `/Users/hunterbr/.omlx/logs/server.log`
