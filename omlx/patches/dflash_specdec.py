"""DFlash Speculative Decoding integration for oMLX.

Wraps the dflash-mlx package to provide speculative decode acceleration
via block diffusion draft models. Uses the already-loaded oMLX model
(reuses weights, no duplicate memory) and only loads the lightweight
draft model from disk.

Requires: dflash-mlx >= 0.1.0
"""

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level draft model cache (loaded once, reused across requests)
_cache_lock = threading.Lock()
_cached_drafts: dict[str, Any] = {}


def _ensure_dflash_available() -> bool:
    """Check if dflash-mlx is installed and importable."""
    try:
        import dflash_mlx  # noqa: F401
        return True
    except ImportError:
        return False


def load_draft_model(draft_ref: str) -> Any:
    """Load or retrieve cached DFlash draft model.

    Only the lightweight draft model is loaded from disk (~3.5GB).
    The target model is reused from oMLX's already-loaded instance.

    Args:
        draft_ref: HuggingFace ID or local path for the DFlash draft model.

    Returns:
        The draft model.
    """
    with _cache_lock:
        if draft_ref in _cached_drafts:
            logger.debug(f"DFlash draft cache hit for {draft_ref}")
            return _cached_drafts[draft_ref]

    from dflash_mlx.runtime import load_draft_bundle

    logger.info(f"Loading DFlash draft model: {draft_ref}")
    t0 = time.time()

    draft_model, _ = load_draft_bundle(draft_ref, lazy=True)

    elapsed = time.time() - t0
    logger.info(f"DFlash draft model loaded in {elapsed:.1f}s")

    with _cache_lock:
        _cached_drafts[draft_ref] = draft_model
        return draft_model


def clear_dflash_cache() -> None:
    """Clear cached DFlash models (e.g., when switching models)."""
    with _cache_lock:
        _cached_drafts.clear()
    logger.info("DFlash model cache cleared")


def run_dflash_generation(
    target_ref: str,
    draft_ref: str,
    prompt: str,
    max_tokens: int,
    *,
    use_chat_template: bool = True,
    block_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    target_model: Any = None,
    tokenizer: Any = None,
):
    """Run DFlash generation, yielding events as they arrive.

    This is a synchronous generator designed to run on the MLX executor thread.
    Yields events (token, summary, error) one at a time for real-time streaming.

    Args:
        target_ref: Target model reference (HF ID or local path).
        draft_ref: DFlash draft model reference (HF ID or local path).
        prompt: Input prompt string.
        max_tokens: Maximum tokens to generate.
        use_chat_template: Whether to apply chat template.
        block_tokens: Block size for draft generation.
        stop_token_ids: Token IDs that stop generation.
        target_model: Pre-loaded target model (avoids double-loading ~54GB).
        tokenizer: Pre-loaded tokenizer (paired with target_model).

    Yields:
        Dicts with 'event' key ('token', 'summary', or 'error').
    """
    from dflash_mlx.runtime import (
        stream_dflash_generate,
        load_target_bundle,
    )

    draft_model = load_draft_model(draft_ref)

    if target_model is not None and tokenizer is not None:
        # Use pre-loaded target model from oMLX engine (zero-copy, saves ~54GB RAM).
        # dflash_mlx needs its own model structure (with speculative hooks installed),
        # so we load the structure from load_target_bundle and replace weight arrays
        # with oMLX's already-loaded arrays (zero-copy sharing).
        logger.info("DFlash attempting zero-copy weight sharing with target model")
        weight_sharing_ok = False
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from dflash_mlx.runtime import load_target_bundle

            # Load the dflash-compatible model structure (with hooks installed).
            # lazy=True means weights are placeholders — we'll replace them below.
            df_model, df_tokenizer, _ = load_target_bundle(target_ref, lazy=True)

            # Unwrap oMLX's VLM model to the ForCausalLM level.
            omlx_ca = target_model
            if hasattr(omlx_ca, '_vlm_model'):
                omlx_ca = omlx_ca._vlm_model
            if hasattr(omlx_ca, 'language_model'):
                omlx_ca = omlx_ca.language_model

            df_ca = df_model
            # df_model from load_target_bundle is already ForCausalLM

            # MLX parameters() returns a nested dict, not flat dotted paths.
            # Flatten to get dotted-name -> array mapping for matching.
            def _flatten_params(d, prefix=''):
                items = {}
                for k, v in d.items():
                    full = f'{prefix}.{k}' if prefix else k
                    if isinstance(v, dict):
                        items.update(_flatten_params(v, full))
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            items.update(_flatten_params(item, f'{full}.{i}'))
                    elif isinstance(v, mx.array):
                        items[full] = v
                return items

            omlx_weights = _flatten_params(omlx_ca.parameters())
            df_weights = _flatten_params(df_ca.parameters())

            omlx_names = set(omlx_weights.keys())
            df_names = set(df_weights.keys())

            logger.info(
                f"DFlash weight sharing: omlx has {len(omlx_names)} params "
                f"({type(omlx_ca).__name__}), df has {len(df_names)} params "
                f"({type(df_ca).__name__})"
            )

            if not omlx_names:
                raise RuntimeError("oMLX model has no parameters (not loaded yet?)")
            if not df_names:
                raise RuntimeError("DFlash model has no parameters")

            # Detect and strip common prefix mismatches between model hierarchies.
            # e.g., omlx has "lm_head.weight" while df has "language_model.lm_head.weight"
            _common_prefixes = ("language_model.", "model.")
            df_stripped = {}  # stripped_name -> original_name mapping for df
            for prefix in _common_prefixes:
                if all(n.startswith(prefix) for n in df_names) and not any(
                    n.startswith(prefix) for n in omlx_names
                ):
                    logger.info(f"DFlash stripping df prefix '{prefix}' for matching")
                    df_stripped = {k[len(prefix):]: k for k in df_weights}
                    break
            if not df_stripped:
                df_stripped = {k: k for k in df_weights}

            # Build lookup: stripped_name -> omlx_weight
            omlx_lookup = {k: v for k, v in omlx_weights.items()}

            # Replace dflash model's weights with oMLX's shared arrays.
            # Navigate the ORIGINAL (unstripped) path in df_model, but match
            # using stripped names.
            shared = 0
            skipped = 0
            for stripped_name, original_name in df_stripped.items():
                if stripped_name not in omlx_lookup:
                    skipped += 1
                    continue
                # Navigate the original df path (with prefix) to set the attribute
                parts = original_name.split('.')
                obj = df_ca
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                if obj is not None and hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], omlx_lookup[stripped_name])
                    shared += 1
                else:
                    skipped += 1

            logger.info(
                f"DFlash weight sharing: {shared}/{len(df_weights)} params shared, "
                f"{skipped} skipped"
            )
            if shared == 0:
                raise RuntimeError(
                    f"Zero params shared — model structures incompatible "
                    f"(omlx type={type(omlx_ca).__name__}, df type={type(df_ca).__name__})"
                )

            target_model = df_model
            tokenizer = df_tokenizer
            weight_sharing_ok = True
        except Exception as e:
            logger.warning(f"DFlash zero-copy weight sharing failed: {e}, loading from disk")
            target_model = None

        if not weight_sharing_ok:
            # Fallback: load target model from disk (uses extra ~54GB).
            cache_key = f"{target_ref}::{draft_ref}"
            with _cache_lock:
                if cache_key not in _cached_drafts:
                    _cached_drafts[cache_key] = {}
                target_cache = _cached_drafts[cache_key]
                if "target" not in target_cache:
                    logger.info(f"Loading DFlash target model: {target_ref}")
                    t0 = time.time()
                    target_model, tokenizer, _ = load_target_bundle(target_ref, lazy=True)
                    elapsed = time.time() - t0
                    logger.info(f"DFlash target model loaded in {elapsed:.1f}s")
                    target_cache["target"] = target_model
                    target_cache["tokenizer"] = tokenizer
                target_model = target_cache["target"]
                tokenizer = target_cache["tokenizer"]

    try:
        for event in stream_dflash_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=prompt,
            max_new_tokens=max_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids or [],
        ):
            yield event
    except Exception as e:
        logger.error(f"DFlash generation error: {e}")
        yield {"event": "error", "error": str(e)}


def get_eligible_draft_models() -> list[dict[str, str]]:
    """Return list of known DFlash draft models with their compatible targets.

    Returns:
        List of dicts with 'draft_model', 'target_model', 'description' keys.
    """
    return [
        {
            "draft_model": "z-lab/Qwen3.5-27B-DFlash",
            "target_patterns": ["Qwen3.5-27B", "qwen3.5-27b"],
            "description": "DFlash draft for Qwen3.5-27B (all quantizations)",
        },
        {
            "draft_model": "z-lab/Qwen3.5-35B-A3B-DFlash",
            "target_patterns": ["Qwen3.5-35B-A3B", "qwen3.5-35b-a3b"],
            "description": "DFlash draft for Qwen3.5-35B-A3B MoE",
        },
    ]


def suggest_draft_model(model_id: str) -> Optional[str]:
    """Suggest a DFlash draft model for the given target model ID.

    Args:
        model_id: The target model identifier.

    Returns:
        Suggested draft model HuggingFace ID, or None.
    """
    model_id_lower = model_id.lower()
    for entry in get_eligible_draft_models():
        for pattern in entry["target_patterns"]:
            if pattern.lower() in model_id_lower:
                return entry["draft_model"]
    return None


def is_draft_compatible(target_model_id: str, draft_ref: str) -> bool:
    """Check if a draft model is compatible with the target model.

    Args:
        target_model_id: The target model identifier.
        draft_ref: The draft model reference (HuggingFace ID or path).

    Returns:
        True if the draft model is compatible with the target.
    """
    target_lower = target_model_id.lower()
    for entry in get_eligible_draft_models():
        if entry["draft_model"] == draft_ref:
            for pattern in entry["target_patterns"]:
                if pattern.lower() in target_lower:
                    return True
            return False
    # Unknown draft model — allow but warn
    logger.warning(f"Unknown DFlash draft model '{draft_ref}', allowing without compatibility check")
    return True
