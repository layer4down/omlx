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

    Yields:
        Dicts with 'event' key ('token', 'summary', or 'error').
    """
    from dflash_mlx.runtime import stream_dflash_generate, load_target_bundle

    draft_model = load_draft_model(draft_ref)

    # Load target model with DFlash's own loader (ensures compatible weight format).
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
