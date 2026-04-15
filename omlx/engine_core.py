# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Engine Core for oMLX continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

import asyncio
import concurrent.futures
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

import mlx.core as mx

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .output_collector import RequestOutputCollector, RequestStreamState
from .model_registry import get_registry, ModelOwnershipError

logger = logging.getLogger(__name__)

_global_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None


def get_mlx_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the global MLX executor (lazy singleton).

    mlx-lm's BatchGenerator uses a module-level Metal stream
    (generation_stream), so ALL MLX GPU operations across all models
    MUST be serialized onto one thread to prevent Metal command buffer
    races that cause segfaults. See issue #85.
    """
    global _global_mlx_executor
    if _global_mlx_executor is None:
        _global_mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-global"
        )
    return _global_mlx_executor


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: Optional[SchedulerConfig] = None
    step_interval: float = 0.001  # 1ms between steps
    stream_interval: int = 1  # Tokens to batch before streaming (1=every token)


class EngineCore:
    """
    Core engine for oMLX inference with continuous batching.

    This engine runs the generation loop and manages request lifecycle.
    It provides both sync and async interfaces for request handling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
        engine_id: Optional[str] = None,
        force_model_ownership: bool = True,
    ):
        """
        Initialize the engine.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Engine configuration
            engine_id: Optional unique ID for this engine (auto-generated if None)
            force_model_ownership: If True (default), forcibly take model ownership
                                   from any existing engine. If False, raises
                                   ModelOwnershipError if model is in use.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        # Acquire model ownership
        registry = get_registry()
        registry.acquire(
            model=model,
            engine=self,
            engine_id=self._engine_id,
            force=force_model_ownership,
        )
        self._owns_model = True

        # Create scheduler
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
        )

        # Output collectors for low-latency streaming (vLLM pattern)
        self._output_collectors: Dict[str, RequestOutputCollector] = {}
        self._stream_states: Dict[str, RequestStreamState] = {}
        self._finished_events: Dict[str, asyncio.Event] = {}

        # Engine state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._steps_executed = 0

        # DFlash speculative decode state
        self._dflash_target_ref: Optional[str] = None
        self._dflash_draft_ref: Optional[str] = None
        self._dflash_enabled: bool = False
        self._dflash_model = None  # Already-loaded model (reused, no dup memory)

        # Global single-thread executor shared across ALL engines.
        # mlx-lm uses a module-level Metal stream, so concurrent MLX calls
        # from different engine threads cause segfaults. See issue #85.
        self._mlx_executor = get_mlx_executor()

        logger.debug(f"Engine {self._engine_id} initialized")

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        """Stop the engine loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop - runs scheduler steps on the MLX executor.

        All scheduler steps run on _mlx_executor (single-worker thread) to
        guarantee that MLX GPU operations are never concurrent.  VLM vision
        encoding also runs on the same executor, so inline scheduler.step()
        on the event loop would race with vision mx.eval() and segfault.
        """
        loop = asyncio.get_running_loop()

        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = (stream_interval == 1)

        while self._running:
            try:
                if self.scheduler.has_requests():
                    output = await loop.run_in_executor(
                        self._mlx_executor, self.scheduler.step
                    )
                    self._steps_executed += 1

                    # Fast path: distribute outputs to collectors
                    outputs = output.outputs
                    if outputs:
                        collectors = self._output_collectors
                        states = self._stream_states
                        events = self._finished_events

                        for req_output in outputs:
                            rid = req_output.request_id
                            collector = collectors.get(rid)

                            if collector is not None:
                                # Optimized: skip stream_interval check when interval=1
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished
                                    ):
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)

                            if req_output.finished:
                                event = events.get(rid)
                                if event:
                                    event.set()
                                # Note: cleanup is handled by stream_outputs() finally block
                                # _delayed_cleanup() was causing double cleanup race condition

                        # Always yield to prevent event loop starvation.
                        # Without this, orphaned requests (client disconnected but
                        # request still in scheduler) block the entire event loop,
                        # making the server unresponsive to all HTTP requests.
                        await asyncio.sleep(0)
                else:
                    # No work, yield control
                    await asyncio.sleep(step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                logger.error(f"Engine loop error: {e}\n{traceback.format_exc()}")
                # Fail all requests and remove from scheduler to prevent
                # infinite loop (has_requests() must return False).
                failed_ids = await loop.run_in_executor(
                    self._mlx_executor, self.scheduler.fail_all_requests
                )
                for rid in failed_ids:
                    collector = self._output_collectors.get(rid)
                    if collector is not None:
                        collector.put(
                            RequestOutput(
                                request_id=rid,
                                finished=True,
                                finish_reason="error",
                                error=str(e),
                            )
                        )
                    event = self._finished_events.get(rid)
                    if event:
                        event.set()
                await asyncio.sleep(0.1)

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        vlm_inputs_embeds: Optional[Any] = None,
        vlm_extra_kwargs: Optional[Dict[str, Any]] = None,
        vlm_image_hash: Optional[str] = None,
        specprefill: Optional[bool] = None,
        specprefill_keep_pct: Optional[float] = None,
        specprefill_threshold: Optional[int] = None,
        specprefill_system_end: Optional[int] = None,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal
            vlm_inputs_embeds: Pre-computed vision+text embeddings for VLM
            vlm_extra_kwargs: Model-specific VLM kwargs (e.g., position_ids)
            vlm_image_hash: SHA256 hash of images for prefix cache
            specprefill: Per-request SpecPrefill override (True/False/None)
            specprefill_keep_pct: Per-request keep rate override
            specprefill_threshold: Per-request threshold override (min tokens)

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
        )

        # SpecPrefill: resolve per-request settings.
        # The scheduler checks _specprefill_enabled to decide whether to score.
        if specprefill is not None:
            request._specprefill_enabled = specprefill
        elif self.scheduler._specprefill_draft_model is not None:
            # Draft model is loaded → enable by default
            request._specprefill_enabled = True
        if specprefill_keep_pct is not None:
            request._specprefill_keep_pct = specprefill_keep_pct
        if specprefill_threshold is not None:
            request._specprefill_threshold = specprefill_threshold
        if specprefill_system_end is not None and specprefill_system_end > 0:
            request.specprefill_system_end = specprefill_system_end

        # Setup output collector with stream_interval from config
        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[request_id] = asyncio.Event()

        # Add to scheduler — route through the MLX executor so that
        # prefix cache reconstruction (mx.load, mx.concatenate) never
        # races with scheduler.step() on the Metal stream.  See #95.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._mlx_executor, self.scheduler.add_request, request
        )

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request.

        Uses deferred abort pattern: scheduler.abort_request() just enqueues
        the request ID into a thread-safe set. The actual abort is processed
        at the start of the next scheduler.step() call, ensuring it runs in
        the same execution context as generation (no race conditions).

        Signals the consumer (stream_outputs/generate) with an abort error
        so it can exit gracefully. Cleanup is handled by the consumer's
        finally block, NOT here -- calling _cleanup_request() immediately
        after put() would clear the output before the consumer can drain it.
        """
        result = self.scheduler.abort_request(request_id)

        # Signal consumer with abort error so any waiting
        # stream_outputs() / generate() can exit gracefully.
        # Matches abort_all_requests() pattern.
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(
                RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="abort",
                    error="Request aborted",
                )
            )
        event = self._finished_events.get(request_id)
        if event is not None:
            event.set()

        return result

    async def abort_all_requests(self) -> int:
        """Abort all active requests without stopping the engine.

        Sends error output to all active collectors and marks requests
        for deferred abort in the scheduler. Cleanup is handled by
        the consumer (stream_outputs/generate).
        """
        request_ids = list(self._output_collectors.keys())
        for rid in request_ids:
            self.scheduler.abort_request(rid)
            collector = self._output_collectors.get(rid)
            if collector is not None:
                error_msg = (
                    "Request aborted: process memory limit exceeded. "
                    "Increase --max-process-memory or reduce context size."
                )
                collector.put(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        new_text=f"\n\n[Error: {error_msg}]",
                        error=error_msg,
                    )
                )
            event = self._finished_events.get(rid)
            if event is not None:
                event.set()
        if request_ids:
            logger.warning(
                f"Aborted {len(request_ids)} requests due to memory pressure"
            )
        return len(request_ids)

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up request tracking.

        Only cleans engine-core level state (collectors, events).
        Scheduler state is cleaned by _do_abort_request (deferred abort)
        or _cleanup_finished (normal completion).
        """
        collector = self._output_collectors.pop(request_id, None)
        if collector:
            collector.clear()
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)

    async def _delayed_cleanup(self, request_id: str, delay: float = 5.0) -> None:
        """
        Cleanup request after delay if not already cleaned.

        This handles the case where a client disconnects before consuming
        the stream_outputs() generator, which would prevent the finally
        block from running.
        """
        await asyncio.sleep(delay)
        if request_id in self._output_collectors:
            logger.debug(f"Delayed cleanup for request {request_id}")
            self._cleanup_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request with low-latency non-blocking pattern.

        Uses the vLLM pattern: get_nowait() or await get()
        This avoids unnecessary task switches when output is available.

        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds

        Yields:
            RequestOutput objects as tokens are generated
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            # Request might not be added yet or already cleaned up
            return

        try:
            while True:
                try:
                    # Non-blocking drain pattern from vLLM
                    # Try get_nowait first to avoid task switch if output ready
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(
                                collector.get(),
                                timeout=timeout
                            )
                    else:
                        output = collector.get_nowait() or await collector.get()

                    yield output

                    if output.error:
                        raise RuntimeError(output.error)

                    if output.finished:
                        break

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for request {request_id}")
                    break

        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate a complete response (non-streaming).

        This method is optimized to avoid streaming overhead when
        you only need the final result.

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            request_id: Optional request ID

        Returns:
            Final RequestOutput with complete text
        """
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

        # Wait for completion using event instead of streaming
        # This avoids the waiting_consumer tracking overhead
        event = self._finished_events.get(request_id)
        if event is None:
            raise RuntimeError(f"No event for request {request_id}")

        try:
            # Wait for the request to finish
            await event.wait()
        except asyncio.CancelledError:
            # Client disconnected or task was cancelled - abort the request
            # to free scheduler/GPU resources (prevents orphaned requests)
            logger.info(f"Request {request_id} cancelled, aborting")
            await self.abort_request(request_id)
            self._cleanup_request(request_id)
            raise

        # Get the final output from collector
        collector = self._output_collectors.get(request_id)
        if collector is None:
            raise RuntimeError(f"No collector for request {request_id}")

        # Drain all outputs and get the last one
        final_output = None
        while True:
            output = collector.get_nowait()
            if output is None:
                break
            final_output = output

        # Cleanup
        self._cleanup_request(request_id)

        if final_output is None:
            raise RuntimeError(f"No output for request {request_id}")

        if final_output.error:
            raise RuntimeError(final_output.error)

        return final_output

    def generate_batch_sync(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses synchronously for maximum throughput.

        This bypasses the async engine loop entirely, running the scheduler
        directly for optimal batching performance. Use this when you don't
        need streaming and want maximum throughput.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters (same for all)

        Returns:
            List of RequestOutput in same order as prompts
        """
        from .request import Request
        import uuid as uuid_module

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add all requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = str(uuid_module.uuid4())
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            self.scheduler.add_request(request)
            request_ids.append(request_id)

        # Process until all done - direct scheduler access, no async overhead
        results: Dict[str, RequestOutput] = {}
        while self.scheduler.has_requests():
            output = self.scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    results[req_output.request_id] = req_output

        # Cleanup
        for rid in request_ids:
            self.scheduler.remove_finished_request(rid)

        # Return in original order
        return [results[rid] for rid in request_ids]

    # ── DFlash Speculative Decode ──────────────────────────────────────────

    def configure_dflash(
        self,
        enabled: bool,
        target_ref: Optional[str] = None,
        draft_ref: Optional[str] = None,
    ) -> None:
        """Configure DFlash speculative decoding for this engine.

        Args:
            enabled: Whether to enable DFlash.
            target_ref: Target model reference (HF ID or local path).
            draft_ref: DFlash draft model reference (HF ID or local path).
        """
        self._dflash_enabled = enabled
        self._dflash_target_ref = target_ref
        self._dflash_draft_ref = draft_ref
        if enabled:
            logger.info(
                f"DFlash speculative decode enabled: "
                f"target={target_ref}, draft={draft_ref}"
            )
        else:
            logger.info("DFlash speculative decode disabled")

    def is_dflash_enabled(self) -> bool:
        """Check if DFlash speculative decode is active."""
        return self._dflash_enabled

    async def add_dflash_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        use_chat_template: bool = True,
    ) -> str:
        """Add a request using DFlash speculative decoding.

        This bypasses the normal scheduler/BatchGenerator path and runs
        DFlash generation directly on the MLX executor. Tokens are emitted
        through the same RequestOutputCollector mechanism for SSE streaming.

        Args:
            prompt: Input prompt (string or token IDs).
            sampling_params: Generation parameters (max_tokens used).
            request_id: Optional custom request ID.
            use_chat_template: Whether to apply chat template.

        Returns:
            The request ID.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Setup output collector
        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[request_id] = asyncio.Event()

        max_tokens = sampling_params.max_tokens or 4096
        prompt_str = prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)

        # Run DFlash generation in background
        asyncio.create_task(
            self._run_dflash_generation(
                request_id=request_id,
                prompt=prompt_str,
                max_tokens=max_tokens,
                use_chat_template=use_chat_template,
            )
        )

        return request_id

    async def _run_dflash_generation(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        use_chat_template: bool = True,
    ) -> None:
        """Run DFlash generation and emit tokens through the collector.

        Runs on the MLX executor thread for GPU serialization.
        """
        loop = asyncio.get_running_loop()
        collector = self._output_collectors.get(request_id)
        if collector is None:
            return

        def _generate():
            from .patches.dflash_specdec import run_dflash_generation

            try:
                gen_start = time.perf_counter()
                completion_tokens = 0
                prompt_token_count = 0
                prefill_duration = 0.0
                acceptance = 0.0
                tokenizer = None

                # run_dflash_generation is now a generator — yields events as they arrive
                # so tokens stream to the collector in real-time instead of batching
                # Unwrap VLMModelAdapter to get the raw model dflash_mlx expects.
                # dflash_mlx needs model.language_model (VLM) or model.model (mlx-lm).
                raw_model = self.model
                if hasattr(raw_model, '_vlm_model'):
                    raw_model = raw_model._vlm_model

                for event in run_dflash_generation(
                    target_ref=self._dflash_target_ref,
                    draft_ref=self._dflash_draft_ref,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    use_chat_template=use_chat_template,
                    target_model=raw_model,
                    tokenizer=self.tokenizer,
                ):
                    if event.get("event") == "token":
                        # Lazily resolve tokenizer on first token
                        if tokenizer is None:
                            from .patches.dflash_specdec import _cached_drafts, _cache_lock
                            cache_key = f"{self._dflash_target_ref}::{self._dflash_draft_ref}"
                            with _cache_lock:
                                tokenizer = _cached_drafts.get(cache_key, {}).get("tokenizer")
                            if tokenizer is None:
                                # Fallback: use engine's own tokenizer
                                tokenizer = self.tokenizer
                        token_id = int(event["token_id"])
                        text = tokenizer.decode([token_id])
                        completion_tokens += 1
                        collector.put(RequestOutput(
                            request_id=request_id,
                            new_text=text,
                            finished=False,
                            completion_tokens=completion_tokens,
                        ))
                    elif event.get("event") == "summary":
                        prompt_token_count = event.get("prompt_token_count", 0)
                        acceptance = event.get("acceptance_ratio", 0.0)
                        phase_timings = event.get("phase_timings_us", {})
                        prefill_duration = phase_timings.get("prefill", 0.0) / 1_000_000.0
                    elif event.get("event") == "error":
                        logger.error(f"DFlash event error: {event.get('error', 'unknown')}")

                gen_elapsed = time.perf_counter() - gen_start
                logger.info(
                    f"DFlash request {request_id} completed: "
                    f"{completion_tokens} tokens in {gen_elapsed:.2f}s "
                    f"({completion_tokens / gen_elapsed:.1f} tok/s), "
                    f"prompt={prompt_token_count}, "
                    f"acceptance={acceptance:.1%}"
                )

                # Signal completion with accurate metrics
                collector.put(RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="stop" if completion_tokens > 0 else "length",
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_tokens,
                    prefill_duration_override=prefill_duration if prefill_duration > 0 else None,
                    generation_duration_override=gen_elapsed if gen_elapsed > 0 else None,
                ))

            except Exception as e:
                import traceback
                logger.error(f"DFlash generation failed: {e}\n{traceback.format_exc()}")
                collector.put(RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error=str(e),
                    new_text=f"\n\n[DFlash error: {e}]",
                ))

        try:
            await loop.run_in_executor(self._mlx_executor, _generate)
        finally:
            event = self._finished_events.get(request_id)
            if event:
                event.set()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_collectors),
            "stream_interval": self.config.stream_interval,
            **scheduler_stats,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.scheduler.get_cache_stats()

    def _release_model(self) -> None:
        """Release model ownership."""
        if self._owns_model and not self._closed:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

    def close(self) -> None:
        """
        Explicitly close the engine and release resources.

        This should be called when done using the engine, especially
        if you plan to create another engine with the same model.
        """
        if self._closed:
            return

        # Release model ownership BEFORE setting _closed
        # (_release_model checks not self._closed)
        if self._owns_model:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

        self._closed = True

        # Shutdown scheduler (clears paged SSD cache if configured)
        self.scheduler.shutdown()

        # Reset scheduler to clear BatchGenerator and all caches
        self.scheduler.deep_reset()

        # Clear output collectors
        for collector in self._output_collectors.values():
            collector.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        # Release model and tokenizer references for GC
        self.model = None
        self.tokenizer = None
        self.scheduler = None

        logger.debug(f"Engine {self._engine_id} closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._release_model()
        except Exception:
            # Ignore errors during garbage collection
            pass

    @property
    def engine_id(self) -> str:
        """Get the engine ID."""
        return self._engine_id


class AsyncEngineCore:
    """
    Async context manager wrapper for EngineCore.

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            request_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs(request_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[EngineConfig] = None,
    ):
        self.engine = EngineCore(model, tokenizer, config)

    @property
    def _mlx_executor(self):
        """Expose the MLX executor for VLM vision encoding."""
        return self.engine._mlx_executor

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.engine.stop()

    def start(self) -> None:
        """Start engine (creates task in current loop)."""
        asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        """Stop the engine."""
        await self.engine.stop()

    async def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Add a request."""
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        return await self.engine.abort_request(request_id)

    async def abort_all_requests(self) -> int:
        """Abort all active requests without stopping the engine."""
        return await self.engine.abort_all_requests()

    async def stream_outputs(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs."""
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> RequestOutput:
        """Generate complete response."""
        return await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine stats."""
        return self.engine.get_stats()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get prefix cache statistics."""
        return self.engine.get_cache_stats()

    # DFlash speculative decode proxies
    def configure_dflash(self, enabled: bool, target_ref=None, draft_ref=None):
        """Configure DFlash speculative decoding."""
        self.engine.configure_dflash(enabled=enabled, target_ref=target_ref, draft_ref=draft_ref)

    def is_dflash_enabled(self) -> bool:
        """Check if DFlash speculative decoding is enabled."""
        return self.engine.is_dflash_enabled()

    async def add_dflash_request(self, prompt, sampling_params=None, request_id=None, use_chat_template=True) -> str:
        """Add a DFlash speculative decode request."""
        return await self.engine.add_dflash_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            use_chat_template=use_chat_template,
        )
