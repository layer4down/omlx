# SPDX-License-Identifier: Apache-2.0
"""Tests for model alias resolution in audio endpoints (INV-04, INV-03, INV-05).

Verifies that audio endpoints correctly resolve model aliases to their actual
model IDs before loading engines, matching the behavior of chat/completion
endpoints.

This addresses GitHub issue #489: Audio endpoints do not resolve model aliases.
"""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid WAV bytes (silence)."""
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


TINY_WAV = _make_wav_bytes()


def _make_mock_stt_engine() -> MagicMock:
    """Build a mock STTEngine."""
    from omlx.engine.stt import STTEngine

    engine = MagicMock(spec=STTEngine)
    engine.transcribe = AsyncMock(
        return_value={
            "text": "hello world",
            "language": "en",
            "duration": 0.1,
            "segments": [],
        }
    )
    return engine


def _make_mock_tts_engine() -> MagicMock:
    """Build a mock TTSEngine."""
    from omlx.engine.tts import TTSEngine

    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=_make_wav_bytes())
    return engine


def _make_mock_sts_engine() -> MagicMock:
    """Build a mock STSEngine."""
    from omlx.engine.sts import STSEngine

    engine = MagicMock(spec=STSEngine)
    engine.process = AsyncMock(return_value=_make_wav_bytes())
    return engine


def _make_mock_pool_with_alias(alias: str, actual: str, model_type: str = "audio_stt"):
    """Build a mock EnginePool that tracks resolve_model_id and get_engine calls."""
    pool = MagicMock()

    if model_type == "audio_stt":
        pool.get_engine = AsyncMock(return_value=_make_mock_stt_engine())
        pool.get_entry = MagicMock(
            return_value=MagicMock(
                model_type="audio_stt",
                engine_type="stt",
            )
        )
    elif model_type == "audio_tts":
        pool.get_engine = AsyncMock(return_value=_make_mock_tts_engine())
        pool.get_entry = MagicMock(
            return_value=MagicMock(
                model_type="audio_tts",
                engine_type="tts",
            )
        )
    else:  # audio_sts
        pool.get_engine = AsyncMock(return_value=_make_mock_sts_engine())
        pool.get_entry = MagicMock(
            return_value=MagicMock(
                model_type="audio_sts",
                engine_type="sts",
            )
        )

    pool.get_model_ids.return_value = [actual]
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()

    def resolve_side_effect(model_id, settings_manager):
        if model_id == alias:
            return actual
        return model_id

    pool.resolve_model_id = MagicMock(side_effect=resolve_side_effect)
    return pool


def _ensure_audio_routes(app):
    """Register audio routes if not already present."""
    from omlx.api.audio_routes import router as audio_router

    audio_paths = {"/v1/audio/transcriptions", "/v1/audio/speech", "/v1/audio/process"}
    existing = {getattr(r, "path", "") for r in app.routes}
    if not audio_paths & existing:
        app.include_router(audio_router)


@pytest.fixture
def audio_alias_client(model_type: str = "audio_stt"):
    """TestClient with mocked pool that has alias configured."""
    from omlx.server import app

    _ensure_audio_routes(app)

    alias = "qwen3-tts"
    actual = "mlx-community--Qwen3-TTS-mlx"

    mock_pool = _make_mock_pool_with_alias(alias, actual, model_type)

    with patch("omlx.server._server_state") as mock_state:
        mock_state.engine_pool = mock_pool
        mock_state.global_settings = None
        mock_state.process_memory_enforcer = None
        mock_state.hf_downloader = None
        mock_state.ms_downloader = None
        mock_state.mcp_manager = None
        mock_state.api_key = None
        mock_state.settings_manager = MagicMock()
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool, alias, actual


# ---------------------------------------------------------------------------
# Test Alias Resolution in Audio Endpoints
# ---------------------------------------------------------------------------


class TestAudioAliasResolution:
    """Verify audio endpoints resolve model aliases before loading engines."""

    def test_stt_resolves_alias_before_get_engine(self, audio_alias_client):
        """STT endpoint calls resolve_model_id before get_engine."""
        client, mock_pool, alias, actual = audio_alias_client

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": alias},
        )

        assert response.status_code == 200

        resolve_call = mock_pool.resolve_model_id.call_args
        get_engine_call = mock_pool.get_engine.call_args

        assert resolve_call is not None, "resolve_model_id was not called"
        assert get_engine_call is not None, "get_engine was not called"

        assert resolve_call.args[0] == alias
        assert get_engine_call.args[0] == actual, (
            f"get_engine should be called with resolved model '{actual}', got '{get_engine_call.args[0]}'"
        )

    def test_tts_resolves_alias_before_get_engine(self):
        """TTS endpoint calls resolve_model_id before get_engine."""
        from omlx.server import app

        _ensure_audio_routes(app)

        alias = "qwen3-tts"
        actual = "mlx-community--Qwen3-TTS-mlx"

        mock_pool = _make_mock_pool_with_alias(alias, actual, "audio_tts")

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={"model": alias, "input": "Hello, world!"},
                )

                assert response.status_code == 200

                resolve_call = mock_pool.resolve_model_id.call_args
                get_engine_call = mock_pool.get_engine.call_args

                assert resolve_call is not None, "resolve_model_id was not called"
                assert get_engine_call is not None, "get_engine was not called"

                assert resolve_call.args[0] == alias
                assert get_engine_call.args[0] == actual, (
                    f"get_engine should be called with resolved model '{actual}', got '{get_engine_call.args[0]}'"
                )

    def test_sts_resolves_alias_before_get_engine(self):
        """STS/audio processing endpoint calls resolve_model_id before get_engine."""
        from omlx.server import app

        _ensure_audio_routes(app)

        alias = "my-denoiser"
        actual = "mlx-community--DeepFilterNet-mlx"

        mock_pool = _make_mock_pool_with_alias(alias, actual, "audio_sts")

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/process",
                    files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
                    data={"model": alias},
                )

                assert response.status_code == 200

                resolve_call = mock_pool.resolve_model_id.call_args
                get_engine_call = mock_pool.get_engine.call_args

                assert resolve_call is not None, "resolve_model_id was not called"
                assert get_engine_call is not None, "get_engine was not called"

                assert resolve_call.args[0] == alias
                assert get_engine_call.args[0] == actual, (
                    f"get_engine should be called with resolved model '{actual}', got '{get_engine_call.args[0]}'"
                )

    def test_stt_with_actual_model_id_no_resolution_needed(self, audio_alias_client):
        """STT endpoint works when actual model ID is provided (no alias)."""
        client, mock_pool, alias, actual = audio_alias_client

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": actual},
        )

        assert response.status_code == 200

        resolve_call = mock_pool.resolve_model_id.call_args
        get_engine_call = mock_pool.get_engine.call_args

        assert resolve_call is not None
        assert resolve_call.args[0] == actual
        assert get_engine_call.args[0] == actual

    def test_tts_with_actual_model_id_no_resolution_needed(self):
        """TTS endpoint works when actual model ID is provided (no alias)."""
        from omlx.server import app

        _ensure_audio_routes(app)

        actual = "mlx-community--Qwen3-TTS-mlx"

        mock_pool = _make_mock_pool_with_alias("qwen3-tts", actual, "audio_tts")

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={"model": actual, "input": "Hello"},
                )

                assert response.status_code == 200

                resolve_call = mock_pool.resolve_model_id.call_args
                get_engine_call = mock_pool.get_engine.call_args

                assert resolve_call is not None
                assert resolve_call.args[0] == actual
                assert get_engine_call.args[0] == actual


# ---------------------------------------------------------------------------
# Test Alias Resolution Helper Function
# ---------------------------------------------------------------------------


class TestResolveModelIdHelper:
    """Unit tests for the _resolve_model_id helper function."""

    def test_resolve_model_id_with_alias(self):
        """_resolve_model_id resolves alias to actual model ID."""
        from omlx.api.audio_routes import _resolve_model_id

        alias = "my-tts"
        actual = "mlx-community--Qwen3-TTS-mlx"

        mock_pool = MagicMock()
        mock_pool.resolve_model_id = MagicMock(return_value=actual)

        mock_settings = MagicMock()

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.settings_manager = mock_settings

            result = _resolve_model_id(alias)

            assert result == actual
            mock_pool.resolve_model_id.assert_called_once_with(alias, mock_settings)

    def test_resolve_model_id_without_alias(self):
        """_resolve_model_id returns model ID unchanged if no alias configured."""
        from omlx.api.audio_routes import _resolve_model_id

        model_id = "mlx-community--Qwen3-TTS-mlx"

        mock_pool = MagicMock()
        mock_pool.resolve_model_id = MagicMock(return_value=model_id)

        mock_settings = MagicMock()

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.settings_manager = mock_settings

            result = _resolve_model_id(model_id)

            assert result == model_id

    def test_resolve_model_id_with_none_pool(self):
        """_resolve_model_id returns original ID if pool is None."""
        from omlx.api.audio_routes import _resolve_model_id

        model_id = "some-model"

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = None

            result = _resolve_model_id(model_id)

            assert result == model_id
