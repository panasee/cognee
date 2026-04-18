from unittest.mock import MagicMock


def _make_adapter(monkeypatch, **kwargs):
    monkeypatch.setattr(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter.instructor.from_litellm",
        lambda *args, **kw: MagicMock(),
    )

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
        GenericAPIAdapter,
    )

    defaults = {
        "api_key": "test-key",
        "model": "gemini-3-flash-preview",
        "max_completion_tokens": 4096,
        "name": "Custom",
        "endpoint": "http://localhost:8765",
        "instructor_mode": "json_mode",
    }
    defaults.update(kwargs)
    return GenericAPIAdapter(**defaults)


def test_custom_adapter_normalizes_openai_compatible_endpoint(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    assert adapter.model == "openai/gemini-3-flash-preview"
    assert adapter.endpoint == "http://localhost:8765/v1"


def test_custom_adapter_respects_explicit_provider_override(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        llm_args={"custom_llm_provider": "bedrock"},
    )

    assert adapter.model == "gemini-3-flash-preview"
    assert adapter.endpoint == "http://localhost:8765"


def test_custom_adapter_preserves_prefixed_models(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        model="openai/gpt-4o-mini",
        endpoint="http://localhost:8765/v1/chat/completions",
    )

    assert adapter.model == "openai/gpt-4o-mini"
    assert adapter.endpoint == "http://localhost:8765/v1/chat/completions"
