import json
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from doc_extractor.provider import (
    OllamaProvider,
    ProviderConfig,
    StructuredOutputError,
    load_providers,
)
from doc_extractor.models import SchemaExtraction


async def test_valid_content_parses_into_response_model():
    """(a) valid content parses into response model."""
    config = ProviderConfig(model="qwen3.5:27b")
    provider = OllamaProvider(config)

    fake_content = json.dumps({"fields": [{"name": "title", "value": "Test Report"}]})
    fake_resp = AsyncMock(spec=httpx.Response)
    fake_resp.json.return_value = {"message": {"content": fake_content}}

    client_ctx = AsyncMock()
    client_ctx.post = AsyncMock(return_value=fake_resp)
    client_ctx.__aenter__ = AsyncMock(return_value=client_ctx)
    client_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("doc_extractor.provider.httpx.AsyncClient", return_value=client_ctx):
        result = await provider.structured("Test prompt", SchemaExtraction)

    assert isinstance(result, SchemaExtraction)
    assert len(result.fields) == 1
    assert result.fields[0].name == "title"
    assert result.fields[0].value == "Test Report"


async def test_invalid_json_then_valid_succeeds():
    """(b) invalid JSON first then valid second -> succeeds (retry works)."""
    config = ProviderConfig(model="qwen3.5:27b")
    provider = OllamaProvider(config)

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        resp = AsyncMock(spec=httpx.Response)
        if call_count == 1:
            resp.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
        else:
            fake_content = json.dumps({"fields": [{"name": "publisher", "value": "McKinsey"}]})
            resp.json.return_value = {"message": {"content": fake_content}}
        return resp

    client_ctx = AsyncMock()
    client_ctx.post = AsyncMock(side_effect=side_effect)
    client_ctx.__aenter__ = AsyncMock(return_value=client_ctx)
    client_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("doc_extractor.provider.httpx.AsyncClient", return_value=client_ctx):
        result = await provider.structured("Test prompt", SchemaExtraction)

    assert call_count == 2
    assert isinstance(result, SchemaExtraction)
    assert result.fields[0].name == "publisher"
    assert result.fields[0].value == "McKinsey"


async def test_invalid_twice_raises_structured_output_error():
    """(c) invalid twice -> StructuredOutputError."""
    config = ProviderConfig(model="qwen3.5:27b")
    provider = OllamaProvider(config)

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        resp = AsyncMock(spec=httpx.Response)
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
        return resp

    client_ctx = AsyncMock()
    client_ctx.post = AsyncMock(side_effect=side_effect)
    client_ctx.__aenter__ = AsyncMock(return_value=client_ctx)
    client_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("doc_extractor.provider.httpx.AsyncClient", return_value=client_ctx):
        with pytest.raises(StructuredOutputError):
            await provider.structured("Test prompt", SchemaExtraction)

    assert call_count == 2


async def test_load_providers_default_gives_2_providers():
    """(d) load_providers default gives 2 providers with expected models."""
    sentinel = "\x00__FAKE_KEY_d02o_default__"
    orig = os.environ.pop(sentinel, None)

    with patch("os.environ.get", return_value=None):
        providers = load_providers(sentinel)

    assert len(providers) == 2
    assert providers[0].name == "qwen3.6:latest"
    assert providers[1].name == "gemma4:26b"

    if orig is not None:
        os.environ[sentinel] = orig


async def test_load_providers_env_override():
    """(e) load_providers env override respected via monkeypatch.setenv."""
    sentinel = "\x00__FAKE_KEY_d02o_env__"
    env_json = json.dumps([{"model": "llama3:8b", "name": "my-llama"}])
    orig = os.environ.pop(sentinel, None)

    with patch("os.environ.get", return_value=env_json):
        providers = load_providers(sentinel)

    assert len(providers) == 1
    assert providers[0].name == "my-llama"
    assert providers[0].base_url == "http://127.0.0.1:11435"

    if orig is not None:
        os.environ[sentinel] = orig
