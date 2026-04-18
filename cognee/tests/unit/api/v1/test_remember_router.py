from types import SimpleNamespace
from uuid import uuid4
import importlib

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute


def _get_route_endpoint(router, path: str, method: str):
    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route.endpoint

    raise AssertionError(f"Route {method} {path} not found")


@pytest.mark.asyncio
async def test_remember_accepts_single_text_form_field(monkeypatch):
    remember_api_module = importlib.import_module("cognee.api.v1.remember")
    remember_router_module = importlib.import_module(
        "cognee.api.v1.remember.routers.get_remember_router"
    )
    usage_logger_module = importlib.import_module("cognee.shared.usage_logger")

    captured = {}

    async def fake_remember(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return SimpleNamespace(to_dict=lambda: {"status": "ok"})

    monkeypatch.setattr(remember_api_module, "remember", fake_remember)
    monkeypatch.setattr(remember_router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        usage_logger_module,
        "get_cache_config",
        lambda: SimpleNamespace(usage_logging=False),
    )

    endpoint = _get_route_endpoint(remember_router_module.get_remember_router(), "", "POST")
    response = await endpoint(
        data=None,
        text="  hello world  ",
        texts=None,
        datasetName="memory",
        datasetId=None,
        node_set=[""],
        run_in_background=False,
        custom_prompt="",
        chunks_per_batch=10,
        user=SimpleNamespace(id=str(uuid4())),
    )

    assert response == {"status": "ok"}
    assert captured["payload"] == "hello world"
    assert captured["kwargs"]["dataset_name"] == "memory"
    assert captured["kwargs"]["node_set"] is None


@pytest.mark.asyncio
async def test_remember_accepts_repeated_texts_form_field(monkeypatch):
    remember_api_module = importlib.import_module("cognee.api.v1.remember")
    remember_router_module = importlib.import_module(
        "cognee.api.v1.remember.routers.get_remember_router"
    )
    usage_logger_module = importlib.import_module("cognee.shared.usage_logger")

    captured = {}

    async def fake_remember(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return SimpleNamespace(to_dict=lambda: {"status": "ok"})

    monkeypatch.setattr(remember_api_module, "remember", fake_remember)
    monkeypatch.setattr(remember_router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        usage_logger_module,
        "get_cache_config",
        lambda: SimpleNamespace(usage_logging=False),
    )

    endpoint = _get_route_endpoint(remember_router_module.get_remember_router(), "", "POST")
    response = await endpoint(
        data=None,
        text=None,
        texts=["alpha", " ", "beta"],
        datasetName="memory",
        datasetId=None,
        node_set=["topic"],
        run_in_background=False,
        custom_prompt="",
        chunks_per_batch=10,
        user=SimpleNamespace(id=str(uuid4())),
    )

    assert response == {"status": "ok"}
    assert captured["payload"] == ["alpha", "beta"]
    assert captured["kwargs"]["node_set"] == ["topic"]


@pytest.mark.asyncio
async def test_remember_requires_input_payload(monkeypatch):
    remember_router_module = importlib.import_module(
        "cognee.api.v1.remember.routers.get_remember_router"
    )
    usage_logger_module = importlib.import_module("cognee.shared.usage_logger")

    monkeypatch.setattr(remember_router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        usage_logger_module,
        "get_cache_config",
        lambda: SimpleNamespace(usage_logging=False),
    )

    endpoint = _get_route_endpoint(remember_router_module.get_remember_router(), "", "POST")

    with pytest.raises(HTTPException, match="At least one of data, text, or texts must be provided."):
        await endpoint(
            data=None,
            text=None,
            texts=None,
            datasetName="memory",
            datasetId=None,
            node_set=[""],
            run_in_background=False,
            custom_prompt="",
            chunks_per_batch=10,
            user=SimpleNamespace(id=str(uuid4())),
        )
