from types import SimpleNamespace
from unittest.mock import AsyncMock
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
async def test_update_dataset_graph_feedback_weights_applies_streaming_updates(monkeypatch):
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )

    dataset_id = uuid4()
    owner_id = uuid4()
    graph_engine = SimpleNamespace(
        get_node_feedback_weights=AsyncMock(return_value={"node-1": 0.2, "node-2": 0.5}),
        get_edge_feedback_weights=AsyncMock(return_value={"edge-1": 0.8}),
        set_node_feedback_weights=AsyncMock(return_value={"node-1": True, "node-2": True}),
        set_edge_feedback_weights=AsyncMock(return_value={"edge-1": True}),
    )

    monkeypatch.setattr(
        datasets_router_module,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=dataset_id, owner_id=owner_id)),
    )
    monkeypatch.setattr(
        datasets_router_module,
        "set_database_global_context_variables",
        AsyncMock(),
    )
    monkeypatch.setattr(
        datasets_router_module,
        "get_graph_engine",
        AsyncMock(return_value=graph_engine),
    )

    payload = datasets_router_module.GraphFeedbackWeightsUpdatePayloadDTO(
        node_ids=["node-1", "node-1", "node-2"],
        edge_ids=["edge-1"],
        target=1.0,
        alpha=0.5,
    )
    endpoint = _get_route_endpoint(
        datasets_router_module.get_datasets_router(),
        "/{dataset_id}/graph/feedback-weights",
        "POST",
    )

    response = await endpoint(
        dataset_id=dataset_id,
        payload=payload,
        user=SimpleNamespace(id=str(uuid4())),
    )

    assert response == {
        "processed": 3,
        "applied": 3,
        "skipped": 0,
        "target": 1.0,
        "alpha": 0.5,
        "node_count": 2,
        "edge_count": 1,
    }
    graph_engine.set_node_feedback_weights.assert_awaited_once_with(
        {"node-1": 0.6, "node-2": 0.75}
    )
    graph_engine.set_edge_feedback_weights.assert_awaited_once_with({"edge-1": 0.9})


@pytest.mark.asyncio
async def test_update_dataset_graph_feedback_weights_validates_alpha(monkeypatch):
    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )

    dataset_id = uuid4()
    monkeypatch.setattr(
        datasets_router_module,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=dataset_id, owner_id=uuid4())),
    )

    payload = datasets_router_module.GraphFeedbackWeightsUpdatePayloadDTO(
        node_ids=["node-1"],
        target=0.6,
        alpha=0,
    )
    endpoint = _get_route_endpoint(
        datasets_router_module.get_datasets_router(),
        "/{dataset_id}/graph/feedback-weights",
        "POST",
    )

    with pytest.raises(HTTPException, match="alpha must be in range \\(0, 1]"):
        await endpoint(
            dataset_id=dataset_id,
            payload=payload,
            user=SimpleNamespace(id=str(uuid4())),
        )
