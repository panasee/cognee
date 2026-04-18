from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest


@pytest.mark.asyncio
async def test_get_formatted_graph_data_includes_edge_object_details(monkeypatch):
    graph_data_module = importlib.import_module(
        "cognee.modules.graph.methods.get_formatted_graph_data"
    )

    dataset_id = uuid4()
    owner_id = uuid4()
    graph_engine = SimpleNamespace(
        get_graph_data=AsyncMock(
            return_value=(
                [
                    (
                        "node-1",
                        {
                            "type": "Entity",
                            "name": "Node One",
                            "confidence": 0.8,
                        },
                    )
                ],
                [
                    (
                        "node-1",
                        "node-2",
                        "CONNECTED_TO",
                        {
                            "edge_object_id": "edge-1",
                            "feedback_weight": 0.7,
                        },
                    )
                ],
            )
        )
    )

    monkeypatch.setattr(
        graph_data_module,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=dataset_id, owner_id=owner_id)),
    )
    monkeypatch.setattr(
        graph_data_module,
        "set_database_global_context_variables",
        AsyncMock(),
    )
    monkeypatch.setattr(
        graph_data_module,
        "get_graph_engine",
        AsyncMock(return_value=graph_engine),
    )

    result = await graph_data_module.get_formatted_graph_data(
        dataset_id,
        SimpleNamespace(id=uuid4()),
    )

    assert result["nodes"] == [
        {
            "id": "node-1",
            "label": "Node One",
            "type": "Entity",
            "properties": {"confidence": 0.8},
        }
    ]
    assert result["edges"] == [
        {
            "source": "node-1",
            "target": "node-2",
            "label": "CONNECTED_TO",
            "edge_object_id": "edge-1",
            "properties": {"edge_object_id": "edge-1", "feedback_weight": 0.7},
        }
    ]
