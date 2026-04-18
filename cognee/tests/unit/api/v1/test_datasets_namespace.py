from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest

from cognee.api.v1.exceptions.exceptions import DocumentSubgraphNotFoundError


@pytest.mark.asyncio
async def test_delete_data_hard_mode_tolerates_missing_document_subgraph(monkeypatch):
    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    dataset_id = uuid4()
    data_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    data = SimpleNamespace(id=data_id, datasets=[SimpleNamespace(id=dataset_id)])

    delete_data_mock = AsyncMock()

    monkeypatch.setattr(
        datasets_module,
        "get_authorized_dataset",
        AsyncMock(return_value=dataset),
    )
    monkeypatch.setattr(
        datasets_module,
        "get_dataset_data",
        AsyncMock(side_effect=[[data], [data]]),
    )
    monkeypatch.setattr(
        datasets_module,
        "has_data_related_nodes",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        datasets_module,
        "legacy_delete",
        AsyncMock(side_effect=DocumentSubgraphNotFoundError("missing")),
    )
    monkeypatch.setattr(
        datasets_module,
        "set_database_global_context_variables",
        AsyncMock(),
    )
    monkeypatch.setattr(
        datasets_module,
        "delete_data_nodes_and_edges",
        AsyncMock(),
    )
    monkeypatch.setattr(
        datasets_module,
        "logger",
        SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        datasets_module,
        "has_dataset_data",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        datasets_module,
        "get_default_user",
        AsyncMock(return_value=user),
    )

    monkeypatch.setattr("cognee.modules.data.methods.delete_data", delete_data_mock)

    result = await datasets_module.datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
        mode="hard",
    )

    assert result == {"status": "success"}
    delete_data_mock.assert_awaited_once_with(data, dataset_id)


@pytest.mark.asyncio
async def test_delete_data_soft_mode_still_raises_missing_document_subgraph(monkeypatch):
    datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")

    dataset_id = uuid4()
    data_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    data = SimpleNamespace(id=data_id, datasets=[SimpleNamespace(id=dataset_id)])

    monkeypatch.setattr(
        datasets_module,
        "get_authorized_dataset",
        AsyncMock(return_value=dataset),
    )
    monkeypatch.setattr(
        datasets_module,
        "get_dataset_data",
        AsyncMock(return_value=[data]),
    )
    monkeypatch.setattr(
        datasets_module,
        "has_data_related_nodes",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        datasets_module,
        "legacy_delete",
        AsyncMock(side_effect=DocumentSubgraphNotFoundError("missing")),
    )
    monkeypatch.setattr(
        datasets_module,
        "set_database_global_context_variables",
        AsyncMock(),
    )

    with pytest.raises(DocumentSubgraphNotFoundError, match="missing"):
        await datasets_module.datasets.delete_data(
            dataset_id=dataset_id,
            data_id=data_id,
            user=user,
            mode="soft",
        )
