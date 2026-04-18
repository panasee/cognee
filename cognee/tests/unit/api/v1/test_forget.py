from types import SimpleNamespace
from uuid import uuid4
import importlib

import pytest

from cognee.api.v1.forget.forget import _resolve_dataset_id


@pytest.mark.asyncio
async def test_resolve_dataset_id_accepts_uuid_string(monkeypatch):
    dataset_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    get_authorized_dataset_module = importlib.import_module(
        "cognee.modules.data.methods.get_authorized_dataset"
    )
    data_methods_module = importlib.import_module("cognee.modules.data.methods")

    async def fake_get_authorized_dataset(resolved_user, resolved_dataset_id, permission):
        assert resolved_user is user
        assert resolved_dataset_id == dataset_id
        assert permission == "delete"
        return SimpleNamespace(id=dataset_id)

    async def fake_get_authorized_dataset_by_name(*args, **kwargs):
        raise AssertionError("UUID-like dataset refs should resolve by id before name lookup")

    monkeypatch.setattr(
        get_authorized_dataset_module,
        "get_authorized_dataset",
        fake_get_authorized_dataset,
    )
    monkeypatch.setattr(data_methods_module, "get_authorized_dataset_by_name", fake_get_authorized_dataset_by_name)

    resolved = await _resolve_dataset_id(str(dataset_id), user)

    assert resolved == dataset_id


@pytest.mark.asyncio
async def test_resolve_dataset_id_falls_back_to_name_lookup(monkeypatch):
    dataset_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset_name = "memory"
    get_authorized_dataset_module = importlib.import_module(
        "cognee.modules.data.methods.get_authorized_dataset"
    )
    data_methods_module = importlib.import_module("cognee.modules.data.methods")

    async def fake_get_authorized_dataset(resolved_user, resolved_dataset_id, permission):
        assert resolved_user is user
        assert resolved_dataset_id == dataset_id
        assert permission == "delete"
        return None

    async def fake_get_authorized_dataset_by_name(resolved_name, resolved_user, permission):
        assert resolved_name == dataset_name
        assert resolved_user is user
        assert permission == "delete"
        return SimpleNamespace(id=dataset_id)

    monkeypatch.setattr(
        get_authorized_dataset_module,
        "get_authorized_dataset",
        fake_get_authorized_dataset,
    )
    monkeypatch.setattr(data_methods_module, "get_authorized_dataset_by_name", fake_get_authorized_dataset_by_name)

    resolved = await _resolve_dataset_id(dataset_name, user)

    assert resolved == dataset_id
