from types import SimpleNamespace
from uuid import uuid4
import importlib

import pytest

from cognee.modules.users.models import DatasetDatabase


class _FakeSession:
    def __init__(self, record):
        self.record = record
        self.committed = False
        self.refreshed = False

    async def get(self, model, dataset_id):
        assert model is DatasetDatabase
        assert dataset_id == self.record.dataset_id
        return self.record

    def add(self, record):
        assert record is self.record

    async def commit(self):
        self.committed = True

    async def refresh(self, record):
        assert record is self.record
        self.refreshed = True


class _FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, session):
        self.session = session

    def get_async_session(self):
        return _FakeSessionContext(self.session)


@pytest.mark.asyncio
async def test_refresh_unsupported_dataset_database_rewrites_handlers(monkeypatch):
    database_utils = importlib.import_module(
        "cognee.infrastructure.databases.utils.get_or_create_dataset_database"
    )
    dataset_id = uuid4()
    owner_id = uuid4()
    dataset_database = DatasetDatabase(
        dataset_id=dataset_id,
        owner_id=owner_id,
        vector_database_name="legacy-vector",
        graph_database_name="legacy-graph",
        vector_database_provider="falkor",
        graph_database_provider="falkor",
        graph_dataset_database_handler="falkor_graph_local",
        vector_dataset_database_handler="falkor_vector_local",
        vector_database_url="graph-db",
        graph_database_url="graph-db",
        vector_database_key=None,
        graph_database_key=None,
        graph_database_connection_info={},
        vector_database_connection_info={},
    )
    user = SimpleNamespace(id=owner_id)
    fake_session = _FakeSession(dataset_database)

    async def fake_get_graph_db_info(resolved_dataset_id, resolved_user):
        assert resolved_dataset_id == dataset_id
        assert resolved_user is user
        return {
            "graph_database_name": f"{resolved_dataset_id}.pkl",
            "graph_database_url": None,
            "graph_database_provider": "kuzu",
            "graph_database_key": None,
            "graph_dataset_database_handler": "kuzu",
            "graph_database_connection_info": {
                "graph_database_username": "",
                "graph_database_password": "",
            },
        }

    async def fake_get_vector_db_info(resolved_dataset_id, resolved_user):
        assert resolved_dataset_id == dataset_id
        assert resolved_user is user
        return {
            "vector_database_provider": "lancedb",
            "vector_database_url": f"/tmp/{resolved_dataset_id}.lance.db",
            "vector_database_key": None,
            "vector_database_name": f"{resolved_dataset_id}.lance.db",
            "vector_dataset_database_handler": "lancedb",
        }

    monkeypatch.setattr(database_utils, "get_relational_engine", lambda: _FakeEngine(fake_session))
    monkeypatch.setattr(database_utils, "_get_graph_db_info", fake_get_graph_db_info)
    monkeypatch.setattr(database_utils, "_get_vector_db_info", fake_get_vector_db_info)

    refreshed = await database_utils._refresh_unsupported_dataset_database(
        dataset_database, dataset_id, user
    )

    assert refreshed is dataset_database
    assert refreshed.graph_database_provider == "kuzu"
    assert refreshed.graph_dataset_database_handler == "kuzu"
    assert refreshed.vector_database_provider == "lancedb"
    assert refreshed.vector_dataset_database_handler == "lancedb"
    assert fake_session.committed is True
    assert fake_session.refreshed is True
