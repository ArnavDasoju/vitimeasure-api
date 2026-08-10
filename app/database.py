"""
Azure Cosmos DB client — lazy-initialised singleton.

Containers are created on first access if they don't exist.
"""

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError

from app.config import settings

_client: CosmosClient | None = None
_db = None
_containers: dict = {}

CONTAINERS = {
    "users":         "/email",
    "scans":         "/userId",
    "patches":       "/userId",
    "check_ins":     "/userId",
    "daily_stress":  "/userId",
    "treatments":    "/userId",
}


def _get_db():
    global _client, _db
    if _db is not None:
        return _db
    _client = CosmosClient.from_connection_string(settings.cosmos_db_connection_string)
    _db = _client.create_database_if_not_exists(settings.cosmos_db_database_name)
    return _db


def get_container(name: str):
    if name in _containers:
        return _containers[name]
    db = _get_db()
    pk = CONTAINERS.get(name, "/id")
    try:
        container = db.create_container_if_not_exists(id=name, partition_key=PartitionKey(path=pk))
    except CosmosResourceExistsError:
        container = db.get_container_client(name)
    _containers[name] = container
    return container
