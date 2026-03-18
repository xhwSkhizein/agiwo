"""MongoDB implementation of AgentStateStorage."""

from collections.abc import Collection

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    normalize_statuses,
)
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.codec import (
    deserialize_user_input_for_store,
    deserialize_wake_condition_for_store,
    serialize_user_input_for_store,
    serialize_wake_condition_for_store,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.storage_support.mongo_runtime import (
    MongoCollectionRuntime,
    MongoIndexSpec,
)

logger = get_logger(__name__)


class MongoAgentStateStorage(AgentStateStorage):
    """MongoDB-backed agent state storage."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agiwo",
    ) -> None:
        self._uri = uri
        self._db_name = db_name
        self._runtime = MongoCollectionRuntime(
            uri=uri,
            db_name=db_name,
            logger=logger,
            connect_event="mongo_agent_state_storage_connected",
            disconnect_event="mongo_agent_state_storage_disconnected",
        )
        self._states_collection = None
        self._events_collection = None

    async def _ensure_collections(self) -> None:
        if self._states_collection is not None:
            return

        self._states_collection = await self._runtime.ensure_collection(
            "agent_states",
            indexes=[
                MongoIndexSpec("id", unique=True),
                MongoIndexSpec("session_id"),
                MongoIndexSpec("parent_id"),
                MongoIndexSpec("status"),
                MongoIndexSpec([("session_id", 1), ("status", 1)]),
                MongoIndexSpec("updated_at"),
            ],
        )
        self._events_collection = await self._runtime.ensure_collection(
            "pending_events",
            indexes=[
                MongoIndexSpec("id", unique=True),
                MongoIndexSpec([("target_agent_id", 1), ("session_id", 1)]),
                MongoIndexSpec("created_at"),
            ],
        )

    def _state_to_doc(self, state: AgentState) -> dict:
        wake_condition_data = serialize_wake_condition_for_store(state.wake_condition)
        return {
            "id": state.id,
            "session_id": state.session_id,
            "parent_id": state.parent_id,
            "status": state.status.value,
            "task": serialize_user_input_for_store(state.task),
            "pending_input": (
                serialize_user_input_for_store(state.pending_input)
                if state.pending_input is not None
                else None
            ),
            "config_overrides": state.config_overrides,
            "wake_condition": wake_condition_data,
            "result_summary": state.result_summary,
            "signal_propagated": state.signal_propagated,
            "is_persistent": state.is_persistent,
            "depth": state.depth,
            "wake_count": state.wake_count,
            "agent_config_id": state.agent_config_id,
            "explain": state.explain,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    def _doc_to_state(self, doc: dict) -> AgentState:
        wake_condition = None
        if doc.get("wake_condition"):
            wake_condition = deserialize_wake_condition_for_store(doc["wake_condition"])

        return AgentState(
            id=doc["id"],
            session_id=doc["session_id"],
            status=AgentStateStatus(doc["status"]),
            task=deserialize_user_input_for_store(doc["task"]),
            parent_id=doc.get("parent_id"),
            pending_input=(
                deserialize_user_input_for_store(doc["pending_input"])
                if doc.get("pending_input")
                else None
            ),
            config_overrides=doc.get("config_overrides", {}),
            wake_condition=wake_condition,
            result_summary=doc.get("result_summary"),
            signal_propagated=doc.get("signal_propagated", False),
            agent_config_id=doc.get("agent_config_id"),
            is_persistent=doc.get("is_persistent", False),
            depth=doc.get("depth", 0),
            wake_count=doc.get("wake_count", 0),
            explain=doc.get("explain"),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    def _event_to_doc(self, event: PendingEvent) -> dict:
        return {
            "id": event.id,
            "target_agent_id": event.target_agent_id,
            "session_id": event.session_id,
            "event_type": event.event_type.value,
            "payload": event.payload,
            "source_agent_id": event.source_agent_id,
            "created_at": event.created_at,
        }

    def _doc_to_event(self, doc: dict) -> PendingEvent:
        return PendingEvent(
            id=doc["id"],
            target_agent_id=doc["target_agent_id"],
            session_id=doc["session_id"],
            event_type=SchedulerEventType(doc["event_type"]),
            payload=doc.get("payload", {}),
            source_agent_id=doc.get("source_agent_id"),
            created_at=doc["created_at"],
        )

    async def save_state(self, state: AgentState) -> None:
        await self._ensure_collections()
        doc = self._state_to_doc(state)
        await self._states_collection.replace_one(
            {"id": state.id},
            doc,
            upsert=True,
        )

    async def get_state(self, state_id: str) -> AgentState | None:
        await self._ensure_collections()
        doc = await self._states_collection.find_one({"id": state_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        return self._doc_to_state(doc)

    async def list_states(
        self,
        *,
        statuses: Collection[AgentStateStatus] | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        signal_propagated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        await self._ensure_collections()
        query: dict = {}

        status_filter = normalize_statuses(statuses)
        if status_filter:
            query["status"] = {"$in": [status.value for status in status_filter]}
        if parent_id is not None:
            query["parent_id"] = parent_id
        if session_id is not None:
            query["session_id"] = session_id
        if signal_propagated is not None:
            query["signal_propagated"] = signal_propagated

        cursor = (
            self._states_collection.find(query)
            .sort("updated_at", -1)
            .skip(offset)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        states = []
        for doc in docs:
            doc.pop("_id", None)
            states.append(self._doc_to_state(doc))
        return states

    async def save_event(self, event: PendingEvent) -> None:
        await self._ensure_collections()
        doc = self._event_to_doc(event)
        await self._events_collection.replace_one(
            {"id": event.id},
            doc,
            upsert=True,
        )

    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PendingEvent]:
        await self._ensure_collections()
        query: dict = {}
        if target_agent_id is not None:
            query["target_agent_id"] = target_agent_id
        if session_id is not None:
            query["session_id"] = session_id

        cursor = self._events_collection.find(query).sort("created_at", 1)
        docs = await cursor.to_list(length=None)
        events = []
        for doc in docs:
            doc.pop("_id", None)
            events.append(self._doc_to_event(doc))
        return events

    async def delete_events(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        await self._ensure_collections()
        await self._events_collection.delete_many({"id": {"$in": event_ids}})

    async def close(self) -> None:
        await self._runtime.disconnect()
        self._states_collection = None
        self._events_collection = None
