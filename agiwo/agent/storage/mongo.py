"""
MongoDB implementation of RunStepStorage.
"""

from motor.motor_asyncio import AsyncIOMotorClient

from dataclasses import asdict

from agiwo.agent.schema import Run, StepRecord, MessageRole, RunMetrics, StepMetrics
from agiwo.agent.storage.base import RunStepStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def filter_none_values(data: dict) -> dict:
    """
    Recursively filter out None values from a dictionary.
    """
    filtered = {}
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            filtered[key] = filter_none_values(value)
        elif isinstance(value, list):
            filtered[key] = [
                filter_none_values(item) if isinstance(item, dict) else item
                for item in value
                if item is not None
            ]
        else:
            filtered[key] = value

    return filtered


class MongoRunStepStorage(RunStepStorage):
    """
    MongoDB implementation of RunStepStorage.

    Collections:
    - runs: Stores Run documents
    - steps: Stores Step documents
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agiwo",
    ):
        self.uri = uri
        self.db_name = db_name
        self.client: AsyncIOMotorClient | None = None
        self.db = None
        self.runs_collection = None
        self.steps_collection = None
        self.counters_collection = None

    async def _ensure_connection(self):
        """Ensure database connection is established."""
        if self.client is None:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]
            self.runs_collection = self.db["runs"]
            self.steps_collection = self.db["steps"]
            self.counters_collection = self.db["counters"]

            # Create indexes for runs
            await self.runs_collection.create_index("id", unique=True)
            await self.runs_collection.create_index("agent_id")
            await self.runs_collection.create_index("user_id")
            await self.runs_collection.create_index("session_id")
            await self.runs_collection.create_index("created_at")

            # Create indexes for steps
            await self.steps_collection.create_index("id", unique=True)
            await self.steps_collection.create_index(
                [("session_id", 1), ("sequence", 1)], unique=True
            )
            # Compound indexes matching common query patterns
            await self.steps_collection.create_index(
                [("session_id", 1), ("run_id", 1), ("sequence", 1)]
            )
            await self.steps_collection.create_index(
                [("session_id", 1), ("run_id", 1), ("node_id", 1), ("sequence", 1)]
            )
            await self.steps_collection.create_index(
                [("session_id", 1), ("tool_call_id", 1)]
            )
            await self.steps_collection.create_index("created_at")

            # Create index for counters collection
            await self.counters_collection.create_index("session_id", unique=True)

            logger.info("mongodb_connected", uri=self.uri, db_name=self.db_name)

    def _serialize_dataclass(self, obj) -> dict:
        data = asdict(obj)
        if "role" in data and hasattr(data["role"], "value"):
            data["role"] = data["role"].value
        return filter_none_values(data)

    def _deserialize_run(self, doc: dict) -> Run:
        doc.pop("_id", None)
        metrics = doc.get("metrics")
        if isinstance(metrics, dict):
            doc["metrics"] = RunMetrics(**metrics)
        return Run(**doc)

    def _deserialize_step(self, doc: dict) -> StepRecord:
        doc.pop("_id", None)
        if "role" in doc and isinstance(doc["role"], str):
            doc["role"] = MessageRole(doc["role"])
        metrics = doc.get("metrics")
        if isinstance(metrics, dict):
            doc["metrics"] = StepMetrics(**metrics)
        return StepRecord(**doc)

    async def initialize(self) -> None:
        """Initialize MongoDB connection."""
        await self._ensure_connection()

    async def close(self) -> None:
        """Close MongoDB connection."""
        await self.disconnect()

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.runs_collection = None
            self.steps_collection = None
            self.counters_collection = None
            logger.info("mongodb_disconnected")

    async def save_run(self, run: Run) -> None:
        """Save or update a run."""
        await self._ensure_connection()

        try:
            run_data = self._serialize_dataclass(run)

            await self.runs_collection.update_one(
                {"id": run.id}, {"$set": run_data}, upsert=True
            )
        except Exception as e:
            logger.error("save_run_failed", error=str(e), run_id=run.id)
            raise

    async def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        await self._ensure_connection()

        try:
            doc = await self.runs_collection.find_one({"id": run_id})
            if doc:
                return self._deserialize_run(doc)
            return None
        except Exception as e:
            logger.error("get_run_failed", error=str(e), run_id=run_id)
            raise

    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        """List runs with filtering and pagination."""
        await self._ensure_connection()

        try:
            query = {"agent_id": {"$exists": True}}
            if user_id:
                query["user_id"] = user_id
            if session_id:
                query["session_id"] = session_id

            cursor = (
                self.runs_collection.find(query)
                .sort("created_at", -1)
                .skip(offset)
                .limit(limit)
            )

            runs = []
            async for doc in cursor:
                runs.append(self._deserialize_run(doc))
            return runs
        except Exception as e:
            logger.error("list_runs_failed", error=str(e))
            raise

    async def delete_run(self, run_id: str) -> None:
        """Delete a run and its associated steps."""
        await self._ensure_connection()

        try:
            run = await self.get_run(run_id)
            await self.runs_collection.delete_one({"id": run_id})

            if run and run.session_id:
                await self.steps_collection.delete_many({"session_id": run.session_id})

        except Exception as e:
            logger.error("delete_run_failed", error=str(e), run_id=run_id)
            raise

    # --- Step Operations ---

    async def save_step(self, step: StepRecord) -> None:
        """Save or update a step."""
        await self._ensure_connection()

        step_data = self._serialize_dataclass(step)

        try:
            # Use upsert: update if step.id exists, insert otherwise
            await self.steps_collection.update_one(
                {"id": step.id}, {"$set": step_data}, upsert=True
            )
        except Exception as e:
            logger.error(
                "save_step_failed",
                error=str(e),
                step_id=step.id,
                session_id=step.session_id,
                sequence=step.sequence,
            )
            raise

    async def save_steps_batch(self, steps: list[StepRecord]) -> None:
        """Batch save steps."""
        if not steps:
            return

        await self._ensure_connection()

        try:
            operations = []
            for step in steps:
                step_data = self._serialize_dataclass(step)

                from pymongo import UpdateOne

                operations.append(
                    UpdateOne({"id": step.id}, {"$set": step_data}, upsert=True)
                )

            if operations:
                await self.steps_collection.bulk_write(operations)
        except Exception as e:
            logger.error("save_steps_batch_failed", error=str(e), count=len(steps))
            raise

    async def get_steps(
        self,
        session_id: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepRecord]:
        """Get steps for a session with optional filtering."""
        await self._ensure_connection()

        try:
            query = {"session_id": session_id}

            if start_seq is not None or end_seq is not None:
                query["sequence"] = {}
                if start_seq is not None:
                    query["sequence"]["$gte"] = start_seq
                if end_seq is not None:
                    query["sequence"]["$lte"] = end_seq

            if run_id is not None:
                query["run_id"] = run_id
            if agent_id is not None:
                query["agent_id"] = agent_id

            cursor = self.steps_collection.find(query).sort("sequence", 1).limit(limit)

            steps = []
            async for doc in cursor:
                steps.append(self._deserialize_step(doc))
            return steps
        except Exception as e:
            logger.error("get_steps_failed", error=str(e), session_id=session_id)
            raise

    async def get_last_step(self, session_id: str) -> StepRecord | None:
        """Get the last step of a session."""
        await self._ensure_connection()

        try:
            cursor = (
                self.steps_collection.find({"session_id": session_id})
                .sort("sequence", -1)
                .limit(1)
            )

            async for doc in cursor:
                return self._deserialize_step(doc)
            return None
        except Exception as e:
            logger.error("get_last_step_failed", error=str(e), session_id=session_id)
            raise

    async def delete_steps(self, session_id: str, start_seq: int) -> int:
        """Delete steps from a sequence number onwards."""
        await self._ensure_connection()

        try:
            result = await self.steps_collection.delete_many(
                {"session_id": session_id, "sequence": {"$gte": start_seq}}
            )
            return result.deleted_count
        except Exception as e:
            logger.error("delete_steps_failed", error=str(e), session_id=session_id)
            raise

    async def get_step_count(self, session_id: str) -> int:
        """Get total number of steps for a session."""
        await self._ensure_connection()

        try:
            return await self.steps_collection.count_documents(
                {"session_id": session_id}
            )
        except Exception as e:
            logger.error("get_step_count_failed", error=str(e), session_id=session_id)
            raise

    async def get_max_sequence(self, session_id: str) -> int:
        """
        Get the maximum sequence number in the session.

        Returns:
            Maximum sequence number, or 0 if no steps exist
        """
        await self._ensure_connection()

        try:
            cursor = (
                self.steps_collection.find({"session_id": session_id})
                .sort("sequence", -1)
                .limit(1)
            )

            async for doc in cursor:
                return doc.get("sequence", 0)
            return 0
        except Exception as e:
            logger.error("get_max_sequence_failed", error=str(e), session_id=session_id)
            raise

    async def allocate_sequence(self, session_id: str) -> int:
        """
        Atomically allocate next sequence number using MongoDB's findAndModify.
        Thread-safe and concurrent-safe operation.

        Args:
            session_id: Session ID

        Returns:
            Next sequence number (starting from 1)
        """
        await self._ensure_connection()

        try:
            # First, try to increment existing counter
            result = await self.counters_collection.find_one_and_update(
                {"session_id": session_id},
                {"$inc": {"sequence": 1}},
                return_document=True,
            )

            if result:
                return result["sequence"]

            # Counter doesn't exist, initialize from steps collection
            max_seq = await self.get_max_sequence(session_id)

            # Try to insert initial counter value
            # Use insert to avoid race condition (will fail if another thread inserted first)
            try:
                await self.counters_collection.insert_one(
                    {
                        "session_id": session_id,
                        "sequence": max_seq + 1,
                    }
                )
                return max_seq + 1
            except Exception:
                # Another thread already initialized, retry increment
                result = await self.counters_collection.find_one_and_update(
                    {"session_id": session_id},
                    {"$inc": {"sequence": 1}},
                    return_document=True,
                )
                if result:
                    return result["sequence"]
                # Should never reach here
                raise RuntimeError(
                    f"Failed to allocate sequence for session {session_id}"
                )

        except Exception as e:
            logger.error(
                "allocate_sequence_failed", error=str(e), session_id=session_id
            )
            raise

    async def get_step_by_tool_call_id(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> StepRecord | None:
        """Get a Tool Step by tool_call_id."""
        await self._ensure_connection()

        try:
            doc = await self.steps_collection.find_one(
                {
                    "session_id": session_id,
                    "tool_call_id": tool_call_id,
                }
            )
            if doc:
                return self._deserialize_step(doc)
            return None
        except Exception as e:
            logger.error(
                "get_step_by_tool_call_id_failed",
                error=str(e),
                tool_call_id=tool_call_id,
            )
            raise


__all__ = ["MongoRunStepStorage"]
