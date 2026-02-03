"""
ConsentStore - Authorization record storage.

Stores user consent records (allow/deny patterns) with expiration support.
"""

import asyncio
import fnmatch
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ConsentRecord(BaseModel):
    """User consent record"""

    user_id: str
    tool_name: str | None = None  # None means global pattern
    patterns: list[str]  # allow patterns
    deny_patterns: list[str] = []  # deny patterns
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ConsentStore(ABC):
    """Abstract base class for consent storage"""

    @abstractmethod
    async def check_consent(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Literal["allowed", "denied", None]:
        """
        Check consent for a tool call.

        Returns:
            "allowed" if matched allow pattern
            "denied" if matched deny pattern
            None if no match (requires consent)
        """
        pass

    @abstractmethod
    async def save_consent(
        self,
        user_id: str,
        tool_name: str | None,
        patterns: list[str],
        deny_patterns: list[str] = [],
        expires_at: datetime | None = None,
    ) -> None:
        """Save consent record"""
        pass

    def match_pattern(
        self,
        pattern: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:
        """
        Match a pattern against tool name and arguments.

        Pattern format: "{tool_name}({arg_pattern})"
        Example: "bash(npm run lint)", "file_read(~/.zshrc)"

        Supports glob patterns: *, ?, **
        """
        if not pattern.endswith(")"):
            return False

        open_paren = pattern.find("(")
        if open_paren == -1:
            return False

        pattern_tool_name = pattern[:open_paren]
        arg_pattern = pattern[open_paren + 1 : -1]

        # Match tool name
        if not fnmatch.fnmatch(tool_name, pattern_tool_name):
            return False

        # Serialize arguments for matching
        args_str = self._serialize_args(tool_args)

        # Use glob matching
        return fnmatch.fnmatch(args_str, arg_pattern)

    def _serialize_args(self, args: dict[str, Any]) -> str:
        """Serialize arguments to string for pattern matching"""
        # Exclude tool_call_id
        items = sorted([(k, v) for k, v in args.items() if k != "tool_call_id"])
        return " ".join(f"{k}={v}" for k, v in items)


class MongoConsentStore(ConsentStore):
    """
    MongoDB implementation of ConsentStore.

    Reuses MongoSessionStore's MongoDB connection.
    """

    def __init__(
        self,
        client: AsyncIOMotorClient | None,
        db_name: str,
    ):
        """
        Initialize MongoDB consent store.

        Args:
            client: MongoDB client (reused from MongoSessionStore, can be None)
            db_name: Database name
        """
        self.client = client
        self.db_name = db_name
        self.db: AsyncIOMotorDatabase[Any] | None = None
        self.consents_collection: AsyncIOMotorCollection[Any] | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Ensure collection is initialized"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if self.client is None:
                raise RuntimeError("MongoDB client is not initialized")

            self.db = self.client[self.db_name]
            self.consents_collection = self.db["consents"]

            # Create indexes
            await self.consents_collection.create_index(
                [("user_id", 1), ("tool_name", 1)]
            )
            await self.consents_collection.create_index(
                "expires_at", expireAfterSeconds=0
            )  # TTL index

            self._initialized = True
            logger.info("mongodb_consent_store_initialized", db_name=self.db_name)

    async def check_consent(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Literal["allowed", "denied", None]:
        """Check consent for a tool call"""
        await self._ensure_initialized()

        try:
            # Query consent records for this user and tool
            query = {"user_id": user_id}
            if tool_name:
                query["$or"] = [
                    {"tool_name": tool_name},
                    {"tool_name": None},  # Global patterns
                ]

            if self.consents_collection is None:
                raise RuntimeError("Consents collection not initialized")
            cursor = self.consents_collection.find(query)
            async for doc in cursor:
                record = ConsentRecord.model_validate(doc)

                # Check expiration
                if record.expires_at and record.expires_at < datetime.now():
                    continue

                # First check deny patterns
                for deny_pattern in record.deny_patterns:
                    if self.match_pattern(deny_pattern, tool_name, tool_args):
                        logger.debug(
                            "consent_denied_by_pattern",
                            user_id=user_id,
                            tool_name=tool_name,
                            pattern=deny_pattern,
                        )
                        return "denied"

                # Then check allow patterns
                for pattern in record.patterns:
                    if self.match_pattern(pattern, tool_name, tool_args):
                        logger.debug(
                            "consent_allowed_by_pattern",
                            user_id=user_id,
                            tool_name=tool_name,
                            pattern=pattern,
                        )
                        return "allowed"

            # No match found
            return None

        except Exception as e:
            logger.error(
                "check_consent_failed",
                user_id=user_id,
                tool_name=tool_name,
                error=str(e),
                exc_info=True,
            )
            # On error, return None (requires consent)
            return None

    async def save_consent(
        self,
        user_id: str,
        tool_name: str | None,
        patterns: list[str],
        deny_patterns: list[str] = [],
        expires_at: datetime | None = None,
    ) -> None:
        """Save consent record"""
        await self._ensure_initialized()

        try:
            now = datetime.now()

            # Find existing record or create new one
            query = {"user_id": user_id, "tool_name": tool_name}
            existing = await self.consents_collection.find_one(query)

            if existing:
                # Update existing record
                await self.consents_collection.update_one(
                    query,
                    {
                        "$set": {
                            "patterns": patterns,
                            "deny_patterns": deny_patterns,
                            "expires_at": expires_at.isoformat()
                            if expires_at
                            else None,
                            "updated_at": now.isoformat(),
                        }
                    },
                )
            else:
                # Create new record
                record = ConsentRecord(
                    user_id=user_id,
                    tool_name=tool_name,
                    patterns=patterns,
                    deny_patterns=deny_patterns,
                    expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                )
                await self.consents_collection.insert_one(
                    record.model_dump(mode="json", exclude_none=True)
                )

            logger.info(
                "consent_saved",
                user_id=user_id,
                tool_name=tool_name,
                pattern_count=len(patterns),
            )

        except Exception as e:
            logger.error(
                "save_consent_failed",
                user_id=user_id,
                tool_name=tool_name,
                error=str(e),
                exc_info=True,
            )
            raise


class InMemoryConsentStore(ConsentStore):
    """In-memory implementation for testing"""

    def __init__(self) -> None:
        self._records: dict[str, ConsentRecord] = {}  # key = f"{user_id}:{tool_name}"

    def _make_key(self, user_id: str, tool_name: str | None) -> str:
        """Generate storage key"""
        return f"{user_id}:{tool_name or 'global'}"

    async def check_consent(
        self,
        user_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Literal["allowed", "denied", None]:
        """Check consent for a tool call"""
        # Check tool-specific and global records
        keys_to_check = [
            self._make_key(user_id, tool_name),
            self._make_key(user_id, None),  # Global patterns
        ]

        for key in keys_to_check:
            record = self._records.get(key)
            if not record:
                continue

            # Check expiration
            if record.expires_at and record.expires_at < datetime.now():
                continue

            # First check deny patterns
            for deny_pattern in record.deny_patterns:
                if self.match_pattern(deny_pattern, tool_name, tool_args):
                    return "denied"

            # Then check allow patterns
            for pattern in record.patterns:
                if self.match_pattern(pattern, tool_name, tool_args):
                    return "allowed"

        return None

    async def save_consent(
        self,
        user_id: str,
        tool_name: str | None,
        patterns: list[str],
        deny_patterns: list[str] = [],
        expires_at: datetime | None = None,
    ) -> None:
        """Save consent record"""
        key = self._make_key(user_id, tool_name)
        now = datetime.now()

        record = ConsentRecord(
            user_id=user_id,
            tool_name=tool_name,
            patterns=patterns,
            deny_patterns=deny_patterns,
            expires_at=expires_at,
            created_at=now,
            updated_at=now,
        )

        self._records[key] = record


__all__ = [
    "ConsentStore",
    "MongoConsentStore",
    "InMemoryConsentStore",
    "ConsentRecord",
]
