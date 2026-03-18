"""Tests for channel-specific typed exceptions."""

import pytest

from server.channels.exceptions import (
    BaseAgentNotFoundError,
    ChannelError,
    DefaultAgentNameNotFoundError,
    PreviousTaskRunningError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_channel_error(self) -> None:
        assert issubclass(PreviousTaskRunningError, ChannelError)
        assert issubclass(BaseAgentNotFoundError, ChannelError)
        assert issubclass(DefaultAgentNameNotFoundError, ChannelError)

    def test_channel_error_inherits_from_runtime_error(self) -> None:
        assert issubclass(ChannelError, RuntimeError)


class TestPreviousTaskRunningError:
    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PreviousTaskRunningError):
            raise PreviousTaskRunningError()

    def test_caught_by_channel_error(self) -> None:
        with pytest.raises(ChannelError):
            raise PreviousTaskRunningError()

    def test_caught_by_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise PreviousTaskRunningError()


class TestBaseAgentNotFoundError:
    def test_stores_agent_name(self) -> None:
        err = BaseAgentNotFoundError("my-agent")
        assert err.agent_name == "my-agent"

    def test_message_contains_agent_name(self) -> None:
        err = BaseAgentNotFoundError("my-agent")
        assert "my-agent" in str(err)

    def test_isinstance_check(self) -> None:
        err = BaseAgentNotFoundError("x")
        assert isinstance(err, ChannelError)
        assert isinstance(err, RuntimeError)


class TestDefaultAgentNameNotFoundError:
    def test_stores_agent_name(self) -> None:
        err = DefaultAgentNameNotFoundError("default-bot")
        assert err.agent_name == "default-bot"

    def test_message_contains_agent_name(self) -> None:
        err = DefaultAgentNameNotFoundError("default-bot")
        assert "default-bot" in str(err)

    def test_isinstance_check(self) -> None:
        err = DefaultAgentNameNotFoundError("x")
        assert isinstance(err, ChannelError)
        assert isinstance(err, RuntimeError)
