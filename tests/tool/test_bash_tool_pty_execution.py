"""Integration tests for PTY execution in LocalExecutor."""

import time

import pytest

from agiwo.tool.builtin.bash_tool.local_executor import LocalExecutor


@pytest.fixture
async def executor(tmp_path):
    """Fixture to provide and cleanup a LocalExecutor."""
    executor = LocalExecutor(workspace_dir=str(tmp_path))
    yield executor
    await executor.cleanup()


class TestLocalExecutorPtyExecution:
    """Integration tests for PTY command execution."""

    @pytest.mark.asyncio
    async def test_pty_simple_command(self, executor):
        """Test simple echo command with PTY."""
        result = await executor.execute_command(
            "echo 'hello world'",
            use_pty=True,
        )

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_pty_multiple_lines(self, executor):
        """Test command with multiple output lines."""
        result = await executor.execute_command(
            "for i in 1 2 3; do echo line $i; done",
            use_pty=True,
        )

        assert result.exit_code == 0
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout
        assert "line 3" in result.stdout

    @pytest.mark.asyncio
    async def test_pty_with_timeout(self, executor):
        """Test PTY command with timeout."""
        result = await executor.execute_command(
            "sleep 0.01 && echo 'timed test'",
            use_pty=True,
            timeout=1.0,
        )

        assert result.exit_code == 0
        assert "timed test" in result.stdout

    @pytest.mark.asyncio
    async def test_pty_no_busy_wait(self, executor):
        """Test that PTY execution doesn't busy-wait (regression test)."""
        start = time.time()
        result = await executor.execute_command(
            "sleep 0.2 && echo 'done'",
            use_pty=True,
        )
        elapsed = time.time() - start

        assert result.exit_code == 0
        assert "done" in result.stdout
        # Should take ~0.2s, not significantly more (which would indicate busy-wait)
        assert 0.1 < elapsed < 1.0, f"Execution took {elapsed:.3f}s, expected ~0.2s"

    @pytest.mark.asyncio
    async def test_pty_command_error(self, executor):
        """Test PTY command that exits with error."""
        result = await executor.execute_command(
            "exit 42",
            use_pty=True,
        )

        assert result.exit_code == 42
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_pty_custom_tty_size(self, executor):
        """Test PTY with custom terminal size."""
        result = await executor.execute_command(
            "echo 'size test'",
            use_pty=True,
            pty_cols=80,
            pty_rows=24,
        )

        assert result.exit_code == 0
        assert "size test" in result.stdout

    @pytest.mark.asyncio
    async def test_pty_timeout_exceeded(self, executor):
        """Test PTY command that exceeds timeout."""
        with pytest.raises(TimeoutError):
            await executor.execute_command(
                "sleep 10",
                use_pty=True,
                timeout=0.1,
            )
