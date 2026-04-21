"""Integration tests for PTY execution in LocalExecutor."""

import time

import pytest

from agiwo.tool.builtin.bash_tool.local_executor import LocalExecutor


class TestLocalExecutorPtyExecution:
    """Integration tests for PTY command execution."""

    @pytest.mark.asyncio
    async def test_pty_simple_command(self, tmp_path):
        """Test simple echo command with PTY."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "echo 'hello world'",
            use_pty=True,
        )

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.stderr == ""

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_multiple_lines(self, tmp_path):
        """Test command with multiple output lines."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "for i in 1 2 3; do echo line $i; done",
            use_pty=True,
        )

        assert result.exit_code == 0
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout
        assert "line 3" in result.stdout

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_with_timeout(self, tmp_path):
        """Test PTY command with timeout."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "echo 'timed test'",
            use_pty=True,
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert "timed test" in result.stdout

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_with_stdin(self, tmp_path):
        """Test PTY command with stdin input."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "head -1",
            use_pty=True,
            stdin="test input\n",
        )

        assert result.exit_code == 0
        assert "test input" in result.stdout

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_no_busy_wait(self, tmp_path):
        """Test that PTY execution doesn't busy-wait (regression test)."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        start = time.time()
        result = await executor.execute_command(
            "sleep 0.2 && echo 'done'",
            use_pty=True,
        )
        elapsed = time.time() - start

        assert result.exit_code == 0
        assert "done" in result.stdout
        # Should take ~0.2s, not significantly more (which would indicate busy-wait)
        assert 0.15 < elapsed < 0.5, f"Execution took {elapsed:.3f}s, expected ~0.2s"

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_command_error(self, tmp_path):
        """Test PTY command that exits with error."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "exit 42",
            use_pty=True,
        )

        assert result.exit_code == 42
        assert result.stderr == ""

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_custom_tty_size(self, tmp_path):
        """Test PTY with custom terminal size."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        result = await executor.execute_command(
            "echo 'size test'",
            use_pty=True,
            pty_cols=160,
            pty_rows=50,
        )

        assert result.exit_code == 0
        assert "size test" in result.stdout

        await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pty_timeout_exceeded(self, tmp_path):
        """Test PTY command that exceeds timeout."""
        executor = LocalExecutor(workspace_dir=str(tmp_path))

        with pytest.raises(TimeoutError, match="Command timed out"):
            await executor.execute_command(
                "sleep 10",
                use_pty=True,
                timeout=0.1,
            )

        await executor.cleanup()
