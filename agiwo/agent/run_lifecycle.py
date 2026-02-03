import time

from agiwo.agent.schema import Run, RunStatus, RunOutput
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.side_effect_processor import SideEffectProcessor
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class RunLifecycle:
    """
    Context Manager for Run lifecycle management.

    Usage:
        async with RunLifecycle(agent_id, input_query, context) as run:
            result = await executor.execute(...)
            run_lifecycle.set_output(result)
            return result
    """

    def __init__(
        self,
        agent_id: str,
        input_query: str,
        context: ExecutionContext,
        side_effect_processor: SideEffectProcessor,
    ) -> None:
        self.agent_id = agent_id
        self.input_query = input_query
        self.context = context
        self.side_effect_processor = side_effect_processor
        self.run: Run | None = None
        self._output: RunOutput | None = None

    def _finalize_timing(self) -> None:
        self.run.metrics.end_at = time.time()
        self.run.metrics.duration_ms = (
            self.run.metrics.end_at - self.run.metrics.start_at
        ) * 1000

    def _sync_output_metrics(self) -> None:
        if not self._output or not self._output.metrics:
            return
        self.run.metrics.total_tokens = self._output.metrics.total_tokens
        self.run.metrics.input_tokens = self._output.metrics.input_tokens
        self.run.metrics.output_tokens = self._output.metrics.output_tokens
        self.run.metrics.tool_calls_count = self._output.metrics.tool_calls_count

    def set_output(self, output: RunOutput) -> None:
        """Set the execution output (success case)."""
        self._output = output

    async def _handle_failure(self, exc_val: BaseException) -> None:
        """Handle failure case for the run."""
        self.run.status = RunStatus.FAILED
        logger.error(
            "run_failed",
            run_id=self.run.id,
            agent_id=self.run.agent_id,
            error=str(exc_val),
            exc_info=True,
        )
        await self.side_effect_processor.emit_run_failed(self.run, exc_val)

    async def _handle_success(self) -> None:
        """Handle success case for the run."""
        self.run.status = RunStatus.COMPLETED
        self.run.response_content = self._output.response

        # Sync Metrics
        self._sync_output_metrics()

        await self.side_effect_processor.emit_run_completed(self.run, self._output)

    async def _handle_incomplete(self) -> None:
        """Handle unexpected empty output without exception."""
        self.run.status = RunStatus.FAILED
        msg = "Run exited without output or exception"
        logger.error("run_incomplete", run_id=self.run.id, error=msg)
        await self.side_effect_processor.emit_run_failed(self.run, Exception(msg))

    async def __aenter__(self) -> Run:
        """Start the Run."""
        self.run = Run(
            id=self.context.run_id,
            agent_id=self.context.agent_id,
            session_id=self.context.session_id,
            input_query=self.input_query,
            status=RunStatus.RUNNING,
            parent_run_id=self.context.parent_run_id,
        )
        self.run.metrics.start_at = time.time()

        await self.side_effect_processor.emit_run_started(self.run)
        return self.run

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize the Run."""
        if not self.run:
            return

        # Calculate Duration
        self._finalize_timing()

        if exc_val:
            await self._handle_failure(exc_val)
            return

        if self._output:
            await self._handle_success()
            return

        # This happens if code exited without exception but set_output wasn't called.
        # We treat this as a failure or incomplete run.
        await self._handle_incomplete()
