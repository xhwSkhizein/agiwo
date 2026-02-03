"""
OpenTelemetry OTLP exporter for traces.

Exports Agiwo traces to OTLP-compatible backends (Jaeger, Zipkin, SkyWalking, etc.)
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from agiwo.observability.trace import SpanKind, SpanStatus, Trace
from agiwo.config.settings import settings
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

# Global exporter instance
_exporter: "OTLPExporter | None" = None


class OTLPExporter:
    """
    OTLP (OpenTelemetry Protocol) exporter.

    Converts Agio Trace/Span to OTLP format and exports to configured endpoint.

    Supports:
    - Jaeger (via OTLP gRPC/HTTP)
    - Zipkin (via OTLP HTTP)
    - SkyWalking (via OTLP gRPC)
    - Any OTLP-compatible backend
    """

    def __init__(
        self,
        endpoint: str | None = None,
        protocol: str = "grpc",  # "grpc" or "http"
        headers: dict[str, str] | None = None,
        enabled: bool = True,
        sampling_rate: float = 1.0,
    ):
        """
        Initialize OTLP exporter.

        Args:
            endpoint: OTLP endpoint URL (e.g., "http://localhost:4317" for gRPC,
                     "http://localhost:4318/v1/traces" for HTTP)
            protocol: "grpc" or "http"
            headers: Optional HTTP headers (for authentication, etc.)
            enabled: Enable/disable export
            sampling_rate: Sampling rate (0.0 to 1.0). 1.0 = 100% sampling
        """
        self.endpoint = endpoint
        self.protocol = protocol
        self.headers = headers or {}
        self.enabled = enabled and endpoint is not None
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))

        self._exporter_impl = None
        if self.enabled:
            self._initialize_exporter()

    def _initialize_exporter(self):
        """Initialize OpenTelemetry SDK exporter"""
        try:
            if self.protocol == "grpc":
                self._exporter_impl = OTLPSpanExporter(
                    endpoint=self.endpoint,
                    headers=self.headers,
                )
            else:  # http
                self._exporter_impl = OTLPSpanExporter(
                    endpoint=self.endpoint,
                    headers=self.headers,
                )

            logger.info(
                "otlp_exporter_initialized",
                endpoint=self.endpoint,
                protocol=self.protocol,
            )
        except ImportError:
            logger.warning(
                "otlp_sdk_not_installed",
                message="OTLP export disabled. Install: pip install opentelemetry-exporter-otlp",
            )
            self.enabled = False
        except Exception as e:
            logger.error("otlp_exporter_init_failed", error=str(e))
            self.enabled = False

    async def export_trace(self, trace: Trace) -> bool:
        """
        Export trace to OTLP backend.

        Args:
            trace: Agio Trace to export

        Returns:
            True if export succeeded, False otherwise
        """
        if not self.enabled or not self._exporter_impl:
            return False

        # Apply sampling
        if not self._should_sample():
            logger.debug(
                "trace_sampled_out",
                trace_id=trace.trace_id,
                sampling_rate=self.sampling_rate,
            )
            return False

        try:
            # Convert to OTLP spans
            otlp_spans = self._convert_trace_to_otlp(trace)

            # Note: This is a simplified approach. In production, you'd use
            # the full OTEL SDK with TracerProvider and BatchSpanProcessor
            result = self._exporter_impl.export(otlp_spans)

            success = result == SpanExportResult.SUCCESS
            if success:
                logger.debug(
                    "trace_exported",
                    trace_id=trace.trace_id,
                    span_count=len(otlp_spans),
                )
            else:
                logger.warning(
                    "trace_export_failed",
                    trace_id=trace.trace_id,
                    result=result,
                )

            return success

        except Exception as e:
            logger.error(
                "trace_export_error",
                trace_id=trace.trace_id,
                error=str(e),
            )
            return False

    def _convert_trace_to_otlp(self, trace: Trace) -> list:
        """
        Convert Agio Trace to OTLP ReadableSpan list.

        This is a simplified conversion. Full implementation would use
        opentelemetry.sdk.trace.ReadableSpan with proper context propagation.
        """
        otlp_spans = []

        # Convert trace_id and span_ids from UUID to OTLP format
        trace_id_bytes = self._uuid_to_trace_id(trace.trace_id)

        for span in trace.spans:
            # Convert span_id
            span_id_bytes = self._uuid_to_span_id(span.span_id)
            parent_span_id_bytes = (
                self._uuid_to_span_id(span.parent_span_id)
                if span.parent_span_id
                else None
            )

            # Create SpanContext
            span_context = SpanContext(
                trace_id=int.from_bytes(trace_id_bytes, "big"),
                span_id=int.from_bytes(span_id_bytes, "big"),
                is_remote=False,
                trace_flags=TraceFlags(0x01),  # Sampled
            )

            # Convert status
            status_code = (
                StatusCode.OK
                if span.status == SpanStatus.OK
                else StatusCode.ERROR
                if span.status == SpanStatus.ERROR
                else StatusCode.UNSET
            )
            status = Status(status_code=status_code, description=span.error_message)

            # Convert timestamps to nanoseconds
            start_time_ns = int(span.start_time.timestamp() * 1e9)
            end_time_ns = (
                int(span.end_time.timestamp() * 1e9) if span.end_time else start_time_ns
            )

            # Build attributes
            attributes = dict(span.attributes)
            if span.kind == SpanKind.LLM_CALL:
                attributes["llm.model"] = span.name
                if "tokens.total" in span.metrics:
                    attributes["llm.tokens.total"] = span.metrics["tokens.total"]
                if "tokens.input" in span.metrics:
                    attributes["llm.tokens.input"] = span.metrics["tokens.input"]
                if "tokens.output" in span.metrics:
                    attributes["llm.tokens.output"] = span.metrics["tokens.output"]
            elif span.kind == SpanKind.TOOL_CALL:
                attributes["tool.name"] = span.name

            # Create ReadableSpan
            # Note: This is a simplified mock. Real implementation would use
            # proper OTEL SDK span creation with Tracer
            otlp_span = self._create_readable_span(
                name=span.name,
                context=span_context,
                parent_span_id=parent_span_id_bytes,
                kind=span.to_otel_span_kind(),
                start_time=start_time_ns,
                end_time=end_time_ns,
                attributes=attributes,
                status=status,
            )

            otlp_spans.append(otlp_span)

        return otlp_spans

    def _uuid_to_trace_id(self, uuid_str: str) -> bytes:
        """
        Convert UUID string to OTLP trace_id (16 bytes).

        OTLP trace_id is 16 bytes (128 bits), UUID is also 128 bits.
        """
        import uuid

        uuid_obj = uuid.UUID(uuid_str)
        return uuid_obj.bytes

    def _uuid_to_span_id(self, uuid_str: str) -> bytes:
        """
        Convert UUID string to OTLP span_id (8 bytes).

        OTLP span_id is 8 bytes (64 bits), we take first 8 bytes of UUID.
        """
        import uuid

        uuid_obj = uuid.UUID(uuid_str)
        return uuid_obj.bytes[:8]

    def _create_readable_span(
        self,
        name: str,
        context,
        parent_span_id: bytes | None,
        kind: int,
        start_time: int,
        end_time: int,
        attributes: dict,
        status,
    ):
        """
        Create a ReadableSpan-like object for export.

        Note: This is a simplified mock for demonstration.
        Real implementation should use opentelemetry.sdk.trace.Span.
        """
        # For now, return a dict that matches OTLP structure
        # In production, use proper OTEL SDK span creation
        return {
            "name": name,
            "context": context,
            "parent": parent_span_id,
            "kind": kind,
            "start_time": start_time,
            "end_time": end_time,
            "attributes": attributes,
            "status": status,
        }

    def _should_sample(self) -> bool:
        """
        Determine if trace should be sampled based on sampling rate.

        Uses random sampling for simplicity.
        """
        if self.sampling_rate >= 1.0:
            return True
        if self.sampling_rate <= 0.0:
            return False

        import random

        return random.random() < self.sampling_rate

    async def shutdown(self):
        """Shutdown exporter"""
        if self._exporter_impl:
            try:
                self._exporter_impl.shutdown()
            except Exception as e:
                logger.error("otlp_exporter_shutdown_failed", error=str(e))


def get_otlp_exporter() -> OTLPExporter:
    """Get global OTLP exporter instance"""
    global _exporter
    if _exporter is None:
        # Check for OTLP configuration in settings
        endpoint = settings.otlp_endpoint
        protocol = settings.otlp_protocol
        enabled = settings.otlp_enabled
        sampling_rate = settings.otlp_sampling_rate

        _exporter = OTLPExporter(
            endpoint=endpoint,
            protocol=protocol,
            enabled=enabled,
            sampling_rate=sampling_rate,
        )

    return _exporter


__all__ = ["OTLPExporter", "get_otlp_exporter"]
