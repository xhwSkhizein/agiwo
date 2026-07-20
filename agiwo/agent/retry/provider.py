"""Map provider exceptions to ExecutionFault using structured type/status signals."""

from agiwo.agent.retry.faults import ExecutionFault, FaultDisposition

_AUTH_TYPE_NAMES = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "PermissionError",
    }
)
_RETRYABLE_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ConnectError",
        "ReadTimeout",
        "TimeoutException",
        "BedrockRetryableError",
    }
)
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 422})


def map_provider_exception(
    exc: BaseException,
    *,
    response_observed: bool = False,
    logical_call_id: str | None = None,
    attempt_no: int = 1,
) -> ExecutionFault:
    """Convert a provider exception into a structured fault.

    Disposition is derived from exception type name and numeric status_code
    attributes — not from free-text error message matching.
    """
    type_name = type(exc).__name__
    status = _status_code(exc)
    provider_code = type_name if status is None else f"{type_name}:{status}"

    if type_name in _AUTH_TYPE_NAMES or status in {401, 403}:
        return ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.NON_RETRYABLE,
            run_blocking=True,
            response_observed=response_observed,
            provider_code=provider_code,
            message=str(exc),
            provenance={"exception_type": type_name},
            attempt_no=attempt_no,
            logical_call_id=logical_call_id,
        )

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.RETRYABLE,
            run_blocking=True,
            response_observed=response_observed,
            provider_code=provider_code,
            message=str(exc),
            provenance={"exception_type": type_name},
            attempt_no=attempt_no,
            logical_call_id=logical_call_id,
        )

    if type_name in _RETRYABLE_TYPE_NAMES or status in _RETRYABLE_STATUS:
        return ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.RETRYABLE,
            run_blocking=True,
            response_observed=response_observed,
            provider_code=provider_code,
            message=str(exc),
            provenance={"exception_type": type_name},
            attempt_no=attempt_no,
            logical_call_id=logical_call_id,
        )

    if status in _NON_RETRYABLE_STATUS:
        return ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.NON_RETRYABLE,
            run_blocking=True,
            response_observed=response_observed,
            provider_code=provider_code,
            message=str(exc),
            provenance={"exception_type": type_name},
            attempt_no=attempt_no,
            logical_call_id=logical_call_id,
        )

    # Ambiguous transport: if any bytes may have been accepted, treat as unknown.
    if response_observed:
        return ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.OUTCOME_UNKNOWN,
            run_blocking=True,
            response_observed=True,
            provider_code=provider_code,
            message=str(exc),
            provenance={"exception_type": type_name},
            attempt_no=attempt_no,
            logical_call_id=logical_call_id,
        )

    return ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.NON_RETRYABLE,
        run_blocking=True,
        response_observed=False,
        provider_code=provider_code,
        message=str(exc),
        provenance={"exception_type": type_name},
        attempt_no=attempt_no,
        logical_call_id=logical_call_id,
    )


def _status_code(exc: BaseException) -> int | None:
    raw = getattr(exc, "status_code", None)
    if isinstance(raw, int):
        return raw
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


__all__ = ["map_provider_exception"]
