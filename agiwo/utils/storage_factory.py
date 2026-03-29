"""Generic storage factory helper.

Provides a common dispatch pattern for creating storage backends
from a ``storage_type`` string and a configuration dict.
"""

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def create_storage(
    storage_type: str,
    config: dict[str, Any],
    backends: dict[str, Callable[[dict[str, Any]], T]],
    *,
    label: str = "storage",
) -> T:
    """Dispatch to the registered backend factory.

    Args:
        storage_type: Key selecting which backend to create.
        config: Opaque configuration dict forwarded to the factory.
        backends: Mapping of storage_type → factory callable.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If *storage_type* is not found in *backends*.
    """
    factory = backends.get(storage_type)
    if factory is None:
        known = ", ".join(sorted(backends)) or "(none)"
        raise ValueError(f"Unknown {label} type: {storage_type!r}  (known: {known})")
    return factory(config)


__all__ = ["create_storage"]
