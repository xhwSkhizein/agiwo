"""Deprecated — import from ``agiwo.agent`` or ``agiwo.agent.models`` instead."""

import warnings as _warnings

_warnings.warn(
    "agiwo.agent.types is deprecated; import from agiwo.agent instead",
    DeprecationWarning,
    stacklevel=2,
)

from agiwo.agent.models import *  # noqa: F401, F403, E402
