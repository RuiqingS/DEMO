"""Small environment adapters for legacy experiment entry points."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment override while preserving legacy defaults."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )
