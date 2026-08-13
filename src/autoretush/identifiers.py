from __future__ import annotations

import re

_PAIR_ID_PATTERN = re.compile(r"p_[0-9a-f]{20}\Z")


def validate_pair_id(value: object) -> str:
    """Return a canonical pseudonymous pair ID or reject it as unsafe."""
    if not isinstance(value, str) or _PAIR_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("pair_id must match p_ followed by 20 lowercase hexadecimal characters")
    return value
