from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any


REDACTED_LOG_VALUE = "<redacted>"

_ALWAYS_SENSITIVE_KEY_PARTS = {
    "credential",
    "credentials",
    "passwd",
    "passwds",
    "password",
    "passwords",
    "passphrase",
    "passphrases",
    "secret",
    "secrets",
}
_SENSITIVE_COMPACT_KEYS = {
    "accesskey",
    "accesskeys",
    "accesstoken",
    "accesstokens",
    "apikey",
    "apikeys",
    "auth",
    "authheaders",
    "authorization",
    "bearer",
    "bearertoken",
    "bearertokens",
    "clientsecret",
    "clientsecrets",
    "connectionstring",
    "cookie",
    "cookies",
    "dsn",
    "key",
    "keys",
    "privatekey",
    "privatekeys",
    "refreshtoken",
    "refreshtokens",
    "secretkey",
    "secretkeys",
    "sessioncookie",
    "sessioncookies",
    "setcookie",
    "token",
    "tokens",
}
_SENSITIVE_KEY_MODIFIERS = {
    "access",
    "api",
    "auth",
    "encryption",
    "private",
    "secret",
    "signing",
    "ssh",
}


class RedactedLogValue:
    """JSON-formatted logging view with potential secret values removed."""

    def __init__(self, value: Any):
        self._value = _redact_potential_secrets(value)

    def __str__(self) -> str:
        return json.dumps(
            self._value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return str(self)


def redacted_for_logging(value: Any) -> RedactedLogValue:
    """Wrap a value for structured logging without retaining secret values."""
    return RedactedLogValue(value)


def _redact_potential_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED_LOG_VALUE
                if _is_sensitive_key(key)
                else _redact_potential_secrets(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_potential_secrets(child) for child in value]
    return copy.deepcopy(value)


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    parts = tuple(
        part
        for part in re.split(r"[^a-z0-9]+", separated.lower())
        if part
    )
    if not parts:
        return False
    if any(part in _ALWAYS_SENSITIVE_KEY_PARTS for part in parts):
        return True
    compact = "".join(parts)
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    return (
        parts[-1] in {"key", "keys", "token", "tokens"}
        and any(part in _SENSITIVE_KEY_MODIFIERS for part in parts[:-1])
    )
