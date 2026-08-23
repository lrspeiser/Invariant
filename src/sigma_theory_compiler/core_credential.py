"""Strict, non-persisting credential activation for the live core discovery runtime."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ENV_FILE_POINTER = "INVARIANT_ENV_FILE"
DEFAULT_CREDENTIAL_ENV_VAR = "ANTHROPIC_API_KEY"
MAXIMUM_ENV_FILE_BYTES = 64 * 1024
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


class CredentialActivationError(ValueError):
    """The required core-runtime credential could not be activated safely."""


@dataclass(frozen=True, slots=True)
class CredentialActivation:
    env_var: str
    source_kind: str
    source_locator_sha256: str
    injected_into_process: bool

    def to_evidence(self) -> dict[str, object]:
        return {
            "credential_env_var": self.env_var,
            "credential_persisted": False,
            "credential_value_recorded": False,
            "injected_into_process": self.injected_into_process,
            "source_kind": self.source_kind,
            "source_locator_sha256": self.source_locator_sha256,
        }


def _locator_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_env_value(path: Path, env_var: str) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CredentialActivationError("credential env file could not be read") from error
    if not payload or len(payload) > MAXIMUM_ENV_FILE_BYTES:
        raise CredentialActivationError("credential env file byte budget violated")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CredentialActivationError("credential env file is not UTF-8") from error
    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != env_var:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise CredentialActivationError("credential env file contains an invalid value")
        values.append(value)
    if len(values) != 1:
        raise CredentialActivationError("credential env file must define the credential exactly once")
    return values[0]


def _candidate_files(
    *, environment: Mapping[str, str], project_root: Path, home: Path
) -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = []
    explicit = environment.get(ENV_FILE_POINTER, "").strip()
    if explicit:
        candidates.append(("explicit_env_file", Path(explicit).expanduser()))
    candidates.extend(
        (
            ("project_env_file", project_root / ".env"),
            ("user_invariant_env_file", home / ".invariant.env"),
        )
    )
    unique: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for kind, candidate in candidates:
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append((kind, resolved))
    return tuple(unique)


@contextmanager
def activated_credential(
    *,
    project_root: Path,
    env_var: str = DEFAULT_CREDENTIAL_ENV_VAR,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Iterator[CredentialActivation]:
    """Expose one credential only for the dynamic extent of a live core run.

    Process-environment credentials take precedence.  Otherwise the loader checks an explicitly
    pointed env file, the ignored project ``.env``, and ``~/.invariant.env`` in that order.  It
    parses data and never evaluates shell syntax.
    """

    if _ENV_NAME.fullmatch(env_var) is None:
        raise CredentialActivationError("credential environment variable name is invalid")
    project_root = project_root.resolve()
    source_environment = os.environ if environment is None else environment
    direct = source_environment.get(env_var, "").strip()
    original = os.environ.get(env_var)
    if direct:
        activation = CredentialActivation(
            env_var,
            "process_environment",
            _locator_sha256(env_var),
            False,
        )
        if environment is not None and environment is not os.environ:
            os.environ[env_var] = direct
        try:
            yield activation
        finally:
            if environment is not None and environment is not os.environ:
                if original is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original
        return

    selected: tuple[str, Path] | None = None
    for candidate in _candidate_files(
        environment=source_environment,
        project_root=project_root,
        home=(home or Path.home()).resolve(),
    ):
        if candidate[1].is_file():
            selected = candidate
            break
    if selected is None:
        raise CredentialActivationError(
            f"{env_var} is absent and no supported credential env file exists"
        )
    source_kind, path = selected
    value = _parse_env_value(path, env_var)
    os.environ[env_var] = value
    activation = CredentialActivation(
        env_var,
        source_kind,
        _locator_sha256(os.path.normcase(str(path))),
        True,
    )
    try:
        yield activation
    finally:
        if original is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original
