from __future__ import annotations

import os
from pathlib import Path

import pytest

from sigma_theory_compiler.core_credential import (
    CredentialActivationError,
    activated_credential,
)


def test_user_invariant_env_file_is_activated_only_inside_context(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    secret = "test-anthropic-secret-never-persisted"
    (home / ".invariant.env").write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
    os.environ.pop("ANTHROPIC_API_KEY", None)

    with activated_credential(project_root=project, environment={}, home=home) as activation:
        assert os.environ["ANTHROPIC_API_KEY"] == secret
        assert activation.source_kind == "user_invariant_env_file"
        assert activation.injected_into_process
        assert secret not in str(activation.to_evidence())

    assert "ANTHROPIC_API_KEY" not in os.environ


def test_process_environment_takes_precedence_without_recording_value(tmp_path: Path) -> None:
    secret = "direct-process-secret"
    environment = {"ANTHROPIC_API_KEY": secret}
    with activated_credential(project_root=tmp_path, environment=environment, home=tmp_path) as item:
        assert os.environ["ANTHROPIC_API_KEY"] == secret
        assert item.source_kind == "process_environment"
        assert not item.injected_into_process
        assert secret not in str(item.to_evidence())
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_duplicate_or_missing_env_file_credentials_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "credentials.env"
    path.write_text("ANTHROPIC_API_KEY=one\nANTHROPIC_API_KEY=two\n", encoding="utf-8")
    environment = {"INVARIANT_ENV_FILE": str(path)}
    with pytest.raises(CredentialActivationError, match="exactly once"), activated_credential(
        project_root=tmp_path, environment=environment, home=tmp_path / "missing"
    ):
        pass
    with pytest.raises(CredentialActivationError, match="no supported"), activated_credential(
        project_root=tmp_path / "missing-project", environment={}, home=tmp_path / "missing"
    ):
        pass
