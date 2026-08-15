from __future__ import annotations

import hashlib
import json
import os
import subprocess
import venv
import zipfile
from pathlib import Path

import pytest

from sigma_theory_compiler.standalone_release import (
    MANIFEST_NAME,
    ReleaseBuildError,
    ReleaseVerificationError,
    build_standalone_release,
    verify_release_root,
)

ROOT = Path(__file__).resolve().parents[1]


def _run(
    arguments: list[str], *, cwd: Path, expected: int = 0, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert process.returncode == expected, (process.stdout, process.stderr)
    return process


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(root: Path, name: str) -> Path:
    return root / (f"Scripts/{name}.exe" if os.name == "nt" else f"bin/{name}")


def _snapshot(root: Path, paths: list[str]) -> dict[str, str]:
    return {
        path: hashlib.sha256((root / Path(*path.split("/"))).read_bytes()).hexdigest()
        for path in paths
    }


def test_clean_release_build_refuses_dirty_checkout(tmp_path: Path) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if not status.strip():
        pytest.skip("the checked-out release source is clean")
    with pytest.raises(ReleaseBuildError, match="clean tracked and untracked worktree"):
        build_standalone_release(ROOT, tmp_path)


def test_built_bundle_is_complete_tamper_evident_and_runs_isolated_examples(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    archive_path = build_standalone_release(ROOT, artifacts, allow_dirty=True)
    assert archive_path.name == "invariant-0.1.0-source-release.zip"
    assert zipfile.ZipFile(archive_path).testzip() is None
    repeated = build_standalone_release(ROOT, tmp_path / "artifacts-repeat", allow_dirty=True)
    assert (
        hashlib.sha256(repeated.read_bytes()).digest()
        == hashlib.sha256(archive_path.read_bytes()).digest()
    )

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    release_root = extracted / "Invariant-0.1.0"
    manifest = verify_release_root(release_root)
    assert manifest["counts"]["git_lfs_resources"] == 294
    assert manifest["counts"]["git_tracked_resources"] >= 4000
    assert manifest["claims"] == {
        "editable_checkout_required": False,
        "git_metadata_required_after_extraction": False,
        "lfs_payloads_hydrated": True,
        "network_required_for_formula_cli_examples": False,
    }

    missing_path = release_root / "examples/formula-discovery/pass-exact-polynomial.json"
    missing_bytes = missing_path.read_bytes()
    missing_path.unlink()
    with pytest.raises(ReleaseVerificationError, match="resource is missing"):
        verify_release_root(release_root)
    missing_path.write_bytes(missing_bytes)

    lfs_row = next(row for row in manifest["entries"] if row["kind"] == "git_lfs_resource")
    lfs_path = release_root / Path(*lfs_row["path"].split("/"))
    lfs_bytes = lfs_path.read_bytes()
    lfs_path.write_bytes(lfs_bytes[:-1] + bytes((lfs_bytes[-1] ^ 1,)))
    with pytest.raises(ReleaseVerificationError, match="resource bytes changed"):
        verify_release_root(release_root)
    lfs_path.write_bytes(lfs_bytes)
    assert verify_release_root(release_root)["counts"] == manifest["counts"]

    caller = tmp_path / "caller"
    caller.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PIP_NO_INDEX"] = "1"
    venv_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_root)
    python = _venv_python(venv_root)
    wheel_row = next(row for row in manifest["entries"] if row["kind"] == "python_wheel")
    wheel = release_root / Path(*wheel_row["path"].split("/"))
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
        cwd=caller,
        environment=environment,
    )
    imported = _run(
        [
            str(python),
            "-c",
            "import pathlib,sigma_theory_compiler as s;print(pathlib.Path(s.__file__).resolve())",
        ],
        cwd=caller,
        environment=environment,
    ).stdout.strip()
    assert "site-packages" in imported.replace("\\", "/")
    assert str(ROOT).casefold() not in imported.casefold()

    release_cli = _venv_script(venv_root, "sigma-release")
    formula_cli = _venv_script(venv_root, "sigma-formula-discovery")
    assert release_cli.is_file() and formula_cli.is_file()
    verify = _run(
        [
            str(release_cli),
            "verify",
            "--release-root",
            str(release_root),
        ],
        cwd=caller,
        environment=environment,
    )
    assert json.loads(verify.stdout)["status"] == "VERIFIED"

    examples = release_root / "examples/formula-discovery"
    decisions = (
        ("pass-exact-polynomial.json", "PASS", 0),
        ("reject-heldout-counterexample.json", "REJECT", 10),
    )
    for filename, decision, exit_code in decisions:
        stem = filename.removesuffix(".json")
        result = caller / f"{stem}-result.json"
        report = caller / f"{stem}-report.md"
        command = [str(formula_cli)]
        run = _run(
            [
                *command,
                "run",
                "--problem",
                str(examples / filename),
                "--result",
                str(result),
                "--report",
                str(report),
            ],
            cwd=caller,
            expected=exit_code,
            environment=environment,
        )
        assert json.loads(run.stdout)["decision"] == decision
        replay = _run(
            [
                *command,
                "validate",
                "--problem",
                str(examples / filename),
                "--result",
                str(result),
                "--report",
                str(report),
            ],
            cwd=caller,
            expected=exit_code,
            environment=environment,
        )
        assert json.loads(replay.stdout)["decision"] == decision
        stored = json.loads(result.read_text(encoding="utf-8"))
        assert stored["decision"] == decision
        if decision == "PASS":
            assert stored["discovery_job"]["synthesis"]["expression"] == "x**3 - 2*x + 5"
        else:
            assert stored["discovery_job"]["validation"]["counterexample"] is not None

    materializer = release_root / "scripts/materialize_hash_bound_worktree.py"
    first = _run(
        [str(python), str(materializer), "--project-root", str(release_root)],
        cwd=caller,
        environment=environment,
    )
    first_result = json.loads(first.stdout)
    assert first_result["materialization_passes"] >= 1
    tracked_paths = [
        row["path"]
        for row in manifest["entries"]
        if row["kind"] in {"git_tracked_resource", "git_lfs_resource"}
    ]
    after_first = _snapshot(release_root, tracked_paths)
    second = _run(
        [str(python), str(materializer), "--project-root", str(release_root)],
        cwd=caller,
        environment=environment,
    )
    second_result = json.loads(second.stdout)
    assert second_result["files_rewritten"] == 0
    assert _snapshot(release_root, tracked_paths) == after_first
    assert (release_root / MANIFEST_NAME).is_file()

    _run([str(python), "-m", "pip", "check"], cwd=caller, environment=environment)


def test_script_wrapper_and_project_entry_points_are_registered() -> None:
    source = (ROOT / "scripts/build_standalone_release.py").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "sigma_theory_compiler.standalone_release" in source
    assert (
        'sigma-formula-discovery = "sigma_theory_compiler.formula_discovery_cli:main"' in pyproject
    )
    assert 'sigma-release = "sigma_theory_compiler.standalone_release:main"' in pyproject
    assert 'release = ["setuptools>=75", "wheel"]' in pyproject
