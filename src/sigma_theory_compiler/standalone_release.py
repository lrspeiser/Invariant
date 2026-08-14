"""Build and verify the supported standalone Invariant source release.

The wheel is the non-editable Python installation boundary.  The adjacent source
tree remains the complete resource boundary for configs, formal assets, immutable
receipts, LFS evidence, examples, and the provenance materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "invariant-standalone-source-release-manifest-1.0"
ARCHIVE_FORMAT = "zip-stored-v1"
MANIFEST_NAME = "RELEASE-MANIFEST.json"
MANIFEST_SHA_NAME = "RELEASE-MANIFEST.sha256"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_LFS_LINE = re.compile(r"^([0-9a-f]{64}) [*-] (.+)$")


class StandaloneReleaseError(ValueError):
    """The release cannot be built or verified under the closed contract."""


class ReleaseBuildError(StandaloneReleaseError):
    """The source checkout is not eligible for a standalone release build."""


class ReleaseVerificationError(StandaloneReleaseError):
    """The extracted release does not match its complete manifest."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _run_git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "git command failed"
        raise ReleaseBuildError(message)
    return process.stdout


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise StandaloneReleaseError(f"unsafe release path: {value}")
    return path.as_posix()


def _tracked_modes(root: Path) -> dict[str, int]:
    raw = _run_git(root, "ls-files", "--stage", "-z")
    result: dict[str, int] = {}
    for record in raw.split("\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition("\t")
        if not separator:
            raise ReleaseBuildError("git index record is malformed")
        fields = metadata.split()
        if len(fields) != 3 or fields[2] != "0":
            raise ReleaseBuildError("unmerged index entries cannot be released")
        mode = int(fields[0], 8)
        if mode == 0o160000:
            raise ReleaseBuildError("git submodules are not supported release resources")
        path = _safe_relative(raw_path)
        if path in result:
            raise ReleaseBuildError(f"duplicate tracked path: {path}")
        result[path] = mode
    if not result:
        raise ReleaseBuildError("release has no tracked resources")
    folded: dict[str, str] = {}
    for path in result:
        previous = folded.setdefault(path.casefold(), path)
        if previous != path:
            raise ReleaseBuildError(f"case-insensitive path collision: {previous}, {path}")
    return result


def _lfs_objects(root: Path) -> dict[str, str]:
    output = _run_git(root, "lfs", "ls-files", "--long")
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        match = _LFS_LINE.fullmatch(line)
        if match is None:
            raise ReleaseBuildError(f"unexpected git-lfs record: {line}")
        oid, raw_path = match.groups()
        path = _safe_relative(raw_path)
        result[path] = oid
    return result


def _project_version(root: Path) -> str:
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        project = tomllib.loads(pyproject)["project"]
        version = project["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseBuildError("pyproject version is unavailable") from error
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", version):
        raise ReleaseBuildError("project version is not a supported release version")
    return version


def _build_wheel(root: Path, destination: Path, source_date_epoch: str) -> Path:
    context = destination / "wheel-source"
    context.mkdir()
    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(root / filename, context / filename)
    shutil.copytree(root / "src", context / "src")
    wheels = destination / "wheels"
    wheels.mkdir()
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = source_date_epoch
    process = subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(context),
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    built = sorted(wheels.glob("sigma_theory_compiler-*.whl"))
    if process.returncode != 0 or len(built) != 1:
        detail = process.stderr.strip() or process.stdout.strip() or "wheel build failed"
        raise ReleaseBuildError(detail)
    return built[0]


def _entry(
    path: str,
    source: Path,
    *,
    kind: str,
    mode: int,
    lfs_oid: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "bytes": source.stat().st_size,
        "kind": kind,
        "mode": f"{mode & 0o777:04o}",
        "path": _safe_relative(path),
        "sha256": _sha256_file(source),
    }
    if lfs_oid is not None:
        row["lfs_oid_sha256"] = lfs_oid
    return row


def _zip_write(archive: zipfile.ZipFile, arcname: str, source: Path, mode: int) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ((stat.S_IFREG | (mode & 0o777)) << 16)
    with source.open("rb") as handle, archive.open(info, "w") as target:
        shutil.copyfileobj(handle, target, length=1024 * 1024)


def _zip_write_bytes(archive: zipfile.ZipFile, arcname: str, payload: bytes, mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ((stat.S_IFREG | mode) << 16)
    archive.writestr(info, payload)


def build_standalone_release(
    project_root: Path, output_directory: Path, *, allow_dirty: bool = False
) -> Path:
    """Build one deterministic, complete, hydrated source-release ZIP."""

    root = project_root.resolve()
    if not (root / ".git").exists():
        raise ReleaseBuildError("project root is not a Git worktree")
    dirty = bool(_run_git(root, "status", "--porcelain=v1").strip())
    if dirty and not allow_dirty:
        raise ReleaseBuildError("release builds require a clean tracked and untracked worktree")
    tracked = _tracked_modes(root)
    lfs = _lfs_objects(root)
    unknown_lfs = sorted(set(lfs) - set(tracked))
    if unknown_lfs:
        raise ReleaseBuildError(f"LFS paths are not tracked: {unknown_lfs[0]}")
    version = _project_version(root)
    commit = _run_git(root, "rev-parse", "HEAD").strip()
    source_date_epoch = _run_git(root, "show", "-s", "--format=%ct", "HEAD").strip()
    if not _GIT_COMMIT.fullmatch(commit):
        raise ReleaseBuildError("source commit is not a full object ID")
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / f"invariant-{version}-source-release.zip"
    if archive_path.exists():
        raise ReleaseBuildError(f"release artifact already exists: {archive_path}")
    root_name = f"Invariant-{version}"

    with tempfile.TemporaryDirectory(prefix="invariant-release-build-") as raw_temporary:
        temporary = Path(raw_temporary)
        wheel = _build_wheel(root, temporary, source_date_epoch)
        sources: dict[str, tuple[Path, int]] = {}
        entries: list[dict[str, Any]] = []
        for path, mode in sorted(tracked.items()):
            source = root / Path(*PurePosixPath(path).parts)
            if not source.is_file() or source.is_symlink():
                raise ReleaseBuildError(f"tracked resource is missing or not regular: {path}")
            oid = lfs.get(path)
            row = _entry(
                path,
                source,
                kind="git_lfs_resource" if oid is not None else "git_tracked_resource",
                mode=mode,
                lfs_oid=oid,
            )
            if oid is not None and row["sha256"] != oid:
                raise ReleaseBuildError(f"LFS resource is missing or tampered: {path}")
            entries.append(row)
            sources[path] = (source, mode)
        wheel_path = f"packages/{wheel.name}"
        wheel_row = _entry(wheel_path, wheel, kind="python_wheel", mode=0o100644)
        entries.append(wheel_row)
        sources[wheel_path] = (wheel, 0o100644)
        entries.sort(key=lambda row: row["path"])
        manifest = {
            "archive_format": ARCHIVE_FORMAT,
            "claims": {
                "editable_checkout_required": False,
                "git_metadata_required_after_extraction": False,
                "lfs_payloads_hydrated": True,
                "network_required_for_formula_cli_examples": False,
            },
            "counts": {
                "git_lfs_resources": len(lfs),
                "git_tracked_resources": len(tracked),
                "payload_entries": len(entries),
                "python_wheels": 1,
            },
            "entries": entries,
            "install_contract": {
                "formula_cli": "sigma-formula-discovery",
                "materializer": "scripts/materialize_hash_bound_worktree.py",
                "resource_root": ".",
                "verifier": "sigma-release verify --release-root .",
                "wheel_directory": "packages",
            },
            "release": {
                "name": "Invariant",
                "python_requires": ">=3.11",
                "root_directory": root_name,
                "source_commit": commit,
                "source_tree_clean": not dirty,
                "version": version,
            },
            "schema_version": MANIFEST_SCHEMA,
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_sha = _sha256_bytes(manifest_bytes)
        manifest_sha_bytes = f"{manifest_sha}  {MANIFEST_NAME}\n".encode("ascii")
        try:
            with zipfile.ZipFile(archive_path, "x", allowZip64=True) as archive:
                _zip_write_bytes(archive, f"{root_name}/{MANIFEST_NAME}", manifest_bytes)
                _zip_write_bytes(archive, f"{root_name}/{MANIFEST_SHA_NAME}", manifest_sha_bytes)
                for path, (source, mode) in sorted(sources.items()):
                    _zip_write(archive, f"{root_name}/{path}", source, mode)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
    return archive_path


def _load_manifest(root: Path) -> tuple[dict[str, Any], bytes]:
    manifest_path = root / MANIFEST_NAME
    sha_path = root / MANIFEST_SHA_NAME
    try:
        payload = manifest_path.read_bytes()
        declared = sha_path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ReleaseVerificationError("release manifest or checksum is missing") from error
    expected_line = f"{_sha256_bytes(payload)}  {MANIFEST_NAME}\n"
    if declared != expected_line:
        raise ReleaseVerificationError("release manifest checksum changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError("release manifest is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError("release manifest must be an object")
    return value, payload


def verify_release_root(release_root: Path) -> dict[str, Any]:
    """Verify every file and LFS object in one extracted standalone release."""

    root = release_root.resolve()
    manifest, _ = _load_manifest(root)
    required = {
        "archive_format",
        "claims",
        "counts",
        "entries",
        "install_contract",
        "release",
        "schema_version",
    }
    if set(manifest) != required or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ReleaseVerificationError("release manifest schema changed")
    if manifest.get("archive_format") != ARCHIVE_FORMAT:
        raise ReleaseVerificationError("release archive format changed")
    release = manifest.get("release")
    if not isinstance(release, dict) or release.get("root_directory") != root.name:
        raise ReleaseVerificationError("release root directory changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReleaseVerificationError("release payload manifest is empty")
    expected_paths = {MANIFEST_NAME, MANIFEST_SHA_NAME}
    lfs_count = tracked_count = wheel_count = 0
    previous = ""
    for row in entries:
        if not isinstance(row, dict):
            raise ReleaseVerificationError("release entry is not an object")
        required_row = {"bytes", "kind", "mode", "path", "sha256"}
        kind = row.get("kind")
        if kind == "git_lfs_resource":
            required_row.add("lfs_oid_sha256")
            lfs_count += 1
            tracked_count += 1
        elif kind == "git_tracked_resource":
            tracked_count += 1
        elif kind == "python_wheel":
            wheel_count += 1
        else:
            raise ReleaseVerificationError("release entry kind changed")
        if set(row) != required_row:
            raise ReleaseVerificationError("release entry schema changed")
        try:
            path = _safe_relative(row["path"])
        except (KeyError, TypeError, StandaloneReleaseError) as error:
            raise ReleaseVerificationError("release entry path is invalid") from error
        if path <= previous or path in expected_paths:
            raise ReleaseVerificationError("release entries are not uniquely sorted")
        previous = path
        expected_paths.add(path)
        source = root / Path(*PurePosixPath(path).parts)
        if not source.is_file() or source.is_symlink():
            raise ReleaseVerificationError(f"release resource is missing: {path}")
        size = row.get("bytes")
        sha = row.get("sha256")
        if (
            not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or not _SHA256.fullmatch(sha)
        ):
            raise ReleaseVerificationError(f"release resource metadata is invalid: {path}")
        if source.stat().st_size != size or _sha256_file(source) != sha:
            raise ReleaseVerificationError(f"release resource bytes changed: {path}")
        if kind == "git_lfs_resource":
            oid = row.get("lfs_oid_sha256")
            if oid != sha:
                raise ReleaseVerificationError(f"release LFS object identity changed: {path}")
    discovered = list(root.rglob("*"))
    symlinks = [candidate for candidate in discovered if candidate.is_symlink()]
    if symlinks:
        detail = symlinks[0].relative_to(root).as_posix()
        raise ReleaseVerificationError(f"release contains a symbolic link: {detail}")
    actual_paths = {
        candidate.relative_to(root).as_posix() for candidate in discovered if candidate.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        detail = missing[0] if missing else extra[0]
        raise ReleaseVerificationError(f"release file set changed: {detail}")
    counts = manifest.get("counts")
    expected_counts = {
        "git_lfs_resources": lfs_count,
        "git_tracked_resources": tracked_count,
        "payload_entries": len(entries),
        "python_wheels": wheel_count,
    }
    if counts != expected_counts or wheel_count != 1:
        raise ReleaseVerificationError("release manifest counts changed")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-release")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--project-root", type=Path, default=Path("."))
    build.add_argument("--output-directory", type=Path, required=True)
    build.add_argument("--allow-dirty", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--release-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            artifact = build_standalone_release(
                arguments.project_root,
                arguments.output_directory,
                allow_dirty=arguments.allow_dirty,
            )
            result = {"artifact": str(artifact), "status": "BUILT"}
        else:
            manifest = verify_release_root(arguments.release_root)
            result = {
                "payload_entries": manifest["counts"]["payload_entries"],
                "source_commit": manifest["release"]["source_commit"],
                "status": "VERIFIED",
                "version": manifest["release"]["version"],
            }
    except StandaloneReleaseError as error:
        print(json.dumps({"error": str(error), "status": "ERROR"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_FORMAT",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SHA_NAME",
    "ReleaseBuildError",
    "ReleaseVerificationError",
    "StandaloneReleaseError",
    "build_standalone_release",
    "main",
    "verify_release_root",
]
