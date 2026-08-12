"""Build a deterministic, host-path-free projection of the formal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-formal-controls-portable-report-config-1.0"
RESULT_SCHEMA = "sigma-formal-controls-portable-report-1.0"
EXPECTED_RESULT_KEYS = {
    "backend_identity",
    "claim_seals",
    "content_sha256",
    "decision",
    "portability",
    "schema_version",
    "scope",
    "semantic_report",
    "source_bindings",
    "source_report",
}
EXPECTED_CLAIM_SEALS = {
    "candidate_formal_pass_inferred": False,
    "candidate_theory_validity_claimed": False,
    "observational_gate_opened": False,
    "scientific_result_inferred": False,
    "host_local_receipt_declared_portable": False,
}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]")
WSL_ABSOLUTE = re.compile(r"/mnt/[a-z]/")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("portable formal report path escapes repository") from error
    return path


def _windows_to_wsl_text(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if len(normalized) < 3 or normalized[1:3] != ":/":
        return None
    return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"


def _substitutions(report: Mapping[str, Any], project_root: Path) -> list[tuple[str, str]]:
    cadabra = report.get("backends", {}).get("cadabra2", {})
    contract_path = Path(str(report.get("field_contract", "")))
    reported_root = contract_path.parent.parent if contract_path.is_absolute() else project_root
    roots = [
        (str(project_root.resolve()), "{PROJECT_ROOT}"),
        (str(reported_root), "{PROJECT_ROOT}"),
        (str(cadabra.get("root", "")), "{CADABRA_ROOT}"),
    ]
    values: list[tuple[str, str]] = []
    for raw, replacement in roots:
        if not raw:
            continue
        variants = {raw, raw.replace("\\", "/")}
        wsl = _windows_to_wsl_text(raw)
        if wsl:
            variants.add(wsl)
        values.extend((variant, replacement) for variant in variants if variant)
    return sorted(set(values), key=lambda item: len(item[0]), reverse=True)


def _normalize(value: Any, substitutions: list[tuple[str, str]]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item, substitutions) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item, substitutions) for item in value]
    if isinstance(value, str):
        normalized = value
        for raw, replacement in substitutions:
            normalized = normalized.replace(raw, replacement)
        return normalized
    return value


def _assert_portable(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str) and (WINDOWS_ABSOLUTE.search(value) or WSL_ABSOLUTE.search(value)):
        raise ValueError("portable formal report retains an absolute host path")


def _backend_hashes(report: Mapping[str, Any]) -> dict[str, str]:
    cadabra = report.get("backends", {}).get("cadabra2", {})
    if not cadabra.get("available") or cadabra.get("mode") != "wsl-local":
        raise ValueError("portable formal report requires the registered wsl-local Cadabra run")
    executable = Path(str(cadabra.get("executable", "")))
    module = Path(str(cadabra.get("python_module", "")))
    if not executable.is_file() or not module.is_file():
        raise ValueError("registered Cadabra executable or module is unavailable")
    return {
        "executable_sha256": _file_sha(executable),
        "python_module_sha256": _file_sha(module),
    }


def portable_projection(
    report: Mapping[str, Any],
    project_root: Path,
    *,
    backend_hashes: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(report) != {
        "backends",
        "candidate_readiness",
        "checks",
        "counts",
        "created_utc",
        "field_contract",
        "interpretation",
        "schema_version",
    }:
        raise ValueError("formal controls report shape changed")
    if (
        report.get("schema_version") != "sigma-formal-controls-1.0"
        or report.get("counts") != {"total": 118, "passed": 118, "failed": 0}
        or len(report.get("checks", [])) != 118
        or len({item.get("name") for item in report.get("checks", [])}) != 118
        or any(item.get("status") != "pass" for item in report.get("checks", []))
    ):
        raise ValueError("formal controls are not the registered 118-of-118 run")
    hashes = dict(backend_hashes or _backend_hashes(report))
    if set(hashes) != {"executable_sha256", "python_module_sha256"} or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()
    ):
        raise ValueError("Cadabra backend hashes are incomplete")
    substitutions = _substitutions(report, project_root)
    normalized = deepcopy(dict(report))
    normalized.pop("created_utc")
    cadabra = normalized["backends"]["cadabra2"]
    for key in ("root", "executable", "python_module"):
        cadabra.pop(key, None)
    cadabra.update(hashes)
    normalized = _normalize(normalized, substitutions)
    normalized["field_contract"] = "configs/covariant_field_contract.json"
    _assert_portable(normalized)
    backend_identity = {
        "cadabra2": {
            "available": True,
            "mode": "wsl-local",
            "version": report["backends"]["cadabra2"]["version"],
            **hashes,
        },
        "sympy": {
            "available": True,
            "version": report["backends"]["sympy"]["version"],
        },
    }
    return normalized, backend_identity


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        set(config) != {"schema_version", "input_report", "output_path", "expected", "claim_seals"}
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("expected")
        != {
            "formal_controls_total": 118,
            "formal_controls_passed": 118,
            "formal_controls_failed": 0,
        }
        or config.get("claim_seals") != EXPECTED_CLAIM_SEALS
    ):
        raise ValueError("portable formal report config changed")
    return config


def build_report(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = _load_config(config_path)
    input_path = _inside(root, str(config["input_report"]))
    report = json.loads(input_path.read_text(encoding="utf-8"))
    semantic_report, backend_identity = portable_projection(report, root)
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_formal_controls_portable_report.py"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "decision": "pass_portable_semantic_projection_118_controls",
        "source_report": {
            "path": input_path.relative_to(root).as_posix(),
            "schema_version": report["schema_version"],
            "created_utc_excluded_from_semantic_projection": True,
            "semantic_sha256": _sha(semantic_report),
            "counts": report["counts"],
        },
        "backend_identity": backend_identity,
        "semantic_report": semantic_report,
        "portability": {
            "absolute_windows_paths": 0,
            "absolute_wsl_paths": 0,
            "host_timestamps_in_semantic_projection": 0,
            "backend_paths_replaced_by_binary_hashes": True,
            "project_paths_replaced_by_repository_relative_paths": True,
            "same_semantics_reproduce_across_root_and_timestamp_changes": True,
        },
        "claim_seals": config["claim_seals"],
        "scope": (
            "portable semantic projection of the registered 118 formal controls; the host-local "
            "execution receipt remains separate and no candidate, theory or observation pass is inferred"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    result["content_sha256"] = _sha(result)
    validate_artifact(result, root, config_path)
    return result


def validate_artifact(result: Mapping[str, Any], root: Path, config_path: Path) -> None:
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("portable formal report content hash changed")
    if set(result) != EXPECTED_RESULT_KEYS:
        raise ValueError("portable formal report result shape changed")
    config_path = config_path.resolve()
    config = _load_config(config_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("decision") != "pass_portable_semantic_projection_118_controls"
        or result.get("claim_seals") != EXPECTED_CLAIM_SEALS
        or any(result.get("claim_seals", {}).values())
    ):
        raise ValueError("portable formal report boundary changed")
    _assert_portable(result)
    input_path = _inside(root, str(config["input_report"]))
    report = json.loads(input_path.read_text(encoding="utf-8"))
    cadabra_identity = result.get("backend_identity", {}).get("cadabra2", {})
    supplied_hashes = {
        "executable_sha256": cadabra_identity.get("executable_sha256"),
        "python_module_sha256": cadabra_identity.get("python_module_sha256"),
    }
    semantic_report, backend_identity = portable_projection(
        report, root, backend_hashes=supplied_hashes
    )
    if (
        result.get("semantic_report") != semantic_report
        or result.get("backend_identity") != backend_identity
        or result.get("source_report")
        != {
            "path": input_path.relative_to(root).as_posix(),
            "schema_version": report["schema_version"],
            "created_utc_excluded_from_semantic_projection": True,
            "semantic_sha256": _sha(semantic_report),
            "counts": report["counts"],
        }
        or result.get("portability")
        != {
            "absolute_windows_paths": 0,
            "absolute_wsl_paths": 0,
            "host_timestamps_in_semantic_projection": 0,
            "backend_paths_replaced_by_binary_hashes": True,
            "project_paths_replaced_by_repository_relative_paths": True,
            "same_semantics_reproduce_across_root_and_timestamp_changes": True,
        }
        or result.get("scope")
        != (
            "portable semantic projection of the registered 118 formal controls; the host-local "
            "execution receipt remains separate and no candidate, theory or observation pass is inferred"
        )
    ):
        raise ValueError("portable formal report semantic projection changed")
    bindings = result.get("source_bindings", {})
    expected_paths = {
        "config": config_path.resolve().relative_to(root.resolve()).as_posix(),
        "source": "src/sigma_theory_compiler/formal_controls_portable_report.py",
        "test": "tests/test_formal_controls_portable_report.py",
    }
    if set(bindings) != set(expected_paths):
        raise ValueError("portable formal report source binding set changed")
    for label, relative in expected_paths.items():
        binding = bindings[label]
        if (
            set(binding) != {"path", "file_sha256"}
            or binding["path"] != relative
            or binding["file_sha256"] != _file_sha(_inside(root, relative))
        ):
            raise ValueError("portable formal report source binding changed")
    del config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_report(args.config)
    config = _load_config(args.config.resolve())
    output = args.output or _inside(args.config.resolve().parents[1], config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
