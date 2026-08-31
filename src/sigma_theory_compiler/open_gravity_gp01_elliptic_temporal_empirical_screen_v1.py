"""Retrospective empirical-first screen of GP01 elliptic and temporal leads."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_gp01_elliptic_temporal_empirical_screen_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_gp01_elliptic_temporal_empirical_screen_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_gp01_elliptic_temporal_empirical_screen_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-gp01-elliptic-temporal-empirical-screen-v1/receipt.json"
)
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-gp01-elliptic-temporal-empirical-screen-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "1cdb29625b254299776c211d4305ec6ea40972822ba0c3bdd93cdc834a8da4fa"
_CONFIG_CONTENT_SHA256 = "1aa1817e07bc6425e5c338cfd98baf2c36ade45dd6c61aabdbf4287165cffbe9"
_IMPLEMENTATION_SEMANTIC_SHA256 = "47aecc38a29585fe18834bd67b3c845f5f9166bd8b6ccf8dbea2a412b7801d04"  # fmt: skip
_TEST_RAW_SHA256 = "0ce5ac26991c0def896321f86b22228c6b3446e8cf5a0e75c1e63b2e6e284175"
_SCHEMA = "invariant-open-gravity-gp01-elliptic-temporal-empirical-screen-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-gp01-elliptic-temporal-empirical-screen-receipt-1.0"
_PIN_RE = re.compile(r'^_IMPLEMENTATION_SEMANTIC_SHA256 = "[^"]+"(?:  # fmt: skip)?$', re.MULTILINE)


class EmpiricalScreenError(RuntimeError):
    """Raised whenever the frozen empirical screen fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmpiricalScreenError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_semantic(payload: bytes) -> str:
    text = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_PIN_RE.finditer(text))
    _require(len(matches) == 1, "implementation pin assignment changed")
    text = _PIN_RE.sub('_IMPLEMENTATION_SEMANTIC_SHA256 = "<SELF_PIN>"  # fmt: skip', text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalScreenError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EmpiricalScreenError("sealed result commit is unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "sealed_result_binding",
        "primary_source_metadata",
        "analysis_contract",
        "required_checks",
        "next_experiments",
        "access_accounting",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-gp01-elliptic-temporal-empirical-screen-v1",
        "package ID changed",
    )
    _require(
        config["status"] == "RETROSPECTIVE_DEVELOPMENT_SCREEN_NOT_CONFIRMATION",
        "status changed",
    )
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(len(config["primary_source_metadata"]) == 2, "primary-source count changed")
    _require(len(config["analysis_contract"]["objects"]) == 8, "object count changed")
    _require(len(config["required_checks"]) == 8, "check count changed")
    _require(config["analysis_contract"]["retrospective"] is True, "scope changed")
    _require(
        config["access_accounting"]["raw_scientific_response_files_read"] == 0,
        "raw-response access changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config() -> dict[str, Any]:
    config_path = (_ROOT / CONFIG_PATH).resolve()
    _require(config_path.is_relative_to(_ROOT), "config escaped repository")
    _require(file_sha256(config_path) == _CONFIG_RAW_SHA256, "config bytes changed")
    _require(
        _implementation_semantic((_ROOT / MODULE_PATH).read_bytes())
        == _IMPLEMENTATION_SEMANTIC_SHA256,
        "implementation semantic seal changed",
    )
    _require(file_sha256(_ROOT / TEST_PATH) == _TEST_RAW_SHA256, "test seal changed")
    config = _read_json(config_path, "empirical-screen config")
    validate_config(config)
    return config


def _load_sealed_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["sealed_result_binding"]
    committed = _git_show(str(binding["commit"]), str(binding["path"]))
    _require(
        hashlib.sha256(committed).hexdigest() == binding["sha256"],
        "committed result ledger changed",
    )
    path = (_ROOT / str(binding["path"])).resolve()
    _require(path.is_relative_to(_ROOT), "result ledger escaped repository")
    _require(file_sha256(path) == binding["sha256"], "working result ledger changed before read")
    return _read_json(path, "sealed result ledger")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        rank = (start + stop) / 2.0 + 1.0
        for offset in range(start, stop + 1):
            ranks[order[offset]] = rank
        start = stop + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    _require(denominator > 0.0, "rank correlation is degenerate")
    return numerator / denominator


def _exact_spearman(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    observed = _pearson(left_rank, right_rank)
    extreme = 0
    permutations = 0
    for permuted in itertools.permutations(right_rank):
        permutations += 1
        if abs(_pearson(left_rank, permuted)) >= abs(observed) - 1.0e-15:
            extreme += 1
    _require(permutations == math.factorial(8), "permutation count changed")
    return {
        "rho": observed,
        "exact_two_sided_p": extreme / permutations,
        "extreme_permutations": extreme,
        "permutations": permutations,
    }


def _nominal_objects(row: Mapping[str, Any]) -> dict[str, float]:
    scenarios = [
        item for item in row["scenario_results"] if item["scenario_id"] == "XCOP-SOURCE-NOMINAL"
    ]
    _require(len(scenarios) == 1 and scenarios[0]["valid"] is True, "nominal scenario changed")
    return {str(item["object"]): float(item["loss"]) for item in scenarios[0]["objects"]}


def _metadata_maps(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {row["id"]: row for row in config["primary_source_metadata"]}
    thermo = by_id["GHIRARDINI_2019_XCOP_THERMODYNAMIC"]["values"]
    morphology = by_id["DUPOURQUE_2023_XCOP_FLUCTUATIONS"]["values"]
    return thermo, morphology


def run_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    ledger = _load_sealed_ledger(config)
    rows = ledger["clusters"]
    elliptic = [
        row for row in rows if row["concept_id"] == "GP01-ELLIPTIC" and row["valid"] is True
    ]
    equilibrium = [row for row in rows if row["cell_id"] == "GP01L-n1"]
    _require(len(elliptic) == 1296 and len(equilibrium) == 1, "frozen GP01 rows changed")
    best = min(elliptic, key=lambda row: (float(row["robust_loss"]), str(row["cell_id"])))
    control = equilibrium[0]
    expected_best = "GP01E-n1-A8-rho10-T10-q2-p2-L0"
    _require(best["cell_id"] == expected_best, "best elliptic cell changed")
    objects = list(config["analysis_contract"]["objects"])
    best_losses = _nominal_objects(best)
    control_losses = _nominal_objects(control)
    _require(set(best_losses) == set(control_losses) == set(objects), "object ledger changed")
    thermo, morphology = _metadata_maps(config)
    object_rows = []
    for object_id in objects:
        ratio = best_losses[object_id] / control_losses[object_id]
        object_rows.append(
            {
                "object": object_id,
                "elliptic_nominal_loss": best_losses[object_id],
                "equilibrium_nominal_loss": control_losses[object_id],
                "elliptic_to_equilibrium_loss_ratio": ratio,
                "elliptic_minus_equilibrium_loss": (
                    best_losses[object_id] - control_losses[object_id]
                ),
                "K0_KEV_CM2": float(thermo[object_id]["K0_KEV_CM2"]),
                "cool_core": bool(thermo[object_id]["cool_core"]),
                **{key: float(value) for key, value in morphology[object_id].items()},
            }
        )
    ratios = [row["elliptic_to_equilibrium_loss_ratio"] for row in object_rows]
    differences = [row["elliptic_minus_equilibrium_loss"] for row in object_rows]
    correlations = {}
    proxy_values = {
        "CENTROID_SHIFT_X1E3": [row["CENTROID_SHIFT_X1E3"] for row in object_rows],
        "C_Z": [row["C_Z"] for row in object_rows],
        "LOG_K0_KEV_CM2": [math.log(row["K0_KEV_CM2"]) for row in object_rows],
    }
    for proxy, values in proxy_values.items():
        correlations[proxy] = {
            "loss_ratio": _exact_spearman(values, ratios),
            "loss_difference": _exact_spearman(values, differences),
        }
    cool = [row["elliptic_to_equilibrium_loss_ratio"] for row in object_rows if row["cool_core"]]
    noncool = [
        row["elliptic_to_equilibrium_loss_ratio"] for row in object_rows if not row["cool_core"]
    ]
    _require(len(cool) == 3 and len(noncool) == 5, "cool-core partition changed")
    observed_split = mean(cool) - mean(noncool)
    all_ratios = tuple(ratios)
    split_differences = []
    for indices in itertools.combinations(range(8), 3):
        selected = set(indices)
        split_differences.append(
            mean(all_ratios[index] for index in selected)
            - mean(all_ratios[index] for index in range(8) if index not in selected)
        )
    split_p = sum(abs(value) >= abs(observed_split) - 1.0e-15 for value in split_differences) / len(
        split_differences
    )
    cool_core_test = {
        "cool_core_mean_loss_ratio": mean(cool),
        "noncool_core_mean_loss_ratio": mean(noncool),
        "difference": observed_split,
        "exact_two_sided_p": split_p,
        "partitions": len(split_differences),
    }
    gates = {
        "SEALED_LEDGER_BOUND_BEFORE_READ": True,
        "EXACT_EIGHT_OBJECTS_AND_1296_ELLIPTIC_CELLS": len(object_rows) == 8
        and len(elliptic) == 1296,
        "GLOBAL_BEST_ELLIPTIC_CELL_REPRODUCED": best["cell_id"] == expected_best,
        "ELLIPTIC_VALID_BUT_WORSE_ON_EVERY_OBJECT": all(value > 1.0 for value in ratios),
        "BOUNDARY_OPTIMUM_REQUIRES_SEPARATE_EXTENSION": all(
            token in str(best["cell_id"])
            for token in ("-A8-", "-rho10-", "-T10-", "-q2-", "-p2-", "-L0")
        ),
        "DYNAMICAL_PROXY_SCREEN_REPRODUCED": correlations["LOG_K0_KEV_CM2"]["loss_ratio"][
            "exact_two_sided_p"
        ]
        < 0.05
        and cool_core_test["exact_two_sided_p"] < 0.05,
        "TEMPORAL_LAG_NOT_IDENTIFIED_FROM_STATIC_SNAPSHOTS": True,
        "NO_NEW_RAW_RESPONSE_OR_MODEL_ACCESS": all(
            config["access_accounting"][key] == 0
            for key in (
                "raw_scientific_response_files_read",
                "raw_scientific_response_rows_read",
                "new_scores_from_raw_rows",
                "package_build_network_calls",
                "model_calls",
                "paid_calls",
            )
        ),
    }
    _require(list(gates) == config["required_checks"], "check order changed")
    _require(all(gates.values()), "empirical screen check failed")
    return {
        "elliptic": {
            "valid_frozen_cells": len(elliptic),
            "best_cell": str(best["cell_id"]),
            "best_robust_loss": float(best["robust_loss"]),
            "equilibrium_control_cell": str(control["cell_id"]),
            "equilibrium_robust_loss": float(control["robust_loss"]),
            "robust_loss_ratio": float(best["robust_loss"]) / float(control["robust_loss"]),
            "beats_equilibrium_on_objects": sum(value < 1.0 for value in ratios),
            "object_count": len(object_rows),
            "boundary_extension_signal": True,
        },
        "object_rows": object_rows,
        "dynamical_proxy_correlations": correlations,
        "cool_core_test": cool_core_test,
        "temporal_interpretation": {
            "direct_telegraph_fit": False,
            "static_equilibrium_data_can_identify_relaxation_time": False,
            "retrospective_history_proxy_signal": True,
            "independent_history_data_required": True,
        },
        "checks": gates,
        "checks_passed": len(gates),
        "checks_failed": 0,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RETROSPECTIVE_ELLIPTIC_FIT_TEMPORAL_PROXY_LEAD_NOT_DIRECT_MEMORY_FIT",
        "bindings": {
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(_ROOT / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(_ROOT / TEST_PATH)},
            "sealed_result": config["sealed_result_binding"],
        },
        "screen": run_screen(config),
        "next_experiments": config["next_experiments"],
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    _require(payload == build_receipt(), "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _canonical_output_path() -> Path:
    _require(OUTPUT_PATH == _CANONICAL_OUTPUT_PATH, "output path changed")
    _require(
        str(load_config()["output_path"]) == _CANONICAL_OUTPUT_PATH.as_posix(),
        "output path changed",
    )
    path = (_ROOT / OUTPUT_PATH).resolve()
    _require(path.is_relative_to(_ROOT), "output escaped repository")
    return path


def write_receipt() -> str:
    path = _canonical_output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    path = _canonical_output_path()
    validate_receipt_payload(_read_json(path, "empirical-screen receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        screen = run_screen(load_config())
        print(
            json.dumps(
                {
                    "best_elliptic_cell": screen["elliptic"]["best_cell"],
                    "elliptic_beats_equilibrium_on_objects": screen["elliptic"][
                        "beats_equilibrium_on_objects"
                    ],
                    "log_k0_ratio_rho": screen["dynamical_proxy_correlations"]["LOG_K0_KEV_CM2"][
                        "loss_ratio"
                    ]["rho"],
                    "log_k0_ratio_exact_p": screen["dynamical_proxy_correlations"][
                        "LOG_K0_KEV_CM2"
                    ]["loss_ratio"]["exact_two_sided_p"],
                    "direct_telegraph_fit": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
