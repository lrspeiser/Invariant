from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite
from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59
from sigma_theory_compiler import open_gravity_campaign_v1 as campaign

CONFIG_PATH = Path("configs/open_gravity_gp01_entropy_holdout_confirmation_v1.json")
CONFIG_CANONICAL_SHA256 = "9d78c31fd39a7082ab8608672275857383684ae5dbdf2047589a1afa07f42b94"


class GP01EntropyHoldoutError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload[key] = ""
    return _content_sha256(payload)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GP01EntropyHoldoutError(f"expected object: {path}")
    return value


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return "EXISTING_IDENTICAL"
        raise GP01EntropyHoldoutError(f"refusing to replace existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        if path.read_bytes() == encoded:
            return "EXISTING_IDENTICAL"
        raise GP01EntropyHoldoutError(f"artifact race changed output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(base, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if _content_sha256(config) != CONFIG_CANONICAL_SHA256:
        raise GP01EntropyHoldoutError("holdout config semantics changed")
    if (
        config.get("schema_version")
        != "invariant-open-gravity-gp01-entropy-holdout-confirmation-1.0"
        or config.get("status") != "FROZEN_BEFORE_HOLDOUT_RESPONSE_DECODE"
    ):
        raise GP01EntropyHoldoutError("unsupported holdout config")
    expected_clusters = ["A2029", "A3158", "A644", "RXC1825"]
    if list(config["scope"]["clusters"]) != expected_clusters:
        raise GP01EntropyHoldoutError("holdout cluster set changed")
    scope = config["scope"]
    for key in (
        "network_calls",
        "model_calls",
        "paid_calls",
        "parameter_tuning",
        "formula_repairs",
        "per_cluster_fits",
        "lensing_rows",
        "inferred_total_mass_rows",
        "group_rows",
    ):
        if int(scope[key]) != 0:
            raise GP01EntropyHoldoutError(f"forbidden scope changed: {key}")
    if config["fixed_formulas"]["elliptic"] != {
        "id": "GP01E-n1-A8-rho10-T10-q2-p2-L0",
        "n": 1,
        "A_max": 8.0,
        "rho_ratio": 10.0,
        "tide_ratio": 10.0,
        "q": 2,
        "tide_power": 2,
        "L_ratio": 0.0,
        "description": "Exact development-selected boundary cell; no parameter may change during this confirmation.",
    }:
        raise GP01EntropyHoldoutError("fixed elliptic formula changed")
    if config["fixed_formulas"]["equilibrium"]["n"] != 1:
        raise GP01EntropyHoldoutError("fixed equilibrium formula changed")
    if [row["cell_id"] for row in config["nuisance_scenarios"]] != [
        "XCOP-SOURCE-LOW",
        "XCOP-SOURCE-NOMINAL",
        "XCOP-SOURCE-HIGH",
    ]:
        raise GP01EntropyHoldoutError("nuisance scenario set changed")
    for relative, expected in config["scientific_freeze"]["dependencies"].items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise GP01EntropyHoldoutError(f"dependency changed: {relative}")


def _artifact_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    path = (root / str(config["paths"][key])).resolve()
    if root.resolve() not in path.parents:
        raise GP01EntropyHoldoutError(f"artifact path escaped repository: {key}")
    return path


def _expected_raw_paths(root: Path, config: Mapping[str, Any]) -> list[Path]:
    raw_root = (root / str(config["source_contract"]["root"])).resolve()
    rows: list[Path] = []
    for cluster in config["scope"]["clusters"]:
        for name in config["source_contract"]["expected_files"][cluster]:
            path = (raw_root / cluster / name).resolve()
            if raw_root not in path.parents or not path.is_file():
                raise GP01EntropyHoldoutError(f"missing or unsafe frozen input: {cluster}/{name}")
            rows.append(path)
    if len(rows) != int(config["source_contract"]["expected_file_count"]):
        raise GP01EntropyHoldoutError("frozen input file count changed")
    return rows


def build_preflight(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    raw_paths = _expected_raw_paths(base, config)
    receipt: dict[str, Any] = {
        "schema_version": "invariant-open-gravity-gp01-entropy-holdout-preflight-1.0",
        "experiment_id": config["experiment_id"],
        "status": "READY_FROZEN_ZERO_HOLDOUT_RESPONSE_DECODE",
        "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
        "clusters": list(config["scope"]["clusters"]),
        "fixed_formula_ids": [
            config["fixed_formulas"]["equilibrium"]["id"],
            config["fixed_formulas"]["elliptic"]["id"],
        ],
        "nuisance_scenarios": [row["cell_id"] for row in config["nuisance_scenarios"]],
        "published_entropy_values_kev_cm2": config["published_entropy_covariate"]["values_kev_cm2"],
        "raw_paths_verified_by_metadata_only": len(raw_paths),
        "raw_files_opened": 0,
        "response_rows_decoded": 0,
        "predictions_computed": 0,
        "scores_computed": 0,
        "parameter_tuning": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "preflight_content_sha256": "",
    }
    receipt["preflight_content_sha256"] = _self_hash(receipt, "preflight_content_sha256")
    return receipt


def write_preflight(root: Path | None = None) -> str:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    return _atomic_no_clobber(_artifact_path(base, config, "preflight"), build_preflight(base))


def check_preflight(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    path = _artifact_path(base, config, "preflight")
    stored = _read_json(path)
    expected = build_preflight(base)
    if stored != expected or stored["preflight_content_sha256"] != _self_hash(
        stored, "preflight_content_sha256"
    ):
        raise GP01EntropyHoldoutError("preflight receipt changed")
    return stored


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left = _rankdata(x)
    right = _rankdata(y)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        raise GP01EntropyHoldoutError("undefined Spearman correlation")
    return float(np.corrcoef(left, right)[0, 1])


def _exact_permutation_p(x: Sequence[float], y: Sequence[float], observed: float) -> float:
    correlations = [_spearman(x, permutation) for permutation in itertools.permutations(y)]
    return float(
        sum(abs(value) >= abs(observed) - 1.0e-15 for value in correlations) / len(correlations)
    )


def _access_intent(config: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    intent: dict[str, Any] = {
        "schema_version": "invariant-open-gravity-gp01-entropy-holdout-access-intent-1.0",
        "experiment_id": config["experiment_id"],
        "status": "INTENT_COMMITTED_BEFORE_HOLDOUT_RESPONSE_DECODE",
        "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
        "preflight_content_sha256": preflight["preflight_content_sha256"],
        "clusters": list(config["scope"]["clusters"]),
        "expected_raw_files": int(config["source_contract"]["expected_file_count"]),
        "fixed_formula_ids": [
            config["fixed_formulas"]["equilibrium"]["id"],
            config["fixed_formulas"]["elliptic"]["id"],
        ],
        "parameter_tuning": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "replay_allowed": False,
        "intent_content_sha256": "",
    }
    intent["intent_content_sha256"] = _self_hash(intent, "intent_content_sha256")
    return intent


def _score_cluster(
    packet: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    item59_config: Mapping[str, Any],
) -> dict[str, Any]:
    scaled, state, bundle, gbar = campaign._cluster_state_bundle(packet, scenario, item59_config)
    nuisance = campaign._cluster_nuisance(scenario)
    local_grid = campaign._gp01_l_factor(bundle, 1)
    local_factor = campaign._factor_on_radii(local_grid, bundle, state["radius_m"])
    elliptic_grid, diagnostics = campaign._gp01_elliptic_factor(
        bundle, config["fixed_formulas"]["elliptic"]
    )
    elliptic_factor = campaign._factor_on_radii(elliptic_grid, bundle, state["radius_m"])
    local_predictions = cluster_suite._predictions_from_acceleration(
        scaled, state, local_factor * gbar, nuisance, item59_config
    )
    elliptic_predictions = cluster_suite._predictions_from_acceleration(
        scaled, state, elliptic_factor * gbar, nuisance, item59_config
    )
    floor = float(config["scoring"]["minimum_fractional_error"])
    local_score = campaign._loss_rows(
        local_predictions, scaled["rows"], minimum_fractional_error=floor
    )
    elliptic_score = campaign._loss_rows(
        elliptic_predictions, scaled["rows"], minimum_fractional_error=floor
    )
    local_loss = float(local_score["loss"])
    elliptic_loss = float(elliptic_score["loss"])
    return {
        "cluster": packet["cluster"],
        "scenario_id": scenario["cell_id"],
        "row_count": len(scaled["rows"]),
        "equilibrium_loss": local_loss,
        "elliptic_loss": elliptic_loss,
        "elliptic_to_equilibrium_ratio": elliptic_loss / local_loss,
        "elliptic_minus_equilibrium": elliptic_loss - local_loss,
        "equilibrium_by_observable": local_score["by_observable"],
        "elliptic_by_observable": elliptic_score["by_observable"],
        "equilibrium_worst_row_id": local_score["worst_row_id"],
        "elliptic_worst_row_id": elliptic_score["worst_row_id"],
        "elliptic_operator_diagnostics": diagnostics,
    }


def execute(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    preflight = check_preflight(base)
    result_path = _artifact_path(base, config, "result")
    intent_path = _artifact_path(base, config, "access_intent")
    if result_path.exists() or intent_path.exists():
        raise GP01EntropyHoldoutError("holdout execution is one-shot and already started")
    _atomic_no_clobber(intent_path, _access_intent(config, preflight))

    item59_config = item59.load_config(base)
    packets = []
    for cluster in config["scope"]["clusters"]:
        packet = item59._parse_cluster(base, item59_config, str(cluster))
        clock._add_rows(packet, item59_config)
        packets.append(packet)

    rows = [
        _score_cluster(packet, scenario, config, item59_config)
        for scenario in config["nuisance_scenarios"]
        for packet in packets
    ]
    nominal = [row for row in rows if row["scenario_id"] == "XCOP-SOURCE-NOMINAL"]
    nominal.sort(key=lambda row: str(row["cluster"]))
    entropy = config["published_entropy_covariate"]["values_kev_cm2"]
    x = [float(entropy[row["cluster"]]) for row in nominal]
    ratios = [float(row["elliptic_to_equilibrium_ratio"]) for row in nominal]
    differences = [float(row["elliptic_minus_equilibrium"]) for row in nominal]
    ratio_rho = _spearman(x, ratios)
    difference_rho = _spearman(x, differences)
    ratio_p = _exact_permutation_p(x, ratios, ratio_rho)
    difference_p = _exact_permutation_p(x, differences, difference_rho)
    strong = math.isclose(ratio_rho, -1.0, abs_tol=1.0e-15)
    result: dict[str, Any] = {
        "schema_version": "invariant-open-gravity-gp01-entropy-holdout-result-1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "STRONG_DIRECTIONAL_REPLICATION"
            if strong
            else "DIRECTIONAL_REPLICATION"
            if ratio_rho < 0.0
            else "DIRECTION_NOT_REPLICATED"
        ),
        "config_canonical_sha256": CONFIG_CANONICAL_SHA256,
        "preflight_content_sha256": preflight["preflight_content_sha256"],
        "clusters": list(config["scope"]["clusters"]),
        "published_entropy_values_kev_cm2": entropy,
        "fixed_formula_ids": [
            config["fixed_formulas"]["equilibrium"]["id"],
            config["fixed_formulas"]["elliptic"]["id"],
        ],
        "cluster_scenario_results": rows,
        "nominal_analysis": {
            "spearman_entropy_vs_loss_ratio": ratio_rho,
            "exact_two_sided_permutation_p_ratio": ratio_p,
            "spearman_entropy_vs_loss_difference": difference_rho,
            "exact_two_sided_permutation_p_difference": difference_p,
            "directional_replication": ratio_rho < 0.0,
            "strong_monotonic_replication": strong,
            "elliptic_beats_equilibrium_count": sum(
                float(row["elliptic_loss"]) < float(row["equilibrium_loss"]) for row in nominal
            ),
            "cluster_count": len(nominal),
        },
        "access_accounting": {
            "raw_files_opened": int(config["source_contract"]["expected_file_count"]),
            "clusters_decoded": len(packets),
            "response_rows_decoded": sum(len(packet["rows"]) for packet in packets),
            "cluster_scenario_scores": len(rows) * 2,
            "parameter_tuning": 0,
            "formula_repairs": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "claim_ceiling": {
            "independent_confirmation": False,
            "publication_ready": False,
            "static_elliptic_family_pruned": False,
            "dynamic_history_descendants_pruned": False,
            "finding": "A fixed four-cluster directional holdout check only; interpret with the frozen eight-cluster development signal and physical theory work.",
        },
        "result_content_sha256": "",
    }
    result["result_content_sha256"] = _self_hash(result, "result_content_sha256")
    _atomic_no_clobber(result_path, result)
    return result


def check_result(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    check_preflight(base)
    intent = _read_json(_artifact_path(base, config, "access_intent"))
    result = _read_json(_artifact_path(base, config, "result"))
    if intent["intent_content_sha256"] != _self_hash(intent, "intent_content_sha256"):
        raise GP01EntropyHoldoutError("access intent changed")
    if result["result_content_sha256"] != _self_hash(result, "result_content_sha256"):
        raise GP01EntropyHoldoutError("result changed")
    if result["config_canonical_sha256"] != CONFIG_CANONICAL_SHA256:
        raise GP01EntropyHoldoutError("result config binding changed")
    if result["claim_ceiling"]["publication_ready"] is not False:
        raise GP01EntropyHoldoutError("result overclaim changed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-preflight")
    subparsers.add_parser("check-preflight")
    subparsers.add_parser("execute")
    subparsers.add_parser("check-result")
    args = parser.parse_args(argv)
    if args.command == "write-preflight":
        print(write_preflight())
    elif args.command == "check-preflight":
        print(check_preflight()["status"])
    elif args.command == "execute":
        print(execute()["status"])
    else:
        print(check_result()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
