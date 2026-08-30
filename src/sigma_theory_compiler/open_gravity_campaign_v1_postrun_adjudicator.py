"""Target-free post-run adjudication for the open-gravity continuation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import open_gravity_campaign_v1 as base
from . import open_gravity_campaign_v1_successor as successor

CONFIG_PATH = Path("configs/open_gravity_campaign_v1_postrun_adjudicator.json")
OUTPUT_PATH = Path("runs/gravity/open-gravity-campaign-v1-successor/postrun-adjudication.json")
CONFIG_RAW_SHA256 = "923cc2e007203c807936960537bed8b6054068cf079c1d988b24f9ba28c92886"
CONFIG_CONTENT_SHA256 = "ec8c6ca5fcd70586b6fb87fb939cef7135f4e91af7100504073e3a4be97e6224"
RECEIPT_SCHEMA = "invariant-open-gravity-campaign-postrun-adjudication-1.0"


class OpenGravityPostrunAdjudicatorError(RuntimeError):
    """Raised when the sealed post-run adjudication contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenGravityPostrunAdjudicatorError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityPostrunAdjudicatorError("canonical JSON input is invalid") from error
    _require(base.canonical_bytes(payload) == raw, "canonical JSON encoding changed")
    base._require_finite_json(payload)
    return payload


def load_config(root: Path | None = None) -> dict[str, Any]:
    resolved = (root or _root()).resolve()
    path = (resolved / CONFIG_PATH).resolve()
    _require(path.is_relative_to(resolved), "config path escaped repository")
    _require(_sha256(path) == CONFIG_RAW_SHA256, "post-run config raw hash changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenGravityPostrunAdjudicatorError("post-run config is invalid JSON") from error
    base._require_finite_json(payload)
    _require(
        base.content_sha256(payload) == CONFIG_CONTENT_SHA256,
        "post-run config content changed",
    )
    validate_config(payload)
    return dict(payload)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        base.content_sha256(config) == CONFIG_CONTENT_SHA256,
        "post-run config semantic hash changed",
    )
    _require(
        set(config)
        == {
            "schema_version",
            "adjudicator_id",
            "purpose",
            "source_campaign",
            "diagnosis",
            "access_scope",
            "output_path",
            "claim_ceiling",
        },
        "post-run config key set changed",
    )
    _require(
        config.get("schema_version") == "invariant-open-gravity-campaign-postrun-adjudicator-1.0"
        and config.get("adjudicator_id") == "OPEN-GRAVITY-CAMPAIGN-v1-POSTRUN-ADJUDICATOR-1",
        "post-run identity changed",
    )
    _require(config.get("output_path") == OUTPUT_PATH.as_posix(), "output path changed")
    source = config.get("source_campaign")
    _require(isinstance(source, Mapping), "source campaign contract changed")
    _require(
        source.get("package_commit") == "ccd5eddb921b982208b3ebd9759ec7032f7dec13",
        "source campaign commit changed",
    )
    _require(
        source.get("result_raw_sha256")
        == "e6c9024dd3c3ba88cc3ae3731d3cca17669652b82cda9b153507320c44041874"
        and source.get("adjudication_raw_sha256")
        == "4ac55d8af2d915dd05385f0d7dc004097832590dcb63c447a6d66cca1ac47d9b"
        and source.get("access_intent_raw_sha256")
        == "5526979b353884e0f87ed21eb80a54d120e52ea467d70548ed70922c21d5b215",
        "source result binding changed",
    )
    diagnosis = config.get("diagnosis")
    _require(isinstance(diagnosis, Mapping), "diagnosis changed")
    _require(
        diagnosis.get("failed_gate") == "BYTE_EXACT_ARTIFACT_RECOMPUTATION"
        and float(diagnosis.get("absolute_tolerance", -1)) == 1e-12
        and float(diagnosis.get("relative_tolerance", -1)) == 1e-12
        and diagnosis.get("classification_changes_allowed") == 0
        and diagnosis.get("formula_changes_allowed") == 0
        and diagnosis.get("parameter_changes_allowed") == 0,
        "diagnosis or tolerance changed",
    )
    scope = config.get("access_scope")
    _require(isinstance(scope, Mapping), "access scope changed")
    for key in (
        "raw_scientific_source_files_opened",
        "raw_response_rows_opened",
        "formula_evaluations",
        "new_scores_computed",
        "selection_events",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(scope.get(key) == 0, f"nonzero post-run access scope: {key}")
    ceiling = config.get("claim_ceiling")
    _require(
        ceiling
        == {
            "byte_exact_raw_input_replay_established": False,
            "sealed_object_score_ledger_internally_reaggregated": True,
            "classification_invariance_required": True,
            "development_only": True,
            "confirmation_claim": False,
            "cross_scale_survivor_claim": False,
            "novel_theory_claim": False,
            "physical_time_redshift_capture_or_quantum_claim": False,
        },
        "claim ceiling changed",
    )


def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _verify_source_campaign(root: Path, config: Mapping[str, Any]) -> None:
    source = config["source_campaign"]
    commit = str(source["package_commit"])
    for relative, expected in source["package_files"].items():
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), "source package path escaped")
        _require(_sha256(path) == expected, f"source package file changed: {relative}")
        _require(
            hashlib.sha256(_git_blob(root, commit, relative)).hexdigest() == expected,
            f"source package commit binding changed: {relative}",
        )
    for path_key, hash_key in (
        ("access_intent_path", "access_intent_raw_sha256"),
        ("result_path", "result_raw_sha256"),
        ("adjudication_path", "adjudication_raw_sha256"),
    ):
        path = (root / str(source[path_key])).resolve()
        _require(path.is_relative_to(root), "source output path escaped")
        _require(_sha256(path) == source[hash_key], f"source output changed: {path_key}")


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "mean of empty ledger")
    return float(np.mean(np.asarray(list(values), dtype=np.float64)))


def _close(left: float, right: float, config: Mapping[str, Any]) -> bool:
    diagnosis = config["diagnosis"]
    return math.isclose(
        left,
        right,
        rel_tol=float(diagnosis["relative_tolerance"]),
        abs_tol=float(diagnosis["absolute_tolerance"]),
    )


def _validate_candidate_rows(
    rows: Any,
    domain: str,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(isinstance(rows, list), f"{domain} score ledger changed")
    expected = {
        str(cell["cell_id"]): str(candidate["candidate_id"])
        for candidate, cell in base._eligible_cells(manifest, domain)
    }
    _require(len(rows) == len(expected), f"{domain} cell count changed")
    _require(
        len({str(row.get("cell_id")) for row in rows}) == len(rows)
        and {str(row.get("cell_id")) for row in rows} == set(expected),
        f"{domain} cell identities changed",
    )
    scenarios = {str(row["cell_id"]) for row in base._scenario_rows(base.load_config(), domain)}
    object_key = "SPARC" if domain == "GALAXIES" else "XCOP"
    objects = set(map(str, context["source_predecessor"]["objects"][object_key]))
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    row_keys = {
        "anonymous_formula_id",
        "cell_id",
        "concept_id",
        "domain",
        "gate_failure_count",
        "gate_failures",
        "lane",
        "robust_loss",
        "scenario_results",
        "valid",
    }
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == row_keys, f"{domain} row schema")
        cell_id = str(row["cell_id"])
        _require(
            row["domain"] == domain and row["concept_id"] == expected[cell_id],
            f"{domain} cell link changed",
        )
        scenario_rows = row["scenario_results"]
        _require(
            isinstance(scenario_rows, list)
            and len(scenario_rows) == len(scenarios)
            and {str(item["scenario_id"]) for item in scenario_rows} == scenarios,
            f"{domain} scenario ledger changed",
        )
        failures = row["gate_failures"]
        _require(
            isinstance(failures, list) and row["gate_failure_count"] == len(failures),
            f"{domain} failure count changed",
        )
        if not row["valid"]:
            _require(row["robust_loss"] is None and failures, f"{domain} invalid row promoted")
            flattened_failures = []
            for scenario in scenario_rows:
                scenario_objects = scenario.get("objects")
                scenario_failures = scenario.get("gate_failures")
                object_ids = {str(item["object"]) for item in scenario_objects}
                failure_ids = {str(item["object"]) for item in scenario_failures}
                _require(
                    set(scenario) == {"scenario_id", "valid", "objects", "gate_failures"}
                    and scenario["valid"] is False
                    and isinstance(scenario_objects, list)
                    and isinstance(scenario_failures, list)
                    and scenario_failures
                    and not (object_ids & failure_ids)
                    and object_ids | failure_ids == objects,
                    f"{domain} invalid scenario changed",
                )
                flattened_failures.extend(scenario_failures)
            _require(
                failures == flattened_failures,
                f"{domain} invalid failure chronology changed",
            )
            invalid.append(dict(row))
            continue
        _require(not failures, f"{domain} valid row retained failures")
        means = []
        for scenario in scenario_rows:
            _require(scenario.get("valid") is True, f"{domain} valid scenario changed")
            object_rows = scenario.get("objects")
            _require(
                isinstance(object_rows, list)
                and len(object_rows) == len(objects)
                and len({str(item["object"]) for item in object_rows}) == len(object_rows)
                and {str(item["object"]) for item in object_rows} == objects,
                f"{domain} object ledger changed",
            )
            losses = [float(item["loss"]) for item in object_rows]
            reversals = [float(item["reversal_loss"]) for item in object_rows]
            _require(
                scenario.get("object_count") == len(objects)
                and _close(float(scenario["mean_loss"]), _mean(losses), config)
                and _close(float(scenario["reversal_mean_loss"]), _mean(reversals), config),
                f"{domain} scenario aggregate changed",
            )
            means.append(float(scenario["mean_loss"]))
        _require(
            _close(float(row["robust_loss"]), max(means), config),
            f"{domain} robust loss changed",
        )
        valid.append(dict(row))
    return valid, invalid


def _validate_comparators(
    summary: Any,
    domain: str,
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require(
        isinstance(summary, Mapping)
        and set(summary) == {"domain", "scenario_results"}
        and summary["domain"] == domain,
        f"{domain} comparator summary changed",
    )
    object_key = "SPARC" if domain == "GALAXIES" else "XCOP"
    objects = set(map(str, context["source_predecessor"]["objects"][object_key]))
    expected_scenarios = {
        str(row["cell_id"]) for row in base._scenario_rows(base.load_config(), domain)
    }
    seen: set[tuple[str, str]] = set()
    for row in summary["scenario_results"]:
        pair = (str(row["comparator_id"]), str(row["scenario_id"]))
        _require(pair not in seen and pair[1] in expected_scenarios, "comparator identity changed")
        seen.add(pair)
        object_rows = row["objects"]
        _require(
            len(object_rows) == len(objects)
            and {str(item["object"]) for item in object_rows} == objects,
            f"{domain} comparator object set changed",
        )
        _require(
            _close(
                float(row["mean_loss"]),
                _mean([float(item["loss"]) for item in object_rows]),
                config,
            ),
            f"{domain} comparator mean changed",
        )


def _stable_adjudicate_domain(
    domain: str,
    candidates: Sequence[Mapping[str, Any]],
    comparator_summary: Mapping[str, Any],
    context: Mapping[str, Any],
    campaign_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    strongest = base._strongest_comparator_by_scenario(comparator_summary)
    pilot_set, full_set = base._subset_names(context, campaign_config, domain)
    subsets = (sorted(pilot_set), sorted(full_set))
    minimum_support = (
        int(campaign_config["metrics"]["minimum_galaxy_support"])
        if domain == "GALAXIES"
        else int(campaign_config["metrics"]["minimum_cluster_support"])
    )
    threshold = float(campaign_config["metrics"]["minimum_meaningful_improvement"])
    rows = []
    for candidate in candidates:
        evidence = []
        supported: set[str] | None = None
        loo_min = math.inf
        worst_ratio = 0.0
        subgroup_ratio = 0.0
        for scenario in candidate["scenario_results"]:
            scenario_id = str(scenario["scenario_id"])
            control = strongest[scenario_id]
            control_map = {str(row["object"]): float(row["loss"]) for row in control["objects"]}
            candidate_map = {str(row["object"]): float(row["loss"]) for row in scenario["objects"]}
            names = sorted(candidate_map)
            _require(set(names) == set(control_map), "candidate/comparator objects changed")
            scenario_supported = {
                name
                for name in names
                if candidate_map[name]
                < (1.0 - threshold) * max(control_map[name], np.finfo(float).tiny)
            }
            supported = scenario_supported if supported is None else supported & scenario_supported
            control_mean = float(control["mean_loss"])
            candidate_mean = float(scenario["mean_loss"])
            improvement = (control_mean - candidate_mean) / max(control_mean, np.finfo(float).tiny)
            ratios = [
                candidate_map[name] / max(control_map[name], np.finfo(float).tiny) for name in names
            ]
            worst_ratio = max(worst_ratio, max(ratios))
            for removed in names:
                remaining = [name for name in names if name != removed]
                c_mean = _mean([candidate_map[name] for name in remaining])
                r_mean = _mean([control_map[name] for name in remaining])
                loo_min = min(loo_min, (r_mean - c_mean) / max(r_mean, np.finfo(float).tiny))
            stage_values = []
            for subset in subsets:
                c_mean = _mean([candidate_map[name] for name in subset])
                r_mean = _mean([control_map[name] for name in subset])
                subgroup_ratio = max(subgroup_ratio, c_mean / max(r_mean, np.finfo(float).tiny))
                stage_values.append((c_mean, r_mean))
            evidence.append(
                {
                    "scenario_id": scenario_id,
                    "strongest_comparator": control["comparator_id"],
                    "candidate_mean_loss": candidate_mean,
                    "comparator_mean_loss": control_mean,
                    "fractional_improvement": improvement,
                    "passes_two_percent": improvement >= threshold,
                    "pilot_stage": {
                        "candidate_mean_loss": stage_values[0][0],
                        "comparator_mean_loss": stage_values[0][1],
                        "formula_changes_after_stage": 0,
                        "partial_ranking_released_before_full": False,
                    },
                    "full_development_stage": {
                        "candidate_mean_loss": stage_values[1][0],
                        "comparator_mean_loss": stage_values[1][1],
                        "formula_version_unchanged_from_pilot": True,
                    },
                }
            )
        support = len(supported or set())
        gates = {
            "EVERY_NUISANCE_CASE": all(bool(row["passes_two_percent"]) for row in evidence),
            "OBJECT_BREADTH": support >= minimum_support,
            "LEAVE_ONE_OBJECT_OUT": loo_min
            >= float(campaign_config["metrics"]["leave_one_object_out_minimum_improvement"]),
            "WORST_OBJECT": worst_ratio
            <= float(campaign_config["metrics"]["maximum_worst_object_loss_ratio"]),
            "PILOT_FULL_SUBGROUP": subgroup_ratio
            <= float(campaign_config["metrics"]["maximum_subgroup_loss_ratio"]),
        }
        rows.append(
            {
                "cell_id": str(candidate["cell_id"]),
                "concept_id": str(candidate["concept_id"]),
                "domain": domain,
                "passes": all(gates.values()),
                "gates": gates,
                "support_count": support,
                "minimum_loo_improvement": loo_min,
                "worst_object_loss_ratio": worst_ratio,
                "maximum_subgroup_loss_ratio": subgroup_ratio,
                "scenario_evidence": evidence,
                "counterexample": min(
                    (
                        row
                        for scenario in candidate["scenario_results"]
                        for row in scenario["objects"]
                    ),
                    key=lambda row: (-float(row["loss"]), str(row["object"])),
                ),
            }
        )
    return rows


def _stable_adjudicate_resilient(
    domain: str,
    valid: Sequence[Mapping[str, Any]],
    invalid: Sequence[Mapping[str, Any]],
    comparators: Mapping[str, Any],
    context: Mapping[str, Any],
    campaign_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _stable_adjudicate_domain(domain, valid, comparators, context, campaign_config)
    for candidate in invalid:
        rows.append(
            {
                "cell_id": candidate["cell_id"],
                "concept_id": candidate["concept_id"],
                "domain": domain,
                "passes": False,
                "gates": {
                    "SOURCE_OPERATOR_VALID": False,
                    "EVERY_NUISANCE_CASE": False,
                    "OBJECT_BREADTH": False,
                    "LEAVE_ONE_OBJECT_OUT": False,
                    "WORST_OBJECT": False,
                    "PILOT_FULL_SUBGROUP": False,
                },
                "support_count": 0,
                "minimum_loo_improvement": None,
                "worst_object_loss_ratio": None,
                "maximum_subgroup_loss_ratio": None,
                "scenario_evidence": [],
                "counterexample": {
                    "failure_count": candidate["gate_failure_count"],
                    "first_failure": candidate["gate_failures"][0],
                    "all_failure_codes": sorted(
                        {row["failure_code"] for row in candidate["gate_failures"]}
                    ),
                },
            }
        )
    return sorted(rows, key=lambda row: str(row["cell_id"]))


def _allowed_float_delta(path: str) -> bool:
    path = path.removeprefix(".")
    if not path.startswith("global-cell-ledger.json."):
        return False
    if ".galaxy_adjudication." not in path and ".cluster_adjudication." not in path:
        return False
    return path.endswith(".maximum_subgroup_loss_ratio") or (
        ".scenario_evidence." in path
        and (".pilot_stage." in path or ".full_development_stage." in path)
        and path.endswith((".candidate_mean_loss", ".comparator_mean_loss"))
    )


def _compare_normalized(
    actual: Any,
    expected: Any,
    config: Mapping[str, Any],
    path: str,
    stats: dict[str, Any],
) -> None:
    numeric = (int, float)
    if (
        isinstance(actual, numeric)
        and not isinstance(actual, bool)
        and isinstance(expected, numeric)
        and not isinstance(expected, bool)
    ):
        left = float(actual)
        right = float(expected)
        _require(math.isfinite(left) and math.isfinite(right), f"nonfinite value: {path}")
        _require(_close(left, right, config), f"numeric mismatch exceeds band: {path}")
        if left != right:
            _require(_allowed_float_delta(path), f"unexpected normalized float path: {path}")
            absolute = abs(left - right)
            relative = absolute / max(abs(left), abs(right), 1.0)
            stats["changed_float_count"] += 1
            stats["max_absolute_delta"] = max(stats["max_absolute_delta"], absolute)
            stats["max_relative_delta"] = max(stats["max_relative_delta"], relative)
            stats["changed_path_sha256"].append(hashlib.sha256(path.encode()).hexdigest())
        return
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        _require(set(actual) == set(expected), f"mapping keys changed: {path}")
        for key in sorted(actual):
            _compare_normalized(actual[key], expected[key], config, f"{path}.{key}", stats)
        return
    if isinstance(actual, list) and isinstance(expected, list):
        _require(len(actual) == len(expected), f"list length changed: {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare_normalized(left, right, config, f"{path}.{index}", stats)
        return
    _require(actual == expected, f"non-numeric value changed: {path}")


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    resolved = (root or _root()).resolve()
    config = load_config(resolved)
    _verify_source_campaign(resolved, config)
    result = _read_json(resolved / config["source_campaign"]["result_path"])
    adjudication = _read_json(resolved / config["source_campaign"]["adjudication_path"])
    _require(
        result["result_content_sha256"] == config["source_campaign"]["result_content_sha256"]
        and adjudication["adjudication_content_sha256"]
        == config["source_campaign"]["adjudication_content_sha256"]
        and adjudication["artifact_index_sha256"]
        == config["source_campaign"]["artifact_index_sha256"],
        "source content seals changed",
    )
    manifest, context = base.build_manifest(resolved)
    artifacts = successor._load_artifacts(result, context)
    global_ledger = artifacts["global-cell-ledger.json"]
    comparators = artifacts["comparator-ledger.json"]
    campaign_config = base.load_config(resolved)
    valid_g, invalid_g = _validate_candidate_rows(
        global_ledger["galaxies"], "GALAXIES", manifest, context, config
    )
    valid_c, invalid_c = _validate_candidate_rows(
        global_ledger["clusters"], "CLUSTERS", manifest, context, config
    )
    _validate_comparators(comparators["GALAXIES"], "GALAXIES", context, config)
    _validate_comparators(comparators["CLUSTERS"], "CLUSTERS", context, config)
    stable_g = _stable_adjudicate_resilient(
        "GALAXIES",
        valid_g,
        invalid_g,
        comparators["GALAXIES"],
        context,
        campaign_config,
    )
    stable_c = _stable_adjudicate_resilient(
        "CLUSTERS",
        valid_c,
        invalid_c,
        comparators["CLUSTERS"],
        context,
        campaign_config,
    )
    stable_cross = successor._cross_resilient(stable_g, stable_c)
    dashboards = {
        relative.removeprefix("dashboards/").removesuffix(".json"): payload
        for relative, payload in artifacts.items()
        if relative.startswith("dashboards/")
    }
    stable_artifacts = base._artifact_payloads(
        manifest,
        dashboards,
        global_ledger["galaxies"],
        global_ledger["clusters"],
        stable_g,
        stable_c,
        stable_cross,
        comparators,
    )
    stable_artifacts["repair-ledger.json"] = artifacts["repair-ledger.json"]
    stable_artifacts["lay-summary.json"]["invalid_source_gate_cells"] = {
        "GALAXIES": len(invalid_g),
        "CLUSTERS": len(invalid_c),
    }
    stable_artifacts["lay-summary.json"]["continuation_note"] = artifacts["lay-summary.json"][
        "continuation_note"
    ]
    for row in stable_artifacts["counterexample-ledger.json"]:
        if "SOURCE_OPERATOR_VALID" in row["failed_gates"]:
            row["failure_class"] = "SOURCE_OPERATOR_GATE_FAILURE"
    stats: dict[str, Any] = {
        "changed_float_count": 0,
        "max_absolute_delta": 0.0,
        "max_relative_delta": 0.0,
        "changed_path_sha256": [],
    }
    _compare_normalized(artifacts, stable_artifacts, config, "", stats)
    _require(stats["changed_float_count"] > 0, "declared numerical-order defect not reproduced")
    _compare_normalized(
        result["cross_domain_adjudication"], stable_cross, config, "result.cross_domain", stats
    )
    stable_survivors = [row for row in stable_cross if row["cross_domain_pass"]]
    _require(result["cross_domain_survivors"] == stable_survivors, "survivors changed")
    stable_best = {
        "GALAXIES": min(valid_g, key=lambda row: (float(row["robust_loss"]), row["cell_id"]))[
            "cell_id"
        ],
        "CLUSTERS": min(valid_c, key=lambda row: (float(row["robust_loss"]), row["cell_id"]))[
            "cell_id"
        ],
    }
    _require(result["best_development_cells"] == stable_best, "best cells changed")
    classifications = {
        "published_galaxy_passes": sum(
            bool(row["passes"]) for row in global_ledger["galaxy_adjudication"]
        ),
        "stable_galaxy_passes": sum(bool(row["passes"]) for row in stable_g),
        "published_cluster_passes": sum(
            bool(row["passes"]) for row in global_ledger["cluster_adjudication"]
        ),
        "stable_cluster_passes": sum(bool(row["passes"]) for row in stable_c),
        "published_cross_domain_survivors": len(result["cross_domain_survivors"]),
        "stable_cross_domain_survivors": len(stable_survivors),
    }
    _require(
        classifications["published_galaxy_passes"] == classifications["stable_galaxy_passes"]
        and classifications["published_cluster_passes"] == classifications["stable_cluster_passes"]
        and classifications["published_cross_domain_survivors"]
        == classifications["stable_cross_domain_survivors"],
        "classification changed under stable aggregation",
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "adjudicator_id": config["adjudicator_id"],
        "status": "PASS_STABLE_REAGGREGATION_ZERO_SURVIVORS",
        "source_result_raw_sha256": config["source_campaign"]["result_raw_sha256"],
        "source_result_content_sha256": result["result_content_sha256"],
        "source_artifact_index_sha256": adjudication["artifact_index_sha256"],
        "normalized_artifacts_content_sha256": base.content_sha256(stable_artifacts),
        "normalized_cross_domain_content_sha256": base.content_sha256(stable_cross),
        "diagnosis": {
            "root_cause": config["diagnosis"]["root_cause"],
            "changed_float_count": stats["changed_float_count"],
            "max_absolute_delta": stats["max_absolute_delta"],
            "max_relative_delta": stats["max_relative_delta"],
            "changed_path_set_sha256": base.content_sha256(sorted(stats["changed_path_sha256"])),
            "all_changed_paths_allowlisted": True,
            "all_deltas_within_frozen_band": True,
        },
        "classifications": classifications,
        "counts": result["counts"],
        "best_development_cells": stable_best,
        "invalid_source_gate_cells": {
            "GALAXIES": len(invalid_g),
            "CLUSTERS": len(invalid_c),
        },
        "access_accounting": config["access_scope"],
        "claim_ceiling": config["claim_ceiling"],
        "scientific_conclusion": (
            "No frozen shared parameter cell passed every development gate. GP01L-n1 had the "
            "lowest robust loss in both domains but lost decisively to the strongest declared "
            "comparators and is not a cross-scale survivor."
        ),
        "receipt_content_sha256": "",
    }
    receipt["receipt_content_sha256"] = base._self_hash(receipt, "receipt_content_sha256")
    return base._jsonable(receipt)


def write_receipt() -> str:
    root = _root()
    receipt = build_receipt(root)
    path = (root / OUTPUT_PATH).resolve()
    _require(path.is_relative_to(root), "output path escaped")
    return base._atomic_no_clobber(path, base.canonical_bytes(receipt))


def check_receipt() -> dict[str, Any]:
    root = _root()
    path = (root / OUTPUT_PATH).resolve()
    _require(path == (root / OUTPUT_PATH).resolve(), "output path changed")
    stored = _read_json(path)
    expected = build_receipt(root)
    _require(stored == expected, "post-run receipt failed deterministic rebuild")
    return dict(stored)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        print("VALID" if check_receipt() else "INVALID")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "survivors": receipt["classifications"]["stable_cross_domain_survivors"],
                    "raw_source_opens": receipt["access_accounting"][
                        "raw_scientific_source_files_opened"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
