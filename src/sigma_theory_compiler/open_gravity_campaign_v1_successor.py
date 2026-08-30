"""Fail-retaining continuation of the frozen open-gravity development campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_campaign_v1 as base

CONFIG_PATH = Path("configs/open_gravity_campaign_v1_successor.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_campaign_v1_successor.py")
TEST_PATH = Path("tests/test_open_gravity_campaign_v1_successor.py")
OUTPUT_ROOT = Path("runs/gravity/open-gravity-campaign-v1-successor")
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight.json"
CONTINUATION_LEDGER_PATH = OUTPUT_ROOT / "continuation-ledger.json"
ACCESS_INTENT_PATH = OUTPUT_ROOT / "access-intent.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
ADJUDICATION_PATH = OUTPUT_ROOT / "adjudication.json"
FAILURE_PATH = OUTPUT_ROOT / "failure.json"
ARTIFACT_DIRECTORY = OUTPUT_ROOT / "artifacts"

PREFLIGHT_SCHEMA = "invariant-open-gravity-campaign-successor-preflight-1.0"
LEDGER_SCHEMA = "invariant-open-gravity-campaign-continuation-ledger-1.0"
INTENT_SCHEMA = "invariant-open-gravity-campaign-successor-access-intent-1.0"
RESULT_SCHEMA = "invariant-open-gravity-campaign-successor-result-1.0"
ADJUDICATION_SCHEMA = "invariant-open-gravity-campaign-successor-adjudication-1.0"
FAILURE_SCHEMA = "invariant-open-gravity-campaign-successor-failure-1.0"

EXPECTED_CONFIG_CONTENT_SHA256 = "ae33087195a2fb809ed81fda38be00fb61a863445930aacb0bf4286132662249"
EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256 = "da75fe5370f1dda9bae1f101ef58bb4b12810c6c205dc887dc950abc257f93df"  # fmt: skip
EXPECTED_TEST_FILE_SHA256 = "9eff2fcf9966b0e24b7eaa579f255d702d887aba1bc8a126ed59375aabe49357"
_PIN_RE = base.re.compile(
    r'^EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256 = "[^"]+"(?:  # fmt: skip)?$',
    base.re.MULTILINE,
)


class OpenGravityCampaignSuccessorError(RuntimeError):
    """Fail-closed continuation contract error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenGravityCampaignSuccessorError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_semantic(payload: bytes) -> str:
    text = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_PIN_RE.finditer(text))
    _require(len(matches) == 1, "successor implementation pin assignment changed")
    text = _PIN_RE.sub('EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256 = "<SELF_PIN>"  # fmt: skip', text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config() -> dict[str, Any]:
    root = _root()
    config = _read_json(root / CONFIG_PATH)
    validate_config(config)
    _require(
        _implementation_semantic((root / MODULE_PATH).read_bytes())
        == EXPECTED_IMPLEMENTATION_SEMANTIC_SHA256,
        "successor implementation semantic seal changed",
    )
    _require(
        _file_sha(root / TEST_PATH) == EXPECTED_TEST_FILE_SHA256, "successor test seal changed"
    )
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        base.content_sha256(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "successor config semantic seal changed",
    )
    _require(
        config.get("schema_version") == "invariant-open-gravity-campaign-successor-config-1.0",
        "successor config schema changed",
    )
    _require(
        config.get("continuation_id") == "OPEN-GRAVITY-CAMPAIGN-v1-CONTINUATION-1",
        "successor continuation ID changed",
    )
    _require(config.get("campaign_id") == "OPEN-GRAVITY-CAMPAIGN-v1", "campaign ID changed")
    _require(
        config.get("status") == "FROZEN_UNRUN_SUCCESSOR_AFTER_RETAINED_SOURCE_GATE_FAILURE",
        "successor status changed",
    )
    predecessor = config.get("predecessor")
    _require(
        predecessor
        == {
            "package_commit": "fffe5046cd3b4c7dfc781767bbcbe9d73aa5c2d4",
            "failure_commit": "2af93e40cd56888b9bc576332972a4b69d1fe645",
            "config_path": "configs/open_gravity_campaign_v1.json",
            "config_sha256": ("bc795d5fc57f2b951eeb0762e43778a7ccf81f45e30e3a4cffd275a073f0e023"),
            "module_path": "src/sigma_theory_compiler/open_gravity_campaign_v1.py",
            "module_sha256": ("1838483699cac74e79cd3788cc0f4d5317fd9ba2fa7d1a4fd379ea4a4f8276cc"),
            "test_path": "tests/test_open_gravity_campaign_v1.py",
            "test_sha256": ("5f6cd7b1cdc983d93874f0ba89904b4f3e140f39ca3e8db218405b92d15112c3"),
            "manifest_path": "runs/gravity/open-gravity-campaign-v1/manifest.json",
            "manifest_sha256": "be731f28def324f940c5cb35b15a110bed9354be28ff9e78da55ed39044fcf74",
            "manifest_content_sha256": (
                "4ea98321ef6b5893d4cdf14fd80df7caa7d390a3788f1ed4ffef3422717a18da"
            ),
            "terminal_ledger_path": ("runs/gravity/open-gravity-campaign-v1/terminal-ledger.json"),
            "terminal_ledger_sha256": (
                "f9ffc431f2fa7e42bd853ea64a7e9df06027ccc55c242000447e8251e819c434"
            ),
            "preflight_path": "runs/gravity/open-gravity-campaign-v1/preflight.json",
            "preflight_sha256": (
                "e202456c57634327842feaec8af5d295565b346b4f43f85aa13c153ac46e627c"
            ),
            "access_intent_path": "runs/gravity/open-gravity-campaign-v1/access-intent.json",
            "access_intent_sha256": (
                "d7ba46749350c63f3b958b2f2e25ffaf6b2e8a04ce30fd90bc6f6e2523300f8f"
            ),
            "failure_path": "runs/gravity/open-gravity-campaign-v1/failure.json",
            "failure_sha256": ("3839b8aa75e32ac92fe2d6792f06d9058793d5e815f3a614db0e9569e3e4883c"),
            "failure_status": "TERMINAL_FAILURE_SUCCESSOR_REQUIRED",
            "failure_error_code": "INTERNAL_FAILURE",
            "result_published": False,
            "selection_published": False,
        },
        "predecessor failure binding changed",
    )
    semantics = config.get("continuation_semantics", {})
    _require(
        semantics.get("same_frozen_manifest") is True
        and semantics.get("same_407_candidates") is True
        and semantics.get("same_2486_parameter_cells") is True
        and semantics.get("response_access_attempt_ordinal") == 2
        and semantics.get("completed_campaigns_before_successor") == 0
        and semantics.get("second_formula_campaign") is False
        and semantics.get("automatic_third_attempt") is False,
        "continuation semantics changed",
    )
    for key in (
        "post_freeze_formula_changes",
        "post_freeze_parameter_changes",
        "post_freeze_threshold_changes",
        "post_freeze_source_split_changes",
    ):
        _require(semantics.get(key) == 0, f"forbidden continuation change: {key}")
    policy = config.get("failure_retention_policy", {})
    _require(
        policy
        == {
            "caught_candidate_failure_classes": [
                "OPEN_GRAVITY_STATIC_RADIAL_ADAPTER_ERROR",
                "OPEN_GRAVITY_CAMPAIGN_SCIENTIFIC_GATE_ERROR",
                "FLOATING_POINT_ERROR",
                "OVERFLOW_ERROR",
                "ZERO_DIVISION_ERROR",
            ],
            "evaluate_remaining_objects_and_scenarios_after_cell_failure": True,
            "invalid_cell_has_pseudo_loss": False,
            "invalid_cell_can_rank": False,
            "invalid_cell_can_pass_adjudication": False,
            "invalid_cell_failure_messages_retained": False,
            "invalid_cell_exception_class_names_retained": False,
            "unexpected_implementation_errors_remain_terminal": True,
        },
        "failure retention policy changed",
    )
    scope = config.get("scope", {})
    _require(
        scope.get("galaxy_cells") == 179
        and scope.get("cluster_cells") == 1669
        and scope.get("sparc_objects") == 139
        and scope.get("sparc_rows_scored") == 2720
        and scope.get("xcop_response_rows_scored") == 184,
        "successor scientific scope changed",
    )
    for key in (
        "confirmation_rows",
        "independent_rows",
        "group_rows",
        "lensing_rows",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(scope.get(key) == 0, f"forbidden successor scope: {key}")
    _require(
        config.get("accounting")
        == {
            "predecessor_local_payload_read_operations": 31,
            "successor_execute_local_payload_read_operations": 31,
            "successor_check_result_local_payload_read_operations": 31,
            "cumulative_local_payload_read_operations_after_check": 93,
            "predecessor_partial_cell_scores": "UNKNOWN_NONZERO_NO_RESULT_PUBLISHED",
            "conservative_predecessor_planned_cell_charge": 1848,
            "successor_planned_cell_charge": 1848,
            "cumulative_conservative_cell_charge": 3696,
            "nominal_global_discovery_p_value_allowed": False,
        },
        "successor accounting changed",
    )
    _require(
        config.get("output_paths")
        == {
            "preflight": PREFLIGHT_PATH.as_posix(),
            "continuation_ledger": CONTINUATION_LEDGER_PATH.as_posix(),
            "access_intent": ACCESS_INTENT_PATH.as_posix(),
            "result": RESULT_PATH.as_posix(),
            "adjudication": ADJUDICATION_PATH.as_posix(),
            "failure": FAILURE_PATH.as_posix(),
            "artifact_directory": ARTIFACT_DIRECTORY.as_posix(),
        },
        "successor output paths changed",
    )
    ceiling = config.get("claim_ceiling", {})
    _require(ceiling.get("maximum_label") == "DEVELOPMENT_SIGNAL", "claim ceiling changed")
    _require(
        all(value is False for key, value in ceiling.items() if key.endswith("_claim")),
        "successor scientific claim promoted",
    )


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def verify_predecessor(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["predecessor"]
    rows = []
    for path_key, sha_key in (
        ("config_path", "config_sha256"),
        ("module_path", "module_sha256"),
        ("test_path", "test_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("terminal_ledger_path", "terminal_ledger_sha256"),
        ("preflight_path", "preflight_sha256"),
    ):
        relative = str(predecessor[path_key])
        expected = str(predecessor[sha_key])
        _require(_file_sha(root / relative) == expected, f"predecessor changed: {relative}")
        committed = hashlib.sha256(
            _git_show(root, str(predecessor["package_commit"]), relative)
        ).hexdigest()
        _require(committed == expected, f"package commit binding changed: {relative}")
        rows.append({"path": relative, "sha256": expected})
    for path_key, sha_key in (
        ("access_intent_path", "access_intent_sha256"),
        ("failure_path", "failure_sha256"),
    ):
        relative = str(predecessor[path_key])
        expected = str(predecessor[sha_key])
        _require(_file_sha(root / relative) == expected, f"predecessor changed: {relative}")
        committed = hashlib.sha256(
            _git_show(root, str(predecessor["failure_commit"]), relative)
        ).hexdigest()
        _require(committed == expected, f"predecessor commit binding changed: {relative}")
        rows.append({"path": relative, "sha256": expected})
    failure = _read_json(root / str(predecessor["failure_path"]))
    _require(
        failure.get("status") == predecessor["failure_status"]
        and failure.get("error_code") == predecessor["failure_error_code"]
        and failure.get("result_exists") is False
        and failure.get("adjudication_exists") is False
        and failure.get("replay_allowed") is False,
        "predecessor failure semantics changed",
    )
    _require(not (root / base.RESULT_PATH).exists(), "predecessor result unexpectedly exists")
    return {"verified": rows, "failure_content_sha256": failure["failure_content_sha256"]}


def build_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "continuation_id": config["continuation_id"],
        "campaign_id": config["campaign_id"],
        "session_id": config["session_id"],
        "same_manifest_content_sha256": config["predecessor"]["manifest_content_sha256"],
        "response_access_attempt_ordinal": 2,
        "formula_campaign_ordinal": 1,
        "completed_campaigns_before_continuation": 0,
        "predecessor_failure_retained": True,
        "predecessor_replay_allowed": False,
        "continuation_is_automatic_third_attempt": False,
        "continuation_ledger_content_sha256": "",
    }
    ledger["continuation_ledger_content_sha256"] = base._self_hash(
        ledger, "continuation_ledger_content_sha256"
    )
    return ledger


def build_preflight() -> dict[str, Any]:
    root = _root()
    config = load_config()
    predecessor = verify_predecessor(root, config)
    manifest, _context = base.build_manifest(root)
    _require(
        _file_sha(root / base.MANIFEST_PATH) == config["predecessor"]["manifest_sha256"],
        "live manifest changed",
    )
    ledger = build_ledger(config)
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "continuation_id": config["continuation_id"],
        "decision": "READY_FOR_ONE_FAIL_RETAINING_CONTINUATION_AFTER_COMMIT",
        "status": config["status"],
        "package": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": _file_sha(root / CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": _file_sha(root / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": _file_sha(root / TEST_PATH)},
        },
        "predecessor": predecessor,
        "manifest": {
            "path": base.MANIFEST_PATH.as_posix(),
            "sha256": config["predecessor"]["manifest_sha256"],
            "content_sha256": manifest["manifest_content_sha256"],
            "candidate_count": 407,
            "parameter_cell_count": 2486,
        },
        "continuation_ledger": {
            "path": CONTINUATION_LEDGER_PATH.as_posix(),
            "sha256": hashlib.sha256(base.canonical_bytes(ledger)).hexdigest(),
            "content_sha256": ledger["continuation_ledger_content_sha256"],
        },
        "failure_policy": copy.deepcopy(config["failure_retention_policy"]),
        "accounting": copy.deepcopy(config["accounting"]),
        "zero_new_response_access_at_freeze": copy.deepcopy(base.ZERO_ACCESS),
        "claim_ceiling": copy.deepcopy(config["claim_ceiling"]),
        "preflight_content_sha256": "",
    }
    preflight["preflight_content_sha256"] = base._self_hash(preflight, "preflight_content_sha256")
    return preflight


def write_preflight() -> dict[str, str]:
    root = _root()
    config = load_config()
    ledger = build_ledger(config)
    preflight = build_preflight()
    statuses = {
        "continuation_ledger": base._atomic_no_clobber(
            root / CONTINUATION_LEDGER_PATH, base.canonical_bytes(ledger)
        ),
        "preflight": base._atomic_no_clobber(
            root / PREFLIGHT_PATH, base.canonical_bytes(preflight)
        ),
    }
    _require(
        all(value != "EXISTING_DIFFERENT" for value in statuses.values()),
        "successor preflight no-clobber refusal",
    )
    return statuses


def check_preflight() -> dict[str, Any]:
    root = _root()
    config = load_config()
    expected_ledger = build_ledger(config)
    expected_preflight = build_preflight()
    _require(_read_json(root / CONTINUATION_LEDGER_PATH) == expected_ledger, "ledger changed")
    _require(_read_json(root / PREFLIGHT_PATH) == expected_preflight, "preflight changed")
    for path in (ACCESS_INTENT_PATH, RESULT_PATH, ADJUDICATION_PATH, FAILURE_PATH):
        _require(not (root / path).exists(), f"successor production artifact exists: {path}")
    _require(not (root / ARTIFACT_DIRECTORY).exists(), "successor artifact directory exists")
    return expected_preflight


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_committed_preflight(root: Path) -> str:
    for relative in (CONFIG_PATH, MODULE_PATH, TEST_PATH, PREFLIGHT_PATH, CONTINUATION_LEDGER_PATH):
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative.as_posix()],
            cwd=root,
            check=False,
            capture_output=True,
        )
        _require(listed.returncode == 0, f"successor is not committed: {relative}")
        _require(
            (root / relative).read_bytes() == _git_show(root, "HEAD", relative.as_posix()),
            f"successor differs from HEAD: {relative}",
        )
    return _git_head(root)


def _failure_code(error: BaseException) -> str | None:
    if isinstance(error, base.adapter.OpenGravityStaticRadialAdapterError):
        return "OPEN_GRAVITY_STATIC_RADIAL_ADAPTER_ERROR"
    if isinstance(error, base.OpenGravityCampaignError):
        return "OPEN_GRAVITY_CAMPAIGN_SCIENTIFIC_GATE_ERROR"
    if isinstance(error, FloatingPointError):
        return "FLOATING_POINT_ERROR"
    if isinstance(error, OverflowError):
        return "OVERFLOW_ERROR"
    if isinstance(error, ZeroDivisionError):
        return "ZERO_DIVISION_ERROR"
    return None


def _failure_row(error: BaseException, scenario_id: str, object_name: str) -> dict[str, Any]:
    code = _failure_code(error)
    if code is None:
        raise error
    return {
        "scenario_id": scenario_id,
        "object": object_name,
        "failure_code": code,
        "raw_message_retained": False,
        "raw_exception_class_retained": False,
    }


def _score_sparc_resilient(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    galaxies: Sequence[Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    concepts = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    source_cache = {
        (galaxy.name, str(scenario["cell_id"])): base._sparc_source(galaxy, scenario)
        for scenario in scenarios
        for galaxy in galaxies
    }
    results = []
    for candidate, cell in base._eligible_cells(manifest, "GALAXIES"):
        scenario_rows = []
        cell_failures = []
        for scenario in scenarios:
            scenario_id = str(scenario["cell_id"])
            object_rows = []
            failures = []
            for galaxy in galaxies:
                source = source_cache[(galaxy.name, scenario_id)]
                try:
                    grid_factor, diagnostics = base._factor_for_cell(
                        candidate, cell, concepts, source["bundle"], "SPARC"
                    )
                    factor = base._factor_on_radii(
                        grid_factor, source["bundle"], source["radius_m"]
                    )
                    reverse = base._factor_on_radii(
                        np.asarray(grid_factor)[::-1], source["bundle"], source["radius_m"]
                    )
                    score = base._sparc_score(np.sqrt(factor * np.asarray(source["vbar2"])), source)
                    reverse_score = base._sparc_score(
                        np.sqrt(reverse * np.asarray(source["vbar2"])), source
                    )
                    object_rows.append(
                        {
                            "object": galaxy.name,
                            "loss": float(score["loss"]),
                            "reversal_loss": float(reverse_score["loss"]),
                            "row_count": int(score["row_count"]),
                            "worst_radius": float(score["worst_radius"]),
                            "worst_standardized_square": float(score["worst_standardized_square"]),
                            "operator_diagnostics": diagnostics,
                        }
                    )
                except (
                    base.adapter.OpenGravityStaticRadialAdapterError,
                    base.OpenGravityCampaignError,
                    FloatingPointError,
                    OverflowError,
                    ZeroDivisionError,
                ) as error:
                    failure = _failure_row(error, scenario_id, galaxy.name)
                    failures.append(failure)
                    cell_failures.append(failure)
            if failures:
                scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "valid": False,
                        "objects": object_rows,
                        "gate_failures": failures,
                    }
                )
            else:
                summary = base._aggregate_object_scores(object_rows)
                summary.update(
                    {
                        "scenario_id": scenario_id,
                        "valid": True,
                        "reversal_mean_loss": float(
                            np.mean([row["reversal_loss"] for row in object_rows])
                        ),
                        "objects": object_rows,
                        "gate_failures": [],
                    }
                )
                scenario_rows.append(summary)
        valid = not cell_failures
        results.append(
            {
                "cell_id": str(cell["cell_id"]),
                "concept_id": str(candidate["candidate_id"]),
                "anonymous_formula_id": str(candidate["anonymous_formula_id"]),
                "lane": str(candidate["lane"]),
                "domain": "GALAXIES",
                "valid": valid,
                "robust_loss": (
                    max(float(row["mean_loss"]) for row in scenario_rows) if valid else None
                ),
                "gate_failure_count": len(cell_failures),
                "gate_failures": cell_failures,
                "scenario_results": scenario_rows,
            }
        )
    return results


def _score_xcop_resilient(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    item59_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite

    concepts = {str(row["concept_id"]): row for row in context["packet"]["twell_rows"]}
    source_cache = {
        (str(packet["cluster"]), str(scenario["cell_id"])): base._cluster_state_bundle(
            packet, scenario, item59_config
        )
        for scenario in scenarios
        for packet in packets
    }
    results = []
    for candidate, cell in base._eligible_cells(manifest, "CLUSTERS"):
        scenario_rows = []
        cell_failures = []
        for scenario in scenarios:
            scenario_id = str(scenario["cell_id"])
            nuisance = base._cluster_nuisance(scenario)
            object_rows = []
            failures = []
            for packet in packets:
                cluster = str(packet["cluster"])
                scaled, state, bundle, gbar = source_cache[(cluster, scenario_id)]
                try:
                    grid_factor, diagnostics = base._factor_for_cell(
                        candidate, cell, concepts, bundle, "XCOP_SPHERICAL"
                    )
                    factor = base._factor_on_radii(grid_factor, bundle, state["radius_m"])
                    reverse = base._factor_on_radii(
                        np.asarray(grid_factor)[::-1], bundle, state["radius_m"]
                    )
                    prediction = cluster_suite._predictions_from_acceleration(
                        scaled, state, factor * gbar, nuisance, item59_config
                    )
                    reverse_prediction = cluster_suite._predictions_from_acceleration(
                        scaled, state, reverse * gbar, nuisance, item59_config
                    )
                    score = base._loss_rows(
                        prediction, scaled["rows"], minimum_fractional_error=0.05
                    )
                    reverse_score = base._loss_rows(
                        reverse_prediction, scaled["rows"], minimum_fractional_error=0.05
                    )
                    object_rows.append(
                        {
                            "object": cluster,
                            "loss": float(score["loss"]),
                            "reversal_loss": float(reverse_score["loss"]),
                            "by_observable": score["by_observable"],
                            "row_count": int(score["row_count"]),
                            "worst_radius": float(score["worst_radius"]),
                            "worst_standardized_square": float(score["worst_standardized_square"]),
                            "operator_diagnostics": diagnostics,
                        }
                    )
                except (
                    base.adapter.OpenGravityStaticRadialAdapterError,
                    base.OpenGravityCampaignError,
                    FloatingPointError,
                    OverflowError,
                    ZeroDivisionError,
                ) as error:
                    failure = _failure_row(error, scenario_id, cluster)
                    failures.append(failure)
                    cell_failures.append(failure)
            if failures:
                scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "valid": False,
                        "objects": object_rows,
                        "gate_failures": failures,
                    }
                )
            else:
                summary = base._aggregate_object_scores(object_rows)
                summary.update(
                    {
                        "scenario_id": scenario_id,
                        "valid": True,
                        "reversal_mean_loss": float(
                            np.mean([row["reversal_loss"] for row in object_rows])
                        ),
                        "objects": object_rows,
                        "gate_failures": [],
                    }
                )
                scenario_rows.append(summary)
        valid = not cell_failures
        results.append(
            {
                "cell_id": str(cell["cell_id"]),
                "concept_id": str(candidate["candidate_id"]),
                "anonymous_formula_id": str(candidate["anonymous_formula_id"]),
                "lane": str(candidate["lane"]),
                "domain": "CLUSTERS",
                "valid": valid,
                "robust_loss": (
                    max(float(row["mean_loss"]) for row in scenario_rows) if valid else None
                ),
                "gate_failure_count": len(cell_failures),
                "gate_failures": cell_failures,
                "scenario_results": scenario_rows,
            }
        )
    return results


def _adjudicate_resilient(
    domain: str,
    results: Sequence[Mapping[str, Any]],
    comparators: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    valid = [row for row in results if row["valid"]]
    rows = base._adjudicate_domain(domain, valid, comparators, context, config) if valid else []
    for candidate in results:
        if candidate["valid"]:
            continue
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


def _cross_resilient(
    galaxies: Sequence[Mapping[str, Any]], clusters: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    valid_g = [row for row in galaxies if row["scenario_evidence"]]
    valid_c = [row for row in clusters if row["scenario_evidence"]]
    valid_cross = {row["cell_id"]: row for row in base._cross_domain_adjudication(valid_g, valid_c)}
    gmap = {str(row["cell_id"]): row for row in galaxies}
    cmap = {str(row["cell_id"]): row for row in clusters}
    rows = []
    for cell_id in sorted(set(gmap) & set(cmap)):
        if cell_id in valid_cross:
            rows.append(valid_cross[cell_id])
            continue
        galaxy = gmap[cell_id]
        cluster = cmap[cell_id]
        rows.append(
            {
                "cell_id": cell_id,
                "concept_id": galaxy["concept_id"],
                "galaxies_pass": False,
                "clusters_pass": False,
                "cross_domain_pass": False,
                "combined_worst_fractional_improvement": None,
                "invalid_domains": [
                    domain
                    for domain, row in (("GALAXIES", galaxy), ("CLUSTERS", cluster))
                    if not row["scenario_evidence"]
                ],
            }
        )
    return rows


def _compute() -> dict[str, Any]:
    root = _root()
    config = load_config()
    manifest, context = base.build_manifest(root)
    base_config = base.load_config(root)
    clock_config = base._verify_scientific_input_contracts(root, base_config)
    galaxies, sparc_provenance = base._load_sparc_responses(root, context, base_config)
    packets, _files, clock_config, item59_config = base._load_xcop_responses(
        root, base_config, clock_config
    )
    galaxy_scenarios = base._scenario_rows(base_config, "GALAXIES")
    cluster_scenarios = base._scenario_rows(base_config, "CLUSTERS")
    galaxy_comparator_rows, _ = base._score_sparc_comparators(galaxies, galaxy_scenarios)
    cluster_comparator_rows, _ = base._score_xcop_comparators(
        packets, cluster_scenarios, clock_config, item59_config
    )
    galaxy_results = _score_sparc_resilient(manifest, context, galaxies, galaxy_scenarios)
    cluster_results = _score_xcop_resilient(
        manifest, context, packets, cluster_scenarios, item59_config
    )
    valid_galaxies = [row for row in galaxy_results if row["valid"]]
    valid_clusters = [row for row in cluster_results if row["valid"]]
    _require(valid_galaxies and valid_clusters, "no valid cells remain for a domain")
    galaxy_comparators = base._comparator_summary(galaxy_comparator_rows, "GALAXIES")
    cluster_comparators = base._comparator_summary(cluster_comparator_rows, "CLUSTERS")
    galaxy_adjudication = _adjudicate_resilient(
        "GALAXIES", galaxy_results, galaxy_comparators, context, base_config
    )
    cluster_adjudication = _adjudicate_resilient(
        "CLUSTERS", cluster_results, cluster_comparators, context, base_config
    )
    cross = _cross_resilient(galaxy_adjudication, cluster_adjudication)
    dashboards = base._build_dashboards(
        manifest,
        context,
        galaxies,
        packets,
        valid_galaxies,
        valid_clusters,
        galaxy_comparator_rows,
        cluster_comparator_rows,
        base_config,
        item59_config,
    )
    comparator_summaries = {
        "GALAXIES": galaxy_comparators,
        "CLUSTERS": cluster_comparators,
        "declared_source_or_solver_blocked": [
            row for row in base_config["comparators"] if "BLOCKED" in str(row["status"])
        ],
    }
    artifacts = base._artifact_payloads(
        manifest,
        dashboards,
        galaxy_results,
        cluster_results,
        galaxy_adjudication,
        cluster_adjudication,
        cross,
        comparator_summaries,
    )
    artifacts["repair-ledger.json"] = {
        "post_freeze_formula_repairs": [],
        "post_freeze_parameter_repairs": [],
        "execution_contract_successor": config["continuation_id"],
        "predecessor_failure_retained": True,
        "failure_policy": config["failure_retention_policy"],
        "future_formula_repairs_destination": base.registry.IDEA_RESERVOIR_ID,
    }
    invalid_g = [row for row in galaxy_results if not row["valid"]]
    invalid_c = [row for row in cluster_results if not row["valid"]]
    artifacts["lay-summary.json"]["invalid_source_gate_cells"] = {
        "GALAXIES": len(invalid_g),
        "CLUSTERS": len(invalid_c),
    }
    artifacts["lay-summary.json"]["continuation_note"] = (
        "Cells that failed a frozen source/operator gate were retained as invalid, given no "
        "pseudo-loss, excluded from ranking, and did not abort the remaining cells."
    )
    for row in artifacts["counterexample-ledger.json"]:
        if "SOURCE_OPERATOR_VALID" in row["failed_gates"]:
            row["failure_class"] = "SOURCE_OPERATOR_GATE_FAILURE"
    _require(len(artifacts) == 156, "successor artifact count changed")
    best_g = min(valid_galaxies, key=lambda row: (float(row["robust_loss"]), row["cell_id"]))
    best_c = min(valid_clusters, key=lambda row: (float(row["robust_loss"]), row["cell_id"]))
    counts = {
        "live_candidates": 407,
        "parameter_cells": 2486,
        "galaxy_cells_planned": 179,
        "cluster_cells_planned": 1669,
        "galaxy_cells_valid": len(valid_galaxies),
        "cluster_cells_valid": len(valid_clusters),
        "galaxy_cells_source_gate_invalid": len(invalid_g),
        "cluster_cells_source_gate_invalid": len(invalid_c),
        "galaxies_scored": 139,
        "clusters_scored": 8,
        "sparc_rows_parsed": int(sparc_provenance["point_count"]),
        "sparc_rows_scored": sum(galaxy.count for galaxy in galaxies),
        "xcop_response_rows_scored": sum(len(packet["rows"]) for packet in packets),
        "successor_local_payload_read_operations": 31,
        "cumulative_local_payload_read_operations_before_check": 62,
        "conservative_cumulative_cell_charge": 3696,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "tuning_calls": 0,
        "dashboards": 147,
        "artifacts": 156,
    }
    return {
        "artifacts": base._jsonable(artifacts),
        "counts": counts,
        "cross_domain_adjudication": base._jsonable(cross),
        "cross_domain_survivors": base._jsonable(
            [row for row in cross if row["cross_domain_pass"]]
        ),
        "best_development_cells": {
            "GALAXIES": best_g["cell_id"],
            "CLUSTERS": best_c["cell_id"],
        },
    }


def _artifact_paths(context: Mapping[str, Any]) -> tuple[str, ...]:
    dashboard_ids = [
        *(f"GALAXIES-{name}" for name in context["source_predecessor"]["objects"]["SPARC"]),
        *(f"CLUSTERS-{name}" for name in context["source_predecessor"]["objects"]["XCOP"]),
    ]
    relative = [
        *base.MANDATORY_CAMPAIGN_ARTIFACTS,
        *(f"dashboards/{dashboard_id}.json" for dashboard_id in dashboard_ids),
    ]
    return tuple(sorted((ARTIFACT_DIRECTORY / path).as_posix() for path in relative))


def _load_artifacts(result: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    root = _root()
    expected = _artifact_paths(context)
    index = result.get("artifact_index")
    _require(isinstance(index, list) and len(index) == 156, "artifact index changed")
    _require(
        all(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256"}
            and isinstance(row["path"], str)
            and isinstance(row["sha256"], str)
            and base._SHA256_RE.fullmatch(row["sha256"]) is not None
            for row in index
        ),
        "artifact index row schema changed",
    )
    _require([row.get("path") for row in index] == list(expected), "artifact paths changed")
    actual = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in (root / ARTIFACT_DIRECTORY).rglob("*")
            if path.is_file()
        )
    )
    _require(actual == expected, "artifact directory changed")
    prefix = f"{ARTIFACT_DIRECTORY.as_posix()}/"
    payloads = {}
    for row in index:
        path = (root / str(row["path"])).resolve()
        _require(path.is_relative_to((root / ARTIFACT_DIRECTORY).resolve()), "artifact escaped")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == row["sha256"], "artifact hash changed")
        payload = json.loads(raw.decode("utf-8"))
        _require(base.canonical_bytes(payload) == raw, "artifact encoding changed")
        base._require_finite_json(payload)
        payloads[str(row["path"]).removeprefix(prefix)] = payload
    return payloads


def build_access_intent(package_commit: str) -> dict[str, Any]:
    root = _root()
    config = load_config()
    preflight = build_preflight()
    _require(_read_json(root / PREFLIGHT_PATH) == preflight, "successor preflight changed")
    intent = {
        "schema_version": INTENT_SCHEMA,
        "continuation_id": config["continuation_id"],
        "campaign_id": config["campaign_id"],
        "package_commit": package_commit,
        "preflight_file_sha256": _file_sha(root / PREFLIGHT_PATH),
        "preflight_content_sha256": preflight["preflight_content_sha256"],
        "predecessor_access_intent_sha256": config["predecessor"]["access_intent_sha256"],
        "predecessor_failure_sha256": config["predecessor"]["failure_sha256"],
        "same_manifest_content_sha256": config["predecessor"]["manifest_content_sha256"],
        "response_access_attempt_ordinal": 2,
        "formula_campaign_ordinal": 1,
        "planned_cells": {"GALAXIES": 179, "CLUSTERS": 1669},
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "tuning_calls": 0,
        "automatic_third_attempt": False,
        "intent_content_sha256": "",
    }
    intent["intent_content_sha256"] = base._self_hash(intent, "intent_content_sha256")
    return intent


def _adjudication(result: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": ADJUDICATION_SCHEMA,
        "continuation_id": result["continuation_id"],
        "result_content_sha256": result["result_content_sha256"],
        "artifact_index_sha256": base.content_sha256(result["artifact_index"]),
        "validated_counts": result["counts"],
        "survivor_count": len(result["cross_domain_survivors"]),
        "invalid_source_gate_cells": {
            "GALAXIES": result["counts"]["galaxy_cells_source_gate_invalid"],
            "CLUSTERS": result["counts"]["cluster_cells_source_gate_invalid"],
        },
        "maximum_label": "DEVELOPMENT_SIGNAL",
        "same_formula_campaign": True,
        "automatic_third_attempt": False,
        "adjudication_content_sha256": "",
    }
    value["adjudication_content_sha256"] = base._self_hash(value, "adjudication_content_sha256")
    return value


def execute() -> dict[str, Any]:
    root = _root()
    for path in (ACCESS_INTENT_PATH, RESULT_PATH, ADJUDICATION_PATH, FAILURE_PATH):
        _require(not (root / path).exists(), f"successor replay refused: {path}")
    _require(not (root / ARTIFACT_DIRECTORY).exists(), "successor artifact directory exists")
    package_commit = _require_committed_preflight(root)
    intent = build_access_intent(package_commit)
    _require(
        base._atomic_no_clobber(root / ACCESS_INTENT_PATH, base.canonical_bytes(intent))
        == "CREATED",
        "successor access intent refusal",
    )
    try:
        computed = _compute()
        manifest, _context = base.build_manifest(root)
        artifacts = computed["artifacts"]
        prefix = f"{ARTIFACT_DIRECTORY.as_posix()}/"
        artifact_index = [
            {
                "path": f"{prefix}{relative}",
                "sha256": hashlib.sha256(base.canonical_bytes(payload)).hexdigest(),
            }
            for relative, payload in sorted(artifacts.items())
        ]
        result = {
            "schema_version": RESULT_SCHEMA,
            "continuation_id": "OPEN-GRAVITY-CAMPAIGN-v1-CONTINUATION-1",
            "campaign_id": "OPEN-GRAVITY-CAMPAIGN-v1",
            "status": "DEVELOPMENT_CAMPAIGN_CONTINUATION_COMPLETE",
            "package_commit": package_commit,
            "same_manifest_content_sha256": manifest["manifest_content_sha256"],
            "access_intent_content_sha256": intent["intent_content_sha256"],
            "predecessor_failure_sha256": load_config()["predecessor"]["failure_sha256"],
            "counts": computed["counts"],
            "cross_domain_survivors": computed["cross_domain_survivors"],
            "cross_domain_adjudication": computed["cross_domain_adjudication"],
            "best_development_cells": computed["best_development_cells"],
            "artifact_index": artifact_index,
            "claim_ceiling": load_config()["claim_ceiling"],
            "maximum_label": "DEVELOPMENT_SIGNAL",
            "global_discovery_p_value": None,
            "external_cost_usd": 0.0,
            "result_content_sha256": "",
        }
        result = base._jsonable(result)
        result["result_content_sha256"] = base._self_hash(result, "result_content_sha256")
        for relative, payload in sorted(artifacts.items()):
            _require(
                base._atomic_no_clobber(
                    root / ARTIFACT_DIRECTORY / relative, base.canonical_bytes(payload)
                )
                == "CREATED",
                f"successor artifact publication failed: {relative}",
            )
        _require(
            base._atomic_no_clobber(root / RESULT_PATH, base.canonical_bytes(result)) == "CREATED",
            "successor result publication failed",
        )
        adjudication = _adjudication(result)
        _require(
            base._atomic_no_clobber(root / ADJUDICATION_PATH, base.canonical_bytes(adjudication))
            == "CREATED",
            "successor adjudication publication failed",
        )
        return result
    except BaseException as error:
        code = "UNEXPECTED_IMPLEMENTATION_FAILURE"
        if isinstance(error, (OpenGravityCampaignSuccessorError, base.OpenGravityCampaignError)):
            code = "SUCCESSOR_CONTRACT_FAILURE"
        elif isinstance(error, OSError):
            code = "LOCAL_IO_FAILURE"
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "continuation_id": "OPEN-GRAVITY-CAMPAIGN-v1-CONTINUATION-1",
            "status": "SUCCESSOR_TERMINAL_FAILURE_NO_AUTOMATIC_THIRD_ATTEMPT",
            "error_code": code,
            "raw_exception_message_retained": False,
            "raw_exception_class_retained": False,
            "access_intent_exists": (root / ACCESS_INTENT_PATH).exists(),
            "result_exists": (root / RESULT_PATH).exists(),
            "adjudication_exists": (root / ADJUDICATION_PATH).exists(),
            "automatic_third_attempt": False,
            "failure_content_sha256": "",
        }
        failure["failure_content_sha256"] = base._self_hash(failure, "failure_content_sha256")
        base._atomic_no_clobber(root / FAILURE_PATH, base.canonical_bytes(failure))
        raise


def check_result() -> dict[str, Any]:
    root = _root()
    _require(not (root / FAILURE_PATH).exists(), "successor failure receipt exists")
    package_commit = _require_committed_preflight(root)
    intent = _read_json(root / ACCESS_INTENT_PATH)
    result = _read_json(root / RESULT_PATH)
    adjudication = _read_json(root / ADJUDICATION_PATH)
    _require(intent == build_access_intent(package_commit), "successor access intent changed")
    _require(
        set(result)
        == {
            "schema_version",
            "continuation_id",
            "campaign_id",
            "status",
            "package_commit",
            "same_manifest_content_sha256",
            "access_intent_content_sha256",
            "predecessor_failure_sha256",
            "counts",
            "cross_domain_survivors",
            "cross_domain_adjudication",
            "best_development_cells",
            "artifact_index",
            "claim_ceiling",
            "maximum_label",
            "global_discovery_p_value",
            "external_cost_usd",
            "result_content_sha256",
        },
        "successor result key set changed",
    )
    _require(result.get("schema_version") == RESULT_SCHEMA, "successor result schema changed")
    _require(
        result.get("continuation_id") == "OPEN-GRAVITY-CAMPAIGN-v1-CONTINUATION-1"
        and result.get("campaign_id") == "OPEN-GRAVITY-CAMPAIGN-v1"
        and result.get("status") == "DEVELOPMENT_CAMPAIGN_CONTINUATION_COMPLETE",
        "successor result identity changed",
    )
    _require(result.get("package_commit") == package_commit, "successor result package changed")
    _require(
        result.get("result_content_sha256") == base._self_hash(result, "result_content_sha256"),
        "successor result self hash changed",
    )
    manifest, context = base.build_manifest(root)
    _require(
        result.get("same_manifest_content_sha256") == manifest["manifest_content_sha256"],
        "successor result manifest changed",
    )
    config = load_config()
    _require(
        result.get("access_intent_content_sha256") == intent["intent_content_sha256"]
        and result.get("predecessor_failure_sha256") == config["predecessor"]["failure_sha256"],
        "successor result chronology changed",
    )
    _require(result.get("claim_ceiling") == config["claim_ceiling"], "claim ceiling changed")
    _require(float(result.get("external_cost_usd", -1.0)) == 0.0, "external cost changed")
    base._require_finite_json(result)
    artifacts = _load_artifacts(result, context)
    recomputed = _compute()
    _require(artifacts == recomputed["artifacts"], "successor artifacts failed recomputation")
    for key in (
        "counts",
        "cross_domain_survivors",
        "cross_domain_adjudication",
        "best_development_cells",
    ):
        _require(result.get(key) == recomputed[key], f"successor result changed: {key}")
    _require(adjudication == _adjudication(result), "successor adjudication changed")
    _require(result.get("global_discovery_p_value") is None, "discovery p-value overclaim")
    _require(result.get("maximum_label") == "DEVELOPMENT_SIGNAL", "result overclaim")
    return adjudication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "check-preflight", "status", "execute", "check-result")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(write_preflight(), sort_keys=True))
    elif args.command == "check-preflight":
        value = check_preflight()
        print(
            json.dumps({"decision": value["decision"], "status": value["status"]}, sort_keys=True)
        )
    elif args.command == "status":
        print(
            json.dumps(
                {
                    "access_intent_exists": ACCESS_INTENT_PATH.exists(),
                    "result_exists": RESULT_PATH.exists(),
                    "failure_exists": FAILURE_PATH.exists(),
                },
                sort_keys=True,
            )
        )
    elif args.command == "execute":
        execute()
    else:
        check_result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
