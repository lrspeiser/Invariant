"""Authorization-gated development scoring for the frozen 60-class B+E+N registry.

The ``preflight`` command is metadata-only.  The ``execute`` command validates an exact
authorization and writes an immutable access-intent record before it opens one development
payload.  An access intent cannot be replayed: an interrupted run requires a successor
contract rather than silently opening the data twice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sigma_theory_compiler.gravity_shared_target_blind_ben_synthetic_execution import (
    _find_component,
    evaluate_ast,
    expression,
    formula_ast,
    normalize_ast,
    sha256_value,
    validate_ast,
    validate_registry,
)
from sigma_theory_compiler.real_data_gravity_confrontation import Galaxy
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import (
    CONFIRMATION_FRACTION,
    FULL_SPLIT_RULE,
    FULL_SPLIT_SALT,
    _decimal,
    admit,
    declare_split,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_development_executor_v4.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_development_executor_v4.py")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py"
)
PREFLIGHT_PATH = Path("runs/gravity/shared-target-blind-ben-development-executor-v4-preflight.json")
AUTHORIZATION_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/"
    "authorization-current-unauthorized.json"
)
ACCESS_INTENT_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/access-intent.json"
)
RESULT_PATH = Path("runs/gravity/shared-target-blind-ben-development-executor-v4/result.json")

CONFIG_SCHEMA = "invariant-gravity-ben-development-executor-config-4.0"
PREFLIGHT_SCHEMA = "invariant-gravity-ben-development-executor-preflight-4.0"
AUTHORIZATION_SCHEMA = "invariant-gravity-ben-development-executor-authorization-4.0"
ACCESS_SCHEMA = "invariant-gravity-ben-development-executor-access-intent-4.0"
RESULT_SCHEMA = "invariant-gravity-ben-development-score-result-4.0"
DECISION = "READY_UNAUTHORIZED_ZERO_TARGET_ACCESS"

EXPECTED_XCOP_OBJECTS = [
    "A1644",
    "A1795",
    "A2142",
    "A2255",
    "A2319",
    "A3266",
    "A85",
    "ZW1215",
]
FORBIDDEN_XCOP_OBJECTS = ["A2029", "A3158", "A644", "RXC1825"]
ABLATION_IDS = ["N_zero_ablation", "B_unity_gate_ablation", "A_off_nuisance_ablation"]
XCOP_ROLES = ("density", "pressure", "temperature")
XCOP_HDU = {"density": 2, "pressure": 2, "temperature": 1}
XCOP_COLUMNS = {
    "density": ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"],
    "pressure": ["RW_SZ", "P_SZ", "eP_SZ"],
    "temperature": ["RW_X", "T_X", "eT_X"],
}
ACKNOWLEDGEMENTS = {
    "all_local_sparc_rows_are_development_only_for_this_descendant": True,
    "historical_sparc_score_subset_only": True,
    "xcop_eight_development_objects_only": True,
    "no_confirmation_or_independent_access": True,
    "shape_only_not_absolute_cluster_prediction": True,
    "diagonal_errors_not_full_covariance": True,
    "gas_only_newtonian_cluster_control": True,
    "no_single_counterexample_veto": True,
    "no_formula_family_pruning": True,
    "novelty_labels_non_authoritative": True,
    "no_publication_dark_matter_or_gr_replacement_claim": True,
    "no_network_model_paid_group_or_lensing_access": True,
}


class BENDevelopmentExecutorV4Error(RuntimeError):
    """Raised before a frozen scope, identity, or access boundary can change."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256((canonical_json(unsigned) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(path: Path) -> Path:
    target = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENDevelopmentExecutorV4Error(f"path escaped repository: {path}") from error
    return target


def read_json(path: Path) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise BENDevelopmentExecutorV4Error(f"required artifact absent: {path}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BENDevelopmentExecutorV4Error(f"expected JSON object: {path}")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BENDevelopmentExecutorV4Error(f"{label} keys changed")


def _verify_content_receipt(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("content_sha256") != expected:
        raise BENDevelopmentExecutorV4Error(f"{label} declared content seal changed")
    # These predecessor receipts use several historical content-hash normalizations.  The
    # exact file bytes are already bound and verified above; this check independently
    # requires that the byte-bound artifact still declares the expected semantic seal.


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "verifier_test",
        "source_bindings",
        "candidate_registry",
        "ablation_registry",
        "development_populations",
        "sparc_mapping_and_score",
        "xcop_shape_bridge_and_score",
        "selection_contract",
        "compute_ceiling",
        "authorization_gate",
        "output_paths",
        "zero_access_chronology",
        "claim_ceiling",
    }
    _strict_keys(config, expected, "V4 config")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_unauthorized_zero_target_access"
        or config["implementation_source"] != SOURCE_PATH.as_posix()
        or config["verifier_test"] != TEST_PATH.as_posix()
    ):
        raise BENDevelopmentExecutorV4Error("V4 identity changed")

    registry = config["candidate_registry"]
    ablations = config["ablation_registry"]
    if (
        registry["registry_content_sha256"]
        != "45966eae73d7641ea982a7eea47aad883a9ff344baf121b91b901c32ef819f19"
        or registry["raw_candidates_frozen"] != 240
        or registry["canonical_full_classes"] != 60
        or registry["raw_members_scored"] is not False
        or registry["novelty_labels_authoritative"] is not False
        or registry["post_response_generation_calls"] != 0
        or registry["post_response_repair_calls"] != 0
        or ablations["ordered_ablation_ids"] != ABLATION_IDS
        or ablations["registered_total"] != 180
        or ablations["unique_ablation_asts"] != 51
        or ablations["duplicate_registered_ablation_instances"] != 129
        or ablations["ablation_asts_overlapping_full_classes"] != 33
        or ablations["unique_asts_across_60_full_plus_180_registered_ablations"] != 78
        or ablations["score_every_registered_ablation"] is not True
    ):
        raise BENDevelopmentExecutorV4Error("candidate or ablation accounting changed")

    populations = config["development_populations"]
    sparc = populations["sparc"]
    xcop = populations["xcop"]
    if (
        sparc["all_local_rows_role"] != "development_only_for_this_descendant"
        or sparc["objects"] != 139
        or sparc["rows"] != 2720
        or sparc["rows_outside_subset_scored"] != 0
        or sparc["local_confirmation_role_exists"] is not False
        or xcop["objects"] != EXPECTED_XCOP_OBJECTS
        or xcop["predictor_density_rows"] != 521
        or xcop["response_rows"] != 184
        or xcop["forbidden_objects"] != FORBIDDEN_XCOP_OBJECTS
        or xcop["allowed_roles"] != list(XCOP_ROLES)
        or any(value is not True for value in populations["forbidden"].values())
    ):
        raise BENDevelopmentExecutorV4Error("development row allowlist changed")

    bridge = config["xcop_shape_bridge_and_score"]
    selection = config["selection_contract"]
    if (
        bridge["uses_P500_T500_R500_outer_anchor_or_mass_target"] is not False
        or bridge["matched_nuisance_profile_for_every_candidate_ablation_and_control"] is not True
        or bridge["absolute_amplitude_identified"] is not False
        or bridge["comparators"]
        != ["gas_only_newtonian_shape", "uniform_acceleration_shape_control"]
        or selection["single_counterexample_terminal"] is not False
        or selection["counterexample_count_alone_terminal"] is not False
        or selection["finite_sample_may_prune_formula_family"] is not False
        or selection["retain_all_failures_and_ties"] is not True
        or selection["numeric_improvement_threshold"] is not None
    ):
        raise BENDevelopmentExecutorV4Error("shape, selection, or counterexample policy changed")

    ceiling = config["compute_ceiling"]
    expected_ceiling = {
        "domains": 2,
        "canonical_full_candidates": 60,
        "registered_ablation_variants": 180,
        "domain_specific_comparators": 4,
        "formula_domain_batches_per_backend": 484,
        "cpu_formula_domain_batches": 484,
        "gpu_formula_domain_batches": 484,
        "cpu_gpu_parity_comparisons": 484,
        "sparc_formula_row_cells_per_backend": 658240,
        "xcop_formula_row_cells_per_backend": 126082,
        "total_formula_row_cells_per_backend": 784322,
        "total_formula_row_cells_both_backends": 1568644,
        "xcop_coupled_three_parameter_nuisance_fits": 1936,
        "xcop_zeta_objective_evaluations": 7931792,
        "xcop_analytic_scale_solves": 15863584,
        "maximum_object_score_reductions": 35574,
        "maximum_response_row_score_terms": 702768,
        "candidate_selection_events": 1,
        "threshold_tuning_calls": 0,
        "formula_generation_calls": 0,
        "formula_repair_calls": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "maximum_api_spend_usd": 0.0,
        "maximum_payload_file_opens": 25,
    }
    if ceiling != expected_ceiling:
        raise BENDevelopmentExecutorV4Error("compute ceiling changed")
    gate = config["authorization_gate"]
    if (
        gate["authorization_id"] != "ben-development-score-v4-production-1"
        or gate["authorization_path"] != AUTHORIZATION_PATH.as_posix()
        or gate["current_authorization_expected"] is not False
        or gate["authorization_cannot_expand_scope"] is not True
        or gate["authorization_replay_allowed"] is not False
    ):
        raise BENDevelopmentExecutorV4Error("authorization boundary changed")
    outputs = config["output_paths"]
    if outputs != {
        "preflight_receipt": PREFLIGHT_PATH.as_posix(),
        "authorization": AUTHORIZATION_PATH.as_posix(),
        "access_intent": ACCESS_INTENT_PATH.as_posix(),
        "result": RESULT_PATH.as_posix(),
    }:
        raise BENDevelopmentExecutorV4Error("output path boundary changed")
    if any(
        value != 0
        for key, value in config["zero_access_chronology"].items()
        if key != "contract_frozen_before_target_access"
    ):
        raise BENDevelopmentExecutorV4Error("preflight claims target access")
    if config["zero_access_chronology"]["contract_frozen_before_target_access"] is not True:
        raise BENDevelopmentExecutorV4Error("preflight chronology changed")


def validate_bound_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate only metadata artifacts.  This function must never open a payload."""

    loaded: dict[str, dict[str, Any]] = {}
    payload_paths = {
        Path(config["development_populations"]["sparc"]["payload_path"]),
        Path(config["development_populations"]["xcop"]["raw_directory"]),
    }
    for label, binding in config["source_bindings"].items():
        path = Path(binding["path"])
        if path in payload_paths or any(parent in payload_paths for parent in path.parents):
            raise BENDevelopmentExecutorV4Error("a metadata binding points at a payload")
        target = confined(path)
        if file_sha256(target) != binding["file_sha256"]:
            raise BENDevelopmentExecutorV4Error(f"bound metadata changed: {label}")
        if target.suffix == ".json":
            value = read_json(path)
            if "content_sha256" in binding:
                _verify_content_receipt(value, binding["content_sha256"], label)
            loaded[label] = value

    synthetic = loaded["synthetic_receipt"]
    registry = synthetic["candidate_registry"]
    validate_registry(registry)
    if registry["content_sha256"] != config["candidate_registry"]["registry_content_sha256"]:
        raise BENDevelopmentExecutorV4Error("candidate registry binding changed")
    v2 = loaded["development_v2_receipt"]
    v3 = loaded["shape_v3_receipt"]
    if (
        v2["claims"]["all_local_sparc_rows_development_only_for_descendant"] is not True
        or v2["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or v3["claims"]["all_local_sparc_rows_development_only"] is not True
        or v3["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or v3["claims"]["predictor_only_xcop_shape_basis_frozen"] is not True
        or v3["claims"]["real_scoring_executed"] is not False
    ):
        raise BENDevelopmentExecutorV4Error("V2/V3 access or mapping lineage changed")
    return loaded


def _components_for_representative(registry: Mapping[str, Any], raw_id: str) -> dict[str, Any]:
    raw_rows = {row["raw_id"]: row for row in registry["raw_candidates"]}
    if raw_id not in raw_rows:
        raise BENDevelopmentExecutorV4Error("canonical representative is absent from raw registry")
    ids = raw_rows[raw_id]["component_raw_ids"]
    return {role: _find_component(role, component_id) for role, component_id in ids.items()}


def build_registered_variants(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize 60 full records and 180 named ablations before response access."""

    full: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    for row in sorted(registry["equivalence_classes"], key=lambda item: item["class_id"]):
        full.append(
            {
                "variant_id": f"full:{row['class_id']}",
                "kind": "full",
                "full_class_id": row["class_id"],
                "ablation_id": None,
                "canonical_ast": row["canonical_ast"],
                "canonical_expression": row["canonical_expression"],
                "canonical_expression_sha256": row["canonical_expression_sha256"],
                "representative_raw_id": min(row["raw_member_ids"]),
                "raw_member_count": row["raw_member_count"],
                "provenance_label": row["provenance_label"],
                "provenance_is_authoritative_novelty_finding": False,
            }
        )
        representative = min(row["raw_member_ids"])
        base = _components_for_representative(registry, representative)
        for ablation_id in ABLATION_IDS:
            components = dict(base)
            if ablation_id == "N_zero_ablation":
                components["N_additive_channel"] = _find_component(
                    "N_additive_channel", "N.null_ablation"
                )
            elif ablation_id == "B_unity_gate_ablation":
                components["B_continuous_gate"] = {"ast": {"const": 1.0}}
            elif ablation_id == "A_off_nuisance_ablation":
                components["A_nuisance"] = _find_component("A_nuisance", "A.off")
            ast = normalize_ast(formula_ast(components))
            digest = sha256_value(ast)
            ablations.append(
                {
                    "variant_id": f"ablation:{row['class_id']}:{ablation_id}",
                    "kind": "ablation",
                    "full_class_id": row["class_id"],
                    "ablation_id": ablation_id,
                    "canonical_ast": ast,
                    "canonical_expression": expression(ast),
                    "canonical_expression_sha256": digest,
                    "representative_raw_id": representative,
                    "raw_member_count": 1,
                    "provenance_label": "derived_ablation_not_novelty_assessed",
                    "provenance_is_authoritative_novelty_finding": False,
                }
            )
    variants = full + ablations
    multiplicity = Counter(row["canonical_expression_sha256"] for row in variants)
    for row in variants:
        row["registered_equivalence_multiplicity"] = multiplicity[
            row["canonical_expression_sha256"]
        ]
    ablation_hashes = [row["canonical_expression_sha256"] for row in ablations]
    full_hashes = {row["canonical_expression_sha256"] for row in full}
    if (
        len(full) != 60
        or len(ablations) != 180
        or len(set(ablation_hashes)) != 51
        or len(ablations) - len(set(ablation_hashes)) != 129
        or len(set(ablation_hashes) & full_hashes) != 33
        or len(set(ablation_hashes) | full_hashes) != 78
    ):
        raise BENDevelopmentExecutorV4Error("derived ablation equivalence accounting changed")
    return {
        "full": full,
        "ablations": ablations,
        "variants": variants,
        "accounting": {
            "raw_candidates_frozen": registry["raw_candidate_count"],
            "canonical_full_classes": len(full),
            "registered_ablations": len(ablations),
            "unique_ablation_asts": len(set(ablation_hashes)),
            "duplicate_registered_ablation_instances": len(ablations) - len(set(ablation_hashes)),
            "ablation_asts_overlapping_full_classes": len(set(ablation_hashes) & full_hashes),
            "unique_asts_across_full_and_ablations": len(set(ablation_hashes) | full_hashes),
            "raw_equivalent_members_scored": 0,
        },
    }


def _xcop_inventory(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (row["cluster"], row["role"]): row
        for row in source_receipt["files"]
        if row["role"] in XCOP_ROLES
    }
    inventory = []
    for cluster in EXPECTED_XCOP_OBJECTS:
        for role in XCOP_ROLES:
            row = by_key.get((cluster, role))
            if row is None:
                raise BENDevelopmentExecutorV4Error(f"missing X-COP metadata: {cluster}:{role}")
            expected_member = (
                f"{cluster}/{cluster}_{'density_L1' if role == 'density' else role}.fits"
            )
            if (
                row["member"] != expected_member
                or row["confirmation_response_opened_after_scientific_freeze"] is not False
            ):
                raise BENDevelopmentExecutorV4Error("X-COP development inventory changed")
            inventory.append(
                {
                    "cluster": cluster,
                    "role": role,
                    "relative_path": (
                        Path(config["development_populations"]["xcop"]["raw_directory"])
                        / row["member"]
                    ).as_posix(),
                    "bytes": row["bytes"],
                    "file_sha256": row["sha256"],
                    "hdu": XCOP_HDU[role],
                    "columns": XCOP_COLUMNS[role],
                }
            )
    if len(inventory) != 24:
        raise BENDevelopmentExecutorV4Error("X-COP inventory is not exactly 24 development files")
    return inventory


def build_preflight() -> dict[str, Any]:
    config = load_config()
    loaded = validate_bound_metadata(config)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    inventory = _xcop_inventory(config, loaded["xcop_source_receipt"])
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "frozen_unauthorized_zero_target_access",
        "decision": DECISION,
        "source_bindings": {
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "file_sha256": file_sha256(confined(CONFIG_PATH)),
            },
            "source": {
                "path": SOURCE_PATH.as_posix(),
                "file_sha256": file_sha256(confined(SOURCE_PATH)),
            },
            "test": {"path": TEST_PATH.as_posix(), "file_sha256": file_sha256(confined(TEST_PATH))},
        },
        "bound_metadata_files_validated": len(config["source_bindings"]),
        "candidate_and_ablation_accounting": registered["accounting"],
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "development_populations": {
            "sparc_objects": 139,
            "sparc_rows": 2720,
            "sparc_role": "development_only",
            "xcop_objects": EXPECTED_XCOP_OBJECTS,
            "xcop_predictor_rows": 521,
            "xcop_response_rows": 184,
            "xcop_role": "development_only",
            "forbidden_xcop_objects": FORBIDDEN_XCOP_OBJECTS,
        },
        "xcop_development_file_inventory": inventory,
        "compute_ceiling": config["compute_ceiling"],
        "authorization_contract": {
            "schema_version": AUTHORIZATION_SCHEMA,
            "authorization_id": config["authorization_gate"]["authorization_id"],
            "required_exact_approval_text": config["authorization_gate"]["exact_approval_text"],
            "required_claim_acknowledgements": ACKNOWLEDGEMENTS,
            "authorization_must_precede_payload": True,
            "authorization_replay_allowed": False,
        },
        "zero_access_chronology": config["zero_access_chronology"],
        "claim_ceiling": config["claim_ceiling"],
        "production_executed": False,
        "target_files_opened": 0,
        "target_rows_read": 0,
        "scores_computed": 0,
        "selection_events": 0,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode()
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise BENDevelopmentExecutorV4Error(f"refusing to overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_preflight() -> Path:
    receipt = build_preflight()
    _atomic_no_clobber(PREFLIGHT_PATH, receipt)
    return confined(PREFLIGHT_PATH)


def validate_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = read_json(PREFLIGHT_PATH)
    if receipt.get("schema_version") != PREFLIGHT_SCHEMA or receipt.get("decision") != DECISION:
        raise BENDevelopmentExecutorV4Error("preflight identity changed")
    if receipt.get("content_sha256") != content_sha256(receipt):
        raise BENDevelopmentExecutorV4Error("preflight content seal changed")
    bindings = receipt["source_bindings"]
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings[key] != {"path": path.as_posix(), "file_sha256": file_sha256(confined(path))}:
            raise BENDevelopmentExecutorV4Error(f"preflight {key} binding changed")
    if receipt["compute_ceiling"] != config["compute_ceiling"]:
        raise BENDevelopmentExecutorV4Error("preflight compute ceiling changed")
    if receipt["zero_access_chronology"] != config["zero_access_chronology"]:
        raise BENDevelopmentExecutorV4Error("preflight zero-access chronology changed")
    return receipt


def authorization_template(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorization_id": config["authorization_gate"]["authorization_id"],
        "authorized": False,
        "approved_by": None,
        "approved_at": None,
        "approval_text": config["authorization_gate"]["exact_approval_text"],
        "config_file_sha256": file_sha256(confined(CONFIG_PATH)),
        "source_file_sha256": file_sha256(confined(SOURCE_PATH)),
        "test_file_sha256": file_sha256(confined(TEST_PATH)),
        "preflight_receipt_file_sha256": file_sha256(confined(PREFLIGHT_PATH)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "sparc_role": "development_only",
        "sparc_objects": 139,
        "sparc_rows": 2720,
        "xcop_role": "development_only",
        "xcop_objects": EXPECTED_XCOP_OBJECTS,
        "xcop_predictor_rows": 521,
        "xcop_response_rows": 184,
        "compute_ceiling": config["compute_ceiling"],
        "claim_acknowledgements": ACKNOWLEDGEMENTS,
        "access_intent_path": ACCESS_INTENT_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
    }


def write_unauthorized_template() -> Path:
    config = load_config()
    validate_bound_metadata(config)
    preflight = validate_preflight(config)
    _atomic_no_clobber(AUTHORIZATION_PATH, authorization_template(config, preflight))
    return confined(AUTHORIZATION_PATH)


def validate_authorization(
    path: Path, config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = read_json(path)
    expected = authorization_template(config, preflight)
    _strict_keys(supplied, set(expected), "authorization")
    for key, value in expected.items():
        if key in {"authorized", "approved_by", "approved_at"}:
            continue
        if supplied[key] != value:
            raise BENDevelopmentExecutorV4Error(f"authorization scope changed: {key}")
    if supplied["authorized"] is not True:
        raise BENDevelopmentExecutorV4Error("exact production authorization is required")
    if not isinstance(supplied["approved_by"], str) or not supplied["approved_by"].strip():
        raise BENDevelopmentExecutorV4Error("approved_by must identify the approver")
    if not isinstance(supplied["approved_at"], str):
        raise BENDevelopmentExecutorV4Error("approved_at is absent")
    try:
        datetime.fromisoformat(supplied["approved_at"])
    except ValueError as error:
        raise BENDevelopmentExecutorV4Error("approved_at is not ISO-8601") from error
    return supplied


def _galaxies_from_payload(payload: Mapping[str, Any]) -> list[Galaxy]:
    galaxies = []
    for entry in payload["galaxies"]:
        columns = list(zip(*entry["rows"], strict=True))
        galaxies.append(
            Galaxy(
                name=entry["name"],
                distance_mpc=entry["distance_mpc"],
                radius=tuple(_decimal(value) for value in columns[0]),
                v_obs=tuple(_decimal(value) for value in columns[1]),
                e_v_obs=tuple(_decimal(value) for value in columns[2]),
                v_gas=tuple(_decimal(value) for value in columns[3]),
                v_disk=tuple(_decimal(value) for value in columns[4]),
                v_bul=tuple(_decimal(value) for value in columns[5]),
                published=tuple(tuple(row) for row in entry["rows"]),
            )
        )
    return galaxies


def load_sparc_development(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = config["development_populations"]["sparc"]
    raw = confined(Path(binding["payload_path"])).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["payload_file_sha256"]:
        raise BENDevelopmentExecutorV4Error("SPARC payload file seal changed")
    payload = json.loads(raw.decode("utf-8"))
    validate_dataset(payload)
    if canonical_sha256(payload) != binding["payload_content_sha256"]:
        raise BENDevelopmentExecutorV4Error("SPARC payload semantic seal changed")
    galaxies = _galaxies_from_payload(payload)
    split = declare_split(
        [galaxy.name for galaxy in galaxies],
        count=int(CONFIRMATION_FRACTION * len(galaxies)),
        salt=FULL_SPLIT_SALT,
        rule=FULL_SPLIT_RULE,
    )
    convention = payload["mass_to_light_convention"]
    admitted, _admission = admit(
        galaxies,
        Fraction(convention["disk_3_6um"]),
        Fraction(convention["bulge_3_6um"]),
    )
    allowed = set(split.exploration)
    selected = [galaxy for galaxy in admitted if galaxy.name in allowed]
    if len(selected) != 139 or sum(galaxy.count for galaxy in selected) != 2720:
        raise BENDevelopmentExecutorV4Error("historical SPARC development allowlist changed")
    rows: list[dict[str, Any]] = []
    for galaxy in selected:
        radius = np.asarray([float(value) for value in galaxy.radius], dtype=np.float64)
        vobs = np.asarray([float(value) for value in galaxy.v_obs], dtype=np.float64)
        sigma = np.asarray([float(value) for value in galaxy.e_v_obs], dtype=np.float64)
        vgas = np.asarray([float(value) for value in galaxy.v_gas], dtype=np.float64)
        vdisk = np.asarray([float(value) for value in galaxy.v_disk], dtype=np.float64)
        vbul = np.asarray([float(value) for value in galaxy.v_bul], dtype=np.float64)
        vbar2 = vgas * np.abs(vgas) + 0.5 * vdisk**2 + 0.7 * vbul**2
        state_denom = vgas**2 + 0.5 * vdisk**2 + 0.7 * vbul**2
        geometry_denom = 0.5 * vdisk**2 + 0.7 * vbul**2
        if (
            np.any(radius <= 0)
            or np.any(sigma <= 0)
            or np.any(vbar2 <= 0)
            or np.any(state_denom <= 0)
            or np.any(geometry_denom <= 0)
        ):
            raise BENDevelopmentExecutorV4Error(
                f"SPARC predictor denominator invalid: {galaxy.name}"
            )
        predictors = np.column_stack(
            (
                vbar2 / (radius * 3702.81458),
                radius / np.max(radius),
                vgas**2 / state_denom,
                0.7 * vbul**2 / geometry_denom,
            )
        )
        rows.append(
            {
                "object": galaxy.name,
                "rows": galaxy.count,
                "radius": radius,
                "vobs": vobs,
                "sigma": sigma,
                "vbar2": vbar2,
                "predictors": predictors,
            }
        )
    return rows


def _fits_table_from_bytes(raw: bytes, hdu_index: int, expected_columns: Sequence[str]) -> Any:
    with fits.open(io.BytesIO(raw), memmap=False) as handle:
        data = handle[hdu_index].data.copy()
    if list(data.dtype.names or ()) != list(expected_columns):
        raise BENDevelopmentExecutorV4Error("X-COP FITS column schema changed")
    return data


def load_xcop_development(
    config: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    loaded: dict[tuple[str, str], Any] = {}
    for record in inventory:
        if record["cluster"] not in EXPECTED_XCOP_OBJECTS or record["role"] not in XCOP_ROLES:
            raise BENDevelopmentExecutorV4Error("X-COP inventory escaped development allowlist")
        raw = confined(Path(record["relative_path"])).read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
            raise BENDevelopmentExecutorV4Error("X-COP payload seal changed")
        loaded[(record["cluster"], record["role"])] = _fits_table_from_bytes(
            raw, record["hdu"], record["columns"]
        )
    clusters: list[dict[str, Any]] = []
    density_rows = 0
    response_rows = 0
    for cluster in EXPECTED_XCOP_OBJECTS:
        density = loaded[(cluster, "density")]
        pressure = loaded[(cluster, "pressure")]
        temperature = loaded[(cluster, "temperature")]
        rw = np.asarray(density["RW_X"], dtype=np.float64)
        ne = np.asarray(density["NE"], dtype=np.float64)
        pr = np.asarray(pressure["RW_SZ"], dtype=np.float64)
        py = np.asarray(pressure["P_SZ"], dtype=np.float64)
        pe = np.asarray(pressure["eP_SZ"], dtype=np.float64)
        tr = np.asarray(temperature["RW_X"], dtype=np.float64)
        ty = np.asarray(temperature["T_X"], dtype=np.float64)
        te = np.asarray(temperature["eT_X"], dtype=np.float64)
        arrays = (rw, ne, pr, py, pe, tr, ty, te)
        if any(np.any(~np.isfinite(value)) or np.any(value <= 0) for value in arrays):
            raise BENDevelopmentExecutorV4Error(f"nonpositive X-COP source: {cluster}")
        if np.any(np.diff(rw) <= 0) or np.any(np.diff(pr) <= 0) or np.any(np.diff(tr) <= 0):
            raise BENDevelopmentExecutorV4Error(f"unordered X-COP radii: {cluster}")
        x = rw / np.max(rw)
        q = ne / np.max(ne)
        log_x = np.log(x)
        log_q = np.log(q)
        slope = np.empty_like(q)
        slope[0] = (log_q[1] - log_q[0]) / (log_x[1] - log_x[0])
        slope[-1] = (log_q[-1] - log_q[-2]) / (log_x[-1] - log_x[-2])
        slope[1:-1] = (log_q[2:] - log_q[:-2]) / (log_x[2:] - log_x[:-2])
        px = pr / np.max(rw)
        tx = tr / np.max(rw)
        if (
            len(px) < 3
            or len(tx) < 3
            or px[0] < x[0]
            or px[-1] > x[-1]
            or tx[0] < x[0]
            or tx[-1] > x[-1]
        ):
            raise BENDevelopmentExecutorV4Error(
                f"X-COP response outside density support: {cluster}"
            )
        clusters.append(
            {
                "object": cluster,
                "x": x,
                "q": q,
                "predictors": np.column_stack((q, x, np.abs(slope), np.ones_like(x))),
                "pressure_x": px,
                "pressure_y": py,
                "pressure_sigma": np.maximum(pe, 0.05 * np.abs(py)),
                "temperature_x": tx,
                "temperature_y": ty,
                "temperature_sigma": np.maximum(te, 0.05 * np.abs(ty)),
            }
        )
        density_rows += len(rw)
        response_rows += len(pr) + len(tr)
    if density_rows != 521 or response_rows != 184:
        raise BENDevelopmentExecutorV4Error("X-COP development row counts changed")
    return clusters


def _evaluate_ast_xp(node: Mapping[str, Any], predictors: Any, xp: Any) -> Any:
    validate_ast(node)
    if "const" in node:
        return xp.full(predictors.shape[0], float(node["const"]), dtype=xp.float64)
    if "var" in node:
        order = ("x_source", "x_radial", "x_state", "x_geometry")
        return predictors[:, order.index(str(node["var"]))]
    values = [_evaluate_ast_xp(child, predictors, xp) for child in node["args"]]
    name = node["op"]
    if name == "add":
        return values[0] + values[1]
    if name == "subtract":
        return values[0] - values[1]
    if name == "multiply":
        return values[0] * values[1]
    if name == "divide_safe":
        return values[0] / values[1]
    if name == "sqrt_positive":
        return xp.sqrt(values[0])
    if name == "exp_negative":
        return xp.exp(-values[0])
    raise BENDevelopmentExecutorV4Error(f"unknown AST operator: {name}")


def _parity(cpu: np.ndarray, gpu: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    if cpu.shape != gpu.shape:
        return {"pass": False, "max_abs": None, "max_rel": None, "reason": "shape_mismatch"}
    finite = np.all(np.isfinite(cpu)) and np.all(np.isfinite(gpu))
    if not finite:
        return {"pass": False, "max_abs": None, "max_rel": None, "reason": "nonfinite"}
    absolute = np.abs(cpu - gpu)
    relative = absolute / np.maximum(np.maximum(np.abs(cpu), np.abs(gpu)), 1.0)
    max_abs = float(np.max(absolute, initial=0.0))
    max_rel = float(np.max(relative, initial=0.0))
    selection = config["selection_contract"]
    passed = (
        max_abs <= selection["parity_absolute_tolerance"]
        or max_rel <= selection["parity_relative_tolerance"]
    )
    return {
        "pass": bool(passed),
        "max_abs": format(max_abs, ".12e"),
        "max_rel": format(max_rel, ".12e"),
        "reason": None if passed else "tolerance",
    }


def _loss(prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(np.square((prediction - observed) / sigma)))


def _score_sparc_vector(output: np.ndarray, objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if np.any(~np.isfinite(output)) or np.any(output <= 0):
        return {
            "valid": False,
            "domain_loss": None,
            "per_object": [],
            "failures": ["nonpositive_formula_output"],
        }
    per_object = []
    cursor = 0
    for row in objects:
        count = row["rows"]
        values = output[cursor : cursor + count]
        prediction = np.sqrt(3702.81458 * values * row["radius"])
        loss = _loss(prediction, row["vobs"], row["sigma"])
        per_object.append(
            {
                "object": row["object"],
                "rows": count,
                "loss": format(loss, ".12e"),
                "terminal_veto": False,
            }
        )
        cursor += count
    domain_loss = float(np.mean([float(row["loss"]) for row in per_object]))
    return {
        "valid": True,
        "domain_loss": format(domain_loss, ".12e"),
        "per_object": per_object,
        "failures": [],
    }


def _reverse_integral(x: np.ndarray, q: np.ndarray, f: np.ndarray) -> np.ndarray:
    h = np.zeros_like(x)
    for index in range(len(x) - 2, -1, -1):
        h[index] = h[index + 1] + 0.5 * (q[index] * f[index] + q[index + 1] * f[index + 1]) * (
            x[index + 1] - x[index]
        )
    h[-1] = 0.0
    return h


def _shape_gates(x: np.ndarray, q: np.ndarray, f: np.ndarray, h: np.ndarray) -> list[str]:
    failures = []
    scale = max(1.0, float(np.max(np.abs(h))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(h < -tolerance):
        failures.append("H_negative")
    if h[-1] != 0.0:
        failures.append("H_outer_not_exact_zero")
    derivative = np.diff(h) / np.diff(x)
    expected = -0.5 * (q[:-1] * f[:-1] + q[1:] * f[1:])
    derivative_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(expected))))
    )
    if np.any(derivative >= derivative_tolerance) or not np.allclose(
        derivative, expected, rtol=64.0 * np.finfo(np.float64).eps, atol=derivative_tolerance
    ):
        failures.append("H_derivative_identity")
    if float(np.max(h) - np.min(h)) < math.sqrt(np.finfo(np.float64).eps) * scale:
        failures.append("H_constant")
    return failures


def _fit_shape_nuisance(
    hp: np.ndarray, qp: np.ndarray, ht: np.ndarray, qt: np.ndarray, cluster: Mapping[str, Any]
) -> dict[str, Any]:
    zeta = np.arange(4097, dtype=np.float64) / 4096.0
    phi_p = hp[None, :] + zeta[:, None] * (1.0 - hp[None, :])
    phi_t = ht[None, :] + zeta[:, None] * (1.0 - ht[None, :])
    p_y = cluster["pressure_y"]
    p_s = cluster["pressure_sigma"]
    t_y = cluster["temperature_y"]
    t_s = cluster["temperature_sigma"]
    p_w = 1.0 / p_s**2
    t_w = 1.0 / t_s**2
    p_den = np.sum(p_w[None, :] * phi_p**2, axis=1)
    t_design = phi_t / qt[None, :]
    t_den = np.sum(t_w[None, :] * t_design**2, axis=1)
    if np.any(p_den <= 0) or np.any(t_den <= 0):
        return {"valid": False, "loss": None, "failures": ["zero_norm_design"]}
    b_p = np.maximum(0.0, np.sum(p_w[None, :] * phi_p * p_y[None, :], axis=1) / p_den)
    b_t = np.maximum(0.0, np.sum(t_w[None, :] * t_design * t_y[None, :], axis=1) / t_den)
    p_residual = (b_p[:, None] * phi_p - p_y[None, :]) / p_s[None, :]
    t_residual = (b_t[:, None] * t_design - t_y[None, :]) / t_s[None, :]
    losses = 0.5 * (np.mean(p_residual**2, axis=1) + np.mean(t_residual**2, axis=1))
    index = int(np.argmin(losses))
    zp = float(zeta[index])
    bp = float(b_p[index])
    bt = float(b_t[index])
    pp = bp * phi_p[index]
    failures = []
    if not (zp < 1.0 and bp > 0.0 and bt > 0.0):
        failures.append("neutral_or_nonpositive_nuisance_optimum")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(pp))))
    if np.any(np.diff(pp) > tolerance):
        failures.append("pressure_not_nonincreasing")
    jacobian = np.vstack(
        (
            np.column_stack((bp * (1.0 - hp), phi_p[index], np.zeros_like(hp))),
            np.column_stack((bt * (1.0 - ht) / qt, np.zeros_like(ht), phi_t[index] / qt)),
        )
    )
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms <= 0):
        failures.append("jacobian_zero_column")
        rank = 0
        condition = math.inf
    else:
        singular = np.linalg.svd(jacobian / norms[None, :], compute_uv=False)
        rank = int(np.sum(singular > singular[0] * math.sqrt(np.finfo(np.float64).eps)))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
        if rank != 3 or condition > 67108864.0:
            failures.append("jacobian_rank_or_condition")
    return {
        "valid": not failures,
        "loss": format(float(losses[index]), ".12e"),
        "zeta": format(zp, ".12e"),
        "b_P": format(bp, ".12e"),
        "b_T": format(bt, ".12e"),
        "jacobian_rank": rank,
        "jacobian_condition": None if not math.isfinite(condition) else format(condition, ".12e"),
        "failures": failures,
    }


def _score_xcop_shapes(
    outputs: Sequence[np.ndarray], clusters: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_object = []
    failures = []
    for f_raw, cluster in zip(outputs, clusters, strict=True):
        scoped = []
        if np.any(~np.isfinite(f_raw)) or np.any(f_raw <= 0) or float(np.max(f_raw)) <= 0:
            scoped.append("nonpositive_formula_output")
            per_object.append(
                {
                    "object": cluster["object"],
                    "valid": False,
                    "loss": None,
                    "failures": scoped,
                    "terminal_veto": False,
                }
            )
            failures.append(f"{cluster['object']}:nonpositive_formula_output")
            continue
        f = f_raw / np.max(f_raw)
        h = _reverse_integral(cluster["x"], cluster["q"], f)
        scoped.extend(_shape_gates(cluster["x"], cluster["q"], f, h))
        hp = np.interp(cluster["pressure_x"], cluster["x"], h)
        ht = np.interp(cluster["temperature_x"], cluster["x"], h)
        qp = np.exp(
            np.interp(np.log(cluster["pressure_x"]), np.log(cluster["x"]), np.log(cluster["q"]))
        )
        qt = np.exp(
            np.interp(np.log(cluster["temperature_x"]), np.log(cluster["x"]), np.log(cluster["q"]))
        )
        fit = _fit_shape_nuisance(hp, qp, ht, qt, cluster)
        scoped.extend(fit["failures"])
        for failure in scoped:
            failures.append(f"{cluster['object']}:{failure}")
        per_object.append(
            {
                "object": cluster["object"],
                "response_rows": len(cluster["pressure_y"]) + len(cluster["temperature_y"]),
                "valid": not scoped,
                "loss": fit.get("loss"),
                "nuisance": {key: fit.get(key) for key in ("zeta", "b_P", "b_T")},
                "jacobian_rank": fit.get("jacobian_rank"),
                "jacobian_condition": fit.get("jacobian_condition"),
                "failures": scoped,
                "terminal_veto": False,
            }
        )
    valid_losses = [
        float(row["loss"]) for row in per_object if row["valid"] and row["loss"] is not None
    ]
    valid = len(valid_losses) == len(clusters)
    return {
        "valid": valid,
        "domain_loss": format(float(np.mean(valid_losses)), ".12e") if valid else None,
        "valid_object_scores": len(valid_losses),
        "per_object": per_object,
        "failures": failures,
        "formula_family_pruned": False,
    }


def _split_vector(
    values: np.ndarray, objects: Sequence[Mapping[str, Any]], key: str
) -> list[np.ndarray]:
    output = []
    cursor = 0
    for row in objects:
        count = len(row[key])
        output.append(values[cursor : cursor + count])
        cursor += count
    if cursor != len(values):
        raise BENDevelopmentExecutorV4Error("domain vector split accounting changed")
    return output


def _gas_newtonian_shapes(clusters: Sequence[Mapping[str, Any]], xp: Any) -> Any:
    values = []
    for cluster in clusters:
        x = xp.asarray(cluster["x"], dtype=xp.float64)
        q = xp.asarray(cluster["q"], dtype=xp.float64)
        integrand = q * x**2
        increments = 0.5 * (integrand[:-1] + integrand[1:]) * (x[1:] - x[:-1])
        m = xp.concatenate(
            (xp.asarray([q[0] * x[0] ** 3 / 3.0]), q[0] * x[0] ** 3 / 3.0 + xp.cumsum(increments))
        )
        values.append(m / x**2)
    return xp.concatenate(values)


def _execute_scoring(
    config: Mapping[str, Any],
    registered: Mapping[str, Any],
    sparc: Sequence[Mapping[str, Any]],
    xcop: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - depends on production host
        raise BENDevelopmentExecutorV4Error(
            "CuPy is required by the frozen CPU/GPU parity gate"
        ) from error

    variants = registered["variants"]
    sparc_predictors = np.concatenate([row["predictors"] for row in sparc])
    xcop_predictors = np.concatenate([row["predictors"] for row in xcop])
    sparc_gpu_predictors = cp.asarray(sparc_predictors, dtype=cp.float64)
    xcop_gpu_predictors = cp.asarray(xcop_predictors, dtype=cp.float64)
    sparc_results: dict[str, Any] = {}
    xcop_results: dict[str, Any] = {}
    parity_calls = 0

    for variant in variants:
        ast = variant["canonical_ast"]
        cpu_s = evaluate_ast(ast, sparc_predictors)
        gpu_s = cp.asnumpy(_evaluate_ast_xp(ast, sparc_gpu_predictors, cp))
        parity_s = _parity(cpu_s, gpu_s, config)
        score_s = _score_sparc_vector(cpu_s, sparc)
        score_s["cpu_gpu_parity"] = parity_s
        score_s["valid"] = score_s["valid"] and parity_s["pass"]
        sparc_results[variant["variant_id"]] = score_s
        parity_calls += 1

        cpu_x = evaluate_ast(ast, xcop_predictors)
        gpu_x = cp.asnumpy(_evaluate_ast_xp(ast, xcop_gpu_predictors, cp))
        parity_x = _parity(cpu_x, gpu_x, config)
        score_x = _score_xcop_shapes(_split_vector(cpu_x, xcop, "x"), xcop)
        score_x["cpu_gpu_parity"] = parity_x
        score_x["valid"] = score_x["valid"] and parity_x["pass"]
        xcop_results[variant["variant_id"]] = score_x
        parity_calls += 1

    # Domain-specific controls, with the same CPU/GPU parity and X-COP nuisance profile.
    x_source = sparc_predictors[:, 0]
    controls_s = {
        "control:sparc:newtonian_baryons": x_source,
        "control:sparc:empirical_rar": x_source / (-np.expm1(-np.sqrt(x_source))),
    }
    controls_s_gpu = {
        "control:sparc:newtonian_baryons": sparc_gpu_predictors[:, 0],
        "control:sparc:empirical_rar": sparc_gpu_predictors[:, 0]
        / (-cp.expm1(-cp.sqrt(sparc_gpu_predictors[:, 0]))),
    }
    for key, cpu in controls_s.items():
        gpu = cp.asnumpy(controls_s_gpu[key])
        score = _score_sparc_vector(cpu, sparc)
        score["cpu_gpu_parity"] = _parity(cpu, gpu, config)
        score["valid"] = score["valid"] and score["cpu_gpu_parity"]["pass"]
        sparc_results[key] = score
        parity_calls += 1

    cpu_gas = np.asarray(_gas_newtonian_shapes(xcop, np), dtype=np.float64)
    gpu_gas = cp.asnumpy(_gas_newtonian_shapes(xcop, cp))
    cpu_uniform = np.ones(len(xcop_predictors), dtype=np.float64)
    gpu_uniform = cp.asnumpy(cp.ones(len(xcop_predictors), dtype=cp.float64))
    for key, cpu, gpu in (
        ("control:xcop:gas_only_newtonian_shape", cpu_gas, gpu_gas),
        ("control:xcop:uniform_acceleration_shape_control", cpu_uniform, gpu_uniform),
    ):
        score = _score_xcop_shapes(_split_vector(cpu, xcop, "x"), xcop)
        score["cpu_gpu_parity"] = _parity(cpu, gpu, config)
        score["valid"] = score["valid"] and score["cpu_gpu_parity"]["pass"]
        xcop_results[key] = score
        parity_calls += 1

    decisions = []
    for full in registered["full"]:
        class_id = full["full_class_id"]
        full_id = full["variant_id"]
        s_full = sparc_results[full_id]
        x_full = xcop_results[full_id]
        ablation_ids = [f"ablation:{class_id}:{name}" for name in ABLATION_IDS]
        checks = {
            "sparc_valid": s_full["valid"],
            "xcop_valid": x_full["valid"],
            "sparc_beats_newtonian": s_full["domain_loss"] is not None
            and float(s_full["domain_loss"])
            < float(sparc_results["control:sparc:newtonian_baryons"]["domain_loss"]),
            "sparc_beats_empirical_rar": s_full["domain_loss"] is not None
            and float(s_full["domain_loss"])
            < float(sparc_results["control:sparc:empirical_rar"]["domain_loss"]),
            "xcop_beats_gas_newtonian": x_full["domain_loss"] is not None
            and xcop_results["control:xcop:gas_only_newtonian_shape"]["domain_loss"] is not None
            and float(x_full["domain_loss"])
            < float(xcop_results["control:xcop:gas_only_newtonian_shape"]["domain_loss"]),
            "xcop_beats_uniform": x_full["domain_loss"] is not None
            and xcop_results["control:xcop:uniform_acceleration_shape_control"]["domain_loss"]
            is not None
            and float(x_full["domain_loss"])
            < float(xcop_results["control:xcop:uniform_acceleration_shape_control"]["domain_loss"]),
            "sparc_beats_each_ablation": s_full["domain_loss"] is not None
            and all(
                sparc_results[item]["domain_loss"] is not None
                and float(s_full["domain_loss"]) < float(sparc_results[item]["domain_loss"])
                for item in ablation_ids
            ),
            "xcop_beats_each_ablation": x_full["domain_loss"] is not None
            and all(
                xcop_results[item]["domain_loss"] is not None
                and float(x_full["domain_loss"]) < float(xcop_results[item]["domain_loss"])
                for item in ablation_ids
            ),
        }
        decisions.append(
            {
                "class_id": class_id,
                "eligible": all(checks.values()),
                "checks": checks,
                "formula_family_pruned": False,
                "single_object_terminal_veto": False,
            }
        )
    eligible = [row["class_id"] for row in decisions if row["eligible"]]
    selected = eligible[0] if eligible else None
    if parity_calls != 484:
        raise BENDevelopmentExecutorV4Error("CPU/GPU parity accounting changed")
    return {
        "sparc": sparc_results,
        "xcop": xcop_results,
        "candidate_decisions": decisions,
        "eligible_class_ids": eligible,
        "selected_class_id": selected,
        "selection_events": 1,
        "counterexample_policy": {
            "single_counterexample_terminal": False,
            "counterexample_count_alone_terminal": False,
            "formula_families_pruned": 0,
            "all_scoped_failures_retained": True,
        },
        "actual_compute_accounting": {
            "cpu_formula_domain_batches": 484,
            "gpu_formula_domain_batches": 484,
            "cpu_gpu_parity_comparisons": parity_calls,
            "formula_generation_calls": 0,
            "formula_repair_calls": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "api_spend_usd": 0.0,
        },
    }


def execute(authorization_path: Path) -> Path:
    # Everything through validate_authorization is metadata-only by construction.
    config = load_config()
    loaded = validate_bound_metadata(config)
    preflight = validate_preflight(config)
    authorization = validate_authorization(authorization_path, config, preflight)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    inventory = _xcop_inventory(config, loaded["xcop_source_receipt"])
    if confined(ACCESS_INTENT_PATH).exists() or confined(RESULT_PATH).exists():
        raise BENDevelopmentExecutorV4Error("authorization replay or output overwrite refused")
    access_intent: dict[str, Any] = {
        "schema_version": ACCESS_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": file_sha256(confined(authorization_path)),
        "config_file_sha256": file_sha256(confined(CONFIG_PATH)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "payload_scope": {
            "sparc_files": 1,
            "sparc_score_objects": 139,
            "sparc_score_rows": 2720,
            "xcop_files": 24,
            "xcop_objects": EXPECTED_XCOP_OBJECTS,
            "xcop_predictor_rows": 521,
            "xcop_response_rows": 184,
        },
        "forbidden_scope": config["development_populations"]["forbidden"],
        "payload_files_opened_before_this_record": 0,
        "scores_computed_before_this_record": 0,
        "authorization_replay_allowed": False,
    }
    access_intent["content_sha256"] = content_sha256(access_intent)
    _atomic_no_clobber(ACCESS_INTENT_PATH, access_intent)

    # The first target access in this module occurs here, after immutable access intent.
    sparc = load_sparc_development(config)
    xcop = load_xcop_development(config, inventory)
    scores = _execute_scoring(config, registered, sparc, xcop)
    public_variants = [
        {key: value for key, value in row.items() if key != "canonical_ast"}
        for row in registered["variants"]
    ]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "development_only_scoring_complete",
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": file_sha256(confined(authorization_path)),
        "access_intent_content_sha256": access_intent["content_sha256"],
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "registered_formula_ledger": public_variants,
        "candidate_and_ablation_accounting": registered["accounting"],
        "development_populations": preflight["development_populations"],
        "scores": scores,
        "claim_ceiling": config["claim_ceiling"],
        "claims": {
            "development_only_score": True,
            "fresh_confirmation": False,
            "absolute_cluster_prediction": False,
            "full_covariance": False,
            "historical_novelty_established": False,
            "dark_matter_eliminated": False,
            "alternative_to_gr_established": False,
            "publication_ready": False,
            "formula_family_pruned": False,
        },
    }
    result["content_sha256"] = content_sha256(result)
    _atomic_no_clobber(RESULT_PATH, result)
    return confined(RESULT_PATH)


def check_preflight() -> dict[str, Any]:
    config = load_config()
    validate_bound_metadata(config)
    receipt = validate_preflight(config)
    authorization = read_json(AUTHORIZATION_PATH)
    expected = authorization_template(config, receipt)
    if authorization != expected:
        raise BENDevelopmentExecutorV4Error(
            "current authorization is not the sealed false template"
        )
    if confined(ACCESS_INTENT_PATH).exists() or confined(RESULT_PATH).exists():
        raise BENDevelopmentExecutorV4Error("preflight is no longer zero-access")
    return {
        "ok": True,
        "decision": DECISION,
        "authorization_id": expected["authorization_id"],
        "authorized": False,
        "target_files_opened": 0,
        "scores_computed": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "write-unauthorized", "check-preflight", "execute"),
    )
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(canonical_json({"path": str(write_preflight()), "decision": DECISION}))
        return 0
    if args.command == "write-unauthorized":
        print(canonical_json({"path": str(write_unauthorized_template()), "authorized": False}))
        return 0
    if args.command == "check-preflight":
        print(canonical_json(check_preflight()))
        return 0
    if args.authorization is None:
        raise BENDevelopmentExecutorV4Error("--authorization is required for execute")
    print(canonical_json({"path": str(execute(args.authorization))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
