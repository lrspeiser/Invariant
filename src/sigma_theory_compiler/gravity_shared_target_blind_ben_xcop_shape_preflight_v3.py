"""Metadata-only V3 preflight for the B+E+N X-COP shape mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_xcop_shape_preflight_v3.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_xcop_shape_preflight_v3.py")
AUTHORIZATION_PATH = Path(
    "runs/gravity/shared-target-blind-ben-xcop-shape-preflight-v3/authorization-v1.json"
)
RECEIPT_PATH = Path("runs/gravity/shared-target-blind-ben-xcop-shape-preflight-v3.json")

CONFIG_SCHEMA = "invariant-gravity-ben-xcop-shape-preflight-config-3.0"
AUTHORIZATION_SCHEMA = "invariant-gravity-ben-xcop-shape-authorization-3.0"
RECEIPT_SCHEMA = "invariant-gravity-ben-xcop-shape-preflight-receipt-3.0"
DECISION = "READY_PREDICTOR_ONLY_XCOP_SHAPE_BASIS_RESPONSE_PROFILED_NUISANCE_UNAUTHORIZED_NO_SCORE"
CONFIG_FILE_SHA256 = "c90800e68fc3b8e22f48149072a3b18f03ac91da735d419cf5b8268e386a5da5"
TEST_FILE_SHA256 = "52606503cfa5342c1187912191daacd89c097f852fa6fb7b28df3f82a8c60d67"

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
EXPECTED_ACKNOWLEDGEMENTS = {
    "all_local_sparc_development_only": True,
    "no_local_sparc_confirmation_claim": True,
    "xcop_development_only": True,
    "shape_only_no_absolute_normalization_claim": True,
    "no_single_counterexample_veto": True,
    "no_family_pruning": True,
    "no_publication_or_gr_replacement_claim": True,
    "diagonal_errors_not_full_covariance": True,
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "purpose",
    "implementation_source",
    "verifier_test",
    "source_bindings",
    "lineage",
    "populations",
    "predictor_only_input_mapping",
    "shape_output_mapping",
    "xcop_comparator_contract",
    "nuisance_contract",
    "score_contract",
    "selection_and_ablation",
    "compute_ceiling",
    "zero_access_chronology",
    "production_gate",
    "approval_schema",
    "claim_boundary",
    "output_paths",
}
EXPECTED_AUTHORIZATION_KEYS = {
    "schema_version",
    "authorization_id",
    "authorized",
    "approved_by",
    "approved_at",
    "config_file_sha256",
    "preflight_receipt_file_sha256",
    "preflight_receipt_content_sha256",
    "v2_freeze_commit",
    "candidate_registry_content_sha256",
    "sparc_role",
    "sparc_objects",
    "sparc_rows",
    "xcop_role",
    "xcop_objects",
    "xcop_predictor_rows",
    "xcop_response_rows",
    "compute_ceiling",
    "claim_acknowledgements",
}

SECTION_SHA256 = {
    "source_bindings": "1dbf343c8324f4688d8953cd30c961053897b861915cc7ded150a47e5f427554",
    "lineage": "87a06e875540f5cb0fa8c11f6adf6d0f6c2a44175d95a3b836b7d4f2d8e13a56",
    "populations": "c7a486bf71ab112d908633c817ac6f4711754d78cb91965cbafa45c83dd19c60",
    "predictor_only_input_mapping": "38c264413718692e89b974f9950faa287dcb2fa625982145aba422cafa230842",
    "shape_output_mapping": "4e6606226307e10891988d62761a3cad3d50a084f781d070a6c2522f1282ff28",
    "xcop_comparator_contract": "438dad8aa4c7a3d5bf8c86de23aae1869895fc787d8d085d5be4975bd8380ffa",
    "nuisance_contract": "c90721eb018af0fa2a34cc971f32b578e7e9b252abbc5ba1bcb06704776e095c",
    "score_contract": "c4ef372a3f8904c33ef339444b1ff396ab9814781f0c488ee3c63a460eb8c567",
    "selection_and_ablation": "6429df8df06e5bcfc6883feb65f51b652d71dd0eb89b6be118a73fcc0e905838",
    "compute_ceiling": "aeaab2dd5f81a710d766e79d8756fa2685dadb1c93ef02060070ed9e59abce8e",
    "zero_access_chronology": "3c969cb59e224168825d99033e22d11aa8e7bce8e4a620d6ca053cd210d3988f",
    "production_gate": "18e7626f08047b2450431300d9a33daa04c28700d698c9e53ca78ca2cff343e1",
    "approval_schema": "e814827ae5b6c4985be815b026df34f7371eb33a3c6df8c9517c01c05edda4e9",
    "claim_boundary": "5c6d64aedc35b0bb59e4c270795a0e23c83ceade233ef0e01e3f8b892200ba7d",
    "output_paths": "b11788a9598eab5b06164a04f873693f3649a818467f0bfcd6bd74e40d2618b9",
}


class BENXCOPShapePreflightV3Error(RuntimeError):
    """Raised when the V3 shape mapping or access boundary changes."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def section_sha256(value: Any) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENXCOPShapePreflightV3Error(f"path escaped repository: {path}") from error
    return target


def read_json(path: Path) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise BENXCOPShapePreflightV3Error(f"required artifact absent: {path}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BENXCOPShapePreflightV3Error(f"expected JSON object: {path}")
    return value


def strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BENXCOPShapePreflightV3Error(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    strict_keys(config, EXPECTED_TOP_LEVEL_KEYS, "V3 config")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"]
        != "predictor_only_shape_basis_frozen_response_profiled_nuisance_unauthorized"
        or config["purpose"]
        != (
            "Freeze a predictor-only X-COP pressure-drop shape basis and a target-independent "
            "nuisance/scoring contract for the committed B+E+N registry without opening or "
            "scoring any SPARC or X-COP payload row."
        )
        or config["implementation_source"]
        != "src/sigma_theory_compiler/gravity_shared_target_blind_ben_xcop_shape_preflight_v3.py"
        or config["verifier_test"]
        != {"path": TEST_PATH.as_posix(), "file_sha256": TEST_FILE_SHA256}
    ):
        raise BENXCOPShapePreflightV3Error("V3 identity changed")
    for section, expected in SECTION_SHA256.items():
        if section_sha256(config[section]) != expected:
            raise BENXCOPShapePreflightV3Error(f"frozen {section} changed")

    sparc = config["populations"]["sparc"]
    xcop = config["populations"]["xcop"]
    if (
        sparc["all_locally_accessible_rows_role"] != "development_only_for_this_descendant"
        or sparc["local_confirmation_role_exists"] is not False
        or sparc["local_confirmation_claim_allowed"] is not False
        or sparc["planned_score_subset_objects"] != 139
        or sparc["planned_score_subset_rows"] != 2720
        or sparc["rows_outside_planned_subset_scored"] != 0
        or xcop["objects"] != EXPECTED_XCOP_OBJECTS
        or xcop["predictor_density_rows_metadata_count"] != 521
        or xcop["pressure_temperature_response_rows_metadata_count"] != 184
    ):
        raise BENXCOPShapePreflightV3Error("development population boundary weakened")

    inputs = config["predictor_only_input_mapping"]
    mapping = config["shape_output_mapping"]
    if (
        inputs["response_fields_used_to_construct_candidate_input_or_shape"] != []
        or inputs["object_label_used_as_predictor"] is not False
        or inputs["domain_label_used_as_predictor"] is not False
        or not all(
            inputs[key] is True
            for key in (
                "no_r500_required",
                "no_p500_required",
                "no_t500_required",
                "no_outer_pressure_required",
            )
        )
        or mapping["mapping_ready"] is not True
        or mapping["dimensionless_drop_equation"] != "H_c(x)=integral_from_x_to_1 q(u)*f_c(u) du"
        or mapping["physical_equation"] != "dP_e/dr=-mu_gas_particle*m_p*n_e(r)*g_c(r)"
        or mapping["composition_coefficient"]["symbol"] != "mu_gas_particle"
        or mapping["composition_coefficient"]["value"] != 0.61
        or "not mean mass per electron" not in mapping["composition_coefficient"]["definition"]
        or mapping["shared_shape_equation"] != "phi_c(x;zeta)=zeta+(1-zeta)*H_c(x), with 0<=zeta<=1"
        or mapping["pressure_shape_equation"] != "P_SZ_hat(x)=b_P*phi_c(x;zeta)"
        or mapping["temperature_shape_equation"] != "T_X_hat(x)=b_T*phi_c(x;zeta)/q(x)"
        or mapping["H_c_is_target_independent"] is not True
        or mapping["observable_normalization_and_boundary_are_response_profiled_at_scoring"]
        is not True
        or mapping["profiled_nuisances_do_not_enter_H_c_or_candidate_generation"] is not True
        or mapping["mapping_never_uses"]
        != [
            "P500",
            "T500",
            "outermost_P_SZ_as_anchor",
            "R500",
            "hydrostatic_mass",
            "lensing",
            "inferred_total_mass",
        ]
    ):
        raise BENXCOPShapePreflightV3Error("target-independent shape mapping weakened")

    comparators = config["xcop_comparator_contract"]
    if (
        comparators["frozen_before_response_access"] is not True
        or comparators["cross_domain_comparator_identity_differs"] is not True
        or comparators["sparc_comparators_unchanged"] != ["newtonian_baryons", "empirical_rar"]
        or comparators["xcop_comparators"]
        != ["gas_only_newtonian_shape", "uniform_acceleration_shape_control"]
        or comparators["gas_only_newtonian_shape"]["raw_acceleration_shape"]
        != "f_N_raw(x)=m_q(x)/x^2"
        or comparators["gas_only_newtonian_shape"]["inner_continuation"]
        != "q(u)=q(x_min) for 0<=u<x_min"
        or "gas-only" not in comparators["gas_only_newtonian_shape"]["stellar_handling"]
        or comparators["uniform_acceleration_shape_control"]["equation"]
        != "f_U(x)=1 on every predictor node"
        or comparators["post_response_comparator_choice_allowed"] is not False
    ):
        raise BENXCOPShapePreflightV3Error("X-COP comparator contract changed")

    nuisance = config["nuisance_contract"]
    score = config["score_contract"]
    if (
        nuisance["nuisances_are_never_candidate_inputs"] is not True
        or nuisance["nuisances_are_never_reused_between_candidates_or_comparators"] is not True
        or nuisance["nuisance_fit_is_development_only_not_confirmation"] is not True
        or nuisance["nuisances_per_cluster_per_candidate_or_comparator"] != ["zeta", "b_P", "b_T"]
        or nuisance["zeta_grid_points"] != 4097
        or nuisance["parameter_count_eight_clusters"] != 24
        or nuisance["aggregate_nominal_residual_degrees_of_freedom"] != 160
        or nuisance["four_independent_affine_nuisance_control_scored"] is not False
        or nuisance["matched_control_rule"]
        != (
            "every X-COP full candidate, X-COP ablation, gas-only Newtonian shape control, "
            "and uniform-acceleration shape control receives exactly the same three-parameter "
            "coupled profile; empirical RAR remains SPARC-only"
        )
        or nuisance["response_node_jacobian"]["rank_required"] != 3
        or nuisance["response_node_jacobian"]["relative_singular_value_floor"]
        != 1.4901161193847656e-8
        or nuisance["response_node_jacobian"]["maximum_condition_number"] != 67108864.0
        or nuisance["response_node_jacobian"]["pressure_row"] != ["b_P*(1-H)", "phi", "0"]
        or nuisance["response_node_jacobian"]["temperature_row"] != ["b_T*(1-H)/q", "0", "phi/q"]
        or "response-fitted" not in nuisance["response_node_jacobian"]["evaluation_point"]
        or "not response-independent" not in nuisance["response_node_jacobian"]["weighting"]
        or "generic rank possibility"
        not in nuisance["response_node_jacobian"]["generic_structural_rank_precheck"]
        or "only after authorized scoring"
        not in nuisance["response_node_jacobian"]["conditioning_evidence_timing"]
        or "neither actual rank nor empirical conditioning"
        not in nuisance["response_node_jacobian"]["pre_score_claim"]
        or score["primary_metric_id"] != "equal_object_mean_squared_standardized_residual"
        or score["comparators_by_domain"]
        != {
            "sparc": ["newtonian_baryons", "empirical_rar"],
            "xcop": [
                "gas_only_newtonian_shape",
                "uniform_acceleration_shape_control",
            ],
        }
        or score["numeric_improvement_threshold"] is not None
        or score["single_counterexample_terminal"] is not False
        or score["finite_sample_may_prune_formula_family"] is not False
        or score["xcop_primary_is_joint_coupled_shape_score"] is not True
        or score["not_absolute_joint_physical_prediction"] is not True
    ):
        raise BENXCOPShapePreflightV3Error("nuisance or score boundary weakened")

    selection = config["selection_and_ablation"]
    structure = selection["shape_structure_gates"]
    if (
        "H_nonnegative_outer_zero_negative_derivative_gate" not in selection["eligibility"]
        or "pressure_nonincreasing_outward_gate" not in selection["eligibility"]
        or "response_node_jacobian_rank3_condition_gate" not in selection["eligibility"]
        or "no fixed sign" not in structure["temperature_slope"]
        or "H>=-tol_H" not in structure["nonnegative"]
        or "H(x_outer)" not in structure["outer_boundary"]
        or "<0 within tol_d" not in structure["derivative"]
        or "nonincreasing outward" not in structure["pressure_slope"]
    ):
        raise BENXCOPShapePreflightV3Error("shape structure gates changed")

    ceiling = config["compute_ceiling"]
    if (
        ceiling["ablation_variants"] != 3 * 60
        or ceiling["formula_domain_batches_per_backend"] != (60 + 180) * 2 + 4
        or ceiling["sparc_formula_row_cells_per_backend"] != 242 * 2720
        or ceiling["xcop_formula_row_cells_per_backend"] != 242 * 521
        or ceiling["total_formula_row_cells_both_backends"] != 2 * 784322
        or ceiling["xcop_coupled_three_parameter_nuisance_fits"] != 242 * 8
        or ceiling["xcop_zeta_objective_evaluations"] != 242 * 8 * 4097
        or ceiling["xcop_analytic_scale_solves"] != 2 * 242 * 8 * 4097
        or ceiling["maximum_object_score_reductions"] != 242 * (139 + 8)
        or ceiling["maximum_response_row_score_terms"] != 242 * (2720 + 184)
        or ceiling["maximum_payload_file_opens"] != 1 + 8 * 3
        or ceiling["network_calls"] != 0
        or ceiling["model_calls"] != 0
        or ceiling["paid_calls"] != 0
        or ceiling["maximum_api_spend_usd"] != 0.0
    ):
        raise BENXCOPShapePreflightV3Error("compute ceiling changed")

    chronology = config["zero_access_chronology"]
    if chronology["v3_contract_frozen_before_payload_access"] is not True or any(
        value != 0
        for key, value in chronology.items()
        if key != "v3_contract_frozen_before_payload_access"
    ):
        raise BENXCOPShapePreflightV3Error("zero-access chronology changed")
    gate = config["production_gate"]
    claims = config["claim_boundary"]
    if (
        gate["payload_loader_present_in_v3"] is not False
        or gate["scoring_executor_present_in_v3"] is not False
        or gate["refuse_unauthorized_before_payload_load"] is not True
        or claims["production_authorized"] is not False
        or claims["real_scoring_executed"] is not False
        or claims["absolute_pressure_or_temperature_prediction"] is not False
        or claims["predictor_only_xcop_shape_basis_frozen"] is not True
        or claims["parameter_free_target_independent_observable_mapping"] is not False
        or claims["response_profiled_nuisance_required_at_scoring"] is not True
        or claims["xcop_newtonian_control_is_gas_only"] is not True
        or claims["xcop_empirical_rar_control_defined"] is not False
        or claims["xcop_comparators_frozen_predictor_only"] is not True
        or claims["response_node_identifiability_gate_frozen"] is not True
        or claims["local_sparc_confirmation_claim_survives"] is not False
        or claims["scientific_claim_allowed_now"] is not False
    ):
        raise BENXCOPShapePreflightV3Error("production or claim boundary weakened")


def validate_bound_files(config: Mapping[str, Any]) -> None:
    test = confined(ROOT / str(config["verifier_test"]["path"]))
    if not test.is_file() or file_sha256(test) != TEST_FILE_SHA256:
        raise BENXCOPShapePreflightV3Error("V3 test binding changed")
    for label, binding in config["source_bindings"].items():
        target = confined(ROOT / str(binding["path"]))
        if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
            raise BENXCOPShapePreflightV3Error(f"source binding missing or changed: {label}")
        if "content_sha256" in binding:
            value = read_json(target)
            if value.get("content_sha256") != binding["content_sha256"]:
                raise BENXCOPShapePreflightV3Error(f"source content binding changed: {label}")


def load_config() -> dict[str, Any]:
    target = confined(ROOT / CONFIG_PATH)
    if file_sha256(target) != CONFIG_FILE_SHA256:
        raise BENXCOPShapePreflightV3Error("V3 config hash changed")
    config = read_json(target)
    validate_config(config)
    validate_bound_files(config)
    return config


def validate_authorization(
    authorization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    require_authorized: bool,
) -> None:
    strict_keys(authorization, EXPECTED_AUTHORIZATION_KEYS, "authorization")
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["config_file_sha256"] != CONFIG_FILE_SHA256
        or authorization["v2_freeze_commit"] != config["lineage"]["v2_freeze_commit"]
        or authorization["candidate_registry_content_sha256"]
        != config["lineage"]["v2_candidate_registry_content_sha256"]
        or authorization["sparc_role"] != "development_only"
        or authorization["sparc_objects"] not in (0, 139)
        or authorization["sparc_rows"] not in (0, 2720)
        or authorization["xcop_role"] != "development_only"
        or authorization["xcop_objects"] not in ([], EXPECTED_XCOP_OBJECTS)
        or authorization["xcop_predictor_rows"] not in (0, 521)
        or authorization["xcop_response_rows"] not in (0, 184)
        or authorization["claim_acknowledgements"] != EXPECTED_ACKNOWLEDGEMENTS
    ):
        raise BENXCOPShapePreflightV3Error("authorization boundary changed")
    if authorization["authorized"] is False:
        if require_authorized:
            raise BENXCOPShapePreflightV3Error("UNAUTHORIZED_BEFORE_PAYLOAD_LOAD")
        expected_empty = {
            "authorization_id": None,
            "approved_by": None,
            "approved_at": None,
            "preflight_receipt_file_sha256": None,
            "preflight_receipt_content_sha256": None,
            "sparc_objects": 0,
            "sparc_rows": 0,
            "xcop_objects": [],
            "xcop_predictor_rows": 0,
            "xcop_response_rows": 0,
            "compute_ceiling": {
                "cpu_formula_domain_batches": 0,
                "gpu_formula_domain_batches": 0,
                "cpu_gpu_parity_comparisons": 0,
                "xcop_coupled_three_parameter_nuisance_fits": 0,
                "payload_file_opens": 0,
                "network_calls": 0,
                "model_calls": 0,
                "paid_calls": 0,
                "maximum_api_spend_usd": 0.0,
            },
        }
        if any(authorization[key] != value for key, value in expected_empty.items()):
            raise BENXCOPShapePreflightV3Error("unauthorized manifest grants work")
        return
    if authorization["authorized"] is not True:
        raise BENXCOPShapePreflightV3Error("authorization flag is not boolean")
    receipt = read_json(ROOT / RECEIPT_PATH)
    if (
        not authorization["authorization_id"]
        or not authorization["approved_by"]
        or not authorization["approved_at"]
        or authorization["preflight_receipt_file_sha256"] != file_sha256(ROOT / RECEIPT_PATH)
        or authorization["preflight_receipt_content_sha256"] != receipt["content_sha256"]
        or authorization["sparc_objects"] != 139
        or authorization["sparc_rows"] != 2720
        or authorization["xcop_objects"] != EXPECTED_XCOP_OBJECTS
        or authorization["xcop_predictor_rows"] != 521
        or authorization["xcop_response_rows"] != 184
        or authorization["compute_ceiling"] != config["compute_ceiling"]
    ):
        raise BENXCOPShapePreflightV3Error("authorized artifact is incomplete")


def artifact_binding(path: Path, *, content: bool = False) -> dict[str, Any]:
    target = confined(path)
    binding: dict[str, Any] = {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }
    if content:
        binding["content_sha256"] = read_json(target)["content_sha256"]
    return binding


def build_receipt(config: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "v3_predictor_only_shape_basis_frozen_response_profiled_nuisance_no_payload_access",
        "decision": DECISION,
        "evidence": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "source": artifact_binding(Path(__file__)),
            "tests": artifact_binding(ROOT / TEST_PATH),
            "current_unauthorized_manifest": artifact_binding(ROOT / AUTHORIZATION_PATH),
            **{label: dict(binding) for label, binding in config["source_bindings"].items()},
        },
        "lineage": config["lineage"],
        "populations": config["populations"],
        "predictor_only_input_mapping": config["predictor_only_input_mapping"],
        "shape_output_mapping": config["shape_output_mapping"],
        "xcop_comparator_contract": config["xcop_comparator_contract"],
        "nuisance_contract": config["nuisance_contract"],
        "score_contract": config["score_contract"],
        "selection_and_ablation": config["selection_and_ablation"],
        "future_compute_ceiling": config["compute_ceiling"],
        "zero_access_chronology": config["zero_access_chronology"],
        "production_gate": config["production_gate"],
        "approval_schema": config["approval_schema"],
        "current_authorization": {
            "path": AUTHORIZATION_PATH.as_posix(),
            "authorized": False,
            "authorized_payload_file_opens": 0,
            "authorized_cpu_formula_domain_batches": 0,
            "authorized_gpu_formula_domain_batches": 0,
            "authorized_paid_calls": 0,
        },
        "claims": config["claim_boundary"],
        "limitations": [
            "The mapping tests dimensionless radial shape only; absolute pressure, temperature, acceleration amplitude, and boundary pressure are not identified.",
            "One shared boundary-shape parameter and two observable scales are profiled for every cluster and candidate or comparator, so a future pass is development-only coupled shape compatibility rather than an absolute prediction.",
            "Spherical thermal hydrostatic equilibrium and zero nonthermal pressure are assumptions, and released diagonal errors are not a complete joint covariance.",
            "The X-COP Newtonian control is gas-only; stellar profiles are not opened, and empirical RAR is not defined because absolute g_bar/a0 is unavailable under the predictor-only contract.",
            "All local SPARC data and all eight X-COP objects remain development-only; no confirmation or publication claim survives.",
            "No payload, nuisance fit, metric, score, selection, model call, or paid call exists in V3.",
        ],
    }
    return {**body, "content_sha256": content_sha256(body)}


def write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as error:
        raise BENXCOPShapePreflightV3Error(
            f"atomic no-clobber publication refused existing artifact: {target}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write() -> dict[str, Any]:
    config = load_config()
    authorization = read_json(ROOT / AUTHORIZATION_PATH)
    validate_authorization(authorization, config, require_authorized=False)
    receipt = build_receipt(config, authorization)
    write_json_no_clobber(ROOT / RECEIPT_PATH, receipt)
    return receipt


def check() -> dict[str, Any]:
    config = load_config()
    authorization = read_json(ROOT / AUTHORIZATION_PATH)
    validate_authorization(authorization, config, require_authorized=False)
    receipt = read_json(ROOT / RECEIPT_PATH)
    if receipt != build_receipt(config, authorization):
        raise BENXCOPShapePreflightV3Error("V3 receipt does not reconstruct")
    return {
        "valid": True,
        "decision": receipt["decision"],
        "mapping_ready": True,
        "authorized": False,
        "payload_rows_read": 0,
        "real_candidates_scored": 0,
        "receipt_sha256": file_sha256(ROOT / RECEIPT_PATH),
    }


def production_preflight(authorization_path: Path) -> dict[str, Any]:
    config = load_config()
    stored = read_json(ROOT / RECEIPT_PATH)
    current = read_json(ROOT / AUTHORIZATION_PATH)
    if stored != build_receipt(config, current):
        raise BENXCOPShapePreflightV3Error("frozen V3 preflight receipt changed")
    authorization = read_json(authorization_path)
    validate_authorization(authorization, config, require_authorized=True)
    return {
        "metadata_ready": True,
        "authorization_id": authorization["authorization_id"],
        "payload_loader_present": False,
        "successor_executor_required": True,
        "payload_rows_read": 0,
        "real_candidates_scored": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write")
    commands.add_parser("check")
    production = commands.add_parser("production-preflight")
    production.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write":
        print(json.dumps(write(), sort_keys=True))
    elif args.command == "check":
        print(json.dumps(check(), sort_keys=True))
    else:
        print(json.dumps(production_preflight(args.authorization), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
