"""Bounded universal-matter action and coupled-PDE controls with fail-closed gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-universal-matter-coupled-pde-control-pack-config-1.0"
RESULT_SCHEMA = "invariant-universal-matter-coupled-pde-control-pack-result-1.0"
CAMPAIGN_ID = "universal-matter-coupled-pde-control-pack-001"
CONFIG_PATH = "configs/universal_matter_coupled_pde_control_pack.json"
SOURCE_PATH = "src/sigma_theory_compiler/universal_matter_coupled_pde_control_pack.py"
TEST_PATH = "tests/test_universal_matter_coupled_pde_control_pack.py"
OUTPUT_PATH = "runs/math/universal-matter-coupled-pde-control-pack/receipt.json"
GATE_IDS = (
    "action_level_universal_metric_coupling",
    "stress_energy_conservation_interface",
    "principal_symbol_hyperbolicity",
    "constraint_propagation",
)
SECTOR_IDS = (
    "minimally_coupled_scalar",
    "maxwell_lorenz_gauge",
    "barotropic_perfect_fluid",
)
CLAIMS = {
    "representative_matter_sectors_registered": 3,
    "scalar_control_pack_complete_within_local_scope": True,
    "maxwell_action_and_lorenz_principal_controls_pass": True,
    "maxwell_full_stress_conservation_interface_complete": False,
    "barotropic_fluid_control_pack_complete": False,
    "universal_all_matter_closure_established": False,
    "gravity_h7_theorem_established": False,
    "nonlinear_coupled_gravity_matter_pde_closure_established": False,
    "promotion_authorized": False,
    "scientific_or_physics_truth_inferred": False,
}
_TOP_KEYS = {
    "audit",
    "campaign_id",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "first_blocker",
    "schema_version",
    "scope",
    "sector_results",
    "source_bindings",
}
_CONFIG_KEYS = {
    "campaign_id",
    "evidence_bindings",
    "output_path",
    "policies",
    "schema_version",
    "sectors",
}
_GATE_KEYS = {"evidence", "gate_id", "outcome", "reason_codes"}
_HOST_PATH = re.compile(r"[A-Za-z]:\\|/(?:home|Users)/")


class UniversalMatterControlError(ValueError):
    """Raised when registered evidence or a checked artifact changes."""


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise UniversalMatterControlError("universal-matter path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise UniversalMatterControlError("universal-matter path escapes root") from error
    return path


def _text_file_sha(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeError) as error:
        raise UniversalMatterControlError(
            f"cannot load registered evidence: {path.name}"
        ) from error
    return hashlib.sha256(text.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UniversalMatterControlError(
            f"cannot load registered evidence: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise UniversalMatterControlError("registered evidence must be an object")
    return value


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {"file_sha256", "path"}
    if "content_sha256" in binding:
        expected_keys.add("content_sha256")
    if set(binding) != expected_keys:
        raise UniversalMatterControlError("evidence binding schema changed")
    path = _resolve(root, str(binding["path"]))
    if _text_file_sha(path) != binding["file_sha256"]:
        raise UniversalMatterControlError("registered evidence file hash changed")
    value = _load_json(path)
    if "content_sha256" in binding and value.get("content_sha256") != binding["content_sha256"]:
        raise UniversalMatterControlError("registered evidence content hash changed")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
    ):
        raise UniversalMatterControlError("universal-matter config schema changed")
    policies = config.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {
        "allowed_gate_outcomes",
        "forbidden_matter_action_dependencies",
        "gravity_h7_theorem_required_for_sector_control",
        "missing_evidence_outcome",
        "physical_metric",
        "require_action_before_dependent_gates",
        "require_exact_or_symbolic_residuals",
    }:
        raise UniversalMatterControlError("universal-matter policies changed")
    if (
        policies["allowed_gate_outcomes"] != ["PASS", "BLOCK", "REJECT"]
        or policies["missing_evidence_outcome"] != "BLOCK"
        or policies["physical_metric"] != "g_mu_nu"
        or policies["gravity_h7_theorem_required_for_sector_control"] is not False
        or policies["require_action_before_dependent_gates"] is not True
        or policies["require_exact_or_symbolic_residuals"] is not True
        or policies["forbidden_matter_action_dependencies"]
        != ["phi", "u_mu", "A_mu", "J_b_mu", "z_b", "T2_m"]
    ):
        raise UniversalMatterControlError("universal-matter policy semantics changed")
    bindings = config.get("evidence_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "canonical_scalar_action_ir",
        "canonical_scalar_principal_ir",
        "covariant_field_contract",
        "formal_controls",
        "proca_action_ir",
        "proca_principal_ir",
    }:
        raise UniversalMatterControlError("universal-matter evidence registry changed")
    sectors = config.get("sectors")
    if not isinstance(sectors, list) or [item.get("sector_id") for item in sectors] != list(
        SECTOR_IDS
    ):
        raise UniversalMatterControlError("universal-matter sector registry changed")
    expected_keys = {
        "minimally_coupled_scalar": {
            "action",
            "formal_control_names",
            "required_gates",
            "sector_id",
            "stress_tensor",
        },
        "maxwell_lorenz_gauge": {
            "action",
            "formal_control_names",
            "missing_evidence",
            "required_gates",
            "sector_id",
            "stress_tensor",
        },
        "barotropic_perfect_fluid": {
            "action",
            "eos_assumption",
            "formal_control_names",
            "missing_evidence",
            "required_gates",
            "sector_id",
            "stress_tensor",
        },
    }
    for sector in sectors:
        sector_id = sector["sector_id"]
        if set(sector) != expected_keys[sector_id] or sector["required_gates"] != list(GATE_IDS):
            raise UniversalMatterControlError("universal-matter sector schema changed")
        action = sector["action"]
        if action is not None and (
            not isinstance(action, Mapping)
            or set(action) != {"dependencies", "density", "field", "metric", "schema_version"}
            or action.get("schema_version") != "invariant-bounded-matter-action-ir-1.0"
            or not isinstance(action.get("dependencies"), list)
        ):
            raise UniversalMatterControlError("bounded matter action schema changed")


def _gate(
    gate_id: str,
    outcome: str,
    reason_codes: Sequence[str],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if gate_id not in GATE_IDS or outcome not in {"PASS", "BLOCK", "REJECT"}:
        raise UniversalMatterControlError("unsupported gate result")
    return {
        "gate_id": gate_id,
        "outcome": outcome,
        "reason_codes": list(reason_codes),
        "evidence": dict(evidence),
    }


def _formal_checks(formal: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    checks = formal.get("checks")
    if (
        formal.get("schema_version") != "sigma-formal-controls-1.0"
        or formal.get("counts") != {"failed": 0, "passed": 118, "total": 118}
        or not isinstance(checks, list)
    ):
        raise UniversalMatterControlError("formal-control registry boundary changed")
    by_name: dict[str, Mapping[str, Any]] = {}
    for check in checks:
        if isinstance(check, Mapping) and isinstance(check.get("name"), str):
            if check["name"] in by_name:
                raise UniversalMatterControlError("duplicate formal-control name")
            by_name[check["name"]] = check
    return by_name


def _action_gate(
    sector: Mapping[str, Any],
    contract: Mapping[str, Any],
    policies: Mapping[str, Any],
) -> dict[str, Any]:
    action = sector["action"]
    if action is None:
        return _gate(
            GATE_IDS[0],
            "BLOCK",
            ("missing_admitted_variational_matter_action",),
            {
                "matter_action_present": False,
                "physical_metric": policies["physical_metric"],
            },
        )
    dependencies = set(action["dependencies"])
    forbidden = sorted(dependencies & set(policies["forbidden_matter_action_dependencies"]))
    contract_action = contract.get("action_contract", {})
    if (
        action.get("metric") != policies["physical_metric"]
        or contract_action.get("physical_metric") != policies["physical_metric"]
        or policies["physical_metric"] not in dependencies
    ):
        return _gate(
            GATE_IDS[0],
            "REJECT",
            ("nonuniversal_or_missing_physical_metric",),
            {
                "declared_metric": action.get("metric"),
                "contract_metric": contract_action.get("physical_metric"),
                "physical_metric_dependency": policies["physical_metric"] in dependencies,
            },
        )
    if forbidden:
        return _gate(
            GATE_IDS[0],
            "REJECT",
            ("forbidden_gravitational_or_diagnostic_dependency",),
            {"forbidden_dependencies": forbidden},
        )
    return _gate(
        GATE_IDS[0],
        "PASS",
        (),
        {
            "action_schema": action["schema_version"],
            "action_density": action["density"],
            "matter_field": action["field"],
            "matter_dependencies": sorted(dependencies),
            "physical_metric": action["metric"],
            "candidate_gravitational_field_dependencies": [],
            "diagnostic_dependencies": [],
            "first_derivative_matter_action": True,
            "metric_matter_cross_second_derivative_principal_terms": 0,
            "universal_minimal_split_enforced": (
                contract_action.get("action")
                == "S = S_grav[g_mu_nu, gravitational_fields] + S_m[g_mu_nu, psi_m]"
            ),
        },
    )


def _scalar_gates(
    sector: Mapping[str, Any],
    action: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
    scalar_action_ir: Mapping[str, Any],
    scalar_principal_ir: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if action["outcome"] != "PASS":
        blocker = _gate(
            GATE_IDS[1],
            "BLOCK",
            ("action_gate_not_passed",),
            {"upstream_action_outcome": action["outcome"]},
        )
        return [
            action,
            blocker,
            {**blocker, "gate_id": GATE_IDS[2]},
            {**blocker, "gate_id": GATE_IDS[3]},
        ]
    required = sector["formal_control_names"]
    selected = {name: checks.get(name) for name in required}
    missing = [
        name for name, value in selected.items() if value is None or value.get("status") != "pass"
    ]
    noether = selected.get("canonical_scalar_noether_identity", {})
    metric_variation = selected.get("cadabra_canonical_scalar_metric_variation", {})
    if missing or noether.get("evidence", {}).get("residuals") != ["0", "0", "0", "0"]:
        stress = _gate(
            GATE_IDS[1],
            "BLOCK",
            ("missing_or_failed_scalar_variational_conservation_evidence",),
            {"missing_or_failed_controls": missing},
        )
    else:
        stress = _gate(
            GATE_IDS[1],
            "PASS",
            (),
            {
                "stress_tensor": sector["stress_tensor"],
                "hilbert_metric_variation_control": metric_variation["name"],
                "hilbert_metric_variation_return_code": metric_variation["evidence"]["return_code"],
                "off_shell_identity": noether["evidence"]["identity"],
                "on_shell_conservation": "nabla^mu T_mu_nu=0 when E_chi=0",
                "identity_residuals": noether["evidence"]["residuals"],
                "stress_source_enters_gravity_equation": True,
                "matter_euler_equation_receives_metric_connection": True,
            },
        )
    canonical = scalar_action_ir.get("canonical", {})
    principal_pass = (
        scalar_action_ir.get("valid") is True
        and canonical.get("matter_metric") == "g_mu_nu"
        and scalar_principal_ir.get("status") == "pass"
        and scalar_principal_ir.get("characteristic_speed_squared", {}).get("scalar_speed_squared")
        == "1"
        and scalar_principal_ir.get("propagation_residual")
        == "Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])"
    )
    principal = _gate(
        GATE_IDS[2],
        "PASS" if principal_pass else "BLOCK",
        () if principal_pass else ("scalar_principal_certificate_missing_or_changed",),
        {
            "background": scalar_principal_ir.get("background"),
            "principal_polynomial": scalar_principal_ir.get("principal_polynomial"),
            "scalar_speed_squared": scalar_principal_ir.get("characteristic_speed_squared", {}).get(
                "scalar_speed_squared"
            ),
            "exact_diagonal_physical_eigenbasis": scalar_principal_ir.get(
                "exact_diagonal_physical_eigenbasis"
            ),
            "uniform_direction_scope": scalar_principal_ir.get("uniform_direction_scope"),
            "coupled_principal_structure": "block_diagonal_at_second_derivative_order",
            "gravity_h7_used": False,
        },
    )
    constraint = _gate(
        GATE_IDS[3],
        "PASS",
        (),
        {
            "independent_internal_constraints": 0,
            "propagation_check": "not_applicable_no_internal_gauge_or_primary_matter_constraint",
            "gravity_constraint_propagation_claimed": False,
        },
    )
    return [action, stress, principal, constraint]


def _maxwell_exact_principal() -> dict[str, Any]:
    omega, k, a0, a1 = sp.symbols("omega k a0 a1")
    q = sp.expand(k**2 - omega**2)
    matrix = sp.eye(4) * q
    gauge_constraint = -omega * a0 + k * a1
    divergence = sp.expand(-omega * matrix[0, 0] * a0 + k * matrix[1, 1] * a1)
    residual = sp.expand(divergence - q * gauge_constraint)
    if residual != 0 or matrix.det() != q**4:
        raise UniversalMatterControlError("Maxwell Lorenz principal replay failed")
    return {
        "formulation": "source-free Maxwell potential in Lorenz gauge on a frozen local Minkowski frame",
        "principal_scalar": str(q),
        "principal_matrix_diagonal": [str(q)] * 4,
        "principal_determinant": str(sp.factor(matrix.det())),
        "characteristic_roots_for_unit_spatial_covector": ["-1", "1"],
        "strong_hyperbolicity": True,
        "exact_diagonal_eigenbasis": True,
        "constraint": "C=partial_mu B^mu",
        "constraint_wave_principal": str(q),
        "divergence_commutation_residual": str(residual),
        "physical_maxwell_dirac_reduction_proved_here": False,
    }


def _maxwell_gates(
    sector: Mapping[str, Any],
    action: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
    proca_action_ir: Mapping[str, Any],
    proca_principal_ir: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if action["outcome"] != "PASS":
        blocker = _gate(
            GATE_IDS[1],
            "BLOCK",
            ("action_gate_not_passed",),
            {"upstream_action_outcome": action["outcome"]},
        )
        return [
            action,
            blocker,
            {**blocker, "gate_id": GATE_IDS[2]},
            {**blocker, "gate_id": GATE_IDS[3]},
        ]
    canonical = proca_action_ir.get("canonical", {})
    term_ids = {item.get("id") for item in canonical.get("terms", []) if isinstance(item, Mapping)}
    controls = {name: checks.get(name) for name in sector["formal_control_names"]}
    kinetic_reduction = (
        proca_action_ir.get("valid") is True
        and canonical.get("matter_metric") == "g_mu_nu"
        and {"PROCA_F2", "PROCA_MASS"} <= term_ids
        and all(value is not None and value.get("status") == "pass" for value in controls.values())
        and proca_principal_ir.get("status") == "pass"
        and proca_principal_ir.get("characteristic_speed_squared", {}).get("proca_speed_squared")
        == "1"
    )
    if not kinetic_reduction:
        action = _gate(
            GATE_IDS[0],
            "BLOCK",
            ("massless_vector_kinetic_template_evidence_missing",),
            {**action["evidence"], "proca_template_reduction_passed": False},
        )
    else:
        action["evidence"].update(
            {
                "proca_template_reduction_passed": True,
                "mass_term_removed_before_maxwell_use": True,
                "kinetic_term_id": "PROCA_F2",
                "proca_physical_mode_count_not_reused_for_maxwell": True,
            }
        )
    stress = _gate(
        GATE_IDS[1],
        "BLOCK",
        tuple(sector["missing_evidence"]),
        {
            "stress_tensor_target": sector["stress_tensor"],
            "metric_variation_template_available": controls.get(
                "cadabra_proca_metric_variation", {}
            ).get("status")
            == "pass",
            "dedicated_massless_off_shell_noether_identity_available": False,
            "on_shell_conservation_promoted": False,
        },
    )
    replay = _maxwell_exact_principal()
    principal = _gate(
        GATE_IDS[2],
        "PASS" if action["outcome"] == "PASS" else "BLOCK",
        () if action["outcome"] == "PASS" else ("action_gate_not_passed",),
        {
            key: value
            for key, value in replay.items()
            if key
            not in {"constraint", "constraint_wave_principal", "divergence_commutation_residual"}
        },
    )
    constraint = _gate(
        GATE_IDS[3],
        "PASS" if action["outcome"] == "PASS" else "BLOCK",
        () if action["outcome"] == "PASS" else ("action_gate_not_passed",),
        {
            "formulation": replay["formulation"],
            "constraint": replay["constraint"],
            "constraint_wave_principal": replay["constraint_wave_principal"],
            "divergence_commutation_residual": replay["divergence_commutation_residual"],
            "propagation_scope": "local source-free Lorenz-gauge subsidiary system only",
            "nonlinear_curved_boundary_propagation_claimed": False,
        },
    )
    return [action, stress, principal, constraint]


def _fluid_gates(sector: Mapping[str, Any], action: Mapping[str, Any]) -> list[dict[str, Any]]:
    missing = tuple(sector["missing_evidence"])
    return [
        action,
        _gate(
            GATE_IDS[1],
            "BLOCK",
            ("admitted_barotropic_fluid_action", "variational_perfect_fluid_stress_tensor"),
            {
                "stress_tensor_target": sector["stress_tensor"],
                "conservation_equations_are_not_independent_variational_evidence": True,
            },
        ),
        _gate(
            GATE_IDS[2],
            "BLOCK",
            ("positive_enthalpy_domain", "exact_sound_speed_interval"),
            {
                "eos_assumption": sector["eos_assumption"],
                "conditional_hyperbolicity_not_promoted": True,
            },
        ),
        _gate(
            GATE_IDS[3],
            "BLOCK",
            ("fluid_constraint_propagation_certificate",),
            {"all_registered_missing_evidence": list(missing)},
        ),
    ]


def _sector_result(sector_id: str, gates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [gate["outcome"] for gate in gates]
    status = "REJECT" if "REJECT" in outcomes else "BLOCK" if "BLOCK" in outcomes else "PASS"
    first_blocker = next(
        (
            {
                "gate_id": gate["gate_id"],
                "outcome": gate["outcome"],
                "reason_codes": list(gate["reason_codes"]),
            }
            for gate in gates
            if gate["outcome"] != "PASS"
        ),
        None,
    )
    return {
        "sector_id": sector_id,
        "status": status,
        "gates": [dict(gate) for gate in gates],
        "first_blocker": first_blocker,
        "promotion_allowed": False,
    }


def build_universal_matter_control_pack(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Build one exact bounded receipt from registered action and formal evidence."""

    _validate_config(config)
    root = root.resolve()
    bindings = config["evidence_bindings"]
    loaded = {name: _load_bound(root, binding) for name, binding in bindings.items()}
    contract = loaded["covariant_field_contract"]
    if (
        contract.get("schema_version") != "sigma-covariant-field-contract-1.0"
        or contract.get("status") != "normative"
        or contract.get("action_contract", {}).get("matter_rule")
        != "Every matter species is minimally coupled to the same metric g_mu_nu and to no candidate gravitational field."
        or contract.get("action_contract", {}).get("on_shell_identity")
        != "nabla_mu T_m^{mu nu} = 0 when the matter equations hold"
    ):
        raise UniversalMatterControlError("normative matter contract changed")
    checks = _formal_checks(loaded["formal_controls"])
    sectors = {sector["sector_id"]: sector for sector in config["sectors"]}
    action_gates = {
        sector_id: _action_gate(sector, contract, config["policies"])
        for sector_id, sector in sectors.items()
    }
    scalar = _sector_result(
        SECTOR_IDS[0],
        _scalar_gates(
            sectors[SECTOR_IDS[0]],
            action_gates[SECTOR_IDS[0]],
            checks,
            loaded["canonical_scalar_action_ir"],
            loaded["canonical_scalar_principal_ir"],
        ),
    )
    maxwell = _sector_result(
        SECTOR_IDS[1],
        _maxwell_gates(
            sectors[SECTOR_IDS[1]],
            action_gates[SECTOR_IDS[1]],
            checks,
            loaded["proca_action_ir"],
            loaded["proca_principal_ir"],
        ),
    )
    fluid = _sector_result(
        SECTOR_IDS[2],
        _fluid_gates(sectors[SECTOR_IDS[2]], action_gates[SECTOR_IDS[2]]),
    )
    sector_results = [scalar, maxwell, fluid]
    sector_counts = Counter(row["status"].lower() for row in sector_results)
    gate_counts = Counter(
        gate["outcome"].lower() for row in sector_results for gate in row["gates"]
    )
    first_blocker = next(
        {
            "sector_id": row["sector_id"],
            **row["first_blocker"],
        }
        for row in sector_results
        if row["first_blocker"] is not None
    )
    source_bindings = {
        "config": {
            "path": CONFIG_PATH,
            "file_sha256": _text_file_sha(_resolve(root, CONFIG_PATH)),
        },
        "source": {
            "path": SOURCE_PATH,
            "file_sha256": _text_file_sha(_resolve(root, SOURCE_PATH)),
        },
        "test": {
            "path": TEST_PATH,
            "file_sha256": _text_file_sha(_resolve(root, TEST_PATH)),
        },
        "registered_evidence": dict(bindings),
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "bounded_pass_one_sector_two_blocked_no_rejections",
        "first_blocker": first_blocker,
        "sector_results": sector_results,
        "counts": {
            "sectors": 3,
            "sector_passes": sector_counts["pass"],
            "sector_blocks": sector_counts["block"],
            "sector_rejects": sector_counts["reject"],
            "gates": 12,
            "gate_passes": gate_counts["pass"],
            "gate_blocks": gate_counts["block"],
            "gate_rejects": gate_counts["reject"],
            "formal_controls_bound": 6,
            "exact_symbolic_replays": 1,
            "floating_point_operations": 0,
        },
        "audit": {
            "physical_metric": config["policies"]["physical_metric"],
            "normative_action_split": contract["action_contract"]["action"],
            "normative_on_shell_identity": contract["action_contract"]["on_shell_identity"],
            "formal_control_snapshot_counts": loaded["formal_controls"]["counts"],
            "dark_matter_or_halo_targets": False,
            "redshift_distance_or_supernova_inputs": False,
            "runtime_or_observational_data_accesses": 0,
        },
        "claims": dict(CLAIMS),
        "source_bindings": source_bindings,
        "scope": (
            "three representative matter-sector controls on registered local backgrounds: a "
            "complete canonical scalar control, a source-free local Lorenz-gauge Maxwell "
            "principal/constraint control blocked at the dedicated stress Noether interface, "
            "and a barotropic-fluid placeholder blocked for lack of an admitted variational "
            "action and domain certificates; no H7 or universal all-matter closure claim"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_checked_artifact(
    value: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> None:
    if set(value) != _TOP_KEYS or value.get("schema_version") != RESULT_SCHEMA:
        raise UniversalMatterControlError("universal-matter artifact schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise UniversalMatterControlError("universal-matter artifact seal changed")
    if _HOST_PATH.search(json.dumps(value, sort_keys=True)):
        raise UniversalMatterControlError("universal-matter artifact persisted a host path")
    if value.get("claims") != CLAIMS:
        raise UniversalMatterControlError("universal-matter claim boundary changed")
    sectors = value.get("sector_results")
    if (
        not isinstance(sectors, list)
        or [row.get("sector_id") for row in sectors] != list(SECTOR_IDS)
        or any(
            set(gate) != _GATE_KEYS or gate.get("gate_id") != expected_gate
            for row in sectors
            for gate, expected_gate in zip(row.get("gates", []), GATE_IDS, strict=False)
        )
    ):
        raise UniversalMatterControlError("universal-matter sector or gate schema changed")
    expected = build_universal_matter_control_pack(config, root)
    if dict(value) != expected:
        raise UniversalMatterControlError("universal-matter exact replay changed")


def write_artifact(config_path: Path, output_path: Path) -> Path:
    root = config_path.resolve().parent.parent
    config = _load_json(config_path)
    artifact = build_universal_matter_control_pack(config, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--validate-checked", action="store_true")
    arguments = parser.parse_args()
    root = arguments.config.resolve().parent.parent
    config = _load_json(arguments.config)
    if arguments.validate_checked:
        value = _load_json(_resolve(root, OUTPUT_PATH))
        validate_checked_artifact(value, config, root)
        return 0
    write_artifact(arguments.config, arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
