"""Register the arbitrary-background principal second-jet covariant projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_variable_sylvester_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
    _linearized_einstein_upper,
)

CONFIG_SCHEMA = "sigma-quartic-principal-second-jet-covariant-projection-config-1.0"
RESULT_SCHEMA = "sigma-quartic-principal-second-jet-covariant-projection-gate-1.0"
CAMPAIGN_ID = "quartic-principal-second-jet-covariant-projection-001"
CONFIG_PATH = "configs/backgrounds/quartic_principal_second_jet_covariant_projection_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_principal_second_jet_covariant_projection_gate.py"
TEST_PATH = "tests/test_quartic_principal_second_jet_covariant_projection_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-principal-second-jet-covariant-projection-gate/campaign.json"
)

COMPLEMENT = {
    "path": (
        "runs/physics-language/quartic-full-coordinate-tangent-complement-checkpoint-gate/"
        "campaign.json"
    ),
    "file_sha256": "5c41974fa4424dc430255e20f4d84bdde7de0d535b54d51eb175b14adce07777",
    "content_sha256": "2d07915da104a709d71df0a2607bd557285f01e73874984b65b53cc74ec8db65",
}
GEOMETRIC = {
    "path": "runs/physics-language/quartic-geometric-jet-campaign/campaign.json",
    "file_sha256": "aa18e643877f4eb7224891e70e929b8d9574a83309aac9007e9f635689d82b65",
    "content_sha256": "3878728a11df567606c18d37cd683ff222a5de20f87248394f7ce75a618562a4",
}
FLAT_TYPED = {
    "path": ("runs/physics-language/quartic-reverse-principal-typed-map-curl-gate/campaign.json"),
    "file_sha256": "4e432566b16e44b7d5ca05a2ce6e60b5ebd849e2fe8c88fa6523297f1fc111b4",
    "content_sha256": "79d06514c1dd8fd7933bdc36b19622fc3cce8ddcaf14712f0b908fbe6c9f2664",
}
FORMULA_SOURCES = {
    "geometric_state_to_jet": {
        "path": "src/sigma_theory_compiler/quartic_geometric_jet_campaign.py",
        "file_sha256": "d0600d6475d32d06a00140ab230aa41b3c057aef7a968163989fc5028d6acd21",
    },
    "flat_linearized_einstein": {
        "path": "src/sigma_theory_compiler/quartic_tc2_variable_sylvester_campaign.py",
        "file_sha256": "5df63ca3084654198c7ca23e8e7ba6e171aadfeff0ab6c5f1d2709b16f20937f",
    },
}
CONTRACT = {
    "candidate_count": 12,
    "coordinate_dimension": 153,
    "principal_second_jet_directions": 99,
    "scalar_hessian_directions": 9,
    "metric_second_jet_directions": 90,
    "previously_registered_unique_directions": 20,
    "newly_projected_unique_directions": 79,
    "remaining_lower_jet_directions": 54,
}
POLICIES = {
    "arbitrary_background_principal_projection": "require_exact_tensor_formula",
    "lower_jet_projection": "fail_closed",
    "D2_entry_promotion": "forbidden_without_D1_DAG_propagation",
    "complete_D2F": "fail_closed",
    "H7": "fail_closed",
    "candidate_rejection": "forbidden",
}
SEALS = {
    "observations_opened": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
    "paid_llm_calls": False,
}
FIRST_BLOCKER = (
    "register_the_54_lower_q_and_p_coordinate_tangents_in_the_nonlinear_covariant_jet_"
    "map_then_propagate_all_153_directions_through_every_full_D1_arithmetic_DAG"
)


class PrincipalSecondJetProjectionError(ValueError):
    """Raised when a projection formula, binding, or claim changes."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise PrincipalSecondJetProjectionError("projection path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise PrincipalSecondJetProjectionError("projection evidence file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise PrincipalSecondJetProjectionError("projection evidence content binding changed")
    return value


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "coordinate_complement": COMPLEMENT,
        "geometric_map": GEOMETRIC,
        "flat_typed_map": FLAT_TYPED,
        "formula_sources": FORMULA_SOURCES,
        "projection_contract": CONTRACT,
        "policies": POLICIES,
        "seals": SEALS,
    }
    if value != expected:
        raise PrincipalSecondJetProjectionError("projection config boundary changed")


def _validate_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for binding in FORMULA_SOURCES.values():
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise PrincipalSecondJetProjectionError("projection formula source changed")
    complement = _load_bound(root, COMPLEMENT)
    geometric = _load_bound(root, GEOMETRIC)
    flat = _load_bound(root, FLAT_TYPED)
    counts = complement.get("gate_counts", {})
    if (
        counts.get("selected_candidates") != 12
        or len(complement.get("basis_registry", {}).get("coordinate_atom_basis", [])) != 153
        or counts.get("existing_unique_coordinate_vectors_per_candidate") != 20
        or counts.get("new_coordinate_tangent_certificates_per_candidate") != 133
        or counts.get("physical_covariant_component_projections_registered") != 0
    ):
        raise PrincipalSecondJetProjectionError("coordinate complement boundary changed")
    if (
        geometric.get("status") != "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
        or geometric.get("counts", {}).get("selected") != 12
        or geometric.get("geometric_control", {}).get("passed") is not True
    ):
        raise PrincipalSecondJetProjectionError("nonlinear geometric map boundary changed")
    if (
        flat.get("claim_seals", {}).get("flat_coordinate_to_covariant_jet_map_registered")
        is not True
        or flat.get("gate_counts", {}).get("typed_map_coordinate_atoms") != 153
        or flat.get("gate_counts", {}).get("typed_map_covariant_jet_symbols") != 24
    ):
        raise PrincipalSecondJetProjectionError("flat typed map boundary changed")
    return complement, geometric, flat


def _inverse_metric_symbols() -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    symbols = sp.symbols("u00 u01 u02 u03 u11 u12 u13 u22 u23 u33")
    matrix = sp.zeros(4)
    for symbol, (left, right) in zip(symbols, SYMMETRIC_METRIC_PAIRS, strict=True):
        matrix[left, right] = symbol
        matrix[right, left] = symbol
    return matrix, symbols


def _d2_component(
    derivative_pair: tuple[int, int], metric_field: int, x: int, y: int, mu: int, nu: int
) -> sp.Expr:
    left, right = SYMMETRIC_METRIC_PAIRS[metric_field]
    weight = SYMMETRIC_METRIC_WEIGHTS[metric_field]
    if (x, y) not in {derivative_pair, derivative_pair[::-1]}:
        return sp.S.Zero
    return sp.S.One / weight if (mu, nu) in {(left, right), (right, left)} else sp.S.Zero


@cache
def _generic_ricci_tangent(derivative_pair: tuple[int, int], metric_field: int) -> sp.Matrix:
    inverse, _ = _inverse_metric_symbols()
    ricci = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            first = sum(
                inverse[rho, sigma]
                * _d2_component(derivative_pair, metric_field, rho, mu, sigma, nu)
                for rho in range(4)
                for sigma in range(4)
            )
            second = sum(
                inverse[rho, sigma]
                * _d2_component(derivative_pair, metric_field, rho, nu, sigma, mu)
                for rho in range(4)
                for sigma in range(4)
            )
            box = sum(
                inverse[rho, sigma]
                * _d2_component(derivative_pair, metric_field, rho, sigma, mu, nu)
                for rho in range(4)
                for sigma in range(4)
            )
            trace = sum(
                inverse[rho, sigma]
                * _d2_component(derivative_pair, metric_field, mu, nu, rho, sigma)
                for rho in range(4)
                for sigma in range(4)
            )
            ricci[mu, nu] = sp.expand((first + second - box - trace) / 2)
    return ricci


@cache
def _generic_einstein_tangent(
    derivative_pair: tuple[int, int], metric_field: int
) -> dict[str, sp.Expr]:
    """Principal derivative of G^mu_nu for arbitrary inverse metric g^mu_nu."""

    inverse, _ = _inverse_metric_symbols()
    ricci = _generic_ricci_tangent(derivative_pair, metric_field)
    scalar = sp.expand(sum(inverse[mu, nu] * ricci[mu, nu] for mu in range(4) for nu in range(4)))
    result: dict[str, sp.Expr] = {}
    for mu, nu in SYMMETRIC_METRIC_PAIRS:
        value = sp.factor(
            sp.expand(
                sum(
                    inverse[mu, left] * inverse[nu, right] * ricci[left, right]
                    for left in range(4)
                    for right in range(4)
                )
                - inverse[mu, nu] * scalar / 2
            )
        )
        if value != 0:
            result[f"G_{mu}{nu}"] = value
    return result


def _trace_omission_witness() -> str:
    inverse, _ = _inverse_metric_symbols()
    pair = (0, 1)
    field = 1
    ricci = _generic_ricci_tangent(pair, field)
    scalar = sp.expand(sum(inverse[mu, nu] * ricci[mu, nu] for mu in range(4) for nu in range(4)))
    omitted_trace_difference = sp.factor((inverse[0, 0] * scalar / 2).subs(_flat_substitution()))
    if omitted_trace_difference != sp.sqrt(2) / 2:
        raise PrincipalSecondJetProjectionError("trace omission negative changed")
    return str(omitted_trace_difference)


def _flat_substitution() -> dict[sp.Symbol, int]:
    _, symbols = _inverse_metric_symbols()
    return {
        symbols[0]: -1,
        symbols[1]: 0,
        symbols[2]: 0,
        symbols[3]: 0,
        symbols[4]: 1,
        symbols[5]: 0,
        symbols[6]: 0,
        symbols[7]: 1,
        symbols[8]: 0,
        symbols[9]: 1,
    }


def _parse_atom(atom: str) -> tuple[tuple[int, int], int]:
    family, field_text = atom.split("[")
    field = int(field_text[:-1])
    if family.startswith("s0"):
        pair = (0, int(family[2]))
    elif family.startswith("s") and len(family) == 3:
        pair = (int(family[1]), int(family[2]))
    else:
        raise PrincipalSecondJetProjectionError("principal atom family changed")
    return pair, field


def _projection_record(column: int, atom: str) -> dict[str, Any]:
    pair, field = _parse_atom(atom)
    if field == 10:
        entries = {f"H_{pair[0]}{pair[1]}": "1"}
        theorem = "scalar_covariant_Hessian_principal_second_jet_identity"
        flat_checked = True
    else:
        generic = _generic_einstein_tangent(pair, field)
        entries = {name: str(value) for name, value in sorted(generic.items())}
        expected = _linearized_einstein_upper(pair, field)
        actual = {
            name: sp.factor(value.subs(_flat_substitution()))
            for name, value in generic.items()
            if sp.factor(value.subs(_flat_substitution())) != 0
        }
        flat_checked = actual == expected
        theorem = "arbitrary_inverse_metric_Einstein_principal_second_jet_formula"
    if not flat_checked:
        raise PrincipalSecondJetProjectionError("flat projection replay changed")
    body = {
        "coordinate_atom": atom,
        "coordinate_column": column,
        "covariant_jet_entries": entries,
        "covariant_jet_support_size": len(entries),
        "derivative_pair": list(pair),
        "field_index": field,
        "flat_reference_replay_zero": True,
        "formula_scope": "arbitrary_nonsingular_metric_inverse_at_one_coordinate_point",
        "theorem": theorem,
    }
    return {**body, "content_sha256": _sha(body)}


def _projection_registry(basis: Sequence[str]) -> list[dict[str, Any]]:
    records = [
        _projection_record(column, atom)
        for column, atom in enumerate(basis)
        if atom.startswith("s")
    ]
    if (
        len(records) != 99
        or sum(row["field_index"] == 10 for row in records) != 9
        or sum(row["field_index"] < 10 for row in records) != 90
        or sum(bool(row["covariant_jet_entries"]) for row in records) != 75
        or sum(row["covariant_jet_support_size"] for row in records) != 270
    ):
        raise PrincipalSecondJetProjectionError("principal projection census changed")
    return records


def _candidate_manifests(
    complement: Mapping[str, Any], registry: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_column = {int(row["coordinate_column"]): row for row in registry}
    result = []
    for candidate in complement["candidate_manifests"]:
        obstruction = candidate["alias_obstruction_certificate"]
        duplicate_groups = obstruction["duplicate_groups"]
        alias_records = []
        for group in duplicate_groups:
            column = int(group["coordinate_column"])
            projection = by_column[column]
            alias_body = {
                "coordinate_column": column,
                "formal_coordinate_ordinals": group["formal_coordinate_ordinals"],
                "projection_content_sha256": projection["content_sha256"],
                "same_coordinate_vector_same_covariant_projection": True,
            }
            alias_records.append({**alias_body, "content_sha256": _sha(alias_body)})
        if len(alias_records) != 2:
            raise PrincipalSecondJetProjectionError("formal alias reconciliation changed")
        body = {
            "alias_reconciliation_records": alias_records,
            "candidate_id": candidate["candidate_id"],
            "candidate_projection_registrations": 99,
            "existing_unique_projection_directions_replayed": 20,
            "new_unique_projection_directions_registered": 79,
            "principal_projection_registry_sha256": _sha(
                [row["content_sha256"] for row in registry]
            ),
            "remaining_lower_jet_projection_directions": 54,
            "candidate_decision": "pass_principal_projection_lower_jet_D2_blocked",
            "candidate_rejection_authorized": False,
        }
        result.append({**body, "content_sha256": _sha(body)})
    if len(result) != 12:
        raise PrincipalSecondJetProjectionError("projection candidate set changed")
    return result


def _expected_body(
    root: Path,
    config_path: Path,
    complement: Mapping[str, Any],
    geometric: Mapping[str, Any],
) -> dict[str, Any]:
    basis = complement["basis_registry"]["coordinate_atom_basis"]
    registry = _projection_registry(basis)
    manifests = _candidate_manifests(complement, registry)
    formula_contract = geometric["geometric_control"]["formula_contract_sha256"]
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_99_principal_second_jet_covariant_projections_lower_54_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "full_D2_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "projection_theorem": {
            "name": "arbitrary_background_principal_second_jet_to_covariant_Hessian_Einstein_map",
            "premises": (
                "At one coordinate point the inverse metric is symmetric and nonsingular. "
                "The registered q off-diagonal convention is sqrt(2) times g_mu_nu."
            ),
            "formula": (
                "dR_mn=1/2 g^rs(d_rm h_sn+d_rn h_sm-d_rs h_mn-d_mn h_rs); "
                "dG^mn=g^ma g^nb dR_ab-1/2 g^mn g^ab dR_ab"
            ),
            "conclusion": (
                "All 90 metric and nine scalar principal second-jet coordinate directions "
                "have exact covariant-jet projections on an arbitrary nonsingular background."
            ),
            "boundary": (
                "The 54 q/p lower-jet directions and propagation through every D1 arithmetic "
                "DAG remain unregistered; no D2 entry count is advanced."
            ),
        },
        "geometric_formula_contract_sha256": formula_contract,
        "inverse_metric_symbol_basis": [str(symbol) for symbol in _inverse_metric_symbols()[1]],
        "principal_projection_registry": registry,
        "principal_projection_registry_sha256": _sha([row["content_sha256"] for row in registry]),
        "candidate_manifests": manifests,
        "gate_counts": {
            "selected_candidates": 12,
            "coordinate_basis_dimension": 153,
            "principal_second_jet_projection_directions": 99,
            "active_principal_projection_directions": 75,
            "zero_principal_projection_directions": 24,
            "covariant_projection_scalar_coefficients": 270,
            "existing_unique_projection_directions_replayed_per_candidate": 20,
            "new_unique_projection_directions_registered_per_candidate": 79,
            "candidate_bound_projection_registrations": 1188,
            "formal_alias_groups_reconciled": 24,
            "remaining_lower_jet_projection_directions": 54,
            "D2_entries_registered_per_candidate": 5324,
            "D2_entries_remaining_per_candidate": 252175,
            "complete_D2F_tensors": 0,
            "H7_closures": 0,
        },
        "claim_seals": {
            "arbitrary_background_principal_second_jet_projection_registered": True,
            "all_99_principal_directions_registered": True,
            "formal_22_slot_to_20_unique_alias_reconciled": True,
            "all_153_coordinate_directions_projected": False,
            "lower_54_coordinate_directions_projected": False,
            "D1_DAG_propagation_complete": False,
            "D2_entry_count_advanced": False,
            "complete_D2F": False,
            "global_H7": False,
            "candidate_theory_rejected": False,
        },
        "exact_controls": {
            "flat_limit_all_90_metric_directions_replayed": True,
            "off_diagonal_sqrt2_normalization_bound": True,
            "drop_trace_term_changes_flat_projection": {
                "rejected": True,
                "witness_atom": "s01[1]",
                "witness_output": "G_00",
                "exact_residual": _trace_omission_witness(),
            },
            "promote_principal_projection_to_lower_directions": {"rejected": True},
            "promote_projection_to_D2_values": {"rejected": True},
        },
        "data_seals": dict(SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "coordinate_complement": _copy(COMPLEMENT),
            "geometric_map": _copy(GEOMETRIC),
            "flat_typed_map": _copy(FLAT_TYPED),
            "formula_sources": _copy(FORMULA_SOURCES),
        },
        "scope": (
            "exact arbitrary-background principal second-jet coordinate-to-covariant projection "
            "for 99 of 153 directions across 12 candidates; no lower-jet projection, D1 DAG "
            "propagation, D2 count advance, full tensor, H7, candidate rejection, or observation"
        ),
    }


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    project_root = (root or Path.cwd()).resolve()
    config_path = _inside(project_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    complement, geometric, _flat = _validate_inputs(project_root)
    expected = _expected_body(project_root, config_path, complement, geometric)
    if value.get("content_sha256") != _content_sha(value):
        raise PrincipalSecondJetProjectionError("projection content seal changed")
    if value != {**expected, "content_sha256": _sha(expected)}:
        raise PrincipalSecondJetProjectionError("projection result boundary changed")


def build_campaign(
    config_path: Path | str = CONFIG_PATH, *, root: Path | str | None = None
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    path = _inside(project_root, str(config_path))
    config = json.loads(path.read_text(encoding="utf-8"))
    _validate_config(config)
    complement, geometric, _flat = _validate_inputs(project_root)
    body = _expected_body(project_root, path, complement, geometric)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=project_root)
    return result


def validate_campaign(value: Mapping[str, Any], *, root: Path | str | None = None) -> None:
    _validate_result(value, root=Path(root or Path.cwd()).resolve())


def write_campaign(
    output_path: Path | str = OUTPUT_PATH,
    config_path: Path | str = CONFIG_PATH,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    project_root = Path(root or Path.cwd()).resolve()
    value = build_campaign(config_path, root=project_root)
    path = _inside(project_root, str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        value = json.loads(Path(args.output).read_text(encoding="utf-8"))
        validate_campaign(value)
    else:
        write_campaign(args.output, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
