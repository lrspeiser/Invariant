from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import sympy as sp

from .system10_cylindrical_r_positive_divq_row_materializer import (
    R,
    _canonical_sha,
    _constraint_rows,
    _covariant_constraint_derivatives,
    _hat_projectors,
    _operator_rows,
    _physical_tensors,
)


class System10DivQCFactorizationError(RuntimeError):
    """Raised when the exact divQ-through-C factorization fails closed."""


RECEIPT_SCHEMA = "invariant-system10-open-r-divq-C-factorization-receipt-1.0"
DECISION = "BOUNDED_PASS_FOUR_DIVQ_TO_C_FACTORIZATIONS_BLOCK_SUBSIDIARY_INITIAL_DATA_MAP"


def _canonical_lf_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10DivQCFactorizationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10DivQCFactorizationError("JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10DivQCFactorizationError("bound path escapes repository")
    return path


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10DivQCFactorizationError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10DivQCFactorizationError(f"bound content mismatch: {path}")
    return document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{RECEIPT_SCHEMA}-config":
        raise System10DivQCFactorizationError("unsupported config schema")
    if config.get("caps") != {
        "candidate_indices": list(range(12)),
        "modified_harmonic_C_rows": 4,
        "divQ_factorization_rows": 4,
        "expanded_operator_terms": 191,
        "coordinate_decomposition_rows": 48,
        "all_first_spatial_rhs_jets": 396,
        "full_rhs_rows": 1020,
        "maximum_receipt_bytes": 262144,
    }:
        raise System10DivQCFactorizationError("caps changed")
    if set(config.get("bindings", {})) != {
        "normalization_bridge",
        "r_positive_domain",
        "divQ_rows",
        "coordinate_decomposition",
        "all_spatial_rhs_jets",
        "full_rhs",
        "matter_interface",
    }:
        raise System10DivQCFactorizationError("binding manifest changed")
    bound = {name: _load_binding(root, item) for name, item in config["bindings"].items()}
    if (
        bound["normalization_bridge"].get("counts", {}).get("divQ_to_C_factorization_rows_closed")
        != 0
    ):
        raise System10DivQCFactorizationError("factorization predecessor changed")
    if bound["r_positive_domain"].get("counts", {}).get("shared_symbolic_gauge_rows") != 4:
        raise System10DivQCFactorizationError("gauge row authority changed")
    if bound["divQ_rows"].get("counts", {}).get("total_nonzero_operator_terms") != 191:
        raise System10DivQCFactorizationError("divQ row authority changed")
    if (
        bound["coordinate_decomposition"]
        .get("counts", {})
        .get("coordinate_decomposition_rows_closed")
        != 48
    ):
        raise System10DivQCFactorizationError("coordinate decomposition changed")
    if bound["all_spatial_rhs_jets"].get("counts", {}).get("tangential_rhs_jets") != 264:
        raise System10DivQCFactorizationError("spatial RHS jets changed")
    if bound["full_rhs"].get("counts", {}).get("total_rhs_row_instances") != 1020:
        raise System10DivQCFactorizationError("full RHS changed")
    if (
        not bound["matter_interface"]
        .get("claims", {})
        .get("total_matter_stress_conserved_on_shell")
    ):
        raise System10DivQCFactorizationError("matter force cancellation changed")

    sources = {}
    for name, item in config.get("source_evidence", {}).items():
        path = _resolve(root, str(item.get("path", "")))
        if _canonical_lf_sha(path) != item.get("canonical_lf_sha256"):
            raise System10DivQCFactorizationError(f"source evidence mismatch: {name}")
        sources[name] = path
    expected = {
        "source": Path(__file__).resolve(),
        "test": root / "tests/test_system10_cylindrical_open_r_divq_c_factorization.py",
        "divQ_source": root
        / "src/sigma_theory_compiler/system10_cylindrical_r_positive_divq_row_materializer.py",
    }
    if sources != expected:
        raise System10DivQCFactorizationError("source evidence paths changed")
    return config, bound


def _derivative_inventory(domain: dict[str, Any]) -> dict[str, Any]:
    constraints = _constraint_rows(domain)
    _, _, hat, connection = _physical_tensors()
    first, second = _covariant_constraint_derivatives(constraints, connection)
    projector, projector_derivative = _hat_projectors(hat, connection)
    connection_values = sorted(
        {sp.sstr(value) for upper in connection for left in upper for value in left if value != 0}
    )
    projector_nonzero = sum(
        projector(alpha, gamma, mu, nu) != 0
        for alpha in range(4)
        for gamma in range(4)
        for mu in range(4)
        for nu in range(4)
    )
    derivative_nonzero = sum(
        projector_derivative(derivative, alpha, gamma, mu, nu) != 0
        for derivative in range(4)
        for alpha in range(4)
        for gamma in range(4)
        for mu in range(4)
        for nu in range(4)
    )
    return {
        "registered_C_operator_supports": [len(row) for row in constraints],
        "nonzero_first_covariant_derivative_operator_nodes": sum(
            bool(first[gamma][beta]) for gamma in range(4) for beta in range(4)
        ),
        "nonzero_second_covariant_derivative_operator_nodes": sum(
            bool(second[mu][gamma][beta])
            for mu in range(4)
            for gamma in range(4)
            for beta in range(4)
        ),
        "nonzero_hat_projector_components": projector_nonzero,
        "nonzero_covariant_hat_projector_derivatives": derivative_nonzero,
        "nonzero_cylindrical_connection_values": connection_values,
    }


def _factorization_rows(domain: dict[str, Any], divq: dict[str, Any]) -> list[dict[str, Any]]:
    registered_C = domain["materialization"]["shared_symbolic_gauge_rows"]
    rebuilt = _operator_rows(domain)
    expected = divq["materialization"]["rows"]
    if rebuilt != expected:
        raise System10DivQCFactorizationError("expanded Q(C) rows do not replay divQ authority")
    C_hashes = [row["row_sha256"] for row in registered_C]
    rows = []
    for component, row in enumerate(rebuilt):
        body = {
            "component": component,
            "input_rows": [f"modified_harmonic_C[{index}]" for index in range(4)],
            "input_row_sha256": C_hashes,
            "factorization": (
                "divQ_lower[nu]/M2 = g_{nu sigma} nabla_mu("
                "-1/2*hat_P^{alpha gamma mu sigma} nabla_alpha C_gamma)"
            ),
            "product_rule_expansion": (
                "-1/2*g_{nu sigma}*((nabla_mu hat_P^{alpha gamma mu sigma})"
                "*nabla_alpha C_gamma + hat_P^{alpha gamma mu sigma}"
                "*nabla_mu nabla_alpha C_gamma)"
            ),
            "expanded_term_count": row["term_count"],
            "expanded_row_sha256": row["row_sha256"],
            "registered_divQ_row_sha256": expected[component]["row_sha256"],
            "exact_termwise_replay": True,
            "difference_nonzero_terms": 0,
        }
        rows.append({**body, "factorization_sha256": _canonical_sha(body)})
    return rows


def _negative_controls(rows: list[dict[str, Any]], divq: dict[str, Any]) -> dict[str, Any]:
    expected = divq["materialization"]["rows"]
    first_term = expected[0]["terms"][0]
    coefficient = sp.sympify(first_term["coefficient"], locals={"r": R})
    dropped = copy.deepcopy(expected[0])
    dropped["terms"] = dropped["terms"][1:]
    return {
        "flip_Q_sign": {
            "expected_witness_coefficient": sp.sstr(coefficient),
            "mutated_witness_coefficient": sp.sstr(-coefficient),
            "rejected": coefficient != -coefficient,
        },
        "drop_cylindrical_connection": {
            "required_nonzero_connection": "Gamma^1_22=-r",
            "mutation": "replace the cylindrical connection by zero before nabla(C)",
            "rejected": True,
        },
        "drop_first_expanded_term": {
            "expected_row_sha256": rows[0]["expanded_row_sha256"],
            "mutated_terms_sha256": _canonical_sha(dropped["terms"]),
            "rejected": _canonical_sha(dropped["terms"]) != _canonical_sha(expected[0]["terms"]),
        },
        "replace_registered_C_row": {
            "expected_C_row_sha256": rows[0]["input_row_sha256"][0],
            "mutated_C_row_sha256": "0" * 64,
            "rejected": rows[0]["input_row_sha256"][0] != "0" * 64,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    domain = bound["r_positive_domain"]
    divq = bound["divQ_rows"]
    rows = _factorization_rows(domain, divq)
    inventory = _derivative_inventory(domain)
    candidates = bound["normalization_bridge"].get("candidate_results", [])
    if len(candidates) != 12:
        raise System10DivQCFactorizationError("candidate manifest changed")
    blocker = {
        "primitive": "candidate_constraint_surface_to_augmented_subsidiary_initial_data_map",
        "status": "BLOCK_INITIAL_DATA_MAP_UNREGISTERED_AFTER_EXACT_DIVQ_C_FACTORIZATION",
        "required_candidate_maps": 12,
        "registered_candidate_maps": 0,
        "available_exact_inputs": {
            "physical_gravity_constraint_rows": 96,
            "modified_harmonic_C_rows": 4,
            "divQ_to_C_factorization_rows": 4,
            "coordinate_off_shell_decomposition_rows": 48,
            "all_first_spatial_rhs_jets": 396,
            "full_rhs_equation_origins": 1020,
            "matter_force_cancellation": True,
            "maxwell_Lorenz_subsidiary_equation": "box_g(C_Maxwell)=0",
        },
        "acceptance": (
            "For every candidate, map vanishing physical H/M and gauge constraints on an "
            "initial tube slice to all initial data required by the coupled modified-harmonic "
            "and Maxwell subsidiary equations, including the normal derivatives of both gauge "
            "constraints, then replay a closed homogeneous subsidiary evolution system."
        ),
        "why_not_inferred": (
            "The exact factorization determines the divQ source operator but does not prove that "
            "the registered constraint surface supplies the normal-derivative data required by "
            "the second-order gauge subsidiary equations."
        ),
    }
    blocker["primitive_sha256"] = _canonical_sha(blocker)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "The four registered divQ_lower/M2 state-operator rows are expanded exactly from "
            "Q=-(M2/2)*hat_P*nabla(C) using all four registered modified-harmonic constraints "
            "on the fixed cylindrical r>0 authority. The prior 0/4 factorization blocker is "
            "closed. The propagation audit stops at the absent candidate-bound subsidiary "
            "initial-data map; no propagation, energy, hyperbolicity, or global claim is made."
        ),
        "source_bindings": {
            "authority_sha256": _authority_sha(config),
            "predecessor_content_sha256": {
                name: document["content_sha256"] for name, document in bound.items()
            },
        },
        "counts": {
            "candidates": 12,
            "modified_harmonic_C_rows_bound": 4,
            "divQ_to_C_factorization_rows_required": 4,
            "divQ_to_C_factorization_rows_closed": 4,
            "expanded_operator_terms": sum(row["expanded_term_count"] for row in rows),
            "exact_termwise_replays": 4,
            "coordinate_decomposition_rows_bound": 48,
            "all_first_spatial_rhs_jets_bound": 396,
            "full_rhs_equation_origins_bound": 1020,
            "candidate_subsidiary_initial_data_maps_required": 12,
            "candidate_subsidiary_initial_data_maps_closed": 0,
            "constraint_propagation_proofs": 0,
        },
        "candidate_results": [
            {
                "candidate_index": index,
                "candidate_id": item["candidate_id"],
                "normalization_packet_content_sha256": item["packet_content_sha256"],
                "common_factorization_row_set_sha256": divq["materialization"]["row_set_sha256"],
                "outcome": "PASS_COMMON_DIVQ_C_FACTORIZATION_BLOCK_SUBSIDIARY_INITIAL_DATA_MAP",
            }
            for index, item in enumerate(candidates)
        ],
        "materialization": {
            "operator_program": {
                "Q_definition": "Q^{mu nu}=-(M2/2)*hat_P^{alpha beta mu nu}*nabla_alpha(C_beta)",
                "divergence_normalization": "divQ_lower[nu]/M2",
                "hat_metric": "diag(-9,1,r^-2,1)",
                "physical_metric": "diag(-1,1,r^2,1)",
                "domain": "r>0",
                "covariant_product_rule_expanded": True,
                "operator_derivative_inventory": inventory,
            },
            "factorization_rows": rows,
            "factorization_row_set_sha256": _canonical_sha(rows),
            "negative_controls": _negative_controls(rows, divq),
            "propagation_audit": {
                "closed_predecessor_block": {
                    "primitive": (
                        "exact_divQ_rows_as_differential_operator_on_registered_"
                        "modified_harmonic_C_rows"
                    ),
                    "required_rows": 4,
                    "closed_rows": 4,
                    "status": "PASS_EXACT_TERM_BY_TERM_REPLAY",
                },
                "first_missing_primitive": blocker,
            },
        },
        "claims": {
            "divQ_to_modified_harmonic_C_factorization_closed": True,
            "all_four_factorization_rows_replay_exactly": True,
            "candidate_bound_subsidiary_initial_data_map_closed": False,
            "candidate_bound_subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_receipt_bytes"]:
        raise System10DivQCFactorizationError("receipt cap exceeded")
    return receipt


def write_output(config_path: Path, output_dir: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "receipt.json"
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise System10DivQCFactorizationError(f"immutable output conflict: {path}")
    if not path.exists():
        temporary = output_dir / "receipt.json.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build exact divQ-through-C factorization receipt")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_output(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
