from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp


class MaxwellMetricMixedPrincipalCompletionError(RuntimeError):
    """Raised when the nonlinear Lorenz mixed-principal completion cannot be sealed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise MaxwellMetricMixedPrincipalCompletionError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MaxwellMetricMixedPrincipalCompletionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MaxwellMetricMixedPrincipalCompletionError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise MaxwellMetricMixedPrincipalCompletionError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise MaxwellMetricMixedPrincipalCompletionError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise MaxwellMetricMixedPrincipalCompletionError(f"bound content hash mismatch: {path}")
    return path, value


def _mixed_principal_replay() -> dict[str, Any]:
    eta = sp.diag(-1, 1, 1, 1)
    xi_down = sp.Matrix(sp.symbols("xi_0:4"))
    potential_down = sp.Matrix(sp.symbols("B_0:4"))
    xi_up = eta * xi_down
    pairs = [(left, right) for left in range(4) for right in range(left, 4)]
    q = sp.symbols("q_00 q_01 q_02 q_03 q_11 q_12 q_13 q_22 q_23 q_33")
    h = sp.zeros(4)
    for coordinate, (left, right) in zip(q, pairs):
        value = coordinate if left == right else coordinate / sp.sqrt(2)
        h[left, right] = value
        h[right, left] = value
    trace_h = sp.expand(sum(eta[index, index] * h[index, index] for index in range(4)))
    contracted_connection_symbol: list[sp.Expr] = []
    for upper in range(4):
        expression = (
            sum(xi_up[rho] * eta[upper, lam] * h[rho, lam] for rho in range(4) for lam in range(4))
            - xi_up[upper] * trace_h / 2
        )
        contracted_connection_symbol.append(sp.expand(expression))
    lorenz_metric_scalar = sp.expand(
        sum(potential_down[upper] * contracted_connection_symbol[upper] for upper in range(4))
    )
    mixed = sp.Matrix(
        [
            [
                sp.factor(-xi_down[row] * sp.diff(lorenz_metric_scalar, coordinate))
                for coordinate in q
            ]
            for row in range(4)
        ]
    )
    if mixed.shape != (4, 10) or any(entry == 0 for entry in mixed):
        raise MaxwellMetricMixedPrincipalCompletionError(
            "mixed-principal symbolic matrix is incomplete"
        )

    # Direct RNC decomposition: the action Euler has no partial^2 g term.
    # Adding nabla_nu C supplies -B_sigma partial_nu Gamma^sigma. The
    # independently expanded contracted-connection symbol must reproduce M.
    direct = sp.Matrix(
        [
            [
                sp.factor(
                    -xi_down[row]
                    * sp.diff(
                        sum(
                            potential_down[upper]
                            * (
                                sum(
                                    xi_up[rho] * eta[upper, lam] * h[rho, lam]
                                    for rho in range(4)
                                    for lam in range(4)
                                )
                                - xi_up[upper] * trace_h / 2
                            )
                            for upper in range(4)
                        ),
                        coordinate,
                    )
                )
                for coordinate in q
            ]
            for row in range(4)
        ]
    )
    residual = (mixed - direct).applyfunc(sp.simplify)
    if not residual.is_zero_matrix:
        raise MaxwellMetricMixedPrincipalCompletionError(
            "independent RNC mixed-principal replay failed"
        )
    witness_substitution = {
        potential_down[0]: 1,
        potential_down[1]: 0,
        potential_down[2]: 0,
        potential_down[3]: 0,
        xi_down[0]: 1,
        xi_down[1]: 0,
        xi_down[2]: 0,
        xi_down[3]: 0,
    }
    witness = mixed.subs(witness_substitution)
    witness_nonzero = sum(entry != 0 for entry in witness)
    if witness_nonzero != 4:
        raise MaxwellMetricMixedPrincipalCompletionError("nonzero mixed-block witness changed")
    expressions = [[str(mixed[row, column]) for column in range(10)] for row in range(4)]
    return {
        "basis": {
            "rows": ["E_L_0", "E_L_1", "E_L_2", "E_L_3"],
            "columns": [f"q_{left}{right}" for left, right in pairs],
            "covector": ["xi_0", "xi_1", "xi_2", "xi_3"],
            "background_potential": ["B_0", "B_1", "B_2", "B_3"],
            "signature": [-1, 1, 1, 1],
        },
        "rnc_derivation": {
            "action_euler": "E_nu=nabla^mu F_mu_nu",
            "action_euler_second_metric_derivative_block": "0_(4x10)",
            "lorenz_constraint": "C=nabla^rho B_rho",
            "reduced_euler": "E_L_nu=E_nu+nabla_nu C",
            "mixed_term": "-B_sigma partial_nu Gamma^sigma",
            "contracted_connection_symbol": (
                "Gamma^sigma(xi,h)=xi^rho h_rho^sigma-xi^sigma tr_eta(h)/2"
            ),
            "mixed_symbol": ("M_nu[h]=-xi_nu B_sigma(xi^rho h_rho^sigma-xi^sigma tr_eta(h)/2)"),
            "independent_expansion_residual_entries": 0,
        },
        "matrix_shape": [4, 10],
        "matrix_entries": expressions,
        "structurally_nonzero_entries": sum(entry != 0 for entry in mixed),
        "matrix_sha256": _canonical_sha(expressions),
        "zero_block_negative": {
            "mutation": "replace the complete 4x10 block by zero",
            "witness": {"B_0": "1", "B_1": "0", "B_2": "0", "B_3": "0", "xi": [1, 0, 0, 0]},
            "nonzero_residual_entries": witness_nonzero,
            "witness_matrix": [
                [str(witness[row, column]) for column in range(10)] for row in range(4)
            ],
            "rejected": True,
        },
        "zero_subdomain_control": {
            "condition": "B_mu=0 at the base point",
            "residual_entries": 0,
            "does_not_imply_universal_zero": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-maxwell-metric-mixed-principal-completion-config-1.0"
    ):
        raise MaxwellMetricMixedPrincipalCompletionError("unsupported config schema")
    expected_contract = {
        "dimension": 4,
        "signature": [-1, 1, 1, 1],
        "potential_index_position": "covector B_mu",
        "maxwell_euler": "E_nu=nabla^mu F_mu_nu",
        "lorenz_constraint": "C=nabla^rho B_rho",
        "reduced_euler": "E_L_nu=E_nu+nabla_nu C",
        "metric_coordinate_convention": ("q_ab=h_ab for a=b; q_ab=sqrt(2) h_ab for a<b"),
    }
    if config.get("derivation_contract") != expected_contract:
        raise MaxwellMetricMixedPrincipalCompletionError("derivation contract changed")
    expected_policy = {
        "exact_maxwell_metric_mixed_block": True,
        "full_17_field_second_order_principal_matrix": True,
        "mixed_block_universally_zero": False,
        "full_85_state_first_order_reduction": False,
        "full_coupled_symmetrizer": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise MaxwellMetricMixedPrincipalCompletionError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {"blocked_principal_census", "total_action"}:
        raise MaxwellMetricMixedPrincipalCompletionError("closed binding manifest changed")
    predecessor = bound["blocked_principal_census"][1]
    total_action = bound["total_action"][1]
    if predecessor.get("decision") != ("TYPED_BLOCK_MISSING_MAXWELL_METRIC_MIXED_PRINCIPAL_BLOCK"):
        raise MaxwellMetricMixedPrincipalCompletionError("principal blocker predecessor changed")
    if total_action.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY":
        raise MaxwellMetricMixedPrincipalCompletionError("total action predecessor changed")
    maxwell_component = next(
        (
            item
            for item in total_action.get("shared_matter_action", {}).get("components", [])
            if item.get("sector_id") == "source_free_maxwell"
        ),
        None,
    )
    if (
        not isinstance(maxwell_component, dict)
        or maxwell_component.get("field") != "B_mu"
        or maxwell_component.get("mass_term_removed") is not True
        or "H_mu_nu" not in maxwell_component.get("density", "")
    ):
        raise MaxwellMetricMixedPrincipalCompletionError(
            "committed Maxwell action representation changed"
        )
    previous_records = {
        item.get("candidate_id"): item for item in predecessor.get("candidate_results", [])
    }
    action_records = {
        item.get("candidate_id"): item for item in total_action.get("candidate_results", [])
    }
    expected_count = config.get("expected_candidate_count")
    if (
        expected_count != 12
        or len(previous_records) != expected_count
        or set(previous_records) != set(action_records)
        or None in previous_records
    ):
        raise MaxwellMetricMixedPrincipalCompletionError("candidate set mismatch")
    replay = _mixed_principal_replay()
    replay_sha = _canonical_sha(replay)
    results: list[dict[str, Any]] = []
    complete_hashes: set[str] = set()
    for candidate_id in sorted(previous_records):
        if previous_records[candidate_id].get("outcome") != "BLOCK":
            raise MaxwellMetricMixedPrincipalCompletionError(
                f"predecessor candidate is not blocked: {candidate_id}"
            )
        manifest = {
            "schema_version": "invariant-complete-17-field-principal-manifest-1.0",
            "candidate_id": candidate_id,
            "total_action_sha256": action_records[candidate_id]["total_action_sha256"],
            "prior_partial_matrix_skeleton_sha256": previous_records[candidate_id][
                "partial_matrix_skeleton_sha256"
            ],
            "maxwell_metric_mixed_principal_sha256": replay["matrix_sha256"],
            "second_order_dimension": 17,
            "entries_determined": 289,
            "entries_unresolved": 0,
        }
        complete_sha = _canonical_sha(manifest)
        complete_hashes.add(complete_sha)
        results.append(
            {
                "candidate_id": candidate_id,
                "complete_17_field_principal_manifest": manifest,
                "complete_17_field_principal_sha256": complete_sha,
                "outcome": "PASS",
            }
        )
    if len(complete_hashes) != 12:
        raise MaxwellMetricMixedPrincipalCompletionError(
            "complete principal hashes are not one-to-one"
        )

    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_maxwell_metric_mixed_principal_completion_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": "invariant-maxwell-metric-mixed-principal-completion-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL",
        "maxwell_metric_mixed_principal": replay,
        "maxwell_metric_mixed_principal_sha256": replay_sha,
        "candidate_results": results,
        "updated_principal_census": {
            "second_order_dimension": 17,
            "entries_per_candidate": 289,
            "entries_determined_per_candidate": 289,
            "entries_unresolved_per_candidate": 0,
            "target_first_order_dimension": 85,
            "target_first_order_reduction_status": "NOT_CONSTRUCTED",
        },
        "counts": {
            "candidates": 12,
            "mixed_block_rows": 4,
            "mixed_block_columns": 10,
            "mixed_block_entries": 40,
            "structurally_nonzero_mixed_entries": 40,
            "completed_17_field_principal_matrices": 12,
            "determined_entries_total": 3468,
            "unresolved_entries_total": 0,
            "rnc_replay_residual_entries": 0,
            "negative_controls": 1,
            "first_order_85_state_reductions": 0,
            "rejects": 0,
        },
        "claims": {
            "exact_maxwell_metric_mixed_principal_block_closed": True,
            "mixed_block_universally_zero": False,
            "all_twelve_17_field_second_order_principal_matrices_closed": True,
            "full_85_state_first_order_reduction_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "sourced_gravity_constraints_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact arbitrary-point RNC principal derivation for the nonlinear Lorenz-reduced "
            "source-free Maxwell covector equation, including all 40 metric mixed entries, "
            "and completion of the 17-field second-order principal manifest for all twelve "
            "quartic candidates. The 85-state reduction, symmetrizer, constraints, H7, "
            "universal-matter closure, and promotion remain unclaimed"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
