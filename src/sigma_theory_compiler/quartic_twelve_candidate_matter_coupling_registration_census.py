from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class QuarticMatterCouplingCensusError(RuntimeError):
    """Raised when candidate evidence is missing, inconsistent, or overclaimed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise QuarticMatterCouplingCensusError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QuarticMatterCouplingCensusError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QuarticMatterCouplingCensusError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise QuarticMatterCouplingCensusError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise QuarticMatterCouplingCensusError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise QuarticMatterCouplingCensusError(f"bound content hash mismatch: {path}")
    return path, value


def _records(value: dict[str, Any]) -> list[dict[str, Any]]:
    records = value.get("candidates", value.get("certificates"))
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise QuarticMatterCouplingCensusError("candidate records are absent")
    return records


def _index(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in _records(value):
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in result:
            raise QuarticMatterCouplingCensusError("candidate identifiers are invalid")
        result[candidate_id] = item
    return result


def _require_statuses(records: dict[str, dict[str, Any]], allowed: set[str], label: str) -> None:
    failures = [
        candidate_id
        for candidate_id, record in records.items()
        if record.get("status") not in allowed
    ]
    if failures:
        raise QuarticMatterCouplingCensusError(f"{label} prerequisite failed: {failures[0]}")


def _contract_results() -> list[dict[str, Any]]:
    return [
        {
            "item_id": "total_action_and_euler_same_hash",
            "outcome": "BLOCK",
            "reason_code": "missing_total_matter_action_hash_binding",
            "present_prerequisite": (
                "candidate-specialized quartic action and exact gauge-fixed vacuum Euler "
                "records agree by candidate id and coefficients"
            ),
            "missing_registration": (
                "one hash binding the candidate gravitational action, all three matter "
                "actions, and the sourced gauge-fixed Euler system"
            ),
        },
        {
            "item_id": "exact_total_stress_metric_equation_insertion",
            "outcome": "BLOCK",
            "reason_code": "missing_exact_total_stress_metric_equation_insertion",
            "present_prerequisite": "the three-sector matter stress conservation interface passes",
            "missing_registration": (
                "candidate metric equation containing T_scalar+T_Maxwell+T_fluid with "
                "an exact normalization and sign replay"
            ),
        },
        {
            "item_id": "full_gravity_gauge_three_matter_principal_matrix",
            "outcome": "BLOCK",
            "reason_code": "missing_full_coupled_principal_matrix",
            "present_prerequisite": (
                "candidate vacuum 55-state principal reduction and six-component "
                "matter-side principal direct sum pass independently"
            ),
            "missing_registration": (
                "one candidate-specific matrix with gravity, modified-harmonic gauge, "
                "scalar, Maxwell, fluid, and every cross block"
            ),
        },
        {
            "item_id": "full_coupled_common_time_symmetrizer",
            "outcome": "BLOCK",
            "reason_code": "missing_full_coupled_symmetrizer_and_uniform_domain",
            "present_prerequisite": (
                "candidate vacuum K55 and matter common-time cone certificates pass independently"
            ),
            "missing_registration": (
                "an exact symmetrizer or diagonalizer and uniform bounds for the full "
                "coupled matrix on one common domain"
            ),
        },
        {
            "item_id": "total_matter_sourced_gravity_constraint_propagation",
            "outcome": "BLOCK",
            "reason_code": "missing_total_matter_sourced_gravity_constraint_system",
            "present_prerequisite": (
                "vacuum derivative-definition/curl constraints and the Maxwell "
                "subsidiary constraint pass independently"
            ),
            "missing_registration": (
                "Hamiltonian, momentum, and modified-harmonic gauge-constraint "
                "propagation with total matter sources"
            ),
        },
        {
            "item_id": "corrupted_total_source_constraint_negative",
            "outcome": "BLOCK",
            "reason_code": "missing_sourced_constraint_corruption_negative",
            "present_prerequisite": "sector-local matter negatives exist",
            "missing_registration": (
                "a wrong source sign or omitted matter sector yielding a nonzero "
                "candidate gravitational constraint-propagation residual"
            ),
        },
    ]


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-matter-coupling-registration-census-config-1.0"
    ):
        raise QuarticMatterCouplingCensusError("unsupported config schema")
    expected_policy = {
        "candidate_census_only": True,
        "candidate_coupled_registration_complete": False,
        "total_stress_inserted_in_gravity_equation": False,
        "full_coupled_principal_system": False,
        "full_coupled_symmetrizer": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise QuarticMatterCouplingCensusError("claims policy is absent or broadened")

    predecessor_path, predecessor = _load_binding(repository, config["predecessor"])
    if predecessor.get("decision") != "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK":
        raise QuarticMatterCouplingCensusError("combined matter predecessor is not admissible")
    if (
        predecessor.get("claims", {}).get("combined_three_sector_matter_interface_closed")
        is not True
    ):
        raise QuarticMatterCouplingCensusError("combined matter interface is incomplete")
    predecessor_contract = predecessor.get("gravity_block", {}).get("minimal_registration_contract")
    if not isinstance(predecessor_contract, list) or len(predecessor_contract) != 6:
        raise QuarticMatterCouplingCensusError("six-item predecessor contract is absent")

    bound: dict[str, tuple[Path, dict[str, Any]]] = {
        name: _load_binding(repository, binding)
        for name, binding in config["evidence_bindings"].items()
    }
    action = _index(bound["candidate_action_symbol"][1])
    euler = _index(bound["vacuum_euler"][1])
    reduction = _index(bound["vacuum_first_order_constraints"][1])
    symmetrizer = _index(bound["vacuum_full_symmetrizer"][1])
    expected_count = config.get("expected_candidate_count")
    candidate_sets = [set(records) for records in (action, euler, reduction, symmetrizer)]
    if not isinstance(expected_count, int) or expected_count != 12:
        raise QuarticMatterCouplingCensusError("expected candidate count changed")
    if any(items != candidate_sets[0] for items in candidate_sets[1:]):
        raise QuarticMatterCouplingCensusError("candidate set mismatch across evidence")
    if len(candidate_sets[0]) != expected_count:
        raise QuarticMatterCouplingCensusError("candidate count mismatch")

    _require_statuses(
        action,
        {"pass_exact_11x11_symbol_binding_symmetrizer_unresolved"},
        "action/symbol",
    )
    _require_statuses(
        euler,
        {"pass_exact_local_nonlinear_time_acceleration_elimination"},
        "vacuum Euler",
    )
    _require_statuses(
        reduction,
        {"pass_exact_55_variable_principal_first_order_reduction"},
        "first-order reduction",
    )
    _require_statuses(
        symmetrizer,
        {"pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift"},
        "full symmetrizer",
    )

    candidate_results: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_sets[0]):
        coefficient_views = [
            records[candidate_id].get("coefficients")
            for records in (action, euler, reduction, symmetrizer)
        ]
        if not isinstance(coefficient_views[0], dict) or any(
            item != coefficient_views[0] for item in coefficient_views[1:]
        ):
            raise QuarticMatterCouplingCensusError(
                f"candidate coefficient mismatch: {candidate_id}"
            )
        if reduction[candidate_id].get("definition_and_curl_constraints_propagate") is not True:
            raise QuarticMatterCouplingCensusError(
                f"vacuum reduction constraint prerequisite failed: {candidate_id}"
            )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficient_views[0],
                "prerequisite_census": {
                    "candidate_specialized_action_and_symbol": "PASS",
                    "exact_gauge_fixed_vacuum_euler": "PASS",
                    "exact_vacuum_55_state_first_order_reduction": "PASS",
                    "vacuum_definition_and_curl_constraint_propagation": "PASS",
                    "vacuum_full_55_state_symmetrizer": "PASS",
                    "combined_three_sector_matter_interface": "PASS",
                },
                "contract_results": _contract_results(),
                "first_blocker": "missing_total_matter_action_hash_binding",
                "outcome": "BLOCK",
            }
        )

    item_ids = [item["item_id"] for item in _contract_results()]
    if config.get("contract_item_ids") != item_ids:
        raise QuarticMatterCouplingCensusError("six-item contract identifiers changed")
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_twelve_candidate_matter_coupling_registration_census.py"
    )
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-matter-coupling-registration-census-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_CENSUS_NO_CANDIDATE_COUPLED_REGISTRATION",
        "candidate_results": candidate_results,
        "common_advance": {
            "outcome": "PASS_PREREQUISITE_CENSUS_ONLY",
            "candidate_count": expected_count,
            "passed_prerequisites_per_candidate": 6,
            "conclusion": (
                "all twelve candidates are consistently bound across the exact vacuum "
                "action/symbol, Euler, reduction/constraint, and K55 evidence plus the "
                "shared three-sector matter interface"
            ),
            "coupling_conclusion": (
                "no candidate has yet crossed the first total-matter action/hash gate"
            ),
        },
        "counts": {
            "candidates_audited": 12,
            "prerequisite_passes": 72,
            "contract_items_audited": 72,
            "contract_passes": 0,
            "typed_blocks": 72,
            "candidates_fully_registered": 0,
            "candidates_blocked_at_first_item": 12,
            "rejects": 0,
        },
        "claims": {
            "exact_candidate_specific_prerequisite_census_complete": True,
            "any_candidate_coupled_registration_complete": False,
            "total_stress_inserted_in_any_candidate_metric_equation": False,
            "any_full_coupled_principal_system_closed": False,
            "any_full_coupled_symmetrizer_closed": False,
            "any_sourced_gravity_constraint_system_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "an immutable six-item registration census for the twelve fixed-coefficient "
            "linear-X quartic candidates; it records exact vacuum and matter-side "
            "prerequisites but does not infer a coupled action, sourced metric equation, "
            "coupled principal system, sourced gravity constraints, H7, or promotion"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
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
