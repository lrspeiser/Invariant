from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class Quartic85StateCandidateGravityConstraintBasisError(RuntimeError):
    """Raised when the 85-state constraint-basis audit fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            f"cannot read bound file: {path}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateCandidateGravityConstraintBasisError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateCandidateGravityConstraintBasisError(
            f"JSON root is not an object: {path}"
        )
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "bound path escapes repository root"
        )
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateCandidateGravityConstraintBasisError(
            f"bound file hash mismatch: {path}"
        )
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateCandidateGravityConstraintBasisError(
            f"bound content hash mismatch: {path}"
        )
    return path, value


def _field_basis() -> list[dict[str, Any]]:
    names = [
        "g_00",
        "g_01",
        "g_02",
        "g_03",
        "g_11",
        "g_12",
        "g_13",
        "g_22",
        "g_23",
        "g_33",
        "phi_g",
        "chi_m",
        "B_0",
        "B_1",
        "B_2",
        "B_3",
        "tau",
    ]
    sectors = (
        ["metric"] * 10
        + ["gravity_scalar", "matter_scalar"]
        + ["maxwell"] * 4
        + ["irrotational_fluid"]
    )
    return [
        {"field_index_A": index, "field": name, "sector": sectors[index]}
        for index, name in enumerate(names)
    ]


def _state_coordinates(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = ["q", "v", "w_1", "w_2", "w_3"]
    coordinates: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        for field in fields:
            coordinates.append(
                {
                    "state_index": block_index * 17 + field["field_index_A"],
                    "coordinate": f"{block}[{field['field']}]",
                    "block": block,
                    "field_index_A": field["field_index_A"],
                }
            )
    if [item["state_index"] for item in coordinates] != list(range(85)):
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "85-state coordinate ordering changed"
        )
    return coordinates


def _kinematic_rows(
    fields: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions: list[dict[str, Any]] = []
    curls: list[dict[str, Any]] = []
    for field in fields:
        index = field["field_index_A"]
        for spatial in range(1, 4):
            definitions.append(
                {
                    "row": f"D_{spatial}[{field['field']}]",
                    "sector": field["sector"],
                    "identity": (
                        f"w_{spatial}[{field['field']}]-partial_{spatial} q[{field['field']}]=0"
                    ),
                    "state_terms": [
                        {
                            "state_index": 34 + (spatial - 1) * 17 + index,
                            "operator": "identity",
                            "coefficient": "1",
                        },
                        {
                            "state_index": index,
                            "operator": f"partial_{spatial}",
                            "coefficient": "-1",
                        },
                    ],
                }
            )
        for left, right in ((1, 2), (1, 3), (2, 3)):
            curls.append(
                {
                    "row": f"K_{left}{right}[{field['field']}]",
                    "sector": field["sector"],
                    "identity": (
                        f"partial_{left} w_{right}[{field['field']}]-"
                        f"partial_{right} w_{left}[{field['field']}]=0"
                    ),
                    "state_terms": [
                        {
                            "state_index": 34 + (right - 1) * 17 + index,
                            "operator": f"partial_{left}",
                            "coefficient": "1",
                        },
                        {
                            "state_index": 34 + (left - 1) * 17 + index,
                            "operator": f"partial_{right}",
                            "coefficient": "-1",
                        },
                    ],
                }
            )
    if len(definitions) != 51 or len(curls) != 51:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "kinematic constraint row count changed"
        )
    return definitions, curls


def _flat_maxwell_row() -> dict[str, Any]:
    # C=partial_mu B^mu in signature -+++ and the registered q,v,w_i ordering.
    row = {
        "row": "C_Maxwell",
        "identity": "-v[B_0]+w_1[B_1]+w_2[B_2]+w_3[B_3]=0",
        "state_terms": [
            {"state_index": 29, "coefficient": "-1"},
            {"state_index": 47, "coefficient": "1"},
            {"state_index": 65, "coefficient": "1"},
            {"state_index": 83, "coefficient": "1"},
        ],
        "background": "flat_reference_eta=(-,+,+,+)",
    }
    witness = {29: 2, 47: 3, 65: 5, 83: 7}
    correct = sum(
        int(term["coefficient"]) * witness[term["state_index"]] for term in row["state_terms"]
    )
    corrupted = -witness[29] + witness[47] + witness[65]
    if correct != 13 or corrupted != 6:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "Maxwell coordinate-row witness changed"
        )
    return {
        **row,
        "coordinate_witness": {
            "state_values": {str(index): value for index, value in witness.items()},
            "row_value": str(correct),
        },
        "corruption_negative": {
            "mutation": "omit w_3[B_3] at state index 83",
            "corrupted_row_value": str(corrupted),
            "difference_from_registered_row": str(corrupted - correct),
            "rejected": True,
        },
    }


def _materialize(
    blocker: dict[str, Any],
    reduction: dict[str, Any],
    vacuum: dict[str, Any],
    matter: dict[str, Any],
    field_basis_receipt: dict[str, Any],
) -> dict[str, Any]:
    if blocker.get("decision") != (
        "TYPED_BLOCK_CANDIDATE_GRAVITY_CONSTRAINT_JET_DIVERGENCE_UNREGISTERED"
    ):
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "sourced-constraint predecessor changed"
        )
    if reduction.get("decision") != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE":
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "85-state reduction predecessor changed"
        )
    generic = vacuum.get("generic_reduction_control", {})
    constraints = generic.get("constraints", {})
    if constraints.get("passed") is not True:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "vacuum definition/curl control changed"
        )
    internal = matter.get("combined_matter_certificate", {}).get(
        "internal_matter_constraint_closure", {}
    )
    if (
        internal.get("maxwell_internal_constraints") != 1
        or internal.get("scalar_internal_constraints") != 0
        or internal.get("fluid_internal_constraints") != 0
    ):
        raise Quartic85StateCandidateGravityConstraintBasisError("matter constraint census changed")
    expected_groups = [
        "quartic metric and gravitational scalar (11)",
        "chi_m (1)",
        "B_mu (4)",
        "tau (1)",
    ]
    if (
        field_basis_receipt.get("matrix_census", {}).get("second_order_field_basis")
        != expected_groups
    ):
        raise Quartic85StateCandidateGravityConstraintBasisError("coupled field ordering changed")
    certificate = reduction.get("reduction_certificate", {})
    if certificate.get("state", {}).get("total") != 85 or certificate.get("assembly", {}).get(
        "state_order"
    ) != ["q_A", "v_A", "w_1A", "w_2A", "w_3A"]:
        raise Quartic85StateCandidateGravityConstraintBasisError("85-state ordering changed")
    fields = _field_basis()
    coordinates = _state_coordinates(fields)
    definitions, curls = _kinematic_rows(fields)
    gravity_sectors = {"metric", "gravity_scalar"}
    gravity_kinematic_rows = sum(row["sector"] in gravity_sectors for row in definitions + curls)
    matter_kinematic_rows = len(definitions) + len(curls) - gravity_kinematic_rows
    if gravity_kinematic_rows != 66 or matter_kinematic_rows != 36:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "gravity/matter kinematic row split changed"
        )
    maxwell = _flat_maxwell_row()
    shared_basis = {
        "field_basis": fields,
        "state_coordinates": coordinates,
        "definition_rows": definitions,
        "curl_rows": curls,
        "flat_maxwell_constraint_row": maxwell,
    }
    shared_sha = _canonical_sha(shared_basis)
    candidate_results: list[dict[str, Any]] = []
    candidates = reduction.get("candidate_results", [])
    if len(candidates) != 12:
        raise Quartic85StateCandidateGravityConstraintBasisError("candidate count changed")
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        manifest = {
            "schema_version": "invariant-candidate-85-state-constraint-coordinate-manifest-1.0",
            "candidate_id": candidate["candidate_id"],
            "first_order_manifest_sha256": candidate["first_order_manifest_sha256"],
            "shared_kinematic_matter_basis_sha256": shared_sha,
            "state_dimension": 85,
            "kinematic_constraint_rows": 102,
            "gravity_sector_kinematic_rows": gravity_kinematic_rows,
            "matter_sector_kinematic_rows": matter_kinematic_rows,
            "flat_matter_constraint_rows": 1,
            "physical_gravity_constraint_coordinate_rows": 0,
            "outcome": "TYPED_BLOCK_GRAVITY_GAUGE_ADM_COORDINATE_MAP_UNREGISTERED",
        }
        candidate_results.append(
            {
                **manifest,
                "constraint_coordinate_manifest_sha256": _canonical_sha(manifest),
            }
        )
    if len({item["constraint_coordinate_manifest_sha256"] for item in candidate_results}) != 12:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "candidate constraint manifests are not one-to-one"
        )
    missing_map = {
        "reason_code": "missing_gravity_gauge_adm_to_85_state_coordinate_map",
        "registered_input": (
            "abstract q_A,v_A,w_iA ordering, exact kinematic rows, flat Maxwell row, and "
            "candidate-specific first-order hashes"
        ),
        "missing": [
            {
                "map": "modified_harmonic_C_mu",
                "components_per_candidate": 4,
                "required": (
                    "exact coefficients and lower-order terms mapping the prescribed gauge "
                    "source/reference connection and metric q,v,w coordinates to C_0..C_3"
                ),
            },
            {
                "map": "Hamiltonian_momentum_constraints",
                "components_per_candidate": 4,
                "required": (
                    "candidate-specific sourced normal/tangential Euler projections as "
                    "coordinate-differential rows in q,v,w and matter variables"
                ),
            },
            {
                "map": "coordinate_to_candidate_jet",
                "components_per_candidate": None,
                "required": (
                    "the nonlinear state-to-covariant-jet formula, not only its incidence, "
                    "including connection, spatial-derivative, and source conventions"
                ),
            },
        ],
        "unregistered_gravity_rows_per_candidate": 8,
        "unregistered_gravity_rows_all_candidates": 96,
    }
    return {
        "shared_kinematic_matter_basis": shared_basis,
        "shared_kinematic_matter_basis_sha256": shared_sha,
        "candidate_results": candidate_results,
        "gravity_coordinate_map_block": missing_map,
        "scientific_boundary": (
            "definition/curl identities and the Maxwell Lorenz row do not determine the "
            "modified-harmonic or Hamiltonian/momentum gravity constraints"
        ),
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-candidate-gravity-constraint-basis-config-1.0"
    ):
        raise Quartic85StateCandidateGravityConstraintBasisError("unsupported config schema")
    expected_convention = {
        "second_order_fields": 17,
        "state_order": ["q_A", "v_A", "w_1A", "w_2A", "w_3A"],
        "state_dimension": 85,
        "flat_metric_signature": "-+++",
    }
    if config.get("state_convention") != expected_convention:
        raise Quartic85StateCandidateGravityConstraintBasisError("state convention changed")
    expected_policy = {
        "kinematic_constraint_coordinate_basis": True,
        "flat_maxwell_constraint_coordinate_row": True,
        "candidate_gravity_constraint_coordinate_basis": False,
        "constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateCandidateGravityConstraintBasisError(
            "claims policy is absent or broadened"
        )
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "sourced_constraint_blocker",
        "coupled_85_state_reduction",
        "vacuum_definition_curl",
        "matter_constraint_interface",
        "coupled_field_basis",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateCandidateGravityConstraintBasisError("closed binding manifest changed")
    materialization = _materialize(
        bound["sourced_constraint_blocker"][1],
        bound["coupled_85_state_reduction"][1],
        bound["vacuum_definition_curl"][1],
        bound["matter_constraint_interface"][1],
        bound["coupled_field_basis"][1],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_candidate_gravity_constraint_basis_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-candidate-gravity-constraint-basis-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "state_coordinates": 85,
            "kinematic_definition_rows_per_candidate": 51,
            "kinematic_curl_rows_per_candidate": 51,
            "gravity_sector_kinematic_rows_per_candidate": 66,
            "matter_sector_kinematic_rows_per_candidate": 36,
            "flat_maxwell_rows_per_candidate": 1,
            "hash_bound_kinematic_matter_bases": 12,
            "physical_gravity_constraint_rows_registered": 0,
            "physical_gravity_constraint_rows_required": 96,
            "constraint_propagation_claims": 0,
            "negative_controls": 1,
        },
        "claims": {
            "all_twelve_kinematic_constraint_coordinate_bases_hash_bound": True,
            "flat_maxwell_constraint_coordinate_row_bound": True,
            "candidate_gravity_constraint_coordinate_basis_closed": False,
            "constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact 85-state coordinate indexing, all 102 derivative-definition/curl rows, and "
            "the flat-reference Maxwell Lorenz row are materialized and bound to each of the "
            "12 candidate first-order manifests. The physical gravity constraint basis remains "
            "blocked because no exact candidate map for modified-harmonic C_mu, sourced "
            "Hamiltonian/momentum projections, or the nonlinear coordinate-to-jet formula is "
            "registered. Propagation, candidate-jet, nonlinear/global, H7, universal-matter, "
            "and promotion claims remain false."
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
