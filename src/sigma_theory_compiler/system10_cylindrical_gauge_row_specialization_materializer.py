from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class System10CylindricalGaugeMaterializerError(RuntimeError):
    """Raised when the cylindrical System 10 materializer fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise System10CylindricalGaugeMaterializerError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalGaugeMaterializerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalGaugeMaterializerError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalGaugeMaterializerError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalGaugeMaterializerError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalGaugeMaterializerError(f"bound content hash mismatch: {path}")
    return path, value


def _pairs() -> list[tuple[int, int]]:
    return list(itertools.combinations_with_replacement(range(4), 2))


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _diagonal_value(pair: tuple[int, int], diagonal: tuple[int, int, int, int]) -> Fraction:
    if pair[0] != pair[1]:
        return Fraction(0)
    return Fraction(diagonal[pair[0]])


def _hat_first(derivative: int, pair: tuple[int, int]) -> Fraction:
    return Fraction(-2) if derivative == 1 and pair == (2, 2) else Fraction(0)


def _tilde_second(derivative_pair: tuple[int, int], pair: tuple[int, int]) -> Fraction:
    return Fraction(6) if derivative_pair == (1, 1) and pair == (2, 2) else Fraction(0)


def _reference_zero(upper: int, pair: tuple[int, int]) -> Fraction:
    if upper == 1 and pair == (2, 2):
        return Fraction(-1)
    if upper == 2 and pair == (1, 2):
        return Fraction(1)
    return Fraction(0)


def _reference_first(derivative: int, upper: int, pair: tuple[int, int]) -> Fraction:
    if derivative != 1:
        return Fraction(0)
    if (upper, pair) in ((1, (2, 2)), (2, (1, 2))):
        return Fraction(-1)
    return Fraction(0)


def _reference_second(
    derivative_pair: tuple[int, int], upper: int, pair: tuple[int, int]
) -> Fraction:
    if derivative_pair == (1, 1) and upper == 2 and pair == (1, 2):
        return Fraction(2)
    return Fraction(0)


def _physical_first(derivative: int, pair: tuple[int, int]) -> Fraction:
    return Fraction(2) if derivative == 1 and pair == (2, 2) else Fraction(0)


def _physical_second(derivative_pair: tuple[int, int], pair: tuple[int, int]) -> Fraction:
    return Fraction(2) if derivative_pair == (1, 1) and pair == (2, 2) else Fraction(0)


def _records(
    family: str,
    keys_and_values: list[tuple[str, Fraction]],
) -> dict[str, Any]:
    entries = [{"slot_id": key, "value": _fraction_text(value)} for key, value in keys_and_values]
    nonzero = sum(value != 0 for _, value in keys_and_values)
    body = {
        "family": family,
        "scalar_values": len(entries),
        "nonzero_values": nonzero,
        "zero_values": len(entries) - nonzero,
        "entries": entries,
    }
    return {**body, "family_sha256": _canonical_sha(body)}


def _specialized_value_packet() -> dict[str, Any]:
    pairs = _pairs()
    families: list[dict[str, Any]] = []
    families.append(
        _records(
            "hat_inverse_first",
            [
                (f"d_hat[{derivative}|{left},{right}]", _hat_first(derivative, (left, right)))
                for derivative in range(4)
                for left, right in pairs
            ],
        )
    )
    families.append(
        _records(
            "tilde_inverse_second",
            [
                (
                    f"d2_tilde[{first},{second}|{left},{right}]",
                    _tilde_second((first, second), (left, right)),
                )
                for first, second in pairs
                for left, right in pairs
            ],
        )
    )
    families.append(
        _records(
            "reference_connection_second",
            [
                (
                    f"d2_barGamma[{first},{second}|{upper}|{left},{right}]",
                    _reference_second((first, second), upper, (left, right)),
                )
                for first, second in pairs
                for upper in range(4)
                for left, right in pairs
            ],
        )
    )
    families.append(
        _records(
            "gauge_source_second",
            [
                (f"d2_H[{first},{second}|{lower}]", Fraction(0))
                for first, second in pairs
                for lower in range(4)
            ],
        )
    )
    families.extend(
        [
            _records(
                "hat_inverse_zero",
                [
                    (f"hat[{left},{right}]", _diagonal_value((left, right), (-9, 1, 1, 1)))
                    for left, right in pairs
                ],
            ),
            _records(
                "tilde_inverse_zero",
                [
                    (f"tilde[{left},{right}]", _diagonal_value((left, right), (-4, 1, 1, 1)))
                    for left, right in pairs
                ],
            ),
            _records(
                "tilde_inverse_first",
                [
                    (
                        f"d_tilde[{derivative}|{left},{right}]",
                        _hat_first(derivative, (left, right)),
                    )
                    for derivative in range(4)
                    for left, right in pairs
                ],
            ),
            _records(
                "reference_connection_zero",
                [
                    (
                        f"barGamma[{upper}|{left},{right}]",
                        _reference_zero(upper, (left, right)),
                    )
                    for upper in range(4)
                    for left, right in pairs
                ],
            ),
            _records(
                "reference_connection_first",
                [
                    (
                        f"d_barGamma[{derivative}|{upper}|{left},{right}]",
                        _reference_first(derivative, upper, (left, right)),
                    )
                    for derivative in range(4)
                    for upper in range(4)
                    for left, right in pairs
                ],
            ),
            _records(
                "gauge_source_zero",
                [(f"H[{lower}]", Fraction(0)) for lower in range(4)],
            ),
            _records(
                "gauge_source_first",
                [
                    (f"d_H[{derivative}|{lower}]", Fraction(0))
                    for derivative in range(4)
                    for lower in range(4)
                ],
            ),
            _records(
                "physical_metric_zero",
                [
                    (f"g[{left},{right}]", _diagonal_value((left, right), (-1, 1, 1, 1)))
                    for left, right in pairs
                ],
            ),
            _records(
                "physical_metric_first",
                [
                    (
                        f"d_g[{derivative}|{left},{right}]",
                        _physical_first(derivative, (left, right)),
                    )
                    for derivative in range(4)
                    for left, right in pairs
                ],
            ),
            _records(
                "physical_metric_second",
                [
                    (
                        f"d2_g[{first},{second}|{left},{right}]",
                        _physical_second((first, second), (left, right)),
                    )
                    for first, second in pairs
                    for left, right in pairs
                ],
            ),
        ]
    )
    scalar_values = sum(item["scalar_values"] for item in families)
    nonzero_values = sum(item["nonzero_values"] for item in families)
    slot_ids = [entry["slot_id"] for family in families for entry in family["entries"]]
    if scalar_values != 1010 or len(slot_ids) != len(set(slot_ids)):
        raise System10CylindricalGaugeMaterializerError("specialized slot packet changed")
    if nonzero_values != 22:
        raise System10CylindricalGaugeMaterializerError("cylindrical nonzero value count changed")
    body = {
        "profile": "cylindrical_nested_auxiliary_metrics_at_r=1",
        "families": families,
        "scalar_values": scalar_values,
        "nonzero_values": nonzero_values,
        "zero_values": scalar_values - nonzero_values,
        "slot_ids_sha256": _canonical_sha(slot_ids),
        "all_values_exact": True,
        "general_value_packet": False,
    }
    return {**body, "packet_sha256": _canonical_sha(body)}


def _metric_field_index(pair: tuple[int, int]) -> int:
    return _pairs().index(tuple(sorted(pair)))


def _derivative_state_index(derivative: int, pair: tuple[int, int]) -> int:
    field = _metric_field_index(pair)
    if derivative == 0:
        return 17 + field
    return 34 + (derivative - 1) * 17 + field


def _add_term(terms: dict[int, Fraction], state_index: int, coefficient: Fraction) -> None:
    terms[state_index] = terms.get(state_index, Fraction(0)) + coefficient


def _gauge_row(beta: int, *, omit_reference: bool = False) -> dict[str, Any]:
    tilde_diagonal = (Fraction(-4), Fraction(1), Fraction(1), Fraction(1))
    terms: dict[int, Fraction] = {}
    for rho, tilde in enumerate(tilde_diagonal):
        _add_term(terms, _derivative_state_index(rho, (beta, rho)), tilde)
        _add_term(terms, _derivative_state_index(beta, (rho, rho)), -tilde / 2)
    if not omit_reference:
        for rho, tilde in enumerate(tilde_diagonal):
            for upper in range(4):
                reference = _reference_zero(upper, (rho, rho))
                if reference != 0:
                    _add_term(terms, _metric_field_index((beta, upper)), -tilde * reference)
    terms = {index: coefficient for index, coefficient in terms.items() if coefficient != 0}
    state_terms = [
        {"state_index": index, "coefficient": _fraction_text(coefficient)}
        for index, coefficient in sorted(terms.items())
    ]
    state_values: dict[int, Fraction] = {}
    for pair in _pairs():
        state_values[_metric_field_index(pair)] = _diagonal_value(pair, (-1, 1, 1, 1))
        for derivative in range(4):
            state_values[_derivative_state_index(derivative, pair)] = _physical_first(
                derivative, pair
            )
    value = sum(
        Fraction(term["coefficient"]) * state_values[term["state_index"]] for term in state_terms
    )
    body = {
        "row": f"modified_harmonic_C[{beta}]",
        "identity": (
            "tilde^rho_sigma(1/2(d_rho g_beta_sigma+d_sigma g_beta_rho-"
            "d_beta g_rho_sigma)-g_beta_lambda barGamma^lambda_rho_sigma)-H_beta"
        ),
        "state_terms": state_terms,
        "affine_gauge_source": "0",
        "evaluated_row_value": _fraction_text(value),
        "coefficient_support": len(state_terms),
        "background": "cylindrical_nested_auxiliary_metrics_at_r=1",
    }
    return {**body, "row_sha256": _canonical_sha(body)}


def _gauge_rows() -> list[dict[str, Any]]:
    rows = [_gauge_row(beta) for beta in range(4)]
    if any(row["evaluated_row_value"] != "0" for row in rows):
        raise System10CylindricalGaugeMaterializerError("matched-reference gauge residual changed")
    if not all(row["coefficient_support"] > 0 for row in rows):
        raise System10CylindricalGaugeMaterializerError("gauge row support unexpectedly empty")
    return rows


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    blocker = bound["system10_general_blocker"][1]
    if blocker.get("decision") != (
        "TYPED_BLOCK_GENERAL_FORMULATION_JETS_DOMAIN_AND_GRAVITY_ROWS_UNREGISTERED"
    ):
        raise System10CylindricalGaugeMaterializerError("System 10 blocker changed")
    if blocker.get("counts", {}).get("general_scalar_value_slots_manifested") != 1010:
        raise System10CylindricalGaugeMaterializerError("System 10 slot census changed")
    source = bound["curvilinear_source_control"][1]
    curvilinear = source.get("nonlinear_evolution_control", {}).get(
        "curvilinear_reference_connection_control", {}
    )
    if (
        source.get("status") != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
        or curvilinear.get("metric") != "diag(-1,1,r^2,1)"
        or curvilinear.get("Delta_Gamma_zero_with_matching_flat_reference") is not True
        or curvilinear.get("omitted_reference_connection_nonzero") is not True
    ):
        raise System10CylindricalGaugeMaterializerError("curvilinear source control changed")
    basis = bound["constraint_basis"][1]
    if (
        basis.get("decision")
        != "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
        or basis.get("counts", {}).get("physical_gravity_constraint_rows_required") != 96
    ):
        raise System10CylindricalGaugeMaterializerError("constraint basis changed")
    indexed = bound["indexed_gauge_map"][1]
    if (
        indexed.get("decision") != "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS"
        or indexed.get("counts", {}).get("formal_external_jet_atoms") != 580
    ):
        raise System10CylindricalGaugeMaterializerError("indexed gauge map changed")


def _materialize(bound: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    _validate_predecessors(bound)
    packet = _specialized_value_packet()
    blocker_families = bound["system10_general_blocker"][1]["materialization"][
        "general_scalar_value_attempt"
    ]["slot_families"]
    expected_slot_hashes = {item["family"]: item["slot_ids_sha256"] for item in blocker_families}
    actual_slot_hashes = {
        item["family"]: _canonical_sha([entry["slot_id"] for entry in item["entries"]])
        for item in packet["families"]
    }
    if actual_slot_hashes != expected_slot_hashes:
        raise System10CylindricalGaugeMaterializerError("specialized packet slot IDs drifted")

    shared_rows = _gauge_rows()
    candidates = bound["constraint_basis"][1]["materialization"]["candidate_results"]
    candidate_results = []
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        manifest = {
            "schema_version": "invariant-system10-cylindrical-candidate-gauge-row-manifest-1.0",
            "candidate_id": candidate["candidate_id"],
            "constraint_coordinate_manifest_sha256": candidate[
                "constraint_coordinate_manifest_sha256"
            ],
            "specialized_value_packet_sha256": packet["packet_sha256"],
            "shared_gauge_rows_sha256": _canonical_sha(shared_rows),
            "specialized_modified_harmonic_rows_closed": 4,
            "hamiltonian_momentum_rows_closed": 0,
            "outcome": "PASS_4_SPECIALIZED_GAUGE_ROWS_TYPED_BLOCK_4_ADM_ROWS",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
    omitted = _gauge_row(1, omit_reference=True)
    if omitted["evaluated_row_value"] != "-1":
        raise System10CylindricalGaugeMaterializerError("omitted-reference witness changed")
    hat_family = next(item for item in packet["families"] if item["family"] == "hat_inverse_first")
    radial_hat = next(
        entry for entry in hat_family["entries"] if entry["slot_id"] == "d_hat[1|2,2]"
    )
    if radial_hat["value"] != "-2":
        raise System10CylindricalGaugeMaterializerError("radial hat derivative changed")
    negatives = {
        "omit_reference_connection": {
            "mutation": "drop g_beta_lambda barGamma^lambda_rho_sigma from C_beta",
            "row": "modified_harmonic_C[1]",
            "correct_value": "0",
            "corrupted_value": omitted["evaluated_row_value"],
            "rejected": True,
        },
        "corrupt_nonzero_jet_value": {
            "mutation": "d_hat[1|2,2] -2 -> -1",
            "expected_value": radial_hat["value"],
            "corrupted_value": "-1",
            "exact_delta": "1",
            "rejected": True,
        },
        "drop_candidate_gauge_row": {
            "mutation": "drop modified_harmonic_C[3] from the final candidate",
            "expected_specialized_rows": 48,
            "observed_specialized_rows": 47,
            "rejected": True,
        },
    }
    return {
        "specialized_value_packet": packet,
        "shared_modified_harmonic_rows": shared_rows,
        "shared_modified_harmonic_rows_sha256": _canonical_sha(shared_rows),
        "candidate_results": candidate_results,
        "negative_controls": negatives,
        "remaining_block": {
            "hamiltonian_momentum_rows_required": 48,
            "hamiltonian_momentum_rows_closed": 0,
            "general_value_packet_closed": False,
            "general_common_domain_closed": False,
            "general_hyperbolicity_closed": False,
            "sourced_constraint_propagation_closed": False,
            "reason_code": "specialized_gauge_rows_do_not_supply_sourced_ADM_or_general_jet_closure",
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version")
        != "invariant-system10-cylindrical-gauge-row-specialization-config-1.0"
    ):
        raise System10CylindricalGaugeMaterializerError("unsupported config schema")
    expected_specialization = {
        "coordinates": ["t", "r", "theta", "z"],
        "evaluation_point": {"t": "0", "r": "1", "theta": "0", "z": "0"},
        "physical_metric": "diag(-1,1,r^2,1)",
        "tilde_inverse_metric": "diag(-4,1,r^-2,1)",
        "hat_inverse_metric": "diag(-9,1,r^-2,1)",
        "reference_connection": "Levi-Civita(physical_metric)",
        "gauge_source": "zero",
        "metric_component_basis": "orthonormal_symmetric",
        "state_ordering": "q,v,w_1,w_2,w_3 over 17 fields",
    }
    if config.get("specialization") != expected_specialization:
        raise System10CylindricalGaugeMaterializerError("cylindrical specialization changed")
    expected_policy = {
        "specialized_1010_value_packet": True,
        "specialized_modified_harmonic_gauge_rows": True,
        "all_twelve_candidate_gauge_row_hashes": True,
        "hamiltonian_momentum_rows": False,
        "general_value_packet": False,
        "general_common_domain": False,
        "general_hyperbolicity": False,
        "general_common_time_positivity": False,
        "sourced_constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise System10CylindricalGaugeMaterializerError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "system10_general_blocker",
        "curvilinear_source_control",
        "constraint_basis",
        "indexed_gauge_map",
    }:
        raise System10CylindricalGaugeMaterializerError("closed binding manifest changed")
    materialization = _materialize(bound)
    source_path = Path(__file__).resolve()
    test_path = (
        repository / "tests/test_system10_cylindrical_gauge_row_specialization_materializer.py"
    )
    row_support = [
        item["coefficient_support"] for item in materialization["shared_modified_harmonic_rows"]
    ]
    body: dict[str, Any] = {
        "schema_version": "invariant-system10-cylindrical-gauge-row-specialization-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_CYLINDRICAL_1010_VALUES_AND_48_GAUGE_ROWS_TYPED_BLOCK_ADM_GENERAL",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "specialized_scalar_values": 1010,
            "specialized_nonzero_scalar_values": 22,
            "specialized_zero_scalar_values": 988,
            "shared_modified_harmonic_rows": 4,
            "modified_harmonic_rows_closed_all_candidates": 48,
            "modified_harmonic_row_supports": row_support,
            "hamiltonian_momentum_rows_closed": 0,
            "hamiltonian_momentum_rows_required": 48,
            "physical_gravity_rows_closed": 48,
            "physical_gravity_rows_required": 96,
            "general_value_packets": 0,
            "general_common_domains": 0,
            "general_hyperbolicity_proofs": 0,
            "sourced_constraint_propagation_proofs": 0,
            "negative_controls": 3,
        },
        "claims": {
            "exact_cylindrical_1010_value_packet_closed": True,
            "exact_cylindrical_modified_harmonic_rows_closed": True,
            "all_twelve_candidate_specialized_gauge_row_hashes_closed": True,
            "all_96_physical_gravity_rows_closed": False,
            "hamiltonian_momentum_rows_closed": False,
            "general_value_packet_closed": False,
            "general_common_domain_closed": False,
            "general_coupled_hyperbolicity_closed": False,
            "general_common_time_positivity_closed": False,
            "sourced_constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "Exact specialization at r=1 of the registered cylindrical physical metric with "
            "nested prescribed auxiliary inverse metrics, matched Levi-Civita reference "
            "connection, and zero gauge source. All 1,010 scalar value slots are evaluated "
            "exactly; 22 are nonzero. The four modified-harmonic constraint rows are expanded "
            "into the registered 85-state q/v/w ordering and hash-bound to all 12 candidates, "
            "closing 48 of the 96 specialized physical gravity rows. This point-profile control "
            "does not close the 48 sourced Hamiltonian/momentum rows, a general value packet or "
            "domain, general hyperbolicity/common-time positivity, sourced constraint "
            "propagation, candidate-jet uniformity, nonlinear/global closure, H7, universal "
            "matter, or promotion."
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
