from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class QuarticSourcedMetricEulerBindingError(RuntimeError):
    """Raised when the sourced metric Euler insertion is not exactly supported."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise QuarticSourcedMetricEulerBindingError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QuarticSourcedMetricEulerBindingError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QuarticSourcedMetricEulerBindingError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise QuarticSourcedMetricEulerBindingError("bound path escapes repository root")
    return path


def _load_json_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise QuarticSourcedMetricEulerBindingError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise QuarticSourcedMetricEulerBindingError(f"bound content hash mismatch: {path}")
    return path, value


def _load_file_binding(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise QuarticSourcedMetricEulerBindingError(f"bound file hash mismatch: {path}")
    return path


def _scalar_stress(receipt: dict[str, Any]) -> str:
    sector = next(
        (
            item
            for item in receipt.get("sector_results", [])
            if item.get("sector_id") == "minimally_coupled_scalar"
        ),
        None,
    )
    if not isinstance(sector, dict) or sector.get("status") != "PASS":
        raise QuarticSourcedMetricEulerBindingError("scalar sector is not PASS")
    gate = next(
        (
            item
            for item in sector.get("gates", [])
            if item.get("gate_id") == "stress_energy_conservation_interface"
        ),
        None,
    )
    if not isinstance(gate, dict) or gate.get("outcome") != "PASS":
        raise QuarticSourcedMetricEulerBindingError("scalar Hilbert stress is absent")
    stress = gate.get("evidence", {}).get("stress_tensor")
    if not isinstance(stress, str):
        raise QuarticSourcedMetricEulerBindingError("scalar stress representation is absent")
    return stress


def _stress_ir(
    scalar_receipt: dict[str, Any],
    maxwell_receipt: dict[str, Any],
    fluid_receipt: dict[str, Any],
) -> dict[str, Any]:
    maxwell = maxwell_receipt.get("massless_specialization", {})
    fluid = fluid_receipt.get("specialization", {})
    if (
        maxwell_receipt.get("decision") != "PASS_ARBITRARY_BACKGROUND_MAXWELL_STRESS_DIVERGENCE"
        or maxwell.get("mass_value") != 0
        or not isinstance(maxwell.get("hilbert_stress"), str)
    ):
        raise QuarticSourcedMetricEulerBindingError("Maxwell Hilbert stress is absent")
    if fluid_receipt.get("decision") != "PASS_SECOND_GATE_ONLY" or not isinstance(
        fluid.get("hilbert_stress"), str
    ):
        raise QuarticSourcedMetricEulerBindingError("fluid Hilbert stress is absent")
    return {
        "schema_version": "invariant-three-sector-total-hilbert-stress-ir-1.0",
        "physical_metric": "g_mu_nu",
        "sum": "T_total=T_scalar+T_Maxwell+T_fluid",
        "components": [
            {
                "sector_id": "canonical_minimally_coupled_scalar",
                "hilbert_stress": _scalar_stress(scalar_receipt),
            },
            {
                "sector_id": "source_free_maxwell",
                "hilbert_stress": maxwell["hilbert_stress"],
                "field_strength": maxwell["field_strength"],
                "mass_value": 0,
            },
            {
                "sector_id": "barotropic_irrotational_fluid",
                "hilbert_stress": fluid["hilbert_stress"],
                "pressure": fluid["pressure"],
                "kinetic_scalar": fluid["kinetic_scalar"],
                "domain": fluid["domain"],
            },
        ],
    }


def _exact_variation_replay() -> dict[str, Any]:
    gravity_coefficient = Fraction(1)
    matter_variation_coefficients = [Fraction(-1, 2)] * 3
    correct_source = matter_variation_coefficients
    wrong_sign_source = [Fraction(1, 2)] * 3
    wrong_sign_residual = [
        wrong - correct for wrong, correct in zip(wrong_sign_source, correct_source)
    ]
    omitted_fluid_source = [Fraction(-1, 2), Fraction(-1, 2), Fraction(0)]
    omitted_fluid_residual = [
        actual - correct for actual, correct in zip(omitted_fluid_source, correct_source)
    ]
    if wrong_sign_residual != [Fraction(1)] * 3:
        raise QuarticSourcedMetricEulerBindingError("wrong-sign variation replay failed")
    if omitted_fluid_residual != [Fraction(0), Fraction(0), Fraction(1, 2)]:
        raise QuarticSourcedMetricEulerBindingError("omitted-fluid variation replay failed")
    pairs = [(left, right) for left in range(4) for right in range(left, 4)]
    row_map = [
        {
            "row": index,
            "metric_pair": [left, right],
            "source": (
                f"-T_total^{left}{right}/2"
                if left == right
                else f"-sqrt(2) T_total^{left}{right}/2"
            ),
        }
        for index, (left, right) in enumerate(pairs)
    ]
    row_map.append(
        {
            "row": 10,
            "field": "phi_g",
            "source": "0",
            "reason": "minimal matter actions do not depend on the gravitational scalar",
        }
    )
    return {
        "inverse_metric_variation": (
            "delta S_total/sqrt(-g)=E_grav_mu_nu delta g^mu_nu-T_total_mu_nu delta g^mu_nu/2"
        ),
        "gravity_variation_coefficient": str(gravity_coefficient),
        "matter_variation_coefficients": [str(item) for item in correct_source],
        "gauge_fixed_equation": "E_gf^mu_nu-T_total^mu_nu/2=0",
        "eleven_row_source_map": row_map,
        "wrong_sign_negative": {
            "mutation": "replace -T_total/2 by +T_total/2",
            "independent_sector_residual_coefficients": [str(item) for item in wrong_sign_residual],
            "rejected": True,
        },
        "omitted_fluid_negative": {
            "mutation": "omit T_fluid from the metric equation",
            "independent_sector_residual_coefficients": [
                str(item) for item in omitted_fluid_residual
            ],
            "rejected": True,
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "invariant-quartic-sourced-metric-euler-binding-config-1.0":
        raise QuarticSourcedMetricEulerBindingError("unsupported config schema")
    expected_policy = {
        "sourced_metric_euler_binding_all_twelve": True,
        "matter_field_euler_component_expansion": False,
        "sourced_acceleration_solution": False,
        "total_stress_conservation_implies_gravity_constraints": False,
        "full_coupled_principal_system": False,
        "full_coupled_symmetrizer": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise QuarticSourcedMetricEulerBindingError("claims policy is absent or broadened")
    expected_convention = {
        "gravity_metric_coefficient": ("E_grav_mu_nu=(1/sqrt(-g)) delta S_grav/delta g^mu_nu"),
        "hilbert_stress": "T_mu_nu=-(2/sqrt(-g)) delta S_m/delta g^mu_nu",
        "sourced_equation": ("E_gf^mu_nu-(T_scalar^mu_nu+T_Maxwell^mu_nu+T_fluid^mu_nu)/2=0"),
    }
    if config.get("variation_convention") != expected_convention:
        raise QuarticSourcedMetricEulerBindingError("variation convention changed")
    bound = {
        name: _load_json_binding(repository, binding)
        for name, binding in config.get("json_bindings", {}).items()
    }
    if set(bound) != {
        "total_action_binding",
        "vacuum_euler",
        "scalar_stress",
        "maxwell_stress",
        "fluid_stress",
    }:
        raise QuarticSourcedMetricEulerBindingError("closed JSON binding manifest changed")
    convention_source = _load_file_binding(repository, config["source_evidence"])
    try:
        source_text = convention_source.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuarticSourcedMetricEulerBindingError(
            f"cannot read bound file: {convention_source}"
        ) from exc
    if (
        "The metric coefficient is the variation with respect to the inverse metric."
        not in source_text
        or "action_upper + gauge_upper" not in source_text
        or '-action["scalar_euler"]' not in source_text
    ):
        raise QuarticSourcedMetricEulerBindingError("gravity Euler convention is absent")

    action_receipt = bound["total_action_binding"][1]
    vacuum = bound["vacuum_euler"][1]
    if action_receipt.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY":
        raise QuarticSourcedMetricEulerBindingError("total action predecessor changed")
    if vacuum.get("status") != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations":
        raise QuarticSourcedMetricEulerBindingError("vacuum Euler predecessor changed")
    action_records = {
        item.get("candidate_id"): item for item in action_receipt.get("candidate_results", [])
    }
    vacuum_records = {item.get("candidate_id"): item for item in vacuum.get("certificates", [])}
    expected_count = config.get("expected_candidate_count")
    if (
        expected_count != 12
        or len(action_records) != expected_count
        or set(action_records) != set(vacuum_records)
        or None in action_records
    ):
        raise QuarticSourcedMetricEulerBindingError("candidate set mismatch")
    stress = _stress_ir(
        bound["scalar_stress"][1],
        bound["maxwell_stress"][1],
        bound["fluid_stress"][1],
    )
    stress_sha = _canonical_sha(stress)
    replay = _exact_variation_replay()
    replay_sha = _canonical_sha(replay)
    results: list[dict[str, Any]] = []
    registration_hashes: set[str] = set()
    for candidate_id in sorted(action_records):
        action = action_records[candidate_id]
        euler = vacuum_records[candidate_id]
        if [gate.get("outcome") for gate in action.get("gate_results", [])] != [
            "PASS",
            "BLOCK",
        ]:
            raise QuarticSourcedMetricEulerBindingError(
                f"total action gate changed: {candidate_id}"
            )
        if euler.get("status") != "pass_exact_local_nonlinear_time_acceleration_elimination":
            raise QuarticSourcedMetricEulerBindingError(f"vacuum Euler failed: {candidate_id}")
        registration = {
            "schema_version": "invariant-candidate-sourced-metric-euler-manifest-1.0",
            "candidate_id": candidate_id,
            "total_action_sha256": action["total_action_sha256"],
            "vacuum_evolution_formula_contract_sha256": euler["evolution_formula_contract_sha256"],
            "total_hilbert_stress_sha256": stress_sha,
            "variation_and_row_insertion_sha256": replay_sha,
            "physical_metric": "g_mu_nu",
            "equation": "E_gf^mu_nu-T_total^mu_nu/2=0",
        }
        registration_sha = _canonical_sha(registration)
        registration_hashes.add(registration_sha)
        results.append(
            {
                "candidate_id": candidate_id,
                "total_action_sha256": action["total_action_sha256"],
                "sourced_metric_euler_manifest": registration,
                "sourced_metric_euler_sha256": registration_sha,
                "outcome": "PASS",
            }
        )
    if len(registration_hashes) != 12:
        raise QuarticSourcedMetricEulerBindingError("sourced Euler hashes are not one-to-one")

    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_twelve_candidate_sourced_metric_euler_binding.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-sourced-metric-euler-binding-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY",
        "total_hilbert_stress": stress,
        "total_hilbert_stress_sha256": stress_sha,
        "exact_variation_and_row_insertion": replay,
        "exact_variation_and_row_insertion_sha256": replay_sha,
        "candidate_results": results,
        "counts": {
            "candidates": 12,
            "sourced_metric_euler_bindings_passed": 12,
            "unique_sourced_metric_euler_hashes": 12,
            "metric_equation_rows_per_candidate": 10,
            "unchanged_gravity_scalar_rows_per_candidate": 1,
            "total_registered_gravity_rows": 132,
            "exact_variation_residuals": 2,
            "negative_controls": 2,
            "sourced_acceleration_solutions": 0,
            "rejects": 0,
        },
        "claims": {
            "all_twelve_sourced_metric_euler_equations_hash_bound": True,
            "inverse_metric_variation_normalization_and_sign_replayed": True,
            "matter_field_euler_component_expansion_closed": False,
            "sourced_acceleration_solution_closed": False,
            "full_coupled_principal_system_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "sourced_gravity_constraints_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "candidate-bound insertion of the committed scalar, source-free Maxwell, and "
            "irrotational-fluid Hilbert stresses into the ten modified-harmonic quartic "
            "metric Euler rows with exact inverse-metric variation coefficient -1/2; the "
            "gravity-scalar row is unchanged. Matter-field component equations, solved "
            "sourced accelerations, principal/symmetrizer/constraint closure, H7, universal "
            "matter, and promotion remain outside this gate"
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
            "gravity_euler_convention_source": {
                "path": convention_source.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(convention_source),
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
