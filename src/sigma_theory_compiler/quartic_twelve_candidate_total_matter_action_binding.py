from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class QuarticTotalMatterActionBindingError(RuntimeError):
    """Raised when a composite action cannot be bound exactly and narrowly."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise QuarticTotalMatterActionBindingError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QuarticTotalMatterActionBindingError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QuarticTotalMatterActionBindingError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTotalMatterActionBindingError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise QuarticTotalMatterActionBindingError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise QuarticTotalMatterActionBindingError(f"bound content hash mismatch: {path}")
    return path, value


def _sector(receipt: dict[str, Any], sector_id: str) -> dict[str, Any]:
    result = next(
        (item for item in receipt.get("sector_results", []) if item.get("sector_id") == sector_id),
        None,
    )
    if not isinstance(result, dict):
        raise QuarticTotalMatterActionBindingError(f"missing matter sector: {sector_id}")
    return result


def _action_evidence(sector: dict[str, Any]) -> dict[str, Any]:
    gate = next(
        (
            item
            for item in sector.get("gates", [])
            if item.get("gate_id") == "action_level_universal_metric_coupling"
        ),
        None,
    )
    if not isinstance(gate, dict) or gate.get("outcome") != "PASS":
        raise QuarticTotalMatterActionBindingError("matter action gate is not PASS")
    evidence = gate.get("evidence")
    if not isinstance(evidence, dict):
        raise QuarticTotalMatterActionBindingError("matter action representation is absent")
    return evidence


def _shared_matter_action(
    universal: dict[str, Any], fluid: dict[str, Any], namespace: dict[str, str]
) -> dict[str, Any]:
    scalar = _action_evidence(_sector(universal, "minimally_coupled_scalar"))
    maxwell = _action_evidence(_sector(universal, "maxwell_lorenz_gauge"))
    admitted_fluid = fluid.get("admitted_action")
    if not isinstance(admitted_fluid, dict) or fluid.get("decision") != "PASS_EARLIEST_GATE_ONLY":
        raise QuarticTotalMatterActionBindingError("fluid action representation is absent")
    metric = namespace.get("physical_metric")
    if (
        scalar.get("physical_metric") != metric
        or maxwell.get("physical_metric") != metric
        or admitted_fluid.get("physical_metric") != metric
    ):
        raise QuarticTotalMatterActionBindingError(
            "matter actions do not share the physical metric"
        )
    expected_fields = {
        namespace.get("gravity_scalar"),
        namespace.get("matter_scalar"),
        namespace.get("maxwell_potential"),
        namespace.get("fluid_potential"),
    }
    if None in expected_fields or len(expected_fields) != 4:
        raise QuarticTotalMatterActionBindingError("field namespaces collide or are absent")
    if scalar.get("matter_field") != namespace["matter_scalar"]:
        raise QuarticTotalMatterActionBindingError("scalar field namespace changed")
    if maxwell.get("matter_field") != namespace["maxwell_potential"]:
        raise QuarticTotalMatterActionBindingError("Maxwell field namespace changed")
    if namespace["fluid_potential"] not in admitted_fluid.get("dependencies", []):
        raise QuarticTotalMatterActionBindingError("fluid field namespace changed")
    if maxwell.get("mass_term_removed_before_maxwell_use") is not True:
        raise QuarticTotalMatterActionBindingError("Maxwell massless specialization is absent")
    return {
        "schema_version": "invariant-shared-three-sector-matter-action-ir-1.0",
        "physical_metric": metric,
        "sum_operation": "ordered_additive_action_sum",
        "components": [
            {
                "sector_id": "canonical_minimally_coupled_scalar",
                "field": namespace["matter_scalar"],
                "density": scalar["action_density"],
                "dependencies": scalar["matter_dependencies"],
                "maximum_derivatives_per_matter_field": 1,
            },
            {
                "sector_id": "source_free_maxwell",
                "field": namespace["maxwell_potential"],
                "density": maxwell["action_density"],
                "dependencies": maxwell["matter_dependencies"],
                "mass_term_removed": True,
                "maximum_derivatives_per_matter_field": 1,
            },
            {
                "sector_id": "barotropic_irrotational_fluid",
                "field": namespace["fluid_potential"],
                "density": admitted_fluid["density"],
                "pressure_function": admitted_fluid["pressure_function"],
                "kinetic_scalar": admitted_fluid["kinetic_scalar"],
                "dependencies": admitted_fluid["dependencies"],
                "domain": admitted_fluid["domain"],
                "maximum_derivatives_per_matter_field": admitted_fluid[
                    "maximum_derivatives_per_matter_field"
                ],
            },
        ],
        "forbidden_candidate_gravity_dependencies": [],
    }


def _gravity_action(
    candidate: dict[str, Any], family: dict[str, Any], namespace: dict[str, str]
) -> dict[str, Any]:
    functions = candidate.get("specialized_action")
    if not isinstance(functions, dict) or set(functions) != {"G2", "G3", "G4", "G5"}:
        raise QuarticTotalMatterActionBindingError("candidate action specialization is incomplete")
    if functions["G3"] != "0" or functions["G5"] != "0":
        raise QuarticTotalMatterActionBindingError("candidate left the registered L2-L4 subclass")
    coefficients = candidate.get("coefficients")
    if not isinstance(coefficients, dict):
        raise QuarticTotalMatterActionBindingError("candidate coefficients are absent")
    template = family.get("normalized_action")
    if (
        not isinstance(template, str)
        or family.get("status") != "compiled_formal_adapters_unresolved"
    ):
        raise QuarticTotalMatterActionBindingError("quartic family action template is absent")
    if family.get("l4_differential_completion", {}).get("independent_choice_forbidden") is not True:
        raise QuarticTotalMatterActionBindingError("Horndeski L4 completion is not bound")
    return {
        "schema_version": "invariant-specialized-quartic-gravity-action-ir-1.0",
        "candidate_id": candidate["candidate_id"],
        "physical_metric": namespace["physical_metric"],
        "gravitational_scalar": namespace["gravity_scalar"],
        "source_field_alpha_rename": {"phi": namespace["gravity_scalar"]},
        "normalized_density_template": template,
        "normalization": family["normalization"],
        "functions": functions,
        "coefficients": coefficients,
        "l4_differential_completion": family["l4_differential_completion"],
        "source_family_content_sha256": family["content_sha256"],
    }


def _composite_manifest(
    gravity_sha: str, matter_sha: str, namespace: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "invariant-composite-total-action-manifest-1.0",
        "physical_metric": namespace["physical_metric"],
        "sum_operation": "S_total=S_gravity+S_scalar+S_Maxwell+S_fluid",
        "ordered_components": [
            {"role": "candidate_gravity", "action_sha256": gravity_sha},
            {"role": "shared_three_sector_matter", "action_sha256": matter_sha},
        ],
        "field_namespace": namespace,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "invariant-quartic-total-matter-action-binding-config-1.0":
        raise QuarticTotalMatterActionBindingError("unsupported config schema")
    expected_policy = {
        "total_action_hash_binding_all_twelve": True,
        "sourced_gauge_fixed_euler_binding": False,
        "total_stress_metric_equation_insertion": False,
        "full_coupled_principal_system": False,
        "full_coupled_symmetrizer": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise QuarticTotalMatterActionBindingError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "census",
        "universal_matter",
        "fluid_action",
        "candidate_actions",
        "quartic_family_ir",
    }:
        raise QuarticTotalMatterActionBindingError("closed binding manifest changed")
    census = bound["census"][1]
    universal = bound["universal_matter"][1]
    fluid = bound["fluid_action"][1]
    candidates_artifact = bound["candidate_actions"][1]
    family = bound["quartic_family_ir"][1]
    if census.get("decision") != "TYPED_BLOCK_CENSUS_NO_CANDIDATE_COUPLED_REGISTRATION":
        raise QuarticTotalMatterActionBindingError("candidate census predecessor changed")
    candidates = candidates_artifact.get("candidates")
    expected_count = config.get("expected_candidate_count")
    if (
        not isinstance(candidates, list)
        or len(candidates) != expected_count
        or expected_count != 12
    ):
        raise QuarticTotalMatterActionBindingError("candidate count mismatch")
    census_ids = {item.get("candidate_id") for item in census.get("candidate_results", [])}
    candidate_ids = {item.get("candidate_id") for item in candidates}
    if None in candidate_ids or len(candidate_ids) != 12 or candidate_ids != census_ids:
        raise QuarticTotalMatterActionBindingError("candidate set mismatch")
    if candidates_artifact.get("source_ir_sha256") != family.get("content_sha256"):
        raise QuarticTotalMatterActionBindingError("candidate family action hash mismatch")

    namespace = config.get("field_namespace")
    if not isinstance(namespace, dict):
        raise QuarticTotalMatterActionBindingError("field namespace is absent")
    matter_action = _shared_matter_action(universal, fluid, namespace)
    matter_sha = _canonical_sha(matter_action)
    results: list[dict[str, Any]] = []
    total_hashes: set[str] = set()
    gravity_hashes: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        gravity_action = _gravity_action(candidate, family, namespace)
        gravity_sha = _canonical_sha(gravity_action)
        manifest = _composite_manifest(gravity_sha, matter_sha, namespace)
        total_sha = _canonical_sha(manifest)
        omitted_fluid_action = {
            **matter_action,
            "components": matter_action["components"][:-1],
        }
        omitted_fluid_sha = _canonical_sha(omitted_fluid_action)
        corrupted_manifest = _composite_manifest(gravity_sha, omitted_fluid_sha, namespace)
        corrupted_total_sha = _canonical_sha(corrupted_manifest)
        if total_sha == corrupted_total_sha:
            raise QuarticTotalMatterActionBindingError("omitted-fluid negative did not change hash")
        total_hashes.add(total_sha)
        gravity_hashes.add(gravity_sha)
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "gravity_action": gravity_action,
                "gravity_action_sha256": gravity_sha,
                "shared_matter_action_sha256": matter_sha,
                "total_action_manifest": manifest,
                "total_action_sha256": total_sha,
                "gate_results": [
                    {"gate_id": "total_matter_action_hash_binding", "outcome": "PASS"},
                    {
                        "gate_id": "sourced_gauge_fixed_euler_same_hash",
                        "outcome": "BLOCK",
                        "reason_code": "sourced_euler_not_constructed_in_action_binding_gate",
                    },
                ],
                "omitted_fluid_negative": {
                    "mutated_shared_matter_action_sha256": omitted_fluid_sha,
                    "mutated_total_action_sha256": corrupted_total_sha,
                    "hash_differs": True,
                    "rejected": True,
                },
            }
        )
    if len(total_hashes) != 12 or len(gravity_hashes) != 12:
        raise QuarticTotalMatterActionBindingError("candidate action hashes are not one-to-one")

    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_twelve_candidate_total_matter_action_binding.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-total-matter-action-binding-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY",
        "shared_matter_action": matter_action,
        "shared_matter_action_sha256": matter_sha,
        "candidate_results": results,
        "counts": {
            "candidates": 12,
            "shared_matter_sectors": 3,
            "total_action_hash_bindings_passed": 12,
            "unique_gravity_action_hashes": 12,
            "unique_total_action_hashes": 12,
            "omitted_fluid_hash_negatives_passed": 12,
            "sourced_euler_bindings_passed": 0,
            "six_item_contract_items_fully_closed": 0,
            "blocks": 12,
            "rejects": 0,
        },
        "claims": {
            "all_twelve_total_actions_compositionally_hash_bound": True,
            "shared_physical_metric_and_distinct_field_namespaces_bound": True,
            "sourced_gauge_fixed_euler_bound_to_total_action": False,
            "total_stress_metric_equation_insertion_closed": False,
            "full_coupled_principal_system_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "sourced_gravity_constraints_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact semantic composition and SHA-256 binding of each of the twelve quartic "
            "gravity actions with the committed canonical scalar, source-free Maxwell, and "
            "irrotational P(X)=kappa X^2 matter actions on g_mu_nu; no sourced Euler, stress "
            "insertion, coupled principal, constraint, H7, universal, or promotion claim"
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
