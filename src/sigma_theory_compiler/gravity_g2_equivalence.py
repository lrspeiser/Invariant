"""G2 semantic and behavioral equivalence collapse for all retained G1 formulas."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g1_atlas_repair_v3 import validate_receipt as validate_g1_receipt
from .gravity_g1_pilot import _binding, _file_sha256, _load_json
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import assemble

SCHEMA = "invariant-gravity-g2-equivalence-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g2-equivalence-config-1.0"
CONFIG_PATH = "configs/gravity_g2_equivalence.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g2_equivalence.py"
TEST_PATH = "tests/test_gravity_g2_equivalence.py"
OUTPUT_PATH = "runs/gravity/g2/gravity-formula-equivalence-classes-v1.json"

NATIVE_RANGES = {
    "log_y": (-8.0, 8.0),
    "log_r_over_disk_peak": (-8.0, 8.0),
    "gas_fraction": (-2.0, 3.0),
    "disk_fraction": (-2.0, 3.0),
    "bulge_fraction": (-2.0, 3.0),
    "baryon_log_slope": (-8.0, 8.0),
    "mass_proxy_fraction": (-2.0, 3.0),
    "gas_to_disk": (-4.0, 0.0),
}
FEATURE_IDS = tuple(NATIVE_RANGES)


class GravityG2EquivalenceError(ValueError):
    """The G2 equivalence analysis or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load G2 and validate its complete G1 predecessor chain."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG2EquivalenceError("G2 config schema changed")
    binding = config.get("g1_binding")
    if not isinstance(binding, Mapping):
        raise GravityG2EquivalenceError("G2 G1 binding is missing")
    path = root / str(binding["path"])
    if _file_sha256(path) != binding.get("file_sha256"):
        raise GravityG2EquivalenceError("G2 G1 file binding changed")
    receipt = _load_json(path)
    validate_g1_receipt(receipt, root=root)
    if receipt.get("content_sha256") != binding.get("content_sha256"):
        raise GravityG2EquivalenceError("G2 G1 content binding changed")
    if receipt.get("decision") != binding.get("required_decision"):
        raise GravityG2EquivalenceError("G2 G1 decision changed")
    if receipt.get("counts", {}).get("union_covered_galaxies") != binding.get(
        "required_galaxy_count"
    ):
        raise GravityG2EquivalenceError("G2 G1 galaxy count changed")
    behavior = config.get("behavioral_equivalence", {})
    if behavior.get("adversarial_design_points") != 257 or behavior.get("probe_columns") != 8:
        raise GravityG2EquivalenceError("G2 adversarial design changed")
    if config.get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG2EquivalenceError("G2 permits confirmation access")
    return config


def _semantic_component(component: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in component.items()
        if key not in {"llm_origin_assessment"}
    }


def canonical_formula_ir(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Remove nonsemantic fit fields and canonicalize the two-column span."""

    components = [_semantic_component(row) for row in candidate["components"]]
    components.sort(key=canonical_json_bytes)
    rar_base = "base_family" in candidate or "V_RAR" in str(candidate.get("formula", ""))
    return {
        "base": "empirical_RAR" if rar_base else "newtonian_baryons",
        "coefficient_model": "unordered_two_column_linear_span",
        "components": components,
        "feature_normalization": (
            "within_galaxy_baryonic_minmax_to_minus1_plus1" if rar_base else "native_v3_feature"
        ),
        "output": "circular_velocity_squared",
    }


def structural_signature(candidate: Mapping[str, Any]) -> str:
    return canonical_sha256(canonical_formula_ir(candidate))


def collect_survivors(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect every retained G1 survivor exactly once."""

    root = root.resolve()
    g1 = _load_json(root / str(config["g1_binding"]["path"]))
    repair_candidates = g1["repair"]["retained_pareto"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    checkpoint_dir = root / "runs/gravity/g1-atlas/checkpoints-v1"
    for galaxy in assemble(root).exploration:
        if galaxy.name == "NGC2955":
            candidates = repair_candidates
            source = "repair-v3"
        else:
            checkpoint = _load_json(checkpoint_dir / f"{galaxy.name}.json")
            candidates = checkpoint["retained_pareto"]
            source = "atlas-v1"
        if not candidates:
            raise GravityG2EquivalenceError(f"G2 received no survivors for {galaxy.name}")
        for candidate in candidates:
            segment = str(candidate.get("segment_id", source))
            candidate_id = f"{galaxy.name}:{segment}:{int(candidate['ordinal'])}"
            if candidate_id in seen:
                raise GravityG2EquivalenceError(f"duplicate G1 survivor: {candidate_id}")
            seen.add(candidate_id)
            proposer = candidate.get("origin_assessment")
            if proposer is None:
                component_labels = {
                    row.get("llm_origin_assessment")
                    for row in candidate["components"]
                    if row.get("llm_origin_assessment") is not None
                }
                proposer = (
                    "new_combination_of_known_ideas"
                    if component_labels
                    else "known_family_instance"
                )
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_id": candidate_id,
                    "galaxy": galaxy.name,
                    "proposer_origin_assessment": proposer,
                }
            )
    required = int(config["g1_binding"]["required_survivor_count"])
    if len(rows) != required:
        raise GravityG2EquivalenceError(
            f"G2 survivor count changed: expected {required}, received {len(rows)}"
        )
    return rows


def adversarial_design(point_count: int, probe_count: int) -> tuple[dict[str, np.ndarray], Any]:
    """Build deterministic, non-grid-aligned feature values and projection probes."""

    index = np.arange(point_count, dtype=np.float64)
    strides = (1, 3, 5, 7, 11, 13, 17, 19)
    native: dict[str, np.ndarray] = {}
    normalized: dict[str, np.ndarray] = {}
    for feature_id, stride in zip(FEATURE_IDS, strides, strict=True):
        unit = ((index * stride + 0.5 * stride) % point_count) / (point_count - 1)
        low, high = NATIVE_RANGES[feature_id]
        native[feature_id] = low + (high - low) * unit
        normalized[feature_id] = -1.0 + 2.0 * unit
    probes = np.column_stack(
        [
            np.sin((column + 1) * np.pi * (index + 0.375) / point_count)
            + np.cos((column + 2) * np.sqrt(2.0) * (index + 0.125) / point_count)
            for column in range(probe_count)
        ]
    )
    return {"native": native, "normalized": normalized}, probes


def evaluate_component(
    component: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate one semantic component on the adversarial feature design."""

    family = str(component["family"])
    if family == "chebyshev_feature_product":
        first = features[str(component["first_feature"])]
        second = features[str(component["second_feature"])]
        first_degree = int(component["first_degree"])
        second_degree = int(component["second_degree"])
        first_values = np.polynomial.chebyshev.chebval(
            first, [0.0] * first_degree + [1.0]
        )
        second_values = np.polynomial.chebyshev.chebval(
            second, [0.0] * second_degree + [1.0]
        )
        return first_values * second_values
    value = features[str(component["feature"])]
    if family == "generalized_feature_rbf":
        center = float(component["center"])
        width = float(component["width"])
        q = float(component["q"])
        return np.exp(-(np.abs((value - center) / width) ** q))
    if family == "skew_feature_rbf":
        center = float(component["center"])
        width = float(component["width"])
        kappa = float(component["kappa"])
        z = (value - center) / width
        return np.exp(-(z * z)) * (1.0 + kappa * np.tanh(z))
    if family == "chebyshev":
        degree = int(component["degree"])
        return np.polynomial.chebyshev.chebval(value, [0.0] * degree + [1.0])
    if family == "legendre":
        degree = int(component["degree"])
        return np.polynomial.legendre.legval(value, [0.0] * degree + [1.0])
    if family == "fourier_sine":
        return np.sin(np.pi * float(component["frequency"]) * value)
    if family == "fourier_cosine":
        return np.cos(np.pi * float(component["frequency"]) * value)
    if family == "positive_hinge":
        return np.maximum(0.0, value - float(component["center"])) ** float(
            component["power"]
        )
    if family == "negative_hinge":
        return np.maximum(0.0, float(component["center"]) - value) ** float(
            component["power"]
        )
    if family == "tanh_transition":
        return np.tanh(
            float(component["scale"]) * (value - float(component["center"]))
        )
    orientation = float(component["orientation"])
    exponent = float(component["exponent"])
    scale = float(component["scale"])
    signed = orientation * value
    magnitude = np.abs(value) ** exponent
    if family == "sigmoid_transition":
        return 1.0 / (1.0 + np.exp(np.clip(scale * signed, -60.0, 60.0)))
    if family == "localized_exponential":
        return np.exp(-scale * magnitude)
    if family == "arctan_switch":
        return 0.5 + orientation * np.arctan(scale * value) / np.pi
    if family == "saturating_exponential":
        return 1.0 - np.exp(-scale * np.maximum(signed, 0.0) ** exponent)
    raise GravityG2EquivalenceError(f"unsupported component family: {family}")


def _span_projection(
    matrix: np.ndarray,
    probes: np.ndarray,
    *,
    rank_tolerance: float,
) -> tuple[int, np.ndarray]:
    if np.any(~np.isfinite(matrix)):
        raise GravityG2EquivalenceError("nonfinite adversarial component value")
    u, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
    threshold = rank_tolerance * max(1.0, float(singular[0]))
    rank = int(np.sum(singular > threshold))
    if rank < 1:
        raise GravityG2EquivalenceError("zero-rank survivor component span")
    basis = u[:, :rank]
    return rank, basis @ (basis.T @ probes)


def behavior_record(
    formula_ir: Mapping[str, Any],
    design: Mapping[str, Any],
    probes: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = formula_ir["feature_normalization"].startswith("within_galaxy")
    features = design["normalized" if normalized else "native"]
    matrix = np.column_stack(
        [evaluate_component(component, features) for component in formula_ir["components"]]
    )
    rank, projection = _span_projection(
        matrix,
        probes,
        rank_tolerance=float(config["behavioral_equivalence"]["rank_tolerance"]),
    )
    decimals = int(config["behavioral_equivalence"]["projection_round_decimals"])
    rendered = [format(float(value), f".{decimals}e") for value in projection.ravel()]
    signature = canonical_sha256(
        {
            "base": formula_ir["base"],
            "feature_normalization": formula_ir["feature_normalization"],
            "projection": rendered,
            "rank": rank,
        }
    )
    return {"matrix": matrix, "projection": projection, "rank": rank, "signature": signature}


def authoritative_status(
    formula_ir: Mapping[str, Any], config: Mapping[str, Any]
) -> str:
    mapping = config["authoritative_origin_status"]
    statuses = {mapping[str(component["family"])] for component in formula_ir["components"]}
    return "COMBINATION" if "COMBINATION" in statuses else "KNOWN_FAMILY"


def mutation_controls(
    formula_ir: Mapping[str, Any],
    behavior: Mapping[str, Any],
    design: Mapping[str, Any],
    probes: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the detector merges basis rewrites and separates nearby non-equivalents."""

    matrix = behavior["matrix"]
    reference = behavior["projection"]
    tolerance = float(config["behavioral_equivalence"]["rank_tolerance"])

    def error(other: np.ndarray) -> float:
        _rank, projected = _span_projection(other, probes, rank_tolerance=tolerance)
        return float(np.max(np.abs(projected - reference)))

    positive = {
        "component_swap": error(matrix[:, ::-1]),
        "unequal_nonzero_rescale": error(matrix * np.asarray([3.0, -0.25])[None, :]),
        "invertible_triangular_basis_change": error(
            matrix @ np.asarray([[1.0, 0.75], [0.0, 1.0]])
        ),
    }
    index = np.arange(matrix.shape[0], dtype=np.float64)
    mutation = np.sin(np.sqrt(3.0) * (index + 0.25))
    basis, _ = np.linalg.qr(matrix)
    mutation = mutation - basis @ (basis.T @ mutation)
    mutation /= np.linalg.norm(mutation)
    mutated = matrix.copy()
    mutated[:, 0] += 1e-3 * np.linalg.norm(matrix[:, 0]) * mutation
    component = dict(formula_ir["components"][0])
    feature_keys = [key for key in ("feature", "first_feature", "second_feature") if key in component]
    if not feature_keys:
        raise GravityG2EquivalenceError("mutation control found no feature identity")
    key = feature_keys[0]
    old_feature = str(component[key])
    component[key] = FEATURE_IDS[(FEATURE_IDS.index(old_feature) + 1) % len(FEATURE_IDS)]
    normalized = formula_ir["feature_normalization"].startswith("within_galaxy")
    changed_column = evaluate_component(
        component, design["normalized" if normalized else "native"]
    )
    changed_feature_matrix = matrix.copy()
    changed_feature_matrix[:, 0] = changed_column
    negative = {
        "orthogonal_1e_minus_3_mutation": error(mutated),
        "changed_feature_identity": error(changed_feature_matrix),
    }
    positive_limit = float(config["mutation_controls"]["positive_maximum_projection_error"])
    negative_limit = float(config["mutation_controls"]["negative_minimum_projection_error"])
    return {
        "negative": {name: format(value, ".12e") for name, value in negative.items()},
        "negative_controls_pass": all(value >= negative_limit for value in negative.values()),
        "positive": {name: format(value, ".12e") for name, value in positive.items()},
        "positive_controls_pass": all(value <= positive_limit for value in positive.values()),
    }


def build_receipt(root: Path, *, survivor_limit: int | None = None) -> dict[str, Any]:
    """Build semantic and behavioral classes for every retained G1 survivor."""

    root = root.resolve()
    config = load_config(root)
    survivors = collect_survivors(root, config)
    if survivor_limit is not None:
        survivors = survivors[:survivor_limit]
    structural_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structural_ir: dict[str, dict[str, Any]] = {}
    for row in survivors:
        ir = canonical_formula_ir(row["candidate"])
        signature = canonical_sha256(ir)
        structural_ir[signature] = ir
        structural_members[signature].append(row)
    design, probes = adversarial_design(
        int(config["behavioral_equivalence"]["adversarial_design_points"]),
        int(config["behavioral_equivalence"]["probe_columns"]),
    )
    behavior_by_structure: dict[str, dict[str, Any]] = {}
    behavioral_members: dict[str, list[str]] = defaultdict(list)
    for signature, ir in structural_ir.items():
        behavior = behavior_record(ir, design, probes, config)
        behavior_by_structure[signature] = behavior
        behavioral_members[behavior["signature"]].append(signature)
    maximum_within = 0.0
    for structure_ids in behavioral_members.values():
        reference = behavior_by_structure[structure_ids[0]]["projection"]
        for structure_id in structure_ids[1:]:
            error = float(
                np.max(np.abs(behavior_by_structure[structure_id]["projection"] - reference))
            )
            maximum_within = max(maximum_within, error)
            if error > float(
                config["behavioral_equivalence"]["maximum_projection_error_for_equivalence"]
            ):
                raise GravityG2EquivalenceError("behavioral hash collision exceeded tolerance")
    structural_classes = []
    for signature in sorted(structural_members):
        members = structural_members[signature]
        ir = structural_ir[signature]
        structural_classes.append(
            {
                "authoritative_origin_status": authoritative_status(ir, config),
                "behavioral_class_id": behavior_by_structure[signature]["signature"],
                "canonical_ir": ir,
                "class_id": signature,
                "member_candidate_ids": sorted(row["candidate_id"] for row in members),
                "member_count": len(members),
                "member_galaxies": sorted({row["galaxy"] for row in members}),
                "proposer_origin_assessments": dict(
                    sorted(Counter(row["proposer_origin_assessment"] for row in members).items())
                ),
            }
        )
    behavioral_classes = []
    for signature in sorted(behavioral_members):
        structure_ids = sorted(behavioral_members[signature])
        statuses = {
            authoritative_status(structural_ir[structure_id], config)
            for structure_id in structure_ids
        }
        behavioral_classes.append(
            {
                "authoritative_origin_status": (
                    "COMBINATION" if "COMBINATION" in statuses else "KNOWN_FAMILY"
                ),
                "class_id": signature,
                "member_count": sum(len(structural_members[row]) for row in structure_ids),
                "rank": behavior_by_structure[structure_ids[0]]["rank"],
                "structural_class_ids": structure_ids,
            }
        )
    first_rank_two = next(
        signature
        for signature in sorted(structural_ir)
        if behavior_by_structure[signature]["rank"] == 2
    )
    controls = mutation_controls(
        structural_ir[first_rank_two],
        behavior_by_structure[first_rank_two],
        design,
        probes,
        config,
    )
    full_run = survivor_limit is None
    passed = (
        full_run
        and len(survivors) == int(config["g1_binding"]["required_survivor_count"])
        and controls["positive_controls_pass"]
        and controls["negative_controls_pass"]
    )
    status_counts = Counter(row["authoritative_origin_status"] for row in structural_classes)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G2",
        "decision": "PASS_G2_EQUIVALENCE_COLLAPSE" if passed else "BLOCK_G2_EQUIVALENCE",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "g3_meta_law_authorized": passed,
            "historical_novelty_established": False,
            "structurally_unmatched_formula_found": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "behavioral_classes": len(behavioral_classes),
            "behavioral_collapses": len(structural_classes) - len(behavioral_classes),
            "confirmation_evaluator_accesses": 0,
            "input_survivors": len(survivors),
            "structural_classes": len(structural_classes),
            "structural_duplicates_collapsed": len(survivors) - len(structural_classes),
            "authoritative_statuses": dict(sorted(status_counts.items())),
        },
        "equivalence_validation": {
            "maximum_within_behavioral_class_projection_error": format(maximum_within, ".12e"),
            "mutation_controls": controls,
        },
        "structural_classes": structural_classes,
        "behavioral_classes": behavioral_classes,
        "limitations": [
            "Behavioral equivalence is bounded to the declared 257-point adversarial design and two-column coefficient-span semantics.",
            "No external historical prior-art corpus was searched; statuses are bounded to known grammar families and combinations.",
            "COMBINATION does not mean historically novel, and this receipt makes no novelty claim.",
            "G2 collapses local diagnostic formulas; it does not produce a galaxy-independent law.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Validate the sealed G2 result against its current inputs and implementation."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG2EquivalenceError("G2 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG2EquivalenceError("G2 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG2EquivalenceError("G2 config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG2EquivalenceError(f"G2 {key} binding changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG2EquivalenceError("G2 records confirmation access")
    if receipt.get("claims", {}).get("historical_novelty_established") is not False:
        raise GravityG2EquivalenceError("G2 overstates novelty")
    passed = receipt.get("decision") == "PASS_G2_EQUIVALENCE_COLLAPSE"
    if passed and (
        counts.get("input_survivors") != int(config["g1_binding"]["required_survivor_count"])
        or receipt.get("equivalence_validation", {})
        .get("mutation_controls", {})
        .get("positive_controls_pass")
        is not True
        or receipt.get("equivalence_validation", {})
        .get("mutation_controls", {})
        .get("negative_controls_pass")
        is not True
    ):
        raise GravityG2EquivalenceError("G2 PASS is unsupported")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG2EquivalenceError(f"refusing to overwrite immutable receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--survivor-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root, survivor_limit=args.survivor_limit)
    if args.survivor_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "behavioral_classes": receipt["counts"]["behavioral_classes"],
                "content_sha256": receipt["content_sha256"],
                "decision": receipt["decision"],
                "structural_classes": receipt["counts"]["structural_classes"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G2_EQUIVALENCE_COLLAPSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG2EquivalenceError",
    "adversarial_design",
    "authoritative_status",
    "behavior_record",
    "build_receipt",
    "canonical_formula_ir",
    "collect_survivors",
    "evaluate_component",
    "load_config",
    "mutation_controls",
    "structural_signature",
    "validate_receipt",
]
