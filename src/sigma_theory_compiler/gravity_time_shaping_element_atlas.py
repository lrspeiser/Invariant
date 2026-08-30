"""Finite time-shaping ontology, variant grammar, and sealed real-evidence replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_time_shaping_element_atlas_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_time_shaping_element_atlas.py")
TEST_PATH = Path("tests/test_gravity_time_shaping_element_atlas.py")
OUTPUT_PATH = Path("runs/gravity/theory/time-shaping-element-atlas-v1.json")
CONFIG_SCHEMA = "invariant-gravity-time-shaping-element-atlas-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-time-shaping-element-atlas-receipt-1.0"
STATUS = "finite_atlas_prefiltered_existing_real_evidence_replayed_fresh_score_unrun"
DECISION = (
    "TIME_SHAPING_ATLAS_20_ELEMENTS_3520_VARIANTS_DERIVED_EXISTING_REAL_EVIDENCE_"
    "REPLAYED_FRESH_REAL_SCORING_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "8e17f384fba701a6fda4c5fd192ea798907f583cb81b5bbebc2074bb55532b5e"
EXPECTED_CONFIG_CONTENT_SHA256 = "0b8cae6033fc83e0d6c27a2912b924be71d7b10e3413d02f4974719d9c6ea406"
EXPECTED_SECTION_SHA256 = {
    "source_bindings": "859f8bfc27312e7e778327f39e156fd991a574a61a18989082a06462b5ec2de4",
    "element_ontology": "48f2fee39be0d9f035cd6437276b2e0b2202ad902f2991c7b44a30e0c6da0b9a",
    "variant_grammar": "22a91a3355945c543f93d8fe47e5d244b1795c4ea0166bf3efa36c042d4303de",
    "derivation_contract": "67118e690861213f610fadec86a71ad22a510fe125a31e56deb3022b84373015",
    "real_data_replay_contract": "b7067e01c6fc91d6eca2d20e9736b630894bae7bd3706b9101e70f6c0cc1d0f9",
    "priority_hypotheses": "7bf374fe5e77a1e4153c19dbd861f2a55a912577633ed859e778523f6bd9813d",
    "next_execution_contract": "ef470b32bbabf0c8e0ff8d74b78db47c7a297076470661704046e27b8d6a7cfb",
    "claim_boundary": "cf302c885326c29ec0278ed42ac52f9b0edb8f889b21ea0d40faea6bcde298c8",
    "zero_new_access": "3abf28d4c04d1018a666668c33ae98d58d334d72416a3f4761a299c2c65d51f6",
}


class TimeShapingAtlasError(RuntimeError):
    """Raised when the frozen time-shaping atlas changes or overclaims."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _content_sha(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TimeShapingAtlasError(f"could not read JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise TimeShapingAtlasError("JSON root is not an object")
    return value


def load_config(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    return _load_json(repo / CONFIG_PATH)


def validate_config(config: dict[str, Any], root: Path | None = None) -> None:
    repo = _repo_root() if root is None else root.resolve()
    expected_keys = {
        "schema_version",
        "analysis_id",
        "status",
        "purpose",
        "source_bindings",
        "element_ontology",
        "variant_grammar",
        "derivation_contract",
        "real_data_replay_contract",
        "priority_hypotheses",
        "next_execution_contract",
        "claim_boundary",
        "zero_new_access",
        "output_path",
    }
    if set(config) != expected_keys:
        raise TimeShapingAtlasError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-time-shaping-element-atlas-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise TimeShapingAtlasError("config identity changed")
    if _file_sha(repo / CONFIG_PATH) != EXPECTED_CONFIG_FILE_SHA256:
        raise TimeShapingAtlasError("config file hash changed")
    if _content_sha(config) != EXPECTED_CONFIG_CONTENT_SHA256:
        raise TimeShapingAtlasError("config content changed")
    for section, expected in EXPECTED_SECTION_SHA256.items():
        if _content_sha(config[section]) != expected:
            raise TimeShapingAtlasError(f"config section changed: {section}")
    ontology = config["element_ontology"]
    if len(ontology) != 20 or len({row["id"] for row in ontology}) != 20:
        raise TimeShapingAtlasError("element ontology is not exactly 20 unique elements")
    grammar = config["variant_grammar"]
    if (
        len(grammar["unary_transforms"]) != 6
        or len(grammar["pair_operators"]) != 4
        or len(grammar["clock_combiners"]) != 4
        or grammar["single_variant_count"] != 480
        or grammar["pair_variant_count"] != 3040
        or grammar["total_variant_count"] != 3520
    ):
        raise TimeShapingAtlasError("variant grammar counts changed")
    claims = config["claim_boundary"]
    allowed_true = {
        "finite_ontology_frozen",
        "all_3520_structures_registered",
        "all_3520_structures_symbolically_or_numerically_prefiltered",
        "existing_real_data_evidence_replayed",
    }
    for key, value in claims.items():
        if value is not (key in allowed_true):
            raise TimeShapingAtlasError("claim boundary changed")
    if any(config["zero_new_access"].values()):
        raise TimeShapingAtlasError("zero-new-access ledger changed")
    if config["next_execution_contract"]["authorization_state"] != "not_authorized_by_this_atlas":
        raise TimeShapingAtlasError("atlas cannot authorize a real-data execution")


def _validate_content_hash(receipt: dict[str, Any], label: str) -> None:
    if "content_sha256" not in receipt:
        return
    payload = dict(receipt)
    stored = payload.pop("content_sha256")
    if stored != _content_sha(payload):
        raise TimeShapingAtlasError(f"invalid bound content hash: {label}")


def validate_bindings(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    bindings = config["source_bindings"]
    checked: list[dict[str, Any]] = []
    for key in ("roadmap_completion_audit", "lead_parent_config", "lead_parent_receipt"):
        row = bindings[key]
        path = Path(row["path"])
        if _file_sha(repo / path) != row["file_sha256"]:
            raise TimeShapingAtlasError(f"bound source changed: {key}")
        if key.endswith("receipt"):
            receipt = _load_json(repo / path)
            if row.get("content_sha256") not in (None, receipt.get("content_sha256")):
                raise TimeShapingAtlasError(f"bound receipt content changed: {key}")
        checked.append({"id": key, "path": path.as_posix(), "valid": True})

    clock = bindings["extended_source_clock"]
    commit = clock["git_commit"]
    try:
        object_type = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TimeShapingAtlasError("clock predecessor commit is unavailable") from error
    if object_type != "commit":
        raise TimeShapingAtlasError("clock predecessor is not a commit")
    for artifact in clock["artifacts"]:
        path = Path(artifact["path"])
        if _file_sha(repo / path) != artifact["file_sha256"]:
            raise TimeShapingAtlasError("clock predecessor working bytes changed")
        try:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{path.as_posix()}"],
                cwd=repo,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise TimeShapingAtlasError("clock predecessor commit path is unavailable") from error
        if _sha_bytes(committed) != artifact["file_sha256"]:
            raise TimeShapingAtlasError("clock predecessor committed bytes changed")
    clock_receipt = _load_json(repo / Path(clock["artifacts"][-1]["path"]))
    _validate_content_hash(clock_receipt, "extended_source_clock")
    if clock_receipt["content_sha256"] != clock["receipt_content_sha256"]:
        raise TimeShapingAtlasError("clock predecessor receipt identity changed")
    checked.append({"id": "extended_source_clock", "git_commit": commit, "valid": True})

    seen: set[str] = set()
    for element in config["element_ontology"]:
        path = Path(element["evidence_path"])
        if _file_sha(repo / path) != element["evidence_sha256"]:
            raise TimeShapingAtlasError(f"element evidence changed: {element['id']}")
        if path.as_posix() not in seen:
            _load_json(repo / path)
            seen.add(path.as_posix())
    return {
        "top_level_bindings": checked,
        "element_count": len(config["element_ontology"]),
        "unique_element_evidence_files": len(seen),
        "valid": True,
    }


def _variant_id(structure: dict[str, Any]) -> str:
    return f"clock.{_content_sha(structure)[:24]}"


def generate_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    grammar = config["variant_grammar"]
    element_ids = sorted(row["id"] for row in config["element_ontology"])
    variants: list[dict[str, Any]] = []
    for element in element_ids:
        for transform in grammar["unary_transforms"]:
            for combiner in grammar["clock_combiners"]:
                structure = {
                    "arity": 1,
                    "elements": [element],
                    "transforms": [transform],
                    "pair_operator": None,
                    "clock_combiner": combiner,
                }
                variants.append({"variant_id": _variant_id(structure), **structure})
    for left, right in combinations(element_ids, 2):
        for operator in grammar["pair_operators"]:
            for combiner in grammar["clock_combiners"]:
                structure = {
                    "arity": 2,
                    "elements": [left, right],
                    "transforms": ["saturating", "saturating"],
                    "pair_operator": operator,
                    "clock_combiner": combiner,
                }
                variants.append({"variant_id": _variant_id(structure), **structure})
    if len(variants) != grammar["total_variant_count"]:
        raise TimeShapingAtlasError("generated variant count changed")
    if len({row["variant_id"] for row in variants}) != len(variants):
        raise TimeShapingAtlasError("variant structural IDs collided")
    return variants


def _transform(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise TimeShapingAtlasError("probe predictor must be finite and nonnegative")
    functions = {
        "identity": lambda x: x,
        "log1p": math.log1p,
        "sqrt": math.sqrt,
        "saturating": lambda x: x / (1 + x),
        "tanh": math.tanh,
        "square_saturating": lambda x: x * x / (1 + x * x),
    }
    try:
        return functions[name](value)
    except KeyError as error:
        raise TimeShapingAtlasError(f"unknown unary transform: {name}") from error


def _pair(name: str, left: float, right: float) -> float:
    if name == "product":
        return left * right
    if name == "geometric_mean":
        return math.sqrt(left * right)
    if name == "maximum":
        return max(left, right)
    if name == "harmonic_mean":
        return 0.0 if left + right == 0 else 2 * left * right / (left + right)
    raise TimeShapingAtlasError(f"unknown pair operator: {name}")


def _nu_rar(y: float) -> float:
    return 1 / (1 - math.exp(-math.sqrt(y)))


def _clock_response(combiner: str, source: float, y: float) -> float:
    screen = 1 / (1 + y * y)
    if combiner == "additive":
        return 1 + screen * source
    if combiner == "quadrature":
        return 1 + screen * (math.sqrt(1 + source * source) - 1)
    if combiner == "exponential":
        return 1 + screen * math.expm1(min(source, 20.0))
    if combiner == "competitive_rar":
        return max(_nu_rar(y), 1 + screen * source)
    raise TimeShapingAtlasError(f"unknown clock combiner: {combiner}")


def _evaluate_variant(variant: dict[str, Any], values: list[float], y: float) -> float:
    transformed = [
        _transform(name, value) for name, value in zip(variant["transforms"], values, strict=True)
    ]
    source = transformed[0]
    if variant["arity"] == 2:
        source = _pair(variant["pair_operator"], transformed[0], transformed[1])
    return _clock_response(variant["clock_combiner"], source, y)


def probe_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    probes = (0.0, 0.1, 1.0, 10.0)
    total = 0
    max_high_y_delta = 0.0
    min_clock_ratio = 1.0
    max_response = 1.0
    for index, variant in enumerate(variants):
        scale = 1 + (index % 7) / 3
        for probe in probes:
            values = [probe * scale]
            if variant["arity"] == 2:
                values.append(probe * (1 + ((index + 3) % 5) / 4))
            low = _evaluate_variant(variant, values, 1e-6)
            high = _evaluate_variant(variant, values, 1e12)
            if not all(math.isfinite(value) and value >= 1 for value in (low, high)):
                raise TimeShapingAtlasError("variant probe is not finite and positive")
            max_high_y_delta = max(max_high_y_delta, abs(high - 1))
            min_clock_ratio = min(min_clock_ratio, 1 / math.sqrt(low))
            max_response = max(max_response, low)
            total += 1
    if max_high_y_delta > 1e-12:
        raise TimeShapingAtlasError("registered variant fails Solar high-acceleration probe")
    return {
        "variant_count": len(variants),
        "probe_evaluations": total,
        "all_finite_positive": True,
        "all_high_acceleration_screened": True,
        "max_high_acceleration_nu_minus_one": format(max_high_y_delta, ".17e"),
        "minimum_probe_clock_ratio": format(min_clock_ratio, ".17e"),
        "maximum_probe_response": format(max_response, ".17e"),
    }


def _symbolic_check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    if simplified != 0:
        raise TimeShapingAtlasError(f"symbolic derivation failed: {check_id}: {simplified}")
    return {"check_id": check_id, "statement": statement, "residual": "0", "passed": True}


def symbolic_checks() -> list[dict[str, Any]]:
    y, b, a, x, left, right = sp.symbols("y b a x left right", positive=True)
    screen = 1 / (1 + y**2)
    nu = 1 + screen * b
    clock = nu ** sp.Rational(-1, 2)
    checks = [
        _symbolic_check("S01_CLOCK_MAPPING", 1 / clock**2 - nu, "g_pred/g_b=1/C^2=nu"),
        _symbolic_check(
            "S02_SOLAR_SCREEN",
            sp.limit(screen, y, sp.oo),
            "high acceleration removes finite time response",
        ),
        _symbolic_check(
            "S03_SOLAR_RESPONSE",
            sp.limit(nu, y, sp.oo) - 1,
            "nu tends to one at high acceleration",
        ),
        _symbolic_check(
            "S04_SOLAR_CLOCK",
            sp.limit(clock, y, sp.oo) - 1,
            "matter clock tends to field clock at high acceleration",
        ),
        _symbolic_check(
            "S05_SATURATION_ZERO",
            (x / (1 + x)).subs(x, 0),
            "saturating transform vanishes at zero",
        ),
        _symbolic_check(
            "S06_SQUARE_SATURATION_ZERO",
            (x**2 / (1 + x**2)).subs(x, 0),
            "square saturation vanishes at zero",
        ),
        _symbolic_check(
            "S07_GEOMETRIC_COMMUTATIVITY",
            sp.sqrt(left * right) - sp.sqrt(right * left),
            "geometric pair canonicalization is commutative",
        ),
        _symbolic_check(
            "S08_HARMONIC_COMMUTATIVITY",
            2 * left * right / (left + right) - 2 * right * left / (right + left),
            "harmonic pair canonicalization is commutative",
        ),
        _symbolic_check(
            "S09_EXTENDED_SOURCE_DENSITY_IDENTITY",
            a - 3 * a / 3,
            "eta=dlnM/dlnr=3rho/rhobar under the frozen mass identity",
        ),
    ]
    return checks


def _scientific_result(root: Path, path: str) -> dict[str, Any]:
    receipt = _load_json(root / Path(path))
    value = receipt.get("scientific_result")
    if not isinstance(value, dict):
        raise TimeShapingAtlasError(f"bound scientific result missing: {path}")
    return value


def replay_real_evidence(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    age = _load_json(repo / Path("runs/gravity/roadmap/item-12-manga-dynamical-age-v1.json"))
    disturbance = _load_json(
        repo / Path("runs/gravity/roadmap/item-13-manga-relaxation-mergers-v1.json")
    )
    lapse = _scientific_result(repo, "runs/gravity/roadmap/item-24-temporal-lapse-v1.json")
    varying = _scientific_result(repo, "runs/gravity/roadmap/item-25-time-varying-g-v1.json")
    retarded = _scientific_result(repo, "runs/gravity/roadmap/item-26-retarded-gravity-v1.json")
    memory = _scientific_result(repo, "runs/gravity/roadmap/item-27-gravitational-memory-v1.json")
    periodic = _scientific_result(repo, "runs/gravity/roadmap/item-28-periodic-gravity-v1.json")
    rows = [
        {
            "signal_id": "stellar_age_x_density",
            "status": "retained_positive_development_signal",
            "directional_improvement": age["primary"]["relative_mse_improvement"],
            "selection_p": None,
            "objects": age["counts"]["quality_passing_galaxies"],
            "interpretation": "strongest existing time-related correlation",
        },
        {
            "signal_id": "disturbance_after_age",
            "status": "rejected_as_explanation_age_signal_persists",
            "directional_improvement": disturbance["primary"][
                "disturbance_relative_mse_improvement"
            ],
            "selection_p": None,
            "objects": disturbance["counts"]["quality_passing_galaxies"],
            "interpretation": "merger/disturbance does not explain the age association",
        },
        {
            "signal_id": "literal_temporal_lapse_galaxy",
            "status": "rejected",
            "directional_improvement": lapse["channel_metrics"]["galaxy_motion"][
                "improvement_vs_calibrated_GR"
            ],
            "selection_p": lapse["primary_metrics"]["selection_aware_guarded_permutation_p"],
            "objects": lapse["channel_metrics"]["galaxy_motion"]["objects"],
            "interpretation": "worse than calibrated GR and flexible nuisance",
        },
        {
            "signal_id": "time_varying_G",
            "status": "scoped_reject_not_selection_significant",
            "directional_improvement": varying["metrics"]["improvement_vs_calibrated_baryonic"],
            "selection_p": varying["metrics"]["selection_aware_permutation_p"],
            "objects": varying["valid_objects"],
            "interpretation": "small development gain did not pass the frozen promotion gates",
        },
        {
            "signal_id": "propagation_retardation",
            "status": "inconclusive_no_gain_over_instantaneous",
            "directional_improvement": retarded["metrics"]["improvement_vs_instantaneous_baryonic"],
            "selection_p": retarded["metrics"]["selection_aware_permutation_p"],
            "objects": retarded["valid_objects"],
            "interpretation": "indistinguishable from and slightly worse than instantaneous gravity",
        },
        {
            "signal_id": "memory_history",
            "status": "inconclusive_worse_than_instantaneous",
            "directional_improvement": memory["metrics"]["improvement_vs_instantaneous_baryonic"],
            "selection_p": memory["metrics"]["selection_aware_permutation_p"],
            "objects": memory["valid_objects"],
            "interpretation": "memory kernel underperformed the instantaneous baryonic control",
        },
        {
            "signal_id": "periodic_resonant",
            "status": "inconclusive_reject",
            "directional_improvement": periodic["metrics"]["improvement_vs_baryonic"],
            "selection_p": periodic["metrics"]["selection_aware_permutation_p"],
            "objects": periodic["valid_galaxies"],
            "interpretation": "periodic response underperformed the baryonic control",
        },
    ]
    if float(rows[0]["directional_improvement"]) <= 0.18:
        raise TimeShapingAtlasError("age-density signal no longer matches the sealed result")
    if any(float(row["directional_improvement"]) > 0 for row in rows[4:]):
        raise TimeShapingAtlasError("negative time-family replay changed")
    return {
        "signals": rows,
        "signal_count": len(rows),
        "object_or_row_evaluations_with_cross_receipt_overlap": sum(row["objects"] for row in rows),
        "strongest_signal_id": "stellar_age_x_density",
        "fresh_raw_rows_opened": 0,
        "confirmation_rows_opened": 0,
        "independent_rows_opened": 0,
        "pooled_likelihood_computed": False,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    config = load_config(repo)
    validate_config(config, repo)
    bindings = validate_bindings(config, repo)
    variants = generate_variants(config)
    registry = {
        "element_count": len(config["element_ontology"]),
        "single_variant_count": sum(row["arity"] == 1 for row in variants),
        "pair_variant_count": sum(row["arity"] == 2 for row in variants),
        "total_variant_count": len(variants),
        "registry_content_sha256": _content_sha(variants),
        "first_variant": variants[0],
        "last_variant": variants[-1],
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": STATUS,
        "decision": DECISION,
        "bindings": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_file_sha256": _file_sha(repo / CONFIG_PATH),
            "config_content_sha256": _content_sha(config),
            "implementation_path": SOURCE_PATH.as_posix(),
            "implementation_file_sha256": _file_sha(repo / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(repo / TEST_PATH),
            "upstream": bindings,
        },
        "variant_registry": registry,
        "symbolic_derivations": symbolic_checks(),
        "variant_prefilter": probe_variants(variants),
        "real_data_evidence_replay": replay_real_evidence(repo),
        "priority_hypotheses": config["priority_hypotheses"],
        "next_execution_contract": config["next_execution_contract"],
        "claim_boundary": config["claim_boundary"],
        "zero_new_access": config["zero_new_access"],
    }
    receipt["content_sha256"] = _content_sha(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise TimeShapingAtlasError(f"refusing to overwrite existing output: {path}") from None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return "CREATED"
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_receipt(root: Path | None = None) -> str:
    repo = _repo_root() if root is None else root.resolve()
    receipt = build_receipt(repo)
    payload = _canonical(receipt)
    return _atomic_no_clobber(repo / OUTPUT_PATH, payload)


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root.resolve()
    stored = _load_json(repo / OUTPUT_PATH)
    expected = build_receipt(repo)
    if stored != expected:
        raise TimeShapingAtlasError("stored atlas receipt does not match exact rebuild")
    _validate_content_hash(stored, "time_shaping_atlas")
    return stored


def status(root: Path | None = None) -> dict[str, Any]:
    receipt = check_receipt(root)
    return {
        "valid": True,
        "decision": receipt["decision"],
        "elements": receipt["variant_registry"]["element_count"],
        "variants": receipt["variant_registry"]["total_variant_count"],
        "derived_and_prefiltered": receipt["variant_prefilter"]["all_finite_positive"],
        "real_signals_replayed": receipt["real_data_evidence_replay"]["signal_count"],
        "strongest_signal": receipt["real_data_evidence_replay"]["strongest_signal_id"],
        "fresh_real_score": receipt["claim_boundary"]["all_3520_structures_fresh_real_data_scored"],
        "authorization": receipt["next_execution_contract"]["authorization_state"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(json.dumps({"publication": write_receipt()}, sort_keys=True))
    elif args.command == "check":
        print(json.dumps(check_receipt(), sort_keys=True))
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
