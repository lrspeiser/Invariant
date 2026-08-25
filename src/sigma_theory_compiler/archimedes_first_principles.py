"""Target-blind Archimedes-law discovery and derivation control.

The control deliberately separates three questions:

1. Which dimensionally admissible multiplicative relation is stable in the data?
2. Does it survive nuisance interventions, noise, and a shifted holdout?
3. Is there a first-principles route from a separately discovered pressure law?

All numerical decisions use exact rational arithmetic and are replayed with SymPy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = "configs/archimedes_first_principles.json"
RECEIPT_PATH = "runs/math/archimedes-first-principles/receipt.json"
CONFIG_SCHEMA = "invariant-archimedes-first-principles-config-1.0"
RECEIPT_SCHEMA = "invariant-archimedes-first-principles-receipt-1.0"


class ArchimedesDiscoveryError(ValueError):
    """Raised when the control or its evidence fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _fraction(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ArchimedesDiscoveryError(f"{label} is not an exact rational")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ArchimedesDiscoveryError(f"{label} is not an exact rational") from error
    if result <= 0:
        raise ArchimedesDiscoveryError(f"{label} must be positive")
    return result


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ArchimedesDiscoveryError(f"{label} fields changed")


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(
        value,
        {
            "campaign_id",
            "derivation_policy",
            "hydrostatic_grid",
            "schema_version",
            "search_policy",
            "training_grid",
            "variables",
            "shifted_holdout_grid",
        },
        "Archimedes config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ArchimedesDiscoveryError("Archimedes config schema changed")
    if value["campaign_id"] != "archimedes-first-principles-control-2026-08-24-001":
        raise ArchimedesDiscoveryError("Archimedes campaign identity changed")

    search = value["search_policy"]
    _strict(
        search,
        {
            "force_target_exponent",
            "maximum_absolute_exponent",
            "maximum_l1_norm",
            "noise_multipliers",
            "relative_interval_radius",
        },
        "search policy",
    )
    if search["force_target_exponent"] != 1:
        raise ArchimedesDiscoveryError("target exponent must remain fixed at one")
    maximum_exponent = search["maximum_absolute_exponent"]
    maximum_l1 = search["maximum_l1_norm"]
    if (
        not isinstance(maximum_exponent, int)
        or not 1 <= maximum_exponent <= 3
        or not isinstance(maximum_l1, int)
        or not 4 <= maximum_l1 <= 10
    ):
        raise ArchimedesDiscoveryError("search bounds are invalid")
    noise = search["noise_multipliers"]
    if not isinstance(noise, list) or len(noise) < 3:
        raise ArchimedesDiscoveryError("noise schedule is missing")
    parsed_noise = [_fraction(item, "noise multiplier") for item in noise]
    if min(parsed_noise) >= 1 or max(parsed_noise) <= 1:
        raise ArchimedesDiscoveryError("noise schedule must straddle one")
    radius = _fraction(search["relative_interval_radius"], "interval radius")
    if radius >= 1:
        raise ArchimedesDiscoveryError("interval radius must be below one")

    variables = value["variables"]
    expected_names = [
        "buoyant_force",
        "fluid_density",
        "gravity",
        "displaced_volume",
        "depth",
        "ambient_pressure",
        "object_density",
        "shape_factor",
    ]
    if (
        not isinstance(variables, list)
        or [item.get("name") for item in variables] != expected_names
    ):
        raise ArchimedesDiscoveryError("Archimedes variables changed")
    for variable in variables:
        _strict(variable, {"dimensions", "name"}, "Archimedes variable")
        dimensions = variable["dimensions"]
        if (
            not isinstance(dimensions, list)
            or len(dimensions) != 3
            or any(not isinstance(item, int) for item in dimensions)
        ):
            raise ArchimedesDiscoveryError("variable dimensions changed")

    expected_grid_fields = {
        "ambient_pressure",
        "depth",
        "displaced_volume",
        "fluid_density",
        "gravity",
        "object_density",
        "shape_factor",
    }
    for key in ("training_grid", "shifted_holdout_grid"):
        grid = value[key]
        _strict(grid, expected_grid_fields, key)
        for name, entries in grid.items():
            if not isinstance(entries, list) or len(entries) < 2:
                raise ArchimedesDiscoveryError(f"{key}.{name} has insufficient support")
            parsed = [_fraction(item, f"{key}.{name}") for item in entries]
            if len(set(parsed)) != len(parsed):
                raise ArchimedesDiscoveryError(f"{key}.{name} contains duplicates")

    hydrostatic = value["hydrostatic_grid"]
    _strict(
        hydrostatic,
        {"ambient_pressure", "delta_height", "depth", "fluid_density", "gravity"},
        "hydrostatic grid",
    )
    for name, entries in hydrostatic.items():
        if not isinstance(entries, list) or len(entries) < 2:
            raise ArchimedesDiscoveryError(f"hydrostatic_grid.{name} has insufficient support")
        parsed = [_fraction(item, f"hydrostatic_grid.{name}") for item in entries]
        if len(set(parsed)) != len(parsed):
            raise ArchimedesDiscoveryError(f"hydrostatic_grid.{name} contains duplicates")

    derivation = value["derivation_policy"]
    _strict(
        derivation,
        {
            "require_ambient_pressure_cancellation",
            "require_closed_surface",
            "require_divergence_theorem",
            "require_hydrostatic_discovery",
        },
        "derivation policy",
    )
    if not all(item is True for item in derivation.values()):
        raise ArchimedesDiscoveryError("derivation requirements were weakened")
    return dict(value)


def load_config(root: Path) -> tuple[dict[str, Any], Path]:
    path = (root / CONFIG_PATH).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ArchimedesDiscoveryError("Archimedes config is unavailable")
    return validate_config(json.loads(path.read_text(encoding="utf-8"))), path


def _dimension_residual(
    exponents: Sequence[int], dimensions: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    return tuple(
        sum(exponent * vector[axis] for exponent, vector in zip(exponents, dimensions, strict=True))
        for axis in range(3)
    )


def _candidate_vectors(
    dimensions: Sequence[Sequence[int]], target_exponent: int, bound: int, maximum_l1: int
) -> list[tuple[int, ...]]:
    candidates = []
    for tail in itertools.product(range(-bound, bound + 1), repeat=len(dimensions) - 1):
        candidate = (target_exponent, *tail)
        if sum(abs(item) for item in candidate) > maximum_l1:
            continue
        if not any(_dimension_residual(candidate, dimensions)):
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (sum(abs(value) for value in item), item))


def _expression(names: Sequence[str], exponents: Sequence[int]) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for name, exponent in zip(names, exponents, strict=True):
        if exponent == 0:
            continue
        factor = name if abs(exponent) == 1 else f"{name}**{abs(exponent)}"
        (numerator if exponent > 0 else denominator).append(factor)
    numerator_text = "*".join(numerator) or "1"
    if not denominator:
        return numerator_text
    denominator_text = "*".join(denominator)
    return f"{numerator_text}/({denominator_text})"


def _evaluate_fraction(
    row: Mapping[str, Fraction], names: Sequence[str], exponents: Sequence[int]
) -> Fraction:
    result = Fraction(1)
    for name, exponent in zip(names, exponents, strict=True):
        result *= row[name] ** exponent
    return result


def _evaluate_sympy(
    row: Mapping[str, Fraction], names: Sequence[str], exponents: Sequence[int]
) -> Fraction:
    result = sp.Integer(1)
    for name, exponent in zip(names, exponents, strict=True):
        value = row[name]
        result *= sp.Rational(value.numerator, value.denominator) ** exponent
    result = sp.cancel(result)
    return Fraction(int(sp.numer(result)), int(sp.denom(result)))


def _relative_span(values: Sequence[Fraction]) -> Fraction:
    if not values or min(values) <= 0:
        raise ArchimedesDiscoveryError("candidate evaluation is empty or nonpositive")
    return (max(values) - min(values)) / min(values)


def _score_candidates(
    rows: Sequence[Mapping[str, Fraction]],
    names: Sequence[str],
    candidates: Sequence[Sequence[int]],
    evaluator: Any,
) -> list[dict[str, Any]]:
    scores = []
    for vector in candidates:
        values = [evaluator(row, names, vector) for row in rows]
        scores.append(
            {
                "exponents": list(vector),
                "expression": _expression(names, vector),
                "minimum": _fraction_text(min(values)),
                "maximum": _fraction_text(max(values)),
                "relative_span": _fraction_text(_relative_span(values)),
            }
        )
    return sorted(
        scores,
        key=lambda item: (
            Fraction(item["relative_span"]),
            sum(abs(v) for v in item["exponents"]),
            item["exponents"],
        ),
    )


def _grid_rows(
    grid: Mapping[str, Sequence[Any]], noise: Sequence[Fraction], *, noise_offset: int
) -> list[dict[str, Fraction]]:
    names = list(grid)
    values = [[_fraction(item, f"grid.{name}") for item in grid[name]] for name in names]
    rows = []
    for index, combination in enumerate(itertools.product(*values)):
        row = dict(zip(names, combination, strict=True))
        exact_force = row["fluid_density"] * row["gravity"] * row["displaced_volume"]
        row["buoyant_force"] = exact_force * noise[(index + noise_offset) % len(noise)]
        row["exact_buoyant_force"] = exact_force
        rows.append(row)
    return rows


def _hydrostatic_rows(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Fraction]]:
    names = list(grid)
    values = [[_fraction(item, f"hydrostatic.{name}") for item in grid[name]] for name in names]
    rows = []
    for combination in itertools.product(*values):
        row = dict(zip(names, combination, strict=True))
        pressure_top = (
            row["ambient_pressure"] + row["fluid_density"] * row["gravity"] * row["depth"]
        )
        delta_pressure = row["fluid_density"] * row["gravity"] * row["delta_height"]
        row["pressure_top"] = pressure_top
        row["pressure_bottom"] = pressure_top + delta_pressure
        row["delta_pressure"] = delta_pressure
        rows.append(row)
    return rows


def _dataset_summary(
    rows: Sequence[Mapping[str, Fraction]], public_names: Sequence[str]
) -> dict[str, Any]:
    encoded = [{name: _fraction_text(row[name]) for name in public_names} for row in rows]
    return {
        "row_count": len(rows),
        "content_sha256": _sha(encoded),
        "first_row": encoded[0],
        "last_row": encoded[-1],
    }


def _interval_intersection(
    rows: Sequence[Mapping[str, Fraction]],
    names: Sequence[str],
    exponents: Sequence[int],
    radius: Fraction,
) -> tuple[Fraction, Fraction] | None:
    lower: Fraction | None = None
    upper: Fraction | None = None
    for row in rows:
        center = _evaluate_fraction(row, names, exponents)
        row_lower = center * (1 - radius)
        row_upper = center * (1 + radius)
        lower = row_lower if lower is None else max(lower, row_lower)
        upper = row_upper if upper is None else min(upper, row_upper)
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def _discover(
    rows: Sequence[Mapping[str, Fraction]],
    names: Sequence[str],
    dimensions: Sequence[Sequence[int]],
    search: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _candidate_vectors(
        dimensions,
        search["force_target_exponent"],
        search["maximum_absolute_exponent"],
        search["maximum_l1_norm"],
    )
    primary = _score_candidates(rows, names, candidates, _evaluate_fraction)
    independent = _score_candidates(rows, names, candidates, _evaluate_sympy)
    if primary != independent:
        raise ArchimedesDiscoveryError("Fraction and SymPy candidate rankings disagree")
    return {
        "dimensionally_admissible_candidate_count": len(candidates),
        "winner": primary[0],
        "runner_up": primary[1],
        "top_five": primary[:5],
        "independent_exact_evaluators_agree": True,
    }


def _hydrostatic_discovery(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Fraction]]]:
    rows = _hydrostatic_rows(config["hydrostatic_grid"])
    names = [
        "delta_pressure",
        "fluid_density",
        "gravity",
        "delta_height",
        "depth",
        "ambient_pressure",
    ]
    dimensions = [
        [1, -1, -2],
        [1, -3, 0],
        [0, 1, -2],
        [0, 1, 0],
        [0, 1, 0],
        [1, -1, -2],
    ]
    search = dict(config["search_policy"])
    result = _discover(rows, names, dimensions, search)
    expected = [1, -1, -1, -1, 0, 0]
    if result["winner"]["exponents"] != expected or result["winner"]["relative_span"] != "0":
        raise ArchimedesDiscoveryError("hydrostatic pressure law was not recovered")
    result["dataset"] = _dataset_summary(rows, names + ["pressure_top", "pressure_bottom"])
    return result, rows


def _mutation_controls(
    rows: Sequence[Mapping[str, Fraction]],
    names: Sequence[str],
    dimensions: Sequence[Sequence[int]],
    winner_span: Fraction,
) -> list[dict[str, Any]]:
    mutations = {
        "drop_density": [1, 0, -1, -1, 0, 0, 0, 0],
        "drop_gravity": [1, -1, 0, -1, 0, 0, 0, 0],
        "drop_volume": [1, -1, -1, 0, 0, 0, 0, 0],
        "replace_volume_with_depth_cubed": [1, -1, -1, 0, -3, 0, 0, 0],
        "replace_fluid_with_object_density": [1, 0, -1, -1, 0, 0, -1, 0],
        "add_shape_dependence": [1, -1, -1, -1, 0, 0, 0, 1],
    }
    controls = []
    for control_id, vector in mutations.items():
        residual = _dimension_residual(vector, dimensions)
        values = [_evaluate_fraction(row, names, vector) for row in rows]
        span = _relative_span(values)
        rejected = any(residual) or span > winner_span
        if not rejected:
            raise ArchimedesDiscoveryError(f"mutation {control_id} was not rejected")
        controls.append(
            {
                "control_id": control_id,
                "dimension_residual": list(residual),
                "relative_span": _fraction_text(span),
                "status": "rejected",
            }
        )
    return controls


def _intervention_controls(names: Sequence[str], winner: Sequence[int]) -> list[dict[str, Any]]:
    base = {
        "fluid_density": Fraction(1),
        "gravity": Fraction(981, 100),
        "displaced_volume": Fraction(1, 500),
        "depth": Fraction(1, 2),
        "ambient_pressure": Fraction(101325),
        "object_density": Fraction(3, 4),
        "shape_factor": Fraction(1),
    }
    base["buoyant_force"] = base["fluid_density"] * base["gravity"] * base["displaced_volume"]
    changes = {
        "depth": Fraction(7, 5),
        "ambient_pressure": Fraction(104000),
        "object_density": Fraction(3, 2),
        "shape_factor": Fraction(7, 5),
    }
    baseline = _evaluate_fraction(base, names, winner)
    results = []
    for nuisance, value in changes.items():
        intervened = dict(base)
        intervened[nuisance] = value
        observed = _evaluate_fraction(intervened, names, winner)
        if observed != baseline:
            raise ArchimedesDiscoveryError(f"winner depends on nuisance {nuisance}")
        results.append(
            {
                "intervention": nuisance,
                "invariant_value": _fraction_text(observed),
                "status": "pass",
            }
        )
    return results


def _identifiability_control(
    rows: Sequence[Mapping[str, Fraction]],
    names: Sequence[str],
    dimensions: Sequence[Sequence[int]],
    search: Mapping[str, Any],
) -> dict[str, Any]:
    confounded = [row for row in rows if row["object_density"] == row["fluid_density"]]
    if not confounded:
        raise ArchimedesDiscoveryError("unidentifiable slice is empty")
    result = _discover(confounded, names, dimensions, search)
    best = Fraction(result["winner"]["relative_span"])
    tied = [item for item in result["top_five"] if Fraction(item["relative_span"]) == best]
    if len(tied) < 2:
        raise ArchimedesDiscoveryError("confounded dataset was falsely identified")
    return {
        "row_count": len(confounded),
        "best_score_tie_count_in_top_five": len(tied),
        "status": "blocked_unidentifiable",
        "reason": "fluid_density equals object_density in every retained row",
    }


def _derivation_certificate(
    hydrostatic: Mapping[str, Any], buoyancy: Mapping[str, Any]
) -> dict[str, Any]:
    if hydrostatic["winner"]["exponents"] != [1, -1, -1, -1, 0, 0]:
        raise ArchimedesDiscoveryError("derivation lacks the hydrostatic premise")
    if buoyancy["winner"]["exponents"] != [1, -1, -1, -1, 0, 0, 0, 0]:
        raise ArchimedesDiscoveryError("derivation lacks the buoyancy relation")

    area, height, z0, density, gravity, ambient = sp.symbols("A h z0 rho g p0", positive=True)
    normal_area_sum = sp.simplify(area - area)
    z_normal_area_sum = sp.simplify(area * (z0 + height) - area * z0)
    volume = area * height
    pressure_resultant = sp.simplify(
        ambient * normal_area_sum + density * gravity * z_normal_area_sum
    )
    if normal_area_sum != 0 or sp.simplify(z_normal_area_sum - volume) != 0:
        raise ArchimedesDiscoveryError("closed-surface geometry control failed")
    if sp.simplify(pressure_resultant - density * gravity * volume) != 0:
        raise ArchimedesDiscoveryError("symbolic Archimedes derivation failed")

    return {
        "proof_plan": "hydrostatic_gradient_then_closed_surface_divergence",
        "steps": [
            {
                "step": 1,
                "claim": "delta_pressure = fluid_density * gravity * delta_height",
                "basis": "separately discovered exact sensor relation",
            },
            {
                "step": 2,
                "claim": "F_z = -surface_integral(pressure * normal_z, dA)",
                "basis": "pressure force definition",
            },
            {
                "step": 3,
                "claim": "surface_integral(normal_z, dA) = 0",
                "basis": "closed-surface cancellation; removes ambient pressure and depth offset",
            },
            {
                "step": 4,
                "claim": "surface_integral(z * normal_z, dA) = volume",
                "basis": "divergence theorem applied to the vector field z * e_z",
            },
            {
                "step": 5,
                "claim": "buoyant_force = fluid_density * gravity * displaced_volume",
                "basis": "substitution and exact symbolic simplification",
            },
        ],
        "rectangular_prism_symbolic_control": {
            "closed_surface_normal_sum": str(normal_area_sum),
            "z_normal_surface_sum": str(z_normal_area_sum),
            "volume": str(volume),
            "pressure_resultant": str(pressure_resultant),
            "status": "pass",
        },
        "claim_boundary": (
            "bounded synthetic first-principles rediscovery control; not a historical novelty claim"
        ),
        "status": "pass",
    }


def build_receipt(root: Path) -> dict[str, Any]:
    config, config_path = load_config(root)
    search = config["search_policy"]
    noise = [_fraction(item, "noise multiplier") for item in search["noise_multipliers"]]
    training = _grid_rows(config["training_grid"], noise, noise_offset=0)
    holdout = _grid_rows(config["shifted_holdout_grid"], noise, noise_offset=2)
    hydrostatic, _ = _hydrostatic_discovery(config)

    names = [item["name"] for item in config["variables"]]
    dimensions = [item["dimensions"] for item in config["variables"]]
    discovery = _discover(training, names, dimensions, search)
    winner = discovery["winner"]["exponents"]
    expected = [1, -1, -1, -1, 0, 0, 0, 0]
    if winner != expected:
        raise ArchimedesDiscoveryError("Archimedes relation was not recovered")
    holdout_values = [_evaluate_fraction(row, names, winner) for row in holdout]
    holdout_span = _relative_span(holdout_values)
    if holdout_span != Fraction(discovery["winner"]["relative_span"]):
        raise ArchimedesDiscoveryError("shifted holdout behavior changed")
    interval = _interval_intersection(
        training + holdout,
        names,
        winner,
        _fraction(search["relative_interval_radius"], "interval radius"),
    )
    if interval is None or not interval[0] <= 1 <= interval[1]:
        raise ArchimedesDiscoveryError(
            "noisy intervals exclude the exact first-principles constant"
        )

    discovery.update(
        {
            "training_dataset": _dataset_summary(training, names),
            "shifted_holdout_dataset": _dataset_summary(holdout, names),
            "shifted_holdout_relative_span": _fraction_text(holdout_span),
            "combined_noise_interval_intersection": [
                _fraction_text(interval[0]),
                _fraction_text(interval[1]),
            ],
            "exact_constant_in_interval": True,
            "nuisance_interventions": _intervention_controls(names, winner),
            "mutation_controls": _mutation_controls(
                training,
                names,
                dimensions,
                Fraction(discovery["winner"]["relative_span"]),
            ),
            "identifiability_control": _identifiability_control(
                training, names, dimensions, search
            ),
        }
    )
    derivation = _derivation_certificate(hydrostatic, discovery)
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "normalized_sha256": _file_sha(config_path)},
            "module": {
                "path": "src/sigma_theory_compiler/archimedes_first_principles.py",
                "normalized_sha256": _file_sha(Path(__file__)),
            },
        },
        "hydrostatic_discovery": hydrostatic,
        "buoyancy_discovery": discovery,
        "derivation": derivation,
        "result": {
            "discovered_formula": "buoyant_force = fluid_density * gravity * displaced_volume",
            "training_rows": len(training),
            "shifted_holdout_rows": len(holdout),
            "hydrostatic_sensor_rows": hydrostatic["dataset"]["row_count"],
            "status": "PASS_BOUNDED_FIRST_PRINCIPLES_DISCOVERY_CONTROL",
        },
    }
    return {**payload, "content_sha256": _sha(payload)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = build_receipt(root)
    if receipt != expected:
        raise ArchimedesDiscoveryError("Archimedes receipt does not replay exactly")
    return {
        "status": receipt["result"]["status"],
        "content_sha256": receipt["content_sha256"],
        "discovered_formula": receipt["result"]["discovered_formula"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, default=Path("."))
        subparser.add_argument("--receipt", type=Path, default=Path(RECEIPT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    if args.command == "run":
        receipt = build_receipt(root)
        _write_json(receipt_path, receipt)
        print(json.dumps(validate_receipt(receipt, root), sort_keys=True))
        return 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps(validate_receipt(receipt, root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
