"""Independent read-only audit of the TWELL-400 source-shaped replay v1.

This auditor deliberately does not import the subject module.  It reconstructs
the admitted card set, applies the separately frozen static radial adapter to
the stored source projections, and compares every execution and replay row.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_static_radial_adapter_v1 as adapter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v1"
CARDS = ROOT / "runs/gravity/twell-400-v2-typed-compiler-final-v3/cards.jsonl"

STATIC = frozenset(adapter.STATIC_ARCHITECTURES)
TEMPORAL = frozenset(adapter.TIME_SOURCE_BLOCKS)
DRIVERS = frozenset(adapter.XCOP_DRIVERS)
COMPOUNDS = frozenset(adapter.COMPOUND_IDS)
D13 = "D13_GASF"

EXPECTED_RAW = {
    "configs/open_gravity_twell_400_source_shaped_rebind_replay_v1.json": "12df390acedd7afb650b101b714758b5bd8e1ccf2f5f36da7a1779040533688a",
    "src/sigma_theory_compiler/open_gravity_twell_400_source_shaped_rebind_replay_v1.py": "babbeb727660623bca3816a3ad58d87c319774657ed09f18f00e89c76b2e9e18",
    "tests/test_open_gravity_twell_400_source_shaped_rebind_replay_v1.py": "fdf5ea8ff098ccc3dec5731396501f9374149ae54cb4ba83d484d863a42646cb",
    "runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v1/receipt.json": "bcaf6df58e39a9dc52329e8afd78546c905cba20271858bb68e0e3b4e19bdab4",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    assert array.dtype.name in {"float64", "int64"}
    assert np.all(np.isfinite(array))
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_sealed(rows: list[dict[str, Any]], field: str) -> None:
    for row in rows:
        expected = row[field]
        actual = content_sha({key: value for key, value in row.items() if key != field})
        assert actual == expected, (field, expected, actual)


def classify(card: dict[str, Any]) -> str:
    architecture = card["architecture_id"]
    drivers = set(card["driver_ids"])
    if architecture in TEMPORAL:
        return "TEMPORAL_ARCHITECTURE"
    if card["entry_kind"] == "ATOMIC" and architecture in STATIC and drivers <= DRIVERS:
        return "PROVISIONAL_STATIC"
    if card["concept_id"] in COMPOUNDS:
        return "PROVISIONAL_STATIC"
    return "MISSING_DRIVER_OR_COMPOUND_ADAPTER"


def source_bundle(archive: Any, object_id: str) -> dict[str, Any]:
    prefix = f"source_projection__{object_id}__"
    physical = {
        driver: np.array(archive[prefix + "physical__" + driver.lower()], copy=True)
        for driver in sorted(DRIVERS - {D13})
    }
    normalized = {
        driver: np.array(archive[prefix + "normalized__" + driver.lower()], copy=True)
        for driver in sorted(DRIVERS - {D13})
    }
    return {
        "xi": np.array(archive[prefix + "xi"], copy=True),
        "physical": physical,
        "normalized": normalized,
    }


def card_driver(card: dict[str, Any], source: dict[str, Any]) -> np.ndarray:
    drivers = list(card["driver_ids"])
    if card["entry_kind"] == "ATOMIC":
        assert len(drivers) == 1
        return np.asarray(source["normalized"][drivers[0]], dtype=np.float64)
    return adapter.combine_compound_drivers(
        "XCOP_SPHERICAL",
        card["concept_id"],
        {name: source["normalized"][name] for name in drivers},
    )


def independent_execution(
    card: dict[str, Any], source: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    xi = np.asarray(source["xi"], dtype=np.float64)
    g_b = np.asarray(source["physical"]["D01_ACC"], dtype=np.float64)
    driver = card_driver(card, source)
    primary = adapter.apply_static_architecture(
        card["architecture_id"], driver, g_b, xi, cell["value"]
    )
    xi2 = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    convergence = adapter.apply_static_architecture(
        card["architecture_id"],
        np.interp(xi2, xi, driver),
        np.interp(xi2, xi, g_b),
        xi2,
        cell["value"],
    )
    max_abs = float(
        np.max(np.abs(np.asarray(primary["factor"])[::2] - np.asarray(convergence["factor"])))
    )
    operator = max(
        float(primary["diagnostics"]["operator_residual_max_abs"]),
        float(convergence["diagnostics"]["operator_residual_max_abs"]),
    )
    boundary = max(
        float(primary["diagnostics"]["boundary_residual_max_abs"]),
        float(convergence["diagnostics"]["boundary_residual_max_abs"]),
    )
    valid = bool(
        primary["diagnostics"]["finite"]
        and convergence["diagnostics"]["finite"]
        and primary["diagnostics"]["positive_factor"]
        and convergence["diagnostics"]["positive_factor"]
        and operator <= adapter.OPERATOR_RESIDUAL_TOLERANCE
        and boundary <= adapter.BOUNDARY_RESIDUAL_TOLERANCE
        and max_abs <= adapter.CONVERGENCE_MAX_ABS_TOLERANCE
    )
    factor = np.asarray(primary["factor"], dtype=np.float64)
    effective = np.asarray(primary["g_eff_m_s2"], dtype=np.float64)
    factor_digest = array_sha(factor)
    effective_digest = array_sha(effective)
    diagnostics = {
        "finite": primary["diagnostics"]["finite"] and convergence["diagnostics"]["finite"],
        "positive_factor": primary["diagnostics"]["positive_factor"]
        and convergence["diagnostics"]["positive_factor"],
        "operator_residual_max_abs_hex": operator.hex(),
        "boundary_residual_max_abs_hex": boundary.hex(),
        "primary_vs_convergence_max_abs_hex": max_abs.hex(),
        "primary_vs_convergence_tolerance_hex": float(
            adapter.CONVERGENCE_MAX_ABS_TOLERANCE
        ).hex(),
    }
    health = {
        "kind": "SOURCE_ONLY_NUMERICAL_HEALTH_NOT_SCIENTIFIC_SCORE",
        "factor_min_hex": float(np.min(factor)).hex(),
        "factor_max_hex": float(np.max(factor)).hex(),
        "g_eff_rms_m_s2_hex": float(np.sqrt(np.mean(effective * effective))).hex(),
        "diagnostics": diagnostics,
        "scientific_score": False,
    }
    return {
        "status": "COMPLETED" if valid else "NUMERICAL_INVALID",
        "numerical_valid": valid,
        "factor": factor,
        "g_eff": effective,
        "factor_sha256": factor_digest,
        "g_eff_sha256": effective_digest,
        "prediction_sha256": content_sha(
            {"factor": factor_digest, "g_eff_m_s2": effective_digest}
        ),
        "metric_sha256": content_sha(health),
        "convergence": max_abs,
    }


def main() -> None:
    for relative, expected in EXPECTED_RAW.items():
        assert file_sha(ROOT / relative) == expected

    receipt = json.loads((OUT / "receipt.json").read_text(encoding="utf-8"))
    receipt_without_self = dict(receipt)
    receipt_without_self.pop("content_sha256")
    assert content_sha(receipt_without_self) == receipt["content_sha256"]

    cards = load_jsonl(CARDS)
    assert len(cards) == 400
    assert [card["order_index"] for card in cards] == list(range(400))
    assert len({card["concept_id"] for card in cards}) == 400
    assert sum(card["parameter_cell_count"] for card in cards) == 1184
    assert all(content_sha(card["card"]) == card["card_sha256"] for card in cards)

    card_counts = Counter(classify(card) for card in cards)
    cell_counts = Counter()
    for card in cards:
        cell_counts[classify(card)] += card["parameter_cell_count"]
    assert card_counts == {
        "PROVISIONAL_STATIC": 126,
        "TEMPORAL_ARCHITECTURE": 84,
        "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 190,
    }
    assert cell_counts == {
        "PROVISIONAL_STATIC": 370,
        "TEMPORAL_ARCHITECTURE": 253,
        "MISSING_DRIVER_OR_COMPOUND_ADAPTER": 561,
    }

    compatibility = load_jsonl(OUT / "compatibility-ledger.jsonl")
    dispositions = load_jsonl(OUT / "parameter-cell-disposition-ledger.jsonl")
    bindings = load_jsonl(OUT / "execution-bindings.jsonl")
    source_rows = load_jsonl(OUT / "source-projections.jsonl")
    unique = load_jsonl(OUT / "unique-executions.jsonl")
    replay = load_jsonl(OUT / "replay-ledger.jsonl")
    verify_sealed(compatibility, "compatibility_sha256")
    verify_sealed(dispositions, "disposition_sha256")
    verify_sealed(bindings, "adapter_binding_sha256")
    verify_sealed(source_rows, "source_projection_sha256")
    verify_sealed(unique, "result_sha256")
    verify_sealed(replay, "replay_entry_sha256")
    assert Counter(row["status"] for row in compatibility) == {
        "EXECUTABLE": 110,
        "SOURCE_BLOCKED": 290,
        "INCOMPATIBLE_FEATURE_SET": 1600,
    }
    assert Counter(row["status"] for row in dispositions) == {
        "COMPLETED_WITH_SCENARIO_LEVEL_NUMERICAL_VALIDITY_RETAINED": 324,
        "SOURCE_BLOCKED": 860,
    }
    assert Counter(row["status"] for row in bindings) == {
        "EXECUTABLE": 110,
        "SOURCE_BLOCKED": 16,
    }

    card_by_id = {card["concept_id"]: card for card in cards}
    cell_by_id = {
        cell["cell_id"]: cell
        for card in cards
        for cell in card["card"]["parameter_cells"]
    }
    unique_by_key = {
        (row["object_id"], row["formula_id"], row["cell_id"]): row for row in unique
    }
    assert len(unique_by_key) == len(unique) == 2592
    assert len(replay) == 62208

    mismatches: list[tuple[str, str, str, str]] = []
    invalid: list[tuple[str, str, str]] = []
    convergence_max = 0.0
    with np.load(OUT / "source-projections.npz", allow_pickle=False) as sources, np.load(
        OUT / "predictions.npz", allow_pickle=False
    ) as predictions:
        assert all(name.startswith("source_projection__") for name in sources.files)
        assert len(predictions.files) == 5184
        for row in unique:
            key = (row["object_id"], row["formula_id"], row["cell_id"])
            card = card_by_id[row["formula_id"]]
            cell = cell_by_id[row["cell_id"]]
            result = independent_execution(card, source_bundle(sources, row["object_id"]), cell)
            convergence_max = max(convergence_max, result["convergence"])
            checks = {
                "status": result["status"] == row["status"],
                "valid": result["numerical_valid"] == row["numerical_valid"],
                "factor_hash": result["factor_sha256"] == row["factor_sha256"],
                "g_hash": result["g_eff_sha256"] == row["g_eff_sha256"],
                "prediction_hash": result["prediction_sha256"] == row["prediction_sha256"],
                "metric_hash": result["metric_sha256"] == row["metric_sha256"],
                "factor_array": np.array_equal(
                    result["factor"], predictions[row["factor_value_key"]]
                ),
                "g_array": np.array_equal(result["g_eff"], predictions[row["g_eff_value_key"]]),
            }
            for label, passed in checks.items():
                if not passed:
                    mismatches.append((*key, label))
            if not result["numerical_valid"]:
                invalid.append(key)

    assert not mismatches, mismatches[:10]
    assert Counter(row["status"] for row in unique) == {
        "COMPLETED": 2554,
        "NUMERICAL_INVALID": 38,
    }
    assert Counter(row["status"] for row in replay) == {
        "COMPLETED": 61296,
        "NUMERICAL_INVALID": 912,
    }
    assert len(invalid) == 38
    assert Counter(formula_id for _, formula_id, _ in invalid) == {
        "TW2-A11-D03": 16,
        "TW2-A11-D06": 16,
        "TW2-A11-D04": 5,
        "TW2-A11-D07": 1,
    }

    scenario_ids_by_object: dict[str, set[str]] = defaultdict(set)
    for row in replay:
        scenario_ids_by_object[row["object_id"]].add(row["scenario_id"])
        result = unique_by_key[(row["object_id"], row["formula_id"], row["cell_id"])]
        assert row["result_sha256"] == result["result_sha256"]
        assert row["prediction_sha256"] == result["prediction_sha256"]
        assert row["metric_sha256"] == result["metric_sha256"]
        assert row["response_value_accessed"] is False
        assert row["scientific_score"] is False
    assert all(len(scenarios) == 24 for scenarios in scenario_ids_by_object.values())

    prediction_groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in unique:
        prediction_groups[(row["object_id"], row["prediction_sha256"])].append(
            (row["formula_id"], row["cell_id"])
        )
    tie_groups = sum(len(members) > 1 for members in prediction_groups.values())
    assert tie_groups == 122

    report = {
        "status": "BLOCK_MECHANICAL_FORMAT_GATE_ONLY",
        "subject": "open-gravity-twell-400-source-shaped-rebind-replay-v1",
        "scientific_replay": "PASS_SOURCE_ONLY_SYNTHETIC",
        "raw_hashes": EXPECTED_RAW,
        "card_counts": dict(card_counts),
        "cell_counts": dict(cell_counts),
        "compatibility_counts": dict(Counter(row["status"] for row in compatibility)),
        "unique_execution_counts": dict(Counter(row["status"] for row in unique)),
        "replay_counts": dict(Counter(row["status"] for row in replay)),
        "independent_execution_count": len(unique),
        "independent_array_comparisons": 2 * len(unique),
        "mismatches": len(mismatches),
        "invalid_formula_counts": dict(Counter(formula_id for _, formula_id, _ in invalid)),
        "finite_source_prediction_tie_group_count": tie_groups,
        "maximum_observed_convergence_difference_hex": convergence_max.hex(),
        "response_values_opened": 0,
        "scientific_scores_computed": 0,
        "blockers": [
            "ruff format --check reports that the frozen module and test would be reformatted; receipt limitation says Ruff passed without disclosing this format failure"
        ],
    }
    report["content_sha256"] = content_sha(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
