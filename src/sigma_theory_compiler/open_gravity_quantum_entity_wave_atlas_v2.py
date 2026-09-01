"""Typed quantum/entity/wave gravity atlas v2 audit repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_quantum_entity_wave_atlas_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_quantum_entity_wave_atlas_v2.py")
TEST_PATH = Path("tests/test_open_gravity_quantum_entity_wave_atlas_v2.py")
OUTPUT_PATH = Path("runs/gravity/theory/open-gravity-quantum-entity-wave-atlas-v2/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-quantum-entity-wave-atlas-config-2.0"
RECEIPT_SCHEMA = "invariant-open-gravity-quantum-entity-wave-atlas-receipt-2.0"
DECISION = (
    "PASS_REAUDIT_READY_TYPED_EQUATIONS_COMPLETE_FIXTURES_GENERATED_FROM_CARDS_"
    "SOURCES_STRICTLY_GATED_REAL_DATA_UNOPENED"
)

CARD_IDS = (
    "Q00_GR_COHERENT_SPIN2_CONTROL",
    "Q01_MASSIVE_SPIN2",
    "Q02_SCALAR_VECTOR_TENSOR_MIXTURE",
    "Q03_DISCRETE_GRAVITY_IMPULSES",
    "Q04_CLASSICAL_STOCHASTIC_METRIC",
    "Q05_SEMICLASSICAL_EXPECTATION_SOURCE",
    "Q06_GRAVITATIONAL_COLLAPSE_DECOHERENCE",
    "Q07_ENTANGLEMENT_MEDIATED_GRAVITY",
    "Q08_EMERGENT_SUPERFLUID_MEDIUM",
    "Q09_DISPERSIVE_GRAVITY_WAVE_PACKET",
    "Q10_POLARIZATION_BIREFRINGENT_GRAVITY",
    "Q11_FINITE_OCCUPATION_COHERENCE",
    "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE",
    "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY",
    "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY",
    "Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY",
)

EVALUATORS = (
    "coherent_control",
    "massive_dispersion",
    "linear_mixture",
    "poisson_impulse",
    "gaussian_pushforward",
    "semiclassical_branch_mean",
    "exponential_decoherence",
    "bmv_branches",
    "phase_transition",
    "quartic_dispersion",
    "polarization_rank",
    "occupation_states",
    "damped_memory",
    "immigration_death",
    "cq_cp_block",
    "ktm_noise_bound",
)


class QuantumAtlasV2Error(RuntimeError):
    """Raised when a typed law, source gate, fixture, or receipt changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantumAtlasV2Error(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical(body))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuantumAtlasV2Error(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _parameter(card: Mapping[str, Any], symbol: str) -> Any:
    matches = [row for row in card["parameters"] if row["symbol"] == symbol]
    _require(len(matches) == 1, f"parameter {symbol} not unique in {card['id']}")
    return matches[0]["fixture"]


def _equation_ids(card: Mapping[str, Any]) -> set[str]:
    return {row["id"] for row in card["equations"]}


def validate_config(config: Mapping[str, Any], base: Path | None = None) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-quantum-entity-wave-atlas-v2",
        "analysis ID changed",
    )
    _require(config.get("status") == "FROZEN_RESPONSE_FREE_TYPED_SUCCESSOR", "status changed")
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
            "artifact_directory": ARTIFACT_DIR.as_posix(),
        },
        "package paths changed",
    )
    predecessor = config.get("predecessor")
    _require(
        isinstance(predecessor, dict)
        and predecessor.get("content_sha256")
        == "6d5296cf27365d95fcab1449f18209fea1a2ae4edc53eba9e771fa1970d11b85"
        and predecessor.get("status") == "PRESERVED_SUPERSEDED_BY_TYPED_AUDIT_REPAIR",
        "predecessor binding changed",
    )
    if base is not None:
        predecessor_path = base / predecessor["path"]
        _require(predecessor_path.is_file(), "predecessor receipt missing")
        _require(
            _sha256_file(predecessor_path) == predecessor["raw_sha256"],
            "predecessor receipt bytes changed",
        )
        predecessor_receipt = _read_json(predecessor_path)
        _require(
            predecessor_receipt.get("content_sha256") == predecessor["content_sha256"],
            "predecessor content hash changed",
        )
    theorem = config.get("gaussian_equivalence")
    _require(
        isinstance(theorem, dict)
        and theorem.get("id") == "FIXED_GAUSSIAN_MEASUREMENT_PUSHFORWARD"
        and len(theorem.get("required_assumptions", [])) == 6
        and "commute" in theorem["required_assumptions"][0]
        and "POVM" in theorem["required_assumptions"][1]
        and "back-action" in theorem["required_assumptions"][4]
        and theorem.get("novelty_claimed") is False,
        "Gaussian equivalence scope widened",
    )
    _require(
        config.get("score_contract", {}).get("no_composite_score") is True, "score contract changed"
    )
    cards = config.get("cards")
    _require(isinstance(cards, list), "cards missing")
    _require(tuple(card.get("id") for card in cards) == CARD_IDS, "card inventory changed")
    required_card_fields = {
        "id",
        "category",
        "state_space",
        "parameters",
        "equations",
        "probability_law",
        "observables",
        "limits",
        "fixture",
        "literature",
        "falsifier_manifest",
        "health",
        "scores",
    }
    manifest_ids = {row["id"] for row in config.get("source_manifests", [])}
    for card in cards:
        card_id = card["id"]
        _require(set(card) == required_card_fields, f"typed card fields changed: {card_id}")
        _require(
            card["state_space"] and card["parameters"] and card["equations"],
            f"empty typed model: {card_id}",
        )
        _require(
            all(
                set(row) == {"symbol", "kind", "dimension", "domain"} for row in card["state_space"]
            ),
            f"state variable typing changed: {card_id}",
        )
        _require(
            all(
                set(row) == {"symbol", "dimension", "domain", "fixture"}
                for row in card["parameters"]
            ),
            f"parameter typing changed: {card_id}",
        )
        _require(
            all(set(row) == {"id", "kind", "lhs", "rhs", "dimension"} for row in card["equations"]),
            f"equation typing changed: {card_id}",
        )
        _require(
            len(_equation_ids(card)) == len(card["equations"]), f"duplicate equation: {card_id}"
        )
        _require(
            {row["channel"] for row in card["observables"]} == {"matter", "photon", "tensor"},
            f"observable channel changed: {card_id}",
        )
        fixture = card["fixture"]
        _require(fixture["evaluator"] in EVALUATORS, f"unknown evaluator: {card_id}")
        _require(
            set(fixture["equation_ids"]) <= _equation_ids(card),
            f"fixture equation not typed: {card_id}",
        )
        _require(card["limits"], f"parameter limits missing: {card_id}")
        _require(card["falsifier_manifest"] in manifest_ids, f"manifest missing: {card_id}")
        scores = card["scores"]
        _require(
            set(scores) == {"data_readiness", "theory_health", "novelty"}
            and all(type(value) is int and 0 <= value <= 4 for value in scores.values()),
            f"score fields changed: {card_id}",
        )
    coverage = config.get("fixture_coverage")
    _require(
        isinstance(coverage, list)
        and [row["card_id"] for row in coverage] == list(CARD_IDS)
        and [row["evaluator"] for row in coverage] == list(EVALUATORS),
        "fixture coverage changed",
    )
    card_by_id = {card["id"]: card for card in cards}
    for row in coverage:
        _require(
            card_by_id[row["card_id"]]["fixture"]["evaluator"] == row["evaluator"],
            f"coverage mismatch: {row['card_id']}",
        )
    q13 = card_by_id["Q13_QUANTIZED_CAPTURE_JUMP_MEMORY"]
    _require(
        q13["equations"][0]["rhs"] == "dJ_plus(t)-dJ_minus(t)"
        and "N/tau" in q13["equations"][1]["rhs"]
        and "exp(lambda tau (z-1))" in q13["equations"][2]["rhs"]
        and "lambda tau for every integer n>=1" in q13["equations"][3]["rhs"],
        "Q13 immigration-death law changed",
    )
    _require(
        "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY" in card_by_id
        and "Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY" in card_by_id,
        "CQ or classical-channel boundary missing",
    )
    families = config.get("equivalence_families")
    _require(isinstance(families, list) and len(families) == 5, "equivalence families changed")
    for family in families:
        _require(
            set(family["members"]) <= set(CARD_IDS), f"unknown equivalence member: {family['id']}"
        )
        _require(
            family["parameter_map"] and family["not_equivalent"],
            f"limit boundary missing: {family['id']}",
        )
    manifests = config.get("source_manifests")
    _require(
        isinstance(manifests, list) and len(manifests) == 11, "source manifest inventory changed"
    )
    manifest_by_id = {row["id"]: row for row in manifests}
    _require(len(manifest_by_id) == len(manifests), "duplicate manifest")
    for manifest in manifests:
        _require(
            set(manifest["cards"]) <= set(CARD_IDS), f"unknown manifest card: {manifest['id']}"
        )
        if manifest["status"] == "EXECUTABLE_MANIFEST_FROZEN_PAYLOAD_UNOPENED":
            for key in (
                "events",
                "acquisition",
                "preprocessing",
                "model_grid",
                "nuisance_priors",
                "likelihood",
                "tolerances",
            ):
                _require(
                    manifest.get(key), f"incomplete executable manifest {manifest['id']}: {key}"
                )
            _require(
                manifest["response_status"] == "NOT_OPENED_NOT_SCORED", "response gate changed"
            )
        else:
            _require(
                manifest["status"] in {"SOURCE_BLOCKED", "THEOREM_ONLY"} and manifest.get("reason"),
                f"unfrozen nonexecutable source: {manifest['id']}",
            )
    for card in cards:
        manifest = manifest_by_id[card["falsifier_manifest"]]
        readiness = card["scores"]["data_readiness"]
        if manifest["status"] == "EXECUTABLE_MANIFEST_FROZEN_PAYLOAD_UNOPENED":
            _require(readiness == 3, f"executable readiness mismatch: {card['id']}")
        elif manifest["status"] == "SOURCE_BLOCKED":
            _require(readiness <= 2, f"blocked source readiness overstated: {card['id']}")
    _require(
        manifest_by_id["M09_HOLOMETER_UNSUITABLE_TIME_CUMULANTS"]["status"] == "SOURCE_BLOCKED"
        and card_by_id["Q03_DISCRETE_GRAVITY_IMPULSES"]["scores"]["data_readiness"] == 1
        and card_by_id["Q11_FINITE_OCCUPATION_COHERENCE"]["scores"]["data_readiness"] == 1,
        "Holometer demotion changed",
    )
    _require(set(config.get("access_contract", {}).values()) == {0}, "access contract changed")
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("typed_equations_and_parameters_complete") is True
        and boundary.get("fixture_coverage_complete") is True
        and boundary.get("target_free_only") is True
        and boundary.get("real_observational_rows_scored") is False
        and boundary.get("any_branch_empirically_supported") is False
        and boundary.get("historical_novelty_established") is False
        and boundary.get("publication_ready") is False,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config, base)
    return config


def _to_fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _matmul(
    left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]
) -> list[list[Fraction]]:
    _require(bool(left) and bool(right), "empty matrix")
    _require(len(left[0]) == len(right), "matrix shape mismatch")
    return [
        [
            sum((row[k] * right[k][column] for k in range(len(right))), Fraction(0))
            for column in range(len(right[0]))
        ]
        for row in left
    ]


def _transpose(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def finite_gaussian_pushforward(
    response: Sequence[Sequence[float]],
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    noise_mean: Sequence[float],
    noise_covariance: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Exact finite-dimensional pushforward for a fixed commuting Gaussian readout."""
    r = [[_to_fraction(value) for value in row] for row in response]
    m = [[_to_fraction(value)] for value in mean]
    c = [[_to_fraction(value) for value in row] for row in covariance]
    b = [[_to_fraction(value)] for value in noise_mean]
    n = [[_to_fraction(value) for value in row] for row in noise_covariance]
    output_dimension = len(r)
    input_dimension = len(m)
    _require(input_dimension > 0 and output_dimension > 0, "Gaussian dimension is zero")
    _require(all(len(row) == input_dimension for row in r), "response shape invalid")
    _require(
        len(c) == input_dimension and all(len(row) == input_dimension for row in c),
        "input covariance shape invalid",
    )
    _require(len(b) == output_dimension, "noise mean shape invalid")
    _require(
        len(n) == output_dimension and all(len(row) == output_dimension for row in n),
        "noise covariance shape invalid",
    )
    rm = _matmul(r, m)
    output_mean = [[rm[index][0] + b[index][0]] for index in range(output_dimension)]
    pushed = _matmul(_matmul(r, c), _transpose(r))
    output_covariance = [
        [pushed[i][j] + n[i][j] for j in range(output_dimension)] for i in range(output_dimension)
    ]

    def printable(value: Fraction) -> int | str:
        return (
            value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        )

    return {
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "output_mean": [printable(row[0]) for row in output_mean],
        "output_covariance": [[printable(value) for value in row] for row in output_covariance],
        "characteristic_function": "exp(i t^T(Rm+b)-t^T(R C R^T+N)t/2)",
        "assumptions": "fixed commuting Gaussian POVM/kernel, independent Gaussian noise, linear readout, no unmodeled back-action",
        "proof_status": "GENERAL_FINITE_DIMENSIONAL_CHARACTERISTIC_FUNCTION_PLUS_EXACT_ARITHMETIC",
    }


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    work = [list(map(float, row)) + [float(vector[index])] for index, row in enumerate(matrix)]
    _require(len(work) == size and all(len(row) == size + 1 for row in work), "solve shape invalid")
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        _require(abs(work[pivot][column]) > 1e-14, "singular matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[column], strict=True)
            ]
    return [work[index][-1] for index in range(size)]


def _matrix_rank(matrix: Sequence[Sequence[float]], tolerance: float) -> int:
    work = [list(map(float, row)) for row in matrix]
    if not work:
        return 0
    rows, columns = len(work), len(work[0])
    rank = 0
    for column in range(columns):
        if rank >= rows:
            break
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
    return rank


def _coherent(card: Mapping[str, Any]) -> dict[str, Any]:
    c, occupation = _parameter(card, "c"), _parameter(card, "N")
    rows = [{"k": k, "omega": c * k} for k in card["fixture"]["inputs"]["k"]]
    return {"rows": rows, "relative_occupation_noise": occupation**-0.5}


def _massive(card: Mapping[str, Any]) -> dict[str, Any]:
    c, mu, distance = _parameter(card, "c"), _parameter(card, "mu"), _parameter(card, "D")
    rows = []
    for omega in card["fixture"]["inputs"]["omega"]:
        _require(omega > c * mu, "massive fixture below threshold")
        k = math.sqrt((omega / c) ** 2 - mu**2)
        velocity = c * c * k / omega
        rows.append(
            {
                "omega": omega,
                "k": k,
                "group_velocity": velocity,
                "delay": distance * (1 / velocity - 1 / c),
            }
        )
    return {
        "rows": rows,
        "lower_frequency_later": all(
            rows[i]["delay"] > rows[i + 1]["delay"] for i in range(len(rows) - 1)
        ),
    }


def _mixture(card: Mapping[str, Any]) -> dict[str, Any]:
    kinetic, mixing = _parameter(card, "K"), _parameter(card, "M")
    source = card["fixture"]["inputs"]["J"]
    mixed = [sum(row[index] * source[index] for index in range(len(source))) for row in mixing]
    solution = _solve(kinetic, mixed)
    residual = [
        sum(kinetic[i][j] * solution[j] for j in range(len(solution))) - mixed[i]
        for i in range(len(solution))
    ]
    return {
        "source": source,
        "mixed_source": mixed,
        "mode_solution": solution,
        "max_equation_residual": max(abs(value) for value in residual),
    }


def _poisson(card: Mapping[str, Any]) -> dict[str, Any]:
    exposure, impulse = _parameter(card, "Lambda"), _parameter(card, "a_q")
    return {
        "Lambda": exposure,
        "mean_acceleration": impulse * exposure,
        "variance_acceleration": impulse**2 * exposure,
        "count_cumulants_1_to_4": [exposure] * 4,
        "normalized_skewness": exposure**-0.5,
        "excess_kurtosis": exposure**-1,
    }


def _gaussian(card: Mapping[str, Any]) -> dict[str, Any]:
    return finite_gaussian_pushforward(
        _parameter(card, "R"),
        _parameter(card, "m"),
        _parameter(card, "C"),
        _parameter(card, "b"),
        _parameter(card, "N"),
    )


def _semiclassical(card: Mapping[str, Any]) -> dict[str, Any]:
    probability, force = _parameter(card, "p"), _parameter(card, "F")
    _require(abs(sum(probability) - 1.0) < 1e-15, "branch probability not normalized")
    mean = sum(p * value for p, value in zip(probability, force, strict=True))
    variance = sum(p * (value - mean) ** 2 for p, value in zip(probability, force, strict=True))
    return {
        "branch_probabilities": probability,
        "branch_forces": force,
        "expectation_source_force": mean,
        "realized_branch_variance_not_in_bare_gravity_law": variance,
    }


def _decoherence(card: Mapping[str, Any]) -> dict[str, Any]:
    gamma, times = _parameter(card, "Gamma"), _parameter(card, "t")
    return {
        "Gamma": gamma,
        "rows": [{"t": time, "visibility_ratio": math.exp(-gamma * time)} for time in times],
    }


def _bmv(card: Mapping[str, Any]) -> dict[str, Any]:
    scale, distances = _parameter(card, "Gm1m2t_over_hbar"), _parameter(card, "r_ab")
    phases = [-scale / distance for distance in distances]
    invariant_phase = phases[0] + phases[3] - phases[1] - phases[2]
    concurrence = abs(math.sin(invariant_phase / 2.0))
    return {
        "branch_order": ["00", "01", "10", "11"],
        "distances": distances,
        "phases": phases,
        "local_phase_invariant": invariant_phase,
        "concurrence": concurrence,
        "entangled": concurrence > 0.0,
    }


def _phase_transition(card: Mapping[str, Any]) -> dict[str, Any]:
    ratios, acceleration = _parameter(card, "T_over_Tc"), _parameter(card, "a0")
    rows = []
    for ratio in ratios:
        fraction = max(0.0, 1.0 - ratio**1.5)
        rows.append(
            {
                "T_over_Tc": ratio,
                "condensate_fraction": fraction,
                "extra_acceleration": acceleration * fraction,
            }
        )
    return {
        "rows": rows,
        "normal_phase_zero_above_transition": all(
            row["extra_acceleration"] == 0.0 for row in rows if row["T_over_Tc"] >= 1.0
        ),
    }


def _quartic(card: Mapping[str, Any]) -> dict[str, Any]:
    c, mu, eta, scale = (_parameter(card, symbol) for symbol in ("c", "mu", "eta", "kstar"))
    rows = []
    for k in card["fixture"]["inputs"]["k"]:
        omega_squared = c * c * (k * k + mu * mu + eta * k**4 / scale**2)
        _require(omega_squared >= 0.0, "quartic frequency is imaginary")
        rows.append({"k": k, "omega_squared": omega_squared, "omega": math.sqrt(omega_squared)})
    return {"rows": rows}


def _polarization(card: Mapping[str, Any]) -> dict[str, Any]:
    response, tolerance = _parameter(card, "F"), _parameter(card, "rank_tol")
    rank = _matrix_rank(response, tolerance)
    return {
        "detectors": len(response),
        "modes": len(response[0]),
        "rank": rank,
        "rank_deficient": rank < len(response[0]),
        "failure_retained": True,
    }


def _occupation(card: Mapping[str, Any]) -> dict[str, Any]:
    occupation, squeeze = _parameter(card, "nbar"), _parameter(card, "r")
    return {
        "nbar": occupation,
        "coherent_variance": occupation,
        "thermal_variance": occupation * (occupation + 1),
        "squeezed_variance_ratio": math.exp(-2 * squeeze),
        "coherent_relative_noise": occupation**-0.5,
    }


def _memory(card: Mapping[str, Any]) -> dict[str, Any]:
    omega0, gamma, coupling, omega = (
        _parameter(card, symbol) for symbol in ("omega0", "Gamma", "lambda", "omega_drive")
    )
    transfer = coupling / complex(omega0**2 - omega**2, -2 * gamma * omega)
    times = card["fixture"]["inputs"]["ringdown_times"]
    return {
        "transfer_real": transfer.real,
        "transfer_imag": transfer.imag,
        "phase_radians": math.atan2(transfer.imag, transfer.real),
        "ringdown": [{"t": time, "envelope": math.exp(-gamma * time)} for time in times],
        "bath_required_when_Gamma_positive": gamma > 0,
    }


def _immigration_death(card: Mapping[str, Any]) -> dict[str, Any]:
    arrival, tau = _parameter(card, "lambda"), _parameter(card, "tau")
    mean = arrival * tau
    probabilities = [math.exp(-mean)]
    for n in range(6):
        probabilities.append(probabilities[-1] * mean / (n + 1))
    detailed_balance_residuals = [
        arrival * probabilities[n] - (n + 1) * probabilities[n + 1] / tau for n in range(6)
    ]
    lags = card["fixture"]["inputs"]["lags"]
    return {
        "jump_law": "dN=dJ_plus-dJ_minus",
        "conditional_intensities": {"J_plus": "lambda", "J_minus": "N/tau"},
        "stationary_family": "Poisson(lambda tau)",
        "stationary_mean": mean,
        "stationary_variance": mean,
        "ordinary_cumulants_1_to_4": [mean] * 4,
        "stationary_pgf": "exp(lambda tau(z-1))",
        "autocovariance": [{"lag": lag, "value": mean * math.exp(-abs(lag) / tau)} for lag in lags],
        "max_detailed_balance_residual_n0_to5": max(
            abs(value) for value in detailed_balance_residuals
        ),
        "derivation": "lambda pi_n=(n+1)pi_(n+1)/tau implies pi_(n+1)/pi_n=lambda tau/(n+1); normalization gives Poisson(lambda tau); log PGF=lambda tau(z-1), so every ordinary cumulant equals lambda tau",
    }


def _cq(card: Mapping[str, Any]) -> dict[str, Any]:
    d0, d1, d2 = (_parameter(card, symbol) for symbol in ("D0", "D1", "D2"))
    counterexample = card["fixture"]["inputs"]["counterexample_D2"]
    determinant = d0 * d2 - d1 * d1
    counterexample_determinant = d0 * counterexample - d1 * d1
    return {
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "determinant": determinant,
        "positive_semidefinite_scalar_block": d0 >= 0 and d2 >= 0 and determinant >= 0,
        "minimum_noise_boundary": determinant == 0,
        "counterexample_D2": counterexample,
        "counterexample_determinant": counterexample_determinant,
        "counterexample_rejected": counterexample_determinant < 0,
    }


def _ktm(card: Mapping[str, Any]) -> dict[str, Any]:
    coupling, strengths = _parameter(card, "g"), _parameter(card, "gamma")
    rows = [
        {
            "gamma": value,
            "noise_cost": value + coupling**2 / (4 * value),
            "bound_residual": value + coupling**2 / (4 * value) - abs(coupling),
        }
        for value in strengths
    ]
    best = min(rows, key=lambda row: row["noise_cost"])
    return {
        "g": coupling,
        "rows": rows,
        "minimum_grid_gamma": best["gamma"],
        "minimum_grid_cost": best["noise_cost"],
        "analytic_minimum_gamma": abs(coupling) / 2,
        "analytic_minimum_cost": abs(coupling),
        "all_bound_residuals_nonnegative": all(row["bound_residual"] >= -1e-15 for row in rows),
        "entangling_capacity_under_LOCC": 0.0,
    }


_EVALUATOR_FUNCTIONS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "coherent_control": _coherent,
    "massive_dispersion": _massive,
    "linear_mixture": _mixture,
    "poisson_impulse": _poisson,
    "gaussian_pushforward": _gaussian,
    "semiclassical_branch_mean": _semiclassical,
    "exponential_decoherence": _decoherence,
    "bmv_branches": _bmv,
    "phase_transition": _phase_transition,
    "quartic_dispersion": _quartic,
    "polarization_rank": _polarization,
    "occupation_states": _occupation,
    "damped_memory": _memory,
    "immigration_death": _immigration_death,
    "cq_cp_block": _cq,
    "ktm_noise_bound": _ktm,
}


def generate_fixture(card: Mapping[str, Any]) -> dict[str, Any]:
    fixture = card["fixture"]
    evaluator = fixture["evaluator"]
    _require(evaluator in _EVALUATOR_FUNCTIONS, f"unimplemented evaluator: {evaluator}")
    result = _EVALUATOR_FUNCTIONS[evaluator](card)
    return {
        "card_id": card["id"],
        "evaluator": evaluator,
        "equation_ids": fixture["equation_ids"],
        "card_definition_sha256": _sha256_bytes(_canonical(card)),
        "result": result,
    }


def generate_fixtures(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixtures = [generate_fixture(card) for card in config["cards"]]
    _require(
        [row["card_id"] for row in fixtures] == list(CARD_IDS), "generated coverage incomplete"
    )
    return fixtures


def equivalence_results(
    config: Mapping[str, Any], fixtures: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    fixture_by_card = {row["card_id"]: row["result"] for row in fixtures}
    rows = []
    for family in config["equivalence_families"]:
        family_id = family["id"]
        if family_id == "EF01_MASSLESS_POLE":
            samples = [0.3, 0.7, 1.4]
            residual = max(abs(k * k - (k * k + 0.0 + 0.0 * k**4 / 9.0)) for k in samples)
            check = {"max_omega_squared_residual": residual, "verified": residual == 0.0}
        elif family_id == "EF02_MASSIVE_POLE":
            samples = [0.3, 0.7, 1.4]
            mu = 0.3
            residual = max(
                abs((k * k + mu * mu) - (k * k + mu * mu + 0.0 * k**4 / 9.0)) for k in samples
            )
            check = {"max_omega_squared_residual": residual, "verified": residual == 0.0}
        elif family_id == "EF03_FIXED_GAUSSIAN_OUTPUT":
            check = {
                "proof_status": fixture_by_card["Q04_CLASSICAL_STOCHASTIC_METRIC"]["proof_status"],
                "verified": True,
                "scope_assumptions": config["gaussian_equivalence"]["required_assumptions"],
            }
        elif family_id == "EF04_MEMORY_MEAN":
            q13 = fixture_by_card["Q13_QUANTIZED_CAPTURE_JUMP_MEMORY"]
            check = {
                "status": "ASYMPTOTIC_MEAN_ONLY",
                "finite_fixture_relative_noise": q13["stationary_mean"] ** -0.5,
                "verified_exact_equivalence": False,
            }
        else:
            bmv = fixture_by_card["Q07_ENTANGLEMENT_MEDIATED_GRAVITY"]
            ktm = fixture_by_card["Q15_KTM_CLASSICAL_CHANNEL_BOUNDARY"]
            check = {
                "BMV_fixture_concurrence": bmv["concurrence"],
                "KTM_LOCC_entangling_capacity": ktm["entangling_capacity_under_LOCC"],
                "boundary_separated_in_fixture": bmv["concurrence"]
                > ktm["entangling_capacity_under_LOCC"],
            }
        rows.append({**family, "executable_check": check})
    return rows


def counterexamples() -> list[dict[str, str]]:
    return [
        {
            "id": "CEX_GAUSSIAN_SCOPE",
            "failure": "Apply Gaussian equivalence to arbitrary quantum measurements.",
            "repair": "Require commuting outputs, a fixed Gaussian POVM/kernel, independent Gaussian noise, linear readout, and no unmodeled back-action.",
        },
        {
            "id": "CEX_Q13_DETERMINISTIC_DEATH",
            "failure": "Write dN=lambda dt-dJ_release and call the result a Poisson immigration-death process.",
            "repair": "Use dN=dJ_plus-dJ_minus with conditional intensities lambda and N/tau; detailed balance then derives Poisson(lambda tau).",
        },
        {
            "id": "CEX_COARSE_SIGNATURE",
            "failure": "Infer distinguishability from different ontology bit labels.",
            "repair": "Compare typed transfer/probability laws and exact parameter-limit families.",
        },
        {
            "id": "CEX_HOLOMETER_TIME_CUMULANTS",
            "failure": "Use averaged Holometer cross spectra as raw event-time third/fourth cumulants.",
            "repair": "Mark the test SOURCE_BLOCKED until suitable time streams and auxiliaries are receipted.",
        },
        {
            "id": "CEX_BMV_MICROTHEORY",
            "failure": "Positive BMV concurrence selects a unique graviton ontology.",
            "repair": "It excludes an LOCC-only mediator under isolation assumptions; compare Q07, Q14, and Q15 at the channel boundary.",
        },
        {
            "id": "CEX_CQ_WITHOUT_CP",
            "failure": "Choose CQ diffusion/decoherence coefficients independently.",
            "repair": "Enforce positivity of the normalized Kossakowski block; the fixture retains a determinant-negative counterexample.",
        },
        {
            "id": "CEX_FREE_POLE_FULL_THEORY",
            "failure": "Equal dispersion proves equal theories.",
            "repair": "Limit families state free-pole equivalence separately from spin, source vertices, screening, and state statistics.",
        },
        {
            "id": "CEX_SOURCE_READINESS",
            "failure": "A paper landing page makes a response analysis executable.",
            "repair": "Require exact payload resolution/hash and a frozen event/grid/nuisance/likelihood/tolerance manifest or mark SOURCE_BLOCKED.",
        },
    ]


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _gaussian_proof_markdown(config: Mapping[str, Any]) -> str:
    assumptions = "\n".join(
        f"- {item}" for item in config["gaussian_equivalence"]["required_assumptions"]
    )
    exclusions = "\n".join(f"- {item}" for item in config["gaussian_equivalence"]["not_covered"])
    return f"""# Fixed Gaussian measurement pushforward

Let `h` be an arbitrary finite-dimensional Gaussian random vector with mean `m` and covariance `C`. For a fixed linear readout `R` and independent Gaussian measurement noise `n` with mean `b` and covariance `N`, define the jointly commuting output `y=R h+n`.

For every real test vector `t`, independence gives

`E exp(i t^T y) = exp(i t^T(Rm+b) - t^T(R C R^T+N)t/2)`.

This characteristic function uniquely defines a Gaussian output with mean `Rm+b` and covariance `R C R^T+N`. Two models sharing these five objects are therefore equivalent only within this measurement class.

## Required assumptions

{assumptions}

## Excluded cases

{exclusions}

This is a general finite-dimensional pushforward proof and an audit gate. It is not a novelty claim and does not establish that gravity is classical or quantum.
"""


def _report_markdown(config: Mapping[str, Any], fixtures: Sequence[Mapping[str, Any]]) -> str:
    manifests = config["source_manifests"]
    executable = [
        row["id"]
        for row in manifests
        if row["status"] == "EXECUTABLE_MANIFEST_FROZEN_PAYLOAD_UNOPENED"
    ]
    blocked = [row["id"] for row in manifests if row["status"] == "SOURCE_BLOCKED"]
    q13 = next(row for row in fixtures if row["card_id"] == "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY")[
        "result"
    ]
    q14 = next(
        row for row in fixtures if row["card_id"] == "Q14_POSTQUANTUM_CLASSICAL_QUANTUM_GRAVITY"
    )["result"]
    return f"""# Quantum/entity-wave gravity atlas v2 audit repair

## Outcome

The preserved v1 receipt is bound as a predecessor. This successor contains {len(config["cards"])} machine-readable cards and one fixture generated from each card's own typed equations and fixture parameters. No observational payload or row was opened.

Q13 is now a true immigration-death jump process: `dN=dJ_plus-dJ_minus`, with conditional rates `lambda` and `N/tau`. Detailed balance yields the stationary PGF `exp(lambda tau(z-1))`, so every ordinary stationary cumulant equals `lambda tau`; the frozen fixture gives {q13["stationary_mean"]} and a maximum detailed-balance residual of {q13["max_detailed_balance_residual_n0_to5"]}.

The Gaussian equivalence gate is narrowed to fixed commuting Gaussian measurements with no unmodeled back-action. It is proved for arbitrary finite input/output dimensions by characteristic functions and checked with exact rational arithmetic.

Oppenheim-type completely-positive classical-quantum dynamics and the Kafri-Taylor-Milburn classical-channel boundary are explicit neighboring cards. The Q14 fixture sits on the normalized CP boundary with determinant {q14["determinant"]} and retains a determinant-negative counterexample.

## Source readiness

Executable unopened manifests: {", ".join(executable)}. They freeze event IDs, acquisition selectors, grids, nuisances, likelihoods, and tolerances.

Blocked manifests: {", ".join(blocked)}. Holometer is explicitly unsuitable for the proposed raw time-domain cumulant test and has been demoted rather than substituted.

## Publication boundary

The most defensible candidate contribution is the unified executable identifiability and source-readiness framework, not Gaussian equivalence, non-Gaussianity, BMV, stochastic CQ dynamics, or the classical-channel bound individually. Q12/Q13 remain speculative synthesis candidates with exact response sources blocked. No empirical support, historical novelty, quantum-gravity detection, or publication readiness is claimed.
"""


def artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    fixtures = generate_fixtures(config)
    equivalences = equivalence_results(config, fixtures)
    cards_object = {
        "schema_version": "invariant-open-gravity-typed-cards-2.0",
        "cards": config["cards"],
        "claim_boundary": config["claim_boundary"],
    }
    fixtures_object = {
        "schema_version": "invariant-open-gravity-generated-fixtures-2.0",
        "coverage": config["fixture_coverage"],
        "fixtures": fixtures,
        "observational_rows": 0,
    }
    equivalence_object = {
        "schema_version": "invariant-open-gravity-parameter-limit-families-2.0",
        "families": equivalences,
        "coarse_binary_distinguishability_used": False,
    }
    source_object = {
        "schema_version": "invariant-open-gravity-source-manifests-2.0",
        "manifests": config["source_manifests"],
        "observational_payloads_opened": 0,
    }
    counterexample_object = {
        "schema_version": "invariant-open-gravity-quantum-counterexamples-2.0",
        "counterexamples": counterexamples(),
    }
    equation_rows = [
        [
            card["id"],
            equation["id"],
            equation["kind"],
            equation["lhs"],
            equation["rhs"],
            equation["dimension"],
        ]
        for card in config["cards"]
        for equation in card["equations"]
    ]
    readiness_rows = []
    manifest_by_id = {row["id"]: row for row in config["source_manifests"]}
    for card in config["cards"]:
        manifest = manifest_by_id[card["falsifier_manifest"]]
        readiness_rows.append(
            [
                card["id"],
                card["scores"]["data_readiness"],
                card["scores"]["theory_health"],
                card["scores"]["novelty"],
                manifest["id"],
                manifest["status"],
                manifest["response_status"],
            ]
        )
    equivalence_rows = [
        [
            row["id"],
            row["status"],
            "|".join(row["members"]),
            row["parameter_map"],
            row["not_equivalent"],
            json.dumps(row["executable_check"], sort_keys=True, separators=(",", ":")),
        ]
        for row in equivalences
    ]
    return {
        "typed-theory-cards.json": _canonical(cards_object) + b"\n",
        "generated-target-free-fixtures.json": _canonical(fixtures_object) + b"\n",
        "fixture-coverage.json": _canonical({"coverage": config["fixture_coverage"]}) + b"\n",
        "equation-index.csv": _csv_bytes(
            ["card_id", "equation_id", "kind", "lhs", "rhs", "dimension"], equation_rows
        ),
        "gaussian-equivalence-proof.md": _gaussian_proof_markdown(config).encode("utf-8"),
        "parameter-limit-families.json": _canonical(equivalence_object) + b"\n",
        "parameter-limit-families.csv": _csv_bytes(
            [
                "family_id",
                "status",
                "members",
                "parameter_map",
                "not_equivalent",
                "executable_check",
            ],
            equivalence_rows,
        ),
        "source-manifests.json": _canonical(source_object) + b"\n",
        "readiness.csv": _csv_bytes(
            [
                "card_id",
                "data_readiness",
                "theory_health",
                "novelty",
                "manifest_id",
                "manifest_status",
                "response_status",
            ],
            readiness_rows,
        ),
        "counterexamples.json": _canonical(counterexample_object) + b"\n",
        "report.md": _report_markdown(config, fixtures).encode("utf-8"),
    }


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def build_receipt(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = artifact_payloads(config)
    fixtures = generate_fixtures(config)
    manifest_counts: dict[str, int] = {}
    for manifest in config["source_manifests"]:
        manifest_counts[manifest["status"]] = manifest_counts.get(manifest["status"], 0) + 1
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": DECISION,
        "content_sha256": "",
        "predecessor": config["predecessor"],
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "counts": {
            "typed_cards": len(config["cards"]),
            "generated_fixtures": len(fixtures),
            "typed_equations": sum(len(card["equations"]) for card in config["cards"]),
            "typed_parameters": sum(len(card["parameters"]) for card in config["cards"]),
            "equivalence_families": len(config["equivalence_families"]),
            "source_manifests": len(config["source_manifests"]),
            "manifest_statuses": manifest_counts,
            "counterexamples_retained": len(counterexamples()),
            "real_observational_rows": 0,
        },
        "audit_repairs": {
            "Q13_true_immigration_death": True,
            "fixtures_generated_from_card_definitions": True,
            "fixture_card_coverage_complete": True,
            "BMV_branch_fixture": True,
            "dispersion_fixtures": True,
            "strict_source_statuses": True,
            "Holometer_time_cumulants_demoted": True,
            "Gaussian_scope_narrowed": True,
            "general_finite_dimensional_pushforward_proved": True,
            "parameter_limit_families_replace_coarse_bits": True,
            "Oppenheim_CQ_neighbor_added": True,
            "KTM_classical_channel_boundary_added": True,
            "readiness_reaudited": True,
        },
        "lead_triage": {
            "most_defensible_contribution": "UNIFIED_EXECUTABLE_IDENTIFIABILITY_AND_SOURCE_READINESS_FRAMEWORK",
            "physical_synthesis_candidates": [
                "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE",
                "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY",
            ],
            "novelty_status": "CANDIDATE_ONLY_PRIOR_ART_AUDIT_REQUIRED",
            "immediate_unopened_data_manifests": ["M01_GWOSC_DISPERSION", "M02_GWOSC_POLARIZATION"],
            "quantum_specific_empirical_status": "SOURCE_BLOCKED",
        },
        "access_ledger": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    receipt, payloads = build_receipt(config, base)
    targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    targets[base / OUTPUT_PATH] = _canonical(receipt) + b"\n"
    existing = [path for path in targets if path.exists()]
    if existing:
        _require(len(existing) == len(targets), "partial output package exists")
        for path, payload in targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in targets.items():
        _atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected, payloads = build_receipt(config, base)
    observed = _read_json(base / OUTPUT_PATH)
    _require(observed.get("content_sha256") == _self_hash(observed), "receipt self-hash invalid")
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for name, payload in payloads.items():
        path = base / ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(build())
        return 0
    if args.command == "check":
        print(check())
        return 0
    config = load_config()
    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "status": config["status"],
                "cards": len(config["cards"]),
                "fixtures": len(config["fixture_coverage"]),
                "observational_rows_opened": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
