"""Typed quantum/entity/wave gravity atlas with target-free discriminator gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_quantum_entity_wave_atlas_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_quantum_entity_wave_atlas_v1.py")
TEST_PATH = Path("tests/test_open_gravity_quantum_entity_wave_atlas_v1.py")
OUTPUT_PATH = Path("runs/gravity/theory/open-gravity-quantum-entity-wave-atlas-v1/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-quantum-entity-wave-atlas-config-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-quantum-entity-wave-atlas-receipt-1.0"
DECISION = (
    "PASS_TYPED_ONTOLOGY_ATLAS_AND_EXACT_CLASSICAL_RECORD_EQUIVALENCE_GATE_"
    "TARGET_FREE_FIXTURES_ONLY_REAL_DATA_UNOPENED"
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
)

FIXTURE_IDS = tuple(
    f"F{i:02d}_{name}"
    for i, name in enumerate(
        (
            "GAUSSIAN_PUSHFORWARD",
            "POISSON_CUMULANTS",
            "ENTANGLING_CHANNEL",
            "DISPERSION",
            "POLARIZATION_RANK",
            "MEMORY_HYSTERESIS",
            "CAPTURE_STATIONARY",
            "CLASSICAL_LIMITS",
        ),
        start=1,
    )
)


class QuantumAtlasError(RuntimeError):
    """Raised when a frozen theory-card or receipt invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantumAtlasError(message)


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
        raise QuantumAtlasError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-quantum-entity-wave-atlas-v1",
        "analysis identity changed",
    )
    _require(
        config.get("status") == "FROZEN_RESPONSE_FREE_THEORY_AND_FALSIFIER_ATLAS",
        "status changed",
    )
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
    basis = config.get("observable_basis")
    _require(isinstance(basis, list) and len(basis) == 12, "observable basis changed")
    _require(len(set(basis)) == len(basis), "observable basis is not unique")
    theorem = config.get("equivalence_theorem")
    _require(
        isinstance(theorem, dict)
        and theorem.get("id") == "GAUSSIAN_CLASSICAL_RECORD_EQUIVALENCE"
        and "same transfer R" in theorem.get("statement", "")
        and len(theorem.get("escape_witnesses", [])) == 5
        and len(theorem.get("nearest_prior", [])) == 2,
        "equivalence theorem changed",
    )
    score_contract = config.get("score_contract")
    _require(
        isinstance(score_contract, dict)
        and score_contract.get("no_composite_score") is True
        and "not added" in score_contract.get("rule", ""),
        "score independence changed",
    )
    cards = config.get("theory_cards")
    _require(isinstance(cards, list), "theory cards missing")
    _require(tuple(card.get("id") for card in cards) == CARD_IDS, "card inventory changed")
    required = {
        "id",
        "category",
        "ontology",
        "state_space",
        "dynamics",
        "probability_noise",
        "observables",
        "classical_limit",
        "dimensional_closure",
        "equivalence_relations",
        "nearest_primary_literature",
        "falsifier",
        "health",
        "scores",
        "signature",
    }
    health_axes = {
        "unitarity",
        "causality",
        "lorentz_diffeomorphism",
        "conservation",
        "ghost",
        "cutoff",
        "classical_limit",
    }
    for card in cards:
        _require(set(card) == required, f"typed fields changed: {card.get('id')}")
        _require(
            set(card["observables"]) == {"matter", "photon", "tensor"},
            f"observable channels changed: {card['id']}",
        )
        _require(set(card["health"]) == health_axes, f"health axes changed: {card['id']}")
        scores = card["scores"]
        _require(
            set(scores) == {"data_readiness", "theory_health", "novelty"}
            and all(type(value) is int and 0 <= value <= 4 for value in scores.values()),
            f"score axes changed: {card['id']}",
        )
        signature = card["signature"]
        _require(
            isinstance(signature, list)
            and len(signature) == len(basis)
            and set(signature) <= {0, 1},
            f"signature changed: {card['id']}",
        )
        _require(
            card["falsifier"].get("source_status") and card["falsifier"].get("unchanged_test"),
            f"falsifier incomplete: {card['id']}",
        )
        _require(
            card["nearest_primary_literature"].get("url", "").startswith("https://"),
            f"primary literature URL incomplete: {card['id']}",
        )
    fixtures = config.get("target_free_fixtures")
    _require(
        isinstance(fixtures, list) and tuple(row.get("id") for row in fixtures) == FIXTURE_IDS,
        "fixture inventory changed",
    )
    _require(len(config.get("mandatory_controls", [])) == 12, "controls changed")
    _require(
        config.get("access_contract")
        == {
            "observational_files_opened": 0,
            "observational_rows_read": 0,
            "real_scores_computed": 0,
            "network_downloads_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "parameter_tuning_calls": 0,
        },
        "access contract changed",
    )
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("target_free_fixtures_only") is True
        and boundary.get("real_observational_rows_scored") is False
        and boundary.get("any_branch_empirically_supported") is False
        and boundary.get("historical_novelty_established") is False
        and boundary.get("quantum_nature_of_gravity_established") is False
        and boundary.get("publication_ready") is False,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config)
    return config


def _matmul(
    left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]
) -> list[list[Fraction]]:
    _require(bool(left) and bool(right), "empty matrix")
    _require(len(left[0]) == len(right), "matrix shape mismatch")
    return [
        [
            sum((row[k] * right[k][j] for k in range(len(right))), Fraction(0))
            for j in range(len(right[0]))
        ]
        for row in left
    ]


def _transpose(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def gaussian_pushforward_fixture() -> dict[str, Any]:
    """Exact rational witness for the classical-record equivalence theorem."""
    response = [[Fraction(1), Fraction(2)], [Fraction(0), Fraction(1)]]
    covariance = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
    noise = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(2)]]
    mean = [[Fraction(1)], [Fraction(-1)]]
    output_mean = _matmul(response, mean)
    pushed = _matmul(_matmul(response, covariance), _transpose(response))
    output_covariance = [[pushed[i][j] + noise[i][j] for j in range(2)] for i in range(2)]
    _require(output_mean == [[Fraction(-1)], [Fraction(-1)]], "mean fixture failed")
    _require(
        output_covariance == [[Fraction(19), Fraction(7)], [Fraction(7), Fraction(5)]],
        "covariance fixture failed",
    )
    return {
        "id": "F01_GAUSSIAN_PUSHFORWARD",
        "response": [[int(value) for value in row] for row in response],
        "input_covariance": [[int(value) for value in row] for row in covariance],
        "detector_noise": [[int(value) for value in row] for row in noise],
        "output_mean": [int(row[0]) for row in output_mean],
        "output_covariance": [[int(value) for value in row] for row in output_covariance],
        "models_compared": ["classical_gaussian_metric", "quantum_gaussian_commuting_readout"],
        "classical_record_discriminator_exists": False,
        "proof_status": "EXACT_RATIONAL_PUSHFORWARD_EQUALITY",
    }


def poisson_cumulant_fixture(rate_exposure: int = 25) -> dict[str, Any]:
    _require(rate_exposure > 0, "Poisson exposure must be positive")
    return {
        "id": "F02_POISSON_CUMULANTS",
        "lambda_exposure": rate_exposure,
        "mean": rate_exposure,
        "variance": rate_exposure,
        "connected_kappa3": rate_exposure,
        "connected_kappa4": rate_exposure,
        "normalized_skewness": 1.0 / math.sqrt(rate_exposure),
        "excess_kurtosis": 1.0 / rate_exposure,
        "matched_gaussian_kappa3": 0,
        "matched_gaussian_kappa4": 0,
        "distinguished_beyond_two_point": True,
    }


def entangling_channel_fixture(theta: float = math.pi / 12.0) -> dict[str, Any]:
    _require(math.isfinite(theta), "theta is not finite")
    concurrence = abs(math.sin(2.0 * theta))
    _require(0.0 <= concurrence <= 1.0, "invalid concurrence")
    return {
        "id": "F03_ENTANGLING_CHANNEL",
        "theta_radians": theta,
        "concurrence": concurrence,
        "separable_input": True,
        "entangled_output": concurrence > 0.0,
        "classical_local_channel_bound": 0.0,
        "assumption_boundary": "no direct matter-matter quantum channel and only local operations plus classical communication",
    }


def dispersion_fixture() -> dict[str, Any]:
    mu = 0.3
    distance = 10.0
    frequencies = [0.6, 0.9, 1.5]
    rows = []
    for omega in frequencies:
        k = math.sqrt(omega * omega - mu * mu)
        group_velocity = k / omega
        delay = distance * (1.0 / group_velocity - 1.0)
        rows.append(
            {
                "omega": omega,
                "k": k,
                "group_velocity_over_c": group_velocity,
                "delay_over_light_time_unit": delay,
            }
        )
    delays = [row["delay_over_light_time_unit"] for row in rows]
    _require(delays == sorted(delays, reverse=True), "massive delay is not monotone")
    return {
        "id": "F04_DISPERSION",
        "mu": mu,
        "distance": distance,
        "rows": rows,
        "low_frequency_arrives_later": True,
    }


def _matrix_rank(matrix: Sequence[Sequence[float]], tolerance: float = 1e-12) -> int:
    work = [list(map(float, row)) for row in matrix]
    if not work:
        return 0
    rows, columns = len(work), len(work[0])
    rank = 0
    for column in range(columns):
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]), default=rank)
        if rank >= rows or abs(work[pivot][column]) <= tolerance:
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
        if rank == rows:
            break
    return rank


def polarization_rank_fixture() -> dict[str, Any]:
    two_detector = [[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1]]
    six_detector = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    rank_two = _matrix_rank(two_detector)
    rank_six = _matrix_rank(six_detector)
    _require(rank_two == 2 and rank_six == 6, "polarization ranks changed")
    return {
        "id": "F05_POLARIZATION_RANK",
        "modes": 6,
        "two_detector_rank": rank_two,
        "two_detector_identifiable": False,
        "ideal_six_detector_rank": rank_six,
        "ideal_six_detector_identifiable": True,
        "gate": "never force a six-mode classification when the realized antenna matrix is rank deficient",
    }


def memory_hysteresis_fixture() -> dict[str, Any]:
    omega0 = 1.0
    gamma = 0.2
    drive_omega = 0.5
    denominator_real = omega0**2 - drive_omega**2
    denominator_imag = -2.0 * gamma * drive_omega
    transfer = 1.0 / complex(denominator_real, denominator_imag)
    lag = math.atan2(transfer.imag, transfer.real)
    envelope_at_five = math.exp(-gamma * 5.0)
    _require(lag > 0.0 and envelope_at_five < 1.0, "memory fixture failed")
    return {
        "id": "F06_MEMORY_HYSTERESIS",
        "omega0": omega0,
        "Gamma": gamma,
        "drive_omega": drive_omega,
        "transfer_real": transfer.real,
        "transfer_imag": transfer.imag,
        "response_lag_radians": lag,
        "source_off_envelope_at_t5": envelope_at_five,
        "instantaneous_control_lag_radians": 0.0,
        "distinguishable_from_instantaneous_mean_law": True,
    }


def capture_stationary_fixture() -> dict[str, Any]:
    capture_rate = 4.0
    tau = 3.0
    mean = capture_rate * tau
    autocovariance_at_tau = mean / math.e
    return {
        "id": "F07_CAPTURE_STATIONARY",
        "capture_rate": capture_rate,
        "tau": tau,
        "stationary_mean": mean,
        "stationary_variance": mean,
        "connected_kappa3": mean,
        "connected_kappa4": mean,
        "autocovariance_at_one_tau": autocovariance_at_tau,
        "autocorrelation_at_one_tau": math.exp(-1.0),
    }


def classical_limit_fixture() -> dict[str, Any]:
    occupations = [1, 4, 25, 100, 10000]
    relative_noise = [1.0 / math.sqrt(value) for value in occupations]
    _require(
        all(left > right for left, right in itertools.pairwise(relative_noise)),
        "classical noise limit is not monotone",
    )
    return {
        "id": "F08_CLASSICAL_LIMITS",
        "occupations": occupations,
        "relative_poisson_noise": relative_noise,
        "monotone_to_zero": True,
        "warning": "small relative noise establishes a continuum record, not classical ontology",
    }


def fixture_results() -> list[dict[str, Any]]:
    return [
        gaussian_pushforward_fixture(),
        poisson_cumulant_fixture(),
        entangling_channel_fixture(),
        dispersion_fixture(),
        polarization_rank_fixture(),
        memory_hysteresis_fixture(),
        capture_stationary_fixture(),
        classical_limit_fixture(),
    ]


def pairwise_discriminators(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    basis = config["observable_basis"]
    cards = config["theory_cards"]
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(cards):
        for right in cards[index + 1 :]:
            witnesses = [
                name
                for name, left_value, right_value in zip(
                    basis, left["signature"], right["signature"], strict=True
                )
                if left_value != right_value
            ]
            rows.append(
                {
                    "left": left["id"],
                    "right": right["id"],
                    "distinguishable_in_binary_basis": bool(witnesses),
                    "minimal_declared_witness": witnesses[0] if witnesses else "NONE",
                    "all_declared_witnesses": witnesses,
                    "warning": (
                        "binary signature collision; compare continuous transfer and probability laws"
                        if not witnesses
                        else "a differing flag is a test target, not evidence that either model is true"
                    ),
                }
            )
    return rows


def counterexamples() -> list[dict[str, str]]:
    return [
        {
            "id": "CEX_ONTOLOGY_LABEL",
            "failed_inference": "Calling a signal a wave or a particle distinguishes the theory.",
            "counterexample": "A massive free pole has one dispersion relation whether represented as a particle excitation or a Fourier wave packet.",
            "required_repair": "Specify interaction vertices, state statistics, or number-resolving observables.",
        },
        {
            "id": "CEX_TWO_POINT_QUANTUM_PROOF",
            "failed_inference": "An unexplained Gaussian power spectrum proves quantum gravity.",
            "counterexample": "A classical Gaussian stochastic metric with matched covariance gives exactly the same linear classical record.",
            "required_repair": "Measure a non-Gaussian cumulant, state-conditioned response, noncommuting record, energy quantum, or entanglement witness.",
        },
        {
            "id": "CEX_DISPERSION_GRAVITON",
            "failed_inference": "Frequency-dependent arrival proves individual gravitons.",
            "counterexample": "A classical dispersive medium and a massive quantum mediator can share omega(k).",
            "required_repair": "Combine propagation with polarization, interaction, occupation, or coherence data.",
        },
        {
            "id": "CEX_SINGLE_SYSTEM_DECOHERENCE",
            "failed_inference": "Visibility loss alone proves gravitational collapse.",
            "counterexample": "Classical potential noise can induce the same one-system master equation.",
            "required_repair": "Use registered environmental controls and multi-system/state-conditioned predictions fixed by the collapse noise law.",
        },
        {
            "id": "CEX_BMV_OVERCLAIM",
            "failed_inference": "A positive BMV witness uniquely proves Fock-state gravitons.",
            "counterexample": "It witnesses a nonclassical mediator under locality/isolation assumptions, not one unique microscopic gravity ontology.",
            "required_repair": "Report the channel-level conclusion and retain direct-coupling loopholes.",
        },
        {
            "id": "CEX_SUPERFLUID_LENSING",
            "failed_inference": "A phonon force that fits matter dynamics automatically fits light bending.",
            "counterexample": "A nonmetric baryon phonon coupling can accelerate matter without contributing the same lensing potential.",
            "required_repair": "Derive matter and photon observables from the full stress-energy and metric sector.",
        },
        {
            "id": "CEX_DAMPING_WITHOUT_BATH",
            "failed_inference": "A damped memory field may be used without noise or energy bookkeeping.",
            "counterexample": "Damping removes field energy; a closed description needs a bath, and equilibrium quantum damping links noise to response.",
            "required_repair": "Supply the reservoir, its stress-energy, and the fluctuation-dissipation/noise law.",
        },
        {
            "id": "CEX_EXPECTATION_SOURCE",
            "failed_inference": "The bare expectation-value semiclassical equation already handles macroscopic branch selection.",
            "counterexample": "Page-Geilker reported behavior inconsistent with the simplest expectation-averaged source in its tested setup.",
            "required_repair": "State an independent branch, collapse, or stochastic rule and test it prospectively.",
        },
    ]


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _rankings_markdown(config: Mapping[str, Any]) -> str:
    cards = {card["id"]: card for card in config["theory_cards"]}
    lines = [
        "# Quantum/entity/wave gravity triage",
        "",
        "There is no composite truth score. Data readiness, theory health, and novelty are independent audit axes.",
        "",
        "## Strongest exact result",
        "",
        "`GAUSSIAN_CLASSICAL_RECORD_EQUIVALENCE`: linear classical readout cannot distinguish a classical Gaussian gravity field from a quantum Gaussian field when their mean, transfer, symmetrized covariance, and detector noise match. A power-spectrum excess alone therefore cannot establish that gravity is quantized.",
        "This is a frozen audit gate, not a novelty claim: stochastic-gravity correlation equivalences and non-Gaussian quantum-gravity witnesses have close prior work, including Hu, Roura and Verdaguer and Howl et al.",
        "",
        "## Strongest potentially publishable synthesis",
        "",
        "`Q12_QUANTIZED_TIMEWELL_MEMORY_MODE` combines a causal source-driven memory response with the bath noise required by an open quantum mode. Its paired prediction is harder to fake than a static force excess: one transfer function must jointly fix mean phase lag, source-off ringdown, and state/noise spectrum. Historical novelty is not established.",
        "",
        "## Fastest real-data tests",
        "",
        "- `Q01`, `Q09`, `Q10`: public GWOSC/GWTC-3 dispersion and network-polarization data; these are data-ready but mostly established phenomenology.",
        "- `Q06`: Gran Sasso Diósi-Penrose source spreadsheets; directly tests a specified collapse consequence.",
        "- `Q03`, `Q11`: Holometer/GWOSC cross-spectra can test registered higher-cumulant templates, but detector artifacts are severe controls.",
        "- `Q07`: the cleanest ontology discriminator in principle, but no public gravity-mediated entanglement response exists.",
        "",
        "## Separate audit axes",
        "",
        "| Card | Data readiness | Theory health | Novelty |",
        "|---|---:|---:|---:|",
    ]
    for card_id in CARD_IDS:
        score = cards[card_id]["scores"]
        lines.append(
            f"| {card_id} | {score['data_readiness']} | {score['theory_health']} | {score['novelty']} |"
        )
    lines.extend(
        [
            "",
            "A novelty value of 3 means only that the synthesis merits a dedicated search; it is not a novelty claim. A low theory-health value retains the observable law and lists the completion conflict rather than silently deleting the branch.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_markdown(
    config: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
) -> str:
    collision_count = sum(not row["distinguishable_in_binary_basis"] for row in pairwise)
    lines = [
        "# Quantum/entity-versus-wave gravity atlas v1",
        "",
        "## Outcome",
        "",
        f"Fourteen ontology/dynamics cards and {len(fixtures)} target-free fixtures were frozen. No observational row was opened or scored.",
        "",
        "The main exact result is negative but useful: when a quantum Gaussian gravity field and a classical Gaussian stochastic field have the same linear transfer, mean, covariance, and detector noise, their entire classical measurement-record distribution is identical. Particle/wave language and a power spectrum cannot break that equivalence.",
        "",
        "The required escape observables are higher connected cumulants, state-conditioned response, noncommuting sequential measurements, quantized energy exchange, or gravity-mediated entanglement.",
        "Non-Gaussianity as a quantum-gravity witness is prior work (Howl et al., arXiv:2004.01189). The possible contribution here is the unified executable identifiability ladder across fourteen ontology cards and concrete public-data preflights, not the escape criterion by itself.",
        "",
        "## What is new enough to pursue",
        "",
        "The strongest synthesis candidate is the quantized time-well memory mode (`Q12`). One causal transfer must produce three linked signatures: driven phase lag, source-off ringdown, and fluctuation-dissipation-compatible noise. That linkage makes the law falsifiable even if its eventual covariant completion fails. The discrete capture/jump memory (`Q13`) is more speculative: its distinctive finite-occupation prediction is Poisson-linked mean, variance, third/fourth cumulants, and exponential autocorrelation.",
        "",
        "## What existing data can do",
        "",
        "GWOSC/GWTC-3 can immediately test dispersion and polarization cards without asserting quantumness. The Gran Sasso supplement tests a specified collapse branch. Holometer and off-source GW records can test cross-instrument non-Gaussian templates, although instrumental glitches and photon shot noise are mandatory controls. A BMV/QGEM result remains the cleanest channel-level quantum witness, but no gravity-mediated response dataset is public yet.",
        "",
        "## Identifiability audit",
        "",
        f"The 14 binary signature vectors generate {len(pairwise)} pairwise comparisons and {collision_count} signature collision(s). A collision means the coarse binary basis is insufficient; the continuous transfer and probability laws must be compared. A non-collision is only a candidate discriminator, never evidence for either theory.",
        "",
        "## Claim boundary",
        "",
        "This package derives target-free consequences, public-source preflights, and counterexamples. It does not claim a real-data fit, a quantum detection, a galaxy/cluster explanation, historical novelty, or publication readiness.",
        "",
    ]
    return "\n".join(lines)


def artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    fixtures = fixture_results()
    pairwise = pairwise_discriminators(config)
    basis = config["observable_basis"]
    signature_rows = [
        [
            card["id"],
            card["category"],
            card["scores"]["data_readiness"],
            card["scores"]["theory_health"],
            card["scores"]["novelty"],
            *card["signature"],
            card["falsifier"]["id"],
            card["falsifier"]["source_status"],
        ]
        for card in config["theory_cards"]
    ]
    pairwise_rows = [
        [
            row["left"],
            row["right"],
            row["distinguishable_in_binary_basis"],
            row["minimal_declared_witness"],
            "|".join(row["all_declared_witnesses"]),
            row["warning"],
        ]
        for row in pairwise
    ]
    preflights = [
        {
            "card_id": card["id"],
            "falsifier": card["falsifier"],
            "nearest_primary_literature": card["nearest_primary_literature"],
            "data_readiness": card["scores"]["data_readiness"],
        }
        for card in config["theory_cards"]
    ]
    cards_object = {
        "schema_version": "invariant-open-gravity-typed-theory-cards-1.0",
        "cards": config["theory_cards"],
        "claim_boundary": config["claim_boundary"],
    }
    fixture_object = {
        "schema_version": "invariant-open-gravity-quantum-target-free-fixtures-1.0",
        "fixtures": fixtures,
        "observational_rows_used": 0,
    }
    pairwise_object = {
        "schema_version": "invariant-open-gravity-quantum-pairwise-discriminators-1.0",
        "basis": basis,
        "pairs": pairwise,
        "claim": "binary differences are candidate tests, not empirical evidence",
    }
    counterexample_object = {
        "schema_version": "invariant-open-gravity-quantum-counterexamples-1.0",
        "counterexamples": counterexamples(),
    }
    preflight_object = {
        "schema_version": "invariant-open-gravity-quantum-public-preflights-1.0",
        "rows": preflights,
        "observational_payloads_opened": 0,
    }
    return {
        "theory-cards.json": _canonical(cards_object) + b"\n",
        "target-free-fixtures.json": _canonical(fixture_object) + b"\n",
        "observable-signatures.csv": _csv_bytes(
            [
                "card_id",
                "category",
                "data_readiness",
                "theory_health",
                "novelty",
                *basis,
                "falsifier_id",
                "source_status",
            ],
            signature_rows,
        ),
        "pairwise-discriminators.csv": _csv_bytes(
            [
                "left",
                "right",
                "distinguishable_in_binary_basis",
                "minimal_declared_witness",
                "all_declared_witnesses",
                "warning",
            ],
            pairwise_rows,
        ),
        "pairwise-discriminators.json": _canonical(pairwise_object) + b"\n",
        "counterexamples.json": _canonical(counterexample_object) + b"\n",
        "public-data-preflights.json": _canonical(preflight_object) + b"\n",
        "ranking.md": _rankings_markdown(config).encode("utf-8"),
        "report.md": _report_markdown(config, fixtures, pairwise).encode("utf-8"),
    }


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def build_receipt(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads = artifact_payloads(config)
    fixtures = fixture_results()
    pairwise = pairwise_discriminators(config)
    collisions = [row for row in pairwise if not row["distinguishable_in_binary_basis"]]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": DECISION,
        "content_sha256": "",
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "counts": {
            "theory_cards": len(config["theory_cards"]),
            "target_free_fixtures": len(fixtures),
            "observable_basis_dimensions": len(config["observable_basis"]),
            "pairwise_comparisons": len(pairwise),
            "binary_signature_collisions": len(collisions),
            "counterexamples_retained": len(counterexamples()),
            "public_falsifier_preflights": len(config["theory_cards"]),
            "real_observational_rows": 0,
        },
        "exact_results": {
            "gaussian_record_equivalence": fixtures[0],
            "finite_poisson_escape": fixtures[1],
            "entangling_channel": fixtures[2],
            "polarization_rank_gate": fixtures[4],
            "memory_hysteresis": fixtures[5],
            "capture_stationary_law": fixtures[6],
        },
        "lead_triage": {
            "strongest_exact_result": "GAUSSIAN_CLASSICAL_RECORD_EQUIVALENCE",
            "strongest_synthesis_candidate": "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE",
            "strongest_finite_entity_signature": "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY",
            "cleanest_quantum_channel_witness": "Q07_ENTANGLEMENT_MEDIATED_GRAVITY",
            "fastest_public_data_lanes": [
                "Q01_MASSIVE_SPIN2",
                "Q06_GRAVITATIONAL_COLLAPSE_DECOHERENCE",
                "Q09_DISPERSIVE_GRAVITY_WAVE_PACKET",
                "Q10_POLARIZATION_BIREFRINGENT_GRAVITY",
            ],
            "novelty_status": "CANDIDATE_ONLY_HISTORICAL_AND_CURRENT_AUDIT_NOT_COMPLETE",
        },
        "known_blockers": [
            "no public gravity-mediated entanglement response exists",
            "exact reusable time-stream payloads for source-modulated memory tests are not bound",
            "higher-cumulant searches require detector nonstationarity and glitch controls",
            "Q12 requires a covariant source-plus-memory-plus-bath completion before fundamental interpretation",
            "Q13 lacks an energy-conserving capture/release vertex and preferred-frame resolution",
        ],
        "next_falsifiers": [
            {
                "priority": 1,
                "cards": [
                    "Q01_MASSIVE_SPIN2",
                    "Q09_DISPERSIVE_GRAVITY_WAVE_PACKET",
                    "Q10_POLARIZATION_BIREFRINGENT_GRAVITY",
                ],
                "source": "GWOSC/GWTC-3",
                "test": "one frozen multi-event dispersion and network-polarization transfer with rank-deficient events retained",
            },
            {
                "priority": 2,
                "cards": ["Q06_GRAVITATIONAL_COLLAPSE_DECOHERENCE"],
                "source": "Gran Sasso DP source-data spreadsheets",
                "test": "score every published energy bin for a frozen smearing-radius grid with the published background model",
            },
            {
                "priority": 3,
                "cards": [
                    "Q12_QUANTIZED_TIMEWELL_MEMORY_MODE",
                    "Q13_QUANTIZED_CAPTURE_JUMP_MEMORY",
                ],
                "source": "source-modulated atom interferometer or torsion-balance time streams",
                "test": "same frozen kernel must predict phase lag, source-off ringdown, autocovariance, and finite-occupation cumulants",
            },
        ],
        "score_contract": config["score_contract"],
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
    receipt_payload = _canonical(receipt) + b"\n"
    target = base / OUTPUT_PATH
    all_targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    all_targets[target] = receipt_payload
    existing = [path for path in all_targets if path.exists()]
    if existing:
        _require(len(existing) == len(all_targets), "partial output package exists")
        for path, payload in all_targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in all_targets.items():
        _atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    receipt, payloads = build_receipt(config, base)
    target = base / OUTPUT_PATH
    observed = _read_json(target)
    _require(observed.get("content_sha256") == _self_hash(observed), "receipt self-hash invalid")
    _require(observed == receipt, "receipt differs from deterministic rebuild")
    for name, payload in payloads.items():
        path = base / ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="write the append-only deterministic package")
    subparsers.add_parser("check", help="validate the sealed package without writes")
    subparsers.add_parser("status", help="print the frozen branch and fixture counts")
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
                "cards": len(config["theory_cards"]),
                "fixtures": len(config["target_free_fixtures"]),
                "observational_rows_opened": config["access_contract"]["observational_rows_read"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
