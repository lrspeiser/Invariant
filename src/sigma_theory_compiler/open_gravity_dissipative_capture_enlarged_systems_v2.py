"""Strict-audit successor for dissipative capture and clumping hypotheses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_dissipative_capture_enlarged_systems_v2.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_dissipative_capture_enlarged_systems_v2.py"
)
TEST_PATH = Path("tests/test_open_gravity_dissipative_capture_enlarged_systems_v2.py")
ARTIFACT_DIRECTORY = Path(
    "runs/gravity/open-gravity-dissipative-capture-enlarged-systems-v2/artifacts"
)
OUTPUT_PATH = Path("runs/gravity/open-gravity-dissipative-capture-enlarged-systems-v2/receipt.json")
_SCHEMA = "invariant-open-gravity-dissipative-capture-enlarged-systems-receipt-2.0"
_MECHANISM_IDS = tuple(f"DC{i:02d}_" for i in range(8))


class DissipativeCaptureV2Error(RuntimeError):
    """Raised when a strict capture packet invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DissipativeCaptureV2Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema", "").endswith("2.0"), "wrong schema")
    _require(
        config.get("package_id") == "open-gravity-dissipative-capture-enlarged-systems-v2",
        "wrong package id",
    )
    ids = tuple(row["id"] for row in config["mechanisms"])
    _require(len(ids) == 8, "exactly eight inherited mechanisms are required")
    _require(all(value.startswith(prefix) for value, prefix in zip(ids, _MECHANISM_IDS)), "bad ids")
    fixture = config["dynamics_fixture"]
    _require(fixture["gamma"] >= 0.0, "negative friction violates the entropy gate")
    _require(fixture["temperature"] > 0.0, "receiver temperature must be positive")
    _require(fixture["dt"] > 0.0 and fixture["max_time"] > 0.0, "bad integration grid")
    for key in ("single", "bimodal"):
        mode = fixture[key]
        _require(len(mode["tau"]) == len(mode["weights"]), f"{key} shape mismatch")
        _require(all(value > 0.0 for value in mode["tau"]), f"{key} tau must be positive")
        _require(all(value >= 0.0 for value in mode["weights"]), f"{key} negative weight")
        _require(math.isclose(sum(mode["weights"]), 1.0), f"{key} weights must sum to one")
    access = config["access_contract"]
    for key in (
        "raw_scientific_source_files",
        "raw_scientific_response_files",
        "raw_scientific_rows",
        "new_real_data_scores",
        "model_calls",
        "paid_calls",
    ):
        _require(access[key] == 0, f"forbidden access: {key}")
    _require(
        config["tng_manifest"]["status"]
        == "SOURCE_BLOCKED_API_AUTH_AND_PAYLOAD_CHECKSUMS_UNAVAILABLE",
        "TNG source status must remain honestly blocked",
    )
    _require(config["triage_policy"]["retain_all_failures"] is True, "failures must remain")
    _require(config["artifact_directory"] == ARTIFACT_DIRECTORY.as_posix(), "artifact path drift")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path drift")


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    bound: dict[str, str] = {}
    for row in config["bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing binding: {path.as_posix()}")
        observed = file_sha256(path)
        _require(observed == row["sha256"], f"binding drift: {path.as_posix()}")
        bound[row["role"]] = observed
    prior = Path(config["supersedes"]["path"])
    _require(file_sha256(prior) == config["supersedes"]["file_sha256"], "v1 receipt drift")
    return bound


def _activation(r: float, scale: float) -> float:
    return math.exp(-((r / scale) ** 2))


def _relative_energy(state: Sequence[float], config: Mapping[str, Any]) -> float:
    fixture = config["dynamics_fixture"]
    m1 = fixture["m1"]
    m2 = fixture["m2"]
    mu = m1 * m2 / (m1 + m2)
    r, p_r = state[0], state[1]
    angular_momentum = state[6]
    return (
        p_r * p_r / (2.0 * mu)
        + angular_momentum * angular_momentum / (2.0 * mu * r * r)
        - fixture["G"] * m1 * m2 / r
    )


def _state_invariants(state: Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    fixture = config["dynamics_fixture"]
    total_mass = fixture["m1"] + fixture["m2"]
    rx, ry, px, py, angular_momentum, internal_energy, entropy = state[2:9]
    com_energy = (px * px + py * py) / (2.0 * total_mass)
    return {
        "energy": _relative_energy(state, config) + com_energy + internal_energy,
        "linear_momentum": [px, py],
        "angular_momentum": rx * py - ry * px + angular_momentum,
        "receiver_internal_energy": internal_energy,
        "receiver_entropy": entropy,
        "entropy_identity_error": internal_energy - fixture["temperature"] * entropy,
    }


def _derivative(
    state: Sequence[float],
    config: Mapping[str, Any],
    mode: str,
) -> list[float]:
    fixture = config["dynamics_fixture"]
    m1 = fixture["m1"]
    m2 = fixture["m2"]
    mu = m1 * m2 / (m1 + m2)
    total_mass = m1 + m2
    r, p_r, _rx, _ry, px, py, angular_momentum, _u, _entropy, *memory = state
    radial_velocity = p_r / mu
    activation = _activation(r, fixture["activation_scale"])
    if mode == "conservative":
        effective_memory = 0.0
        tau: list[float] = []
    elif mode in {"single", "bimodal"}:
        mode_config = fixture[mode]
        effective_memory = sum(
            weight * value for weight, value in zip(mode_config["weights"], memory)
        )
        tau = mode_config["tau"]
    elif mode == "compression":
        effective_memory = (
            max(0.0, -radial_velocity / fixture["compression"]["radial_speed_scale"]) * activation
        )
        tau = []
    else:
        raise DissipativeCaptureV2Error(f"unknown mode: {mode}")
    gamma = 0.0 if mode == "conservative" else fixture["gamma"]
    heat_rate = gamma * effective_memory * radial_velocity * radial_velocity
    derivatives = [
        radial_velocity,
        angular_momentum * angular_momentum / (mu * r**3)
        - fixture["G"] * m1 * m2 / r**2
        - gamma * effective_memory * radial_velocity,
        px / total_mass,
        py / total_mass,
        0.0,
        0.0,
        0.0,
        heat_rate,
        heat_rate / fixture["temperature"],
    ]
    derivatives.extend((activation - value) / time for value, time in zip(memory, tau))
    return derivatives


def _rk4_step(
    state: Sequence[float], dt: float, config: Mapping[str, Any], mode: str
) -> list[float]:
    k1 = _derivative(state, config, mode)
    k2 = _derivative([a + 0.5 * dt * b for a, b in zip(state, k1)], config, mode)
    k3 = _derivative([a + 0.5 * dt * b for a, b in zip(state, k2)], config, mode)
    k4 = _derivative([a + dt * b for a, b in zip(state, k3)], config, mode)
    return [
        a + dt * (b + 2.0 * c + 2.0 * d + e) / 6.0 for a, b, c, d, e in zip(state, k1, k2, k3, k4)
    ]


def receiver_dynamics_fixture(
    config: Mapping[str, Any], mode: str, initial_memory: Sequence[float] = ()
) -> dict[str, Any]:
    fixture = config["dynamics_fixture"]
    m1, m2 = fixture["m1"], fixture["m2"]
    mu = m1 * m2 / (m1 + m2)
    state = [
        fixture["initial_r"],
        mu * fixture["initial_radial_velocity"],
        *fixture["com_position"],
        *fixture["com_momentum"],
        fixture["orbital_angular_momentum"],
        0.0,
        0.0,
        *initial_memory,
    ]
    initial_state = list(state)
    initial_invariants = _state_invariants(state, config)
    dt = fixture["dt"]
    maximum_steps = int(fixture["max_time"] / dt)
    passed_pericenter = False
    min_r = state[0]
    step = 0
    for step in range(1, maximum_steps + 1):
        state = _rk4_step(state, dt, config, mode)
        _require(state[0] > 0.0, "radial integration crossed the origin")
        min_r = min(min_r, state[0])
        passed_pericenter = passed_pericenter or state[1] > 0.0
        if passed_pericenter and state[0] >= fixture["outbound_measurement_radius"]:
            break
    final_invariants = _state_invariants(state, config)
    energy_error = final_invariants["energy"] - initial_invariants["energy"]
    momentum_error = [
        end - start
        for end, start in zip(
            final_invariants["linear_momentum"], initial_invariants["linear_momentum"]
        )
    ]
    return {
        "mode": mode,
        "state_equations": "RK4 evolution of (r,p_r,Rx,Ry,Px,Py,L,U,S,h_i)",
        "initial_state": initial_state,
        "final_state": state,
        "elapsed_time": step * dt,
        "minimum_radius": min_r,
        "passed_pericenter": passed_pericenter,
        "initial_relative_energy": _relative_energy(initial_state, config),
        "final_relative_energy": _relative_energy(state, config),
        "captured": _relative_energy(state, config) < 0.0,
        "initial_invariants": initial_invariants,
        "final_invariants": final_invariants,
        "state_derived_errors": {
            "total_energy": energy_error,
            "linear_momentum_norm": math.hypot(*momentum_error),
            "angular_momentum": final_invariants["angular_momentum"]
            - initial_invariants["angular_momentum"],
            "entropy_identity": final_invariants["entropy_identity_error"],
        },
        "receiver_is_posthoc_deficit": False,
        "gate_energy_warning": (
            None
            if mode in {"conservative", "compression"}
            else "h_i has executable dynamics but no declared stress energy; this is not a closed field theory"
        ),
    }


def _three_body_acceleration(
    positions: np.ndarray, masses: np.ndarray, gravitational_constant: float, softening: float
) -> np.ndarray:
    acceleration = np.zeros_like(positions)
    for i in range(3):
        for j in range(i + 1, 3):
            displacement = positions[j] - positions[i]
            denominator = (float(displacement @ displacement) + softening**2) ** 1.5
            acceleration[i] += gravitational_constant * masses[j] * displacement / denominator
            acceleration[j] -= gravitational_constant * masses[i] * displacement / denominator
    return acceleration


def _three_body_invariants(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, g: float, eps: float
) -> dict[str, Any]:
    energy = 0.5 * float(np.sum(masses[:, None] * velocities * velocities))
    for i in range(3):
        for j in range(i + 1, 3):
            separation = positions[i] - positions[j]
            energy -= g * masses[i] * masses[j] / math.sqrt(float(separation @ separation) + eps**2)
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    angular_momentum = np.sum(
        masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])
    )
    return {
        "energy": float(energy),
        "linear_momentum": [float(value) for value in momentum],
        "angular_momentum": float(angular_momentum),
    }


def _visible_pair_energy(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, g: float, eps: float
) -> float:
    relative_position = positions[0] - positions[1]
    relative_velocity = velocities[0] - velocities[1]
    reduced_mass = masses[0] * masses[1] / (masses[0] + masses[1])
    return 0.5 * reduced_mass * float(relative_velocity @ relative_velocity) - (
        g * masses[0] * masses[1] / math.sqrt(float(relative_position @ relative_position) + eps**2)
    )


def three_body_countermodel_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    fixture = config["three_body_countermodel"]
    masses = np.asarray(fixture["masses"], dtype=float)
    positions = np.asarray(fixture["positions"], dtype=float)
    velocities = np.asarray(fixture["velocities"], dtype=float)
    g, eps, dt = fixture["G"], fixture["softening"], fixture["dt"]
    initial_invariants = _three_body_invariants(positions, velocities, masses, g, eps)
    initial_visible_energy = _visible_pair_energy(positions, velocities, masses, g, eps)
    acceleration = _three_body_acceleration(positions, masses, g, eps)
    for _ in range(fixture["steps"]):
        velocities += 0.5 * dt * acceleration
        positions += dt * velocities
        acceleration = _three_body_acceleration(positions, masses, g, eps)
        velocities += 0.5 * dt * acceleration
    final_invariants = _three_body_invariants(positions, velocities, masses, g, eps)
    final_visible_energy = _visible_pair_energy(positions, velocities, masses, g, eps)
    momentum_error = np.asarray(final_invariants["linear_momentum"]) - np.asarray(
        initial_invariants["linear_momentum"]
    )
    return {
        "id": "CM01_EXPLICIT_CONSERVATIVE_THREE_BODY_CAPTURE",
        "state_equations": "velocity-Verlet Newtonian three-body dynamics with softened central forces",
        "initial_visible_pair_energy": initial_visible_energy,
        "final_visible_pair_energy": final_visible_energy,
        "visible_pair_became_bound": bool(initial_visible_energy > 0.0 > final_visible_energy),
        "receiver": "third body's evolved position and momentum",
        "entropy_change": 0.0,
        "initial_invariants": initial_invariants,
        "final_invariants": final_invariants,
        "state_derived_errors": {
            "total_energy": final_invariants["energy"] - initial_invariants["energy"],
            "linear_momentum_norm": float(np.linalg.norm(momentum_error)),
            "angular_momentum": final_invariants["angular_momentum"]
            - initial_invariants["angular_momentum"],
        },
        "falsifies": "visible-pair capture or clumping alone implies dissipation",
    }


def mechanism_triage(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mechanism in config["mechanisms"]:
        metrics = mechanism["structural_metrics"]
        gap_count = len(mechanism["gaps"])
        score = (
            3 * metrics["executable_states"]
            + 3 * metrics["state_derived_invariants"]
            + 2 * metrics["irreversible_receiver"]
            + 2 * metrics["same_state_history_discriminator"]
            + metrics["public_source_contract"]
            - gap_count
            - metrics["free_kernel_modes"]
        )
        rows.append(
            {
                "mechanism_id": mechanism["id"],
                "theory_triage_score": score,
                "gap_count": gap_count,
                "free_kernel_modes": metrics["free_kernel_modes"],
                "execution_grade": mechanism["execution_grade"],
                "theory_grade": mechanism["theory_grade"],
                "candidate_eligible": mechanism["kind"] == "HYPOTHESIS_ENLARGED_RECEIVER",
                "is_data_or_model_score": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["theory_triage_score"],
            row["gap_count"],
            row["free_kernel_modes"],
            row["mechanism_id"],
        )
    )
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def counterexamples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        three_body_countermodel_fixture(config),
        {
            "id": "CM02_COLLISIONLESS_VIOLENT_RELAXATION",
            "execution_grade": "PUBLISHED_COUNTERMODEL_RETAINED",
            "statement": "A time-dependent collisionless potential redistributes individual energies and creates a bound remnant without thermodynamic dissipation.",
            "source_id": "SRC_YOUNG_2018",
        },
        {
            "id": "CM03_ORDINARY_GAS_SHOCK_COOLING",
            "execution_grade": "MANDATORY_NULL",
            "statement": "Measured shock heating and radiative cooling can shorten coalescence without a new gravitational receiver.",
        },
        {
            "id": "CM04_ESTABLISHED_FRICTION_AND_FEEDBACK",
            "execution_grade": "MANDATORY_NULL",
            "statement": "Dynamical friction, satellite mass loss, and feedback history can mimic a memory residual.",
        },
    ]


def synthetic_fixtures(config: Mapping[str, Any]) -> dict[str, Any]:
    fixture = config["dynamics_fixture"]
    return {
        "conservative": receiver_dynamics_fixture(config, "conservative"),
        "single_quiet": receiver_dynamics_fixture(
            config, "single", fixture["single"]["quiet_initial_h"]
        ),
        "single_post_pass": receiver_dynamics_fixture(
            config, "single", fixture["single"]["post_pass_initial_h"]
        ),
        "bimodal_quiet": receiver_dynamics_fixture(
            config, "bimodal", fixture["bimodal"]["quiet_initial_h"]
        ),
        "bimodal_post_pass": receiver_dynamics_fixture(
            config, "bimodal", fixture["bimodal"]["post_pass_initial_h"]
        ),
        "compression": receiver_dynamics_fixture(config, "compression"),
        "three_body_countermodel": three_body_countermodel_fixture(config),
    }


def _ranking_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _report_bytes(
    config: Mapping[str, Any], fixtures: Mapping[str, Any], triage: Sequence[Mapping[str, Any]]
) -> bytes:
    single_quiet = fixtures["single_quiet"]
    single_post = fixtures["single_post_pass"]
    three_body = fixtures["three_body_countermodel"]
    candidate_lead = next(row for row in triage if row["candidate_eligible"])
    lines = [
        "# Dissipative capture strict-audit successor v2",
        "",
        "## Outcome",
        "",
        "Theory repair PASS; real-data source BLOCKED. DC05-DC07 now evolve receiver internal energy and entropy alongside matter. DC02-DC04 are explicitly demoted because their receiver state is not evolved in this packet.",
        "",
        "Energy, momentum, angular momentum, and entropy are evaluated from the initial and final state vectors. No receiver quantity is defined as a negative matter deficit after the fact.",
        "",
        "## Synthetic discriminator",
        "",
        f"At one identical initial matter state, the single-memory quiet history ends with pair energy {single_quiet['final_relative_energy']:.9f} (unbound), while the post-pass state ends at {single_post['final_relative_energy']:.9f} (bound). Both evolve the same ODEs; only the initial memory state differs.",
        "",
        "This remains phenomenology, not a closed gravity theory: the memory gate has dynamics but no declared stress energy or covariant action. That failure is retained in every candidate card.",
        "",
        "## Strongest counterexample",
        "",
        f"A fully conservative three-body fixture changes the visible pair energy from {three_body['initial_visible_pair_energy']:.9f} to {three_body['final_visible_pair_energy']:.9f} while total state-derived energy, momentum, and angular momentum remain conserved and entropy stays zero. Therefore capture of a chosen pair does not identify dissipation.",
        "",
        "## Computed triage",
        "",
        f"The first eligible structural-theory candidate is {candidate_lead['mechanism_id']}. This is expert structural triage computed from declared metrics, never a fit or model score. DC06 is a nested extra-mode extension and pays its additional identifiability gap.",
        "",
        "## TNG falsifier",
        "",
        "The exact simulations, snapshot grid, fields, split, nuisance model, likelihood family, and decision rule are frozen. Authentication, expanded object URLs, byte receipts, and payload hashes are unavailable without source access, so the real-data stage is honestly SOURCE_BLOCKED. No response row was opened.",
        "",
        "Strongest falsifier: after matching TNG100-1 to TNG100-1-Dark and controlling measured shocks, cooling, collisionless history, established friction, feedback, and cadence, the frozen same-current-state prehistory term must improve held-out predictive density with its predicted sign. Failure ends the memory lead without retuning.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Establishes: {value}" for value in config["claim_boundary"]["establishes"])
    lines.extend(
        f"- Does not establish: {value}" for value in config["claim_boundary"]["does_not_establish"]
    )
    return ("\n".join(lines) + "\n").encode()


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    fixtures = synthetic_fixtures(config)
    triage = mechanism_triage(config)
    cards = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in config["mechanisms"]
    )
    return {
        "theory-cards.jsonl": cards,
        "executable-state-fixtures.json": _json_bytes(fixtures),
        "computed-theory-triage.csv": _ranking_bytes(triage),
        "countermodels.json": _json_bytes(counterexamples(config)),
        "tng100-hydro-dmo-source-manifest.json": _json_bytes(config["tng_manifest"]),
        "report.md": _report_bytes(config, fixtures, triage),
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    fixtures = synthetic_fixtures(config)
    triage = mechanism_triage(config)
    candidate_ranking = [row for row in triage if row["candidate_eligible"]]
    artifacts = build_artifacts(config)
    receiver_modes = [
        fixtures[name]
        for name in (
            "single_quiet",
            "single_post_pass",
            "bimodal_quiet",
            "bimodal_post_pass",
            "compression",
        )
    ]
    conservation_pass = all(
        abs(row["state_derived_errors"]["total_energy"]) < 1.0e-8
        and row["state_derived_errors"]["linear_momentum_norm"] < 1.0e-14
        and abs(row["state_derived_errors"]["angular_momentum"]) < 1.0e-12
        and abs(row["state_derived_errors"]["entropy_identity"]) < 1.0e-12
        for row in receiver_modes
    )
    three_body = fixtures["three_body_countermodel"]
    three_body_pass = (
        three_body["visible_pair_became_bound"]
        and abs(three_body["state_derived_errors"]["total_energy"]) < 5.0e-8
        and three_body["state_derived_errors"]["linear_momentum_norm"] < 1.0e-12
        and abs(three_body["state_derived_errors"]["angular_momentum"]) < 1.0e-12
        and three_body["entropy_change"] == 0.0
    )
    receipt: dict[str, Any] = {
        "schema": _SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_THEORY_REPAIR_BLOCK_TNG_SOURCE",
        "bindings": bindings,
        "superseded_v1": config["supersedes"],
        "receiver_audit": {
            "executable_receiver_mechanisms": [
                "DC05_TIMEWELL_SINGLE_MEMORY_BATH",
                "DC06_TIMEWELL_BIMODAL_MEMORY_BATH",
                "DC07_COMPRESSION_GATED_BATH",
            ],
            "demoted_unresolved_receivers": [
                "DC02_INELASTIC_CLOUD_SHOCK",
                "DC03_CHANDRASEKHAR_WAKE_TRANSFER",
                "DC04_GRAVITATIONAL_WAVE_CAPTURE",
            ],
            "posthoc_receiver_deficits": 0,
            "state_derived_conservation_pass": conservation_pass,
            "memory_gate_energy_failure_retained": True,
        },
        "synthetic_fixture_root_sha256": content_sha256(fixtures),
        "same_state_history_conditioned_capture": (
            not fixtures["single_quiet"]["captured"] and fixtures["single_post_pass"]["captured"]
        ),
        "conservative_three_body_countermodel_pass": three_body_pass,
        "triage_label": config["triage_policy"]["label"],
        "triage_formula": config["triage_policy"]["formula"],
        "computed_theory_triage": triage,
        "computed_candidate_ranking": candidate_ranking,
        "strongest_candidate": candidate_ranking[0]["mechanism_id"],
        "strongest_counterexample": three_body,
        "strongest_falsifier": config["tng_manifest"]["likelihood"]["decision_rule"],
        "tng_source_status": config["tng_manifest"]["status"],
        "remaining_blockers": config["tng_manifest"]["unresolved_before_source_open"],
        "published_sources": config["published_sources"],
        "artifact_manifest": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "decision": "REQUEST_INDEPENDENT_REAUDIT_BEFORE_ANY_TNG_RESPONSE_ACCESS",
    }
    _require(conservation_pass, "receiver state conservation failed")
    _require(three_body_pass, "three-body countermodel failed")
    _require(
        receipt["same_state_history_conditioned_capture"], "history fixture not discriminating"
    )
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing artifact differs: {path.as_posix()}")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, f"concurrent artifact differs: {path.as_posix()}")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_packet() -> str:
    config = load_config()
    statuses = [
        _atomic_no_clobber(ARTIFACT_DIRECTORY / name, payload)
        for name, payload in build_artifacts(config).items()
    ]
    statuses.append(_atomic_no_clobber(OUTPUT_PATH, _json_bytes(build_receipt())))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def validate_receipt() -> None:
    expected = build_receipt()
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for name, payload in build_artifacts(load_config()).items():
        _require((ARTIFACT_DIRECTORY / name).read_bytes() == payload, f"artifact drift: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    arguments = parser.parse_args(argv)
    if arguments.action == "build":
        print(write_packet())
    elif arguments.action == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(receipt["status"])
        print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
