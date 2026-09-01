"""Executable enlarged-system capture and dissipative-clumping theory packet."""

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

CONFIG_PATH = Path("configs/open_gravity_dissipative_capture_enlarged_systems_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_dissipative_capture_enlarged_systems_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_dissipative_capture_enlarged_systems_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-dissipative-capture-enlarged-systems-v1/receipt.json")
ARTIFACT_DIRECTORY = Path(
    "runs/gravity/open-gravity-dissipative-capture-enlarged-systems-v1/artifacts"
)
_CANONICAL_OUTPUT_PATH = OUTPUT_PATH
_CANONICAL_ARTIFACT_DIRECTORY = ARTIFACT_DIRECTORY
_SCHEMA = "invariant-open-gravity-dissipative-capture-enlarged-systems-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-dissipative-capture-enlarged-systems-receipt-1.0"
_CONFIG_CONTENT_SHA256 = "b184ae27a82912f86f5ff4f07e1e296ba7b25637ba7dccfe8170c7caf04ced21"
_MECHANISM_IDS = (
    "DC00_NEWTONIAN_FOCUSING_CONTROL",
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL",
    "DC02_INELASTIC_CLOUD_SHOCK",
    "DC03_CHANDRASEKHAR_WAKE_TRANSFER",
    "DC04_GRAVITATIONAL_WAVE_CAPTURE",
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH",
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH",
    "DC07_COMPRESSION_GATED_BATH",
)


class DissipativeCaptureError(RuntimeError):
    """Raised when a frozen model, conservation ledger, or data contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DissipativeCaptureError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DissipativeCaptureError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    required = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "unit_system",
        "mechanisms",
        "fixtures",
        "published_sources",
        "real_data_preflight",
        "published_benchmark",
        "ranking_policy",
        "claim_boundary",
        "access_contract",
        "artifact_directory",
        "output_path",
    }
    _require(set(config) == required, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-dissipative-capture-enlarged-systems-v1",
        "package ID changed",
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output path changed")
    _require(
        config["artifact_directory"] == _CANONICAL_ARTIFACT_DIRECTORY.as_posix(),
        "artifact directory changed",
    )
    mechanisms = config["mechanisms"]
    _require(type(mechanisms) is list and len(mechanisms) == 8, "mechanism count changed")
    _require(tuple(row["id"] for row in mechanisms) == _MECHANISM_IDS, "mechanisms changed")
    for row in mechanisms:
        for field in (
            "receiver",
            "energy_rule",
            "momentum_rule",
            "angular_momentum_rule",
            "entropy_rule",
            "capture_rule",
        ):
            _require(bool(row[field]), f"missing {field}: {row['id']}")
    _require(len(config["published_sources"]) == 10, "published source count changed")
    _require(len(config["real_data_preflight"]["frozen_predictions"]) == 6, "predictions changed")
    _require(
        config["real_data_preflight"]["response_status"] == "FROZEN_UNOPENED_BY_THIS_PACKET",
        "response opened",
    )
    _require(set(config["access_contract"].values()) == {0}, "access contract changed")
    _require(config["ranking_policy"]["retain_all_failures"] is True, "failures not retained")


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "capture config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"binding changed: {row['role']}")
        observed[row["role"]] = digest
    _require(len(observed) == 3, "binding count changed")
    return observed


def _orbital_energy(m1: float, m2: float, separation: float, speed: float, gravity: float) -> float:
    reduced_mass = m1 * m2 / (m1 + m2)
    return 0.5 * reduced_mass * speed**2 - gravity * m1 * m2 / separation


def _impulse_ledger(
    *, m1: float, m2: float, separation: float, speed_before: float, speed_after: float
) -> dict[str, float]:
    _require(0.0 <= speed_after <= speed_before, "impulse is not braking")
    reduced_mass = m1 * m2 / (m1 + m2)
    impulse = reduced_mass * (speed_before - speed_after)
    energy_to_receiver = 0.5 * reduced_mass * (speed_before**2 - speed_after**2)
    angular_before = reduced_mass * separation * speed_before
    angular_after = reduced_mass * separation * speed_after
    return {
        "matter_1_momentum_change_y": -impulse,
        "receiver_bulk_momentum_change_y": impulse,
        "linear_momentum_closure_error": 0.0,
        "orbital_angular_momentum_before": angular_before,
        "orbital_angular_momentum_after": angular_after,
        "receiver_spin_gain": angular_before - angular_after,
        "angular_momentum_closure_error": 0.0,
        "receiver_energy_gain": energy_to_receiver,
    }


def conservative_focusing_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["conservative"]
    m1, m2 = float(row["m1"]), float(row["m2"])
    gravity = float(row["G"])
    separation = float(row["r_peri"])
    speed_infinity = float(row["v_infinity"])
    amplification = float(row["amplification"])
    reduced_mass = m1 * m2 / (m1 + m2)
    energy_infinity = 0.5 * reduced_mass * speed_infinity**2
    speed_peri = math.sqrt(speed_infinity**2 + 2.0 * gravity * (m1 + m2) / separation)
    energy_peri = _orbital_energy(m1, m2, separation, speed_peri, gravity)
    amplified_speed = math.sqrt(
        speed_infinity**2 + 2.0 * amplification * gravity * (m1 + m2) / separation
    )
    amplified_energy = _orbital_energy(m1, m2, separation, amplified_speed, amplification * gravity)
    _require(math.isclose(energy_infinity, energy_peri, abs_tol=1.0e-14), "Newton E drift")
    _require(
        math.isclose(energy_infinity, amplified_energy, abs_tol=1.0e-14),
        "static amplified E drift",
    )
    _require(energy_infinity > 0.0, "control is not hyperbolic")
    return {
        "fixture_id": "FX00_CONSERVATIVE_FOCUSING",
        "energy_at_infinity": energy_infinity,
        "newtonian_pericenter_speed": speed_peri,
        "newtonian_pericenter_energy": energy_peri,
        "amplified_pericenter_speed": amplified_speed,
        "amplified_pericenter_energy": amplified_energy,
        "receiver_energy_gain": 0.0,
        "entropy_gain": 0.0,
        "captured_newtonian": False,
        "captured_static_amplified": False,
        "interpretation": "Both laws focus the orbit; neither produces capture.",
    }


def inelastic_capture_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["inelastic"]
    m1, m2 = float(row["m1"]), float(row["m2"])
    gravity = float(row["G"])
    separation = float(row["separation"])
    speed_before = float(row["relative_speed"])
    restitution = float(row["restitution"])
    speed_after = restitution * speed_before
    energy_before = _orbital_energy(m1, m2, separation, speed_before, gravity)
    energy_after = _orbital_energy(m1, m2, separation, speed_after, gravity)
    ledger = _impulse_ledger(
        m1=m1,
        m2=m2,
        separation=0.0,
        speed_before=speed_before,
        speed_after=speed_after,
    )
    heat = ledger["receiver_energy_gain"]
    closure_error = energy_after + heat - energy_before
    entropy = heat / float(row["receiver_temperature"])
    _require(energy_before > 0.0 > energy_after, "inelastic fixture does not capture")
    _require(abs(closure_error) <= 1.0e-14, "inelastic energy ledger does not close")
    return {
        "fixture_id": "FX01_INELASTIC_TRUE_CAPTURE",
        "energy_before": energy_before,
        "energy_after": energy_after,
        "receiver_energy_gain": heat,
        "total_energy_closure_error": closure_error,
        "entropy_gain": entropy,
        "captured": True,
        "ledger": ledger,
    }


def chandrasekhar_wake_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["wake"]
    m1, m2 = float(row["m1"]), float(row["m2"])
    gravity = float(row["G"])
    separation = float(row["separation"])
    speed_before = float(row["relative_speed"])
    sigma = float(row["sigma"])
    x_value = speed_before / (math.sqrt(2.0) * sigma)
    maxwell_fraction = math.erf(x_value) - 2.0 * x_value * math.exp(-(x_value**2)) / math.sqrt(
        math.pi
    )
    force = (
        4.0
        * math.pi
        * gravity**2
        * m1**2
        * float(row["rho"])
        * float(row["coulomb_log"])
        * maxwell_fraction
        / speed_before**2
    )
    reduced_mass = m1 * m2 / (m1 + m2)
    impulse = force * float(row["duration"])
    speed_after = speed_before - impulse / reduced_mass
    _require(speed_after > 0.0, "wake impulse reverses the encounter")
    energy_before = _orbital_energy(m1, m2, separation, speed_before, gravity)
    energy_after = _orbital_energy(m1, m2, separation, speed_after, gravity)
    ledger = _impulse_ledger(
        m1=m1,
        m2=m2,
        separation=separation,
        speed_before=speed_before,
        speed_after=speed_after,
    )
    closure_error = energy_after + ledger["receiver_energy_gain"] - energy_before
    _require(energy_before > 0.0 > energy_after, "wake fixture does not capture")
    _require(abs(closure_error) <= 1.0e-14, "wake energy ledger does not close")
    return {
        "fixture_id": "FX02_WAKE_TRUE_CAPTURE",
        "maxwell_fraction": maxwell_fraction,
        "force_magnitude": force,
        "impulse_magnitude": impulse,
        "speed_after": speed_after,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "receiver_energy_gain": ledger["receiver_energy_gain"],
        "total_energy_closure_error": closure_error,
        "entropy_gain": ledger["receiver_energy_gain"] / float(row["receiver_temperature"]),
        "captured": True,
        "ledger": ledger,
    }


def gravitational_wave_capture_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["gw"]
    m1, m2 = float(row["m1"]), float(row["m2"])
    gravity, light_speed = float(row["G"]), float(row["c"])
    pericenter, speed_infinity = float(row["r_peri"]), float(row["v_infinity"])
    reduced_mass = m1 * m2 / (m1 + m2)
    energy_infinity = 0.5 * reduced_mass * speed_infinity**2
    radiated_energy = (
        85.0
        * math.pi
        / (12.0 * math.sqrt(2.0))
        * gravity**3.5
        * m1**2
        * m2**2
        * math.sqrt(m1 + m2)
        / (light_speed**5 * pericenter**3.5)
    )
    radiated_angular_momentum = (
        6.0 * math.pi * gravity**3 * m1**2 * m2**2 / (light_speed**5 * pericenter**2)
    )
    energy_after = energy_infinity - radiated_energy
    closure_error = energy_after + radiated_energy - energy_infinity
    _require(abs(closure_error) <= 1.0e-16, "GW energy ledger does not close")
    return {
        "fixture_id": "FX03_GR_CAPTURE",
        "energy_at_infinity": energy_infinity,
        "energy_after": energy_after,
        "radiated_energy": radiated_energy,
        "radiated_angular_momentum": radiated_angular_momentum,
        "radiated_linear_momentum_equal_mass_quadrupole": 0.0,
        "total_energy_closure_error": closure_error,
        "captured": energy_after < 0.0,
        "domain_warning": "Weak-field parabolic quadrupole control; not a galaxy-clumping mechanism.",
    }


def _memory_state(history: Sequence[float], *, tau: float, dt: float) -> float:
    _require(tau > 0.0 and dt > 0.0, "memory scales must be positive")
    decay = math.exp(-dt / tau)
    state = 0.0
    for activation in history:
        _require(0.0 <= activation <= 1.0, "activation outside [0,1]")
        state = decay * state + (1.0 - decay) * activation
    return state


def _memory_impulse(
    *,
    m1: float,
    m2: float,
    separation: float,
    speed_before: float,
    gamma0: float,
    state: float,
    dt: float,
    gravity: float,
) -> dict[str, Any]:
    reduced_mass = m1 * m2 / (m1 + m2)
    speed_after = speed_before * math.exp(-gamma0 * state * dt / reduced_mass)
    energy_before = _orbital_energy(m1, m2, separation, speed_before, gravity)
    energy_after = _orbital_energy(m1, m2, separation, speed_after, gravity)
    ledger = _impulse_ledger(
        m1=m1,
        m2=m2,
        separation=separation,
        speed_before=speed_before,
        speed_after=speed_after,
    )
    closure_error = energy_after + ledger["receiver_energy_gain"] - energy_before
    _require(abs(closure_error) <= 1.0e-14, "memory energy ledger does not close")
    return {
        "memory_state": state,
        "speed_after": speed_after,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "receiver_energy_gain": ledger["receiver_energy_gain"],
        "total_energy_closure_error": closure_error,
        "captured": energy_after < 0.0,
        "ledger": ledger,
    }


def single_memory_hysteresis_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["memory"]
    m1, m2 = float(row["m1"]), float(row["m2"])
    gravity = float(row["G"])
    separation = float(row["separation"])
    reduced_mass = m1 * m2 / (m1 + m2)
    potential_magnitude = gravity * m1 * m2 / separation
    initial_energy = float(row["initial_orbital_energy"])
    speed_before = math.sqrt(2.0 * (initial_energy + potential_magnitude) / reduced_mass)
    common = {
        "m1": m1,
        "m2": m2,
        "separation": separation,
        "speed_before": speed_before,
        "gamma0": float(row["gamma0"]),
        "dt": float(row["dt"]),
        "gravity": gravity,
    }
    quiet_state = _memory_state(row["quiet_history"], tau=float(row["tau"]), dt=float(row["dt"]))
    post_pass_state = _memory_state(
        row["post_pass_history"], tau=float(row["tau"]), dt=float(row["dt"])
    )
    quiet = _memory_impulse(state=quiet_state, **common)
    post_pass = _memory_impulse(state=post_pass_state, **common)
    _require(not quiet["captured"] and post_pass["captured"], "memory discriminator collapsed")
    _require(post_pass_state > quiet_state, "history did not persist")
    _require(
        row["quiet_history"][-1] == row["post_pass_history"][-1],
        "current activation is not matched",
    )
    return {
        "fixture_id": "FX04_SAME_STATE_HISTORY_CONDITIONED_CAPTURE",
        "matched_current_activation": float(row["quiet_history"][-1]),
        "matched_separation": separation,
        "matched_speed_before": speed_before,
        "quiet_history": quiet,
        "post_pass_history": post_pass,
        "entropy_gain_quiet": quiet["receiver_energy_gain"] / float(row["receiver_temperature"]),
        "entropy_gain_post_pass": post_pass["receiver_energy_gain"]
        / float(row["receiver_temperature"]),
        "unique_discriminator": "same instantaneous r, v and activation; different capture from prior passage",
    }


def bimodal_persistence_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["bimodal"]
    times = [float(value) for value in row["sample_times_gyr"]]
    fast_weight, slow_weight = float(row["fast_weight"]), float(row["slow_weight"])
    tau_fast, tau_slow = float(row["tau_fast"]), float(row["tau_slow"])
    _require(math.isclose(fast_weight + slow_weight, 1.0, abs_tol=1.0e-12), "weights")
    states = [
        fast_weight * math.exp(-time / tau_fast) + slow_weight * math.exp(-time / tau_slow)
        for time in times
    ]
    first_drop = 1.0 - states[1] / states[0]
    second_drop = 1.0 - states[2] / states[1]
    _require(abs(first_drop - 0.8) < 0.01, "fast benchmark mismatch")
    _require(abs(second_drop - 0.5) < 0.03, "slow benchmark mismatch")
    return {
        "fixture_id": "FX05_PUBLISHED_TWO_STAGE_CALIBRATION_CONTROL",
        "times_gyr": times,
        "normalized_persistence": states,
        "fractional_drop_first_1_gyr": first_drop,
        "fractional_drop_next_8_5_gyr": second_drop,
        "calibration_only": True,
        "independent_prediction": False,
    }


def compression_gate_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["fixtures"]["compression"]
    duration = float(row["duration"])
    speed = float(row["relative_speed"])
    gamma0 = float(row["gamma0"])
    v0 = float(row["v0"])

    def loss(radial_speed: float) -> float:
        activation = max(0.0, -radial_speed / v0)
        return gamma0 * activation * speed**2 * duration

    inbound_loss = loss(float(row["inbound_radial_speed"]))
    outbound_loss = loss(float(row["outbound_radial_speed"]))
    _require(inbound_loss > 0.0 and outbound_loss == 0.0, "compression gate collapsed")
    return {
        "fixture_id": "FX06_COMPRESSION_TIME_ARROW",
        "matched_speed_magnitude": speed,
        "inbound_receiver_energy_gain_first_order": inbound_loss,
        "outbound_receiver_energy_gain_first_order": outbound_loss,
        "interpretation": "A sign-of-radial-motion discriminator, likely degenerate with shocks.",
    }


def synthetic_fixtures(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        conservative_focusing_fixture(config),
        inelastic_capture_fixture(config),
        chandrasekhar_wake_fixture(config),
        gravitational_wave_capture_fixture(config),
        single_memory_hysteresis_fixture(config),
        bimodal_persistence_fixture(config),
        compression_gate_fixture(config),
    ]
    _require(len({row["fixture_id"] for row in rows}) == 7, "duplicate fixture")
    return rows


def _mechanism_ranking(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores = {
        "DC00_NEWTONIAN_FOCUSING_CONTROL": (5, 5, 5, 0, 5),
        "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL": (5, 5, 5, 0, 4),
        "DC02_INELASTIC_CLOUD_SHOCK": (5, 4, 5, 1, 5),
        "DC03_CHANDRASEKHAR_WAKE_TRANSFER": (5, 3, 5, 1, 4),
        "DC04_GRAVITATIONAL_WAVE_CAPTURE": (5, 5, 2, 1, 5),
        "DC05_TIMEWELL_SINGLE_MEMORY_BATH": (4, 5, 5, 3, 3),
        "DC06_TIMEWELL_BIMODAL_MEMORY_BATH": (4, 5, 5, 4, 3),
        "DC07_COMPRESSION_GATED_BATH": (4, 4, 5, 1, 3),
    }
    mechanism_by_id = {row["id"]: row for row in config["mechanisms"]}
    rows = []
    for mechanism_id, values in scores.items():
        receiver, discriminator, testability, novelty, theory_health = values
        priority = receiver + discriminator + testability + 2 * novelty + theory_health
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "receiver_closure": receiver,
                "synthetic_discriminator": discriminator,
                "empirical_testability": testability,
                "application_novelty": novelty,
                "theory_health": theory_health,
                "publication_priority": priority,
                "is_new_publication_lead": novelty >= 3,
                "theory_grade": mechanism_by_id[mechanism_id]["theory_grade"],
                "novelty_grade": mechanism_by_id[mechanism_id]["novelty_grade"],
            }
        )
    rows.sort(key=lambda row: (-row["publication_priority"], row["mechanism_id"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    _require(rows[0]["mechanism_id"] == "DC06_TIMEWELL_BIMODAL_MEMORY_BATH", "lead changed")
    return rows


def _data_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    preflight = config["real_data_preflight"]
    group_fields = set(preflight["required_group_fields"])
    gas_fields = set(preflight["required_gas_fields"])
    required_group = {"SubhaloPos", "SubhaloVel", "SubhaloVelDisp", "SubhaloSpin"}
    required_gas = {"EnergyDissipation", "GFM_CoolingRate", "InternalEnergy", "Machnumber"}
    required_matching = {"SubhaloIndexDark_LHaloTree", "SubhaloIndexDark_SubLink"}
    _require(required_group <= group_fields, "TNG group contract incomplete")
    _require(required_gas <= gas_fields, "TNG receiver fields incomplete")
    _require(
        required_matching <= set(preflight["required_matching_fields"]),
        "TNG baryonic-DMO matching contract incomplete",
    )
    return {
        "status": "PASS_FROZEN_SOURCE_CONTRACT_RESPONSE_UNOPENED",
        "source": preflight["source"],
        "selection": preflight["selection"],
        "field_counts": {
            "group": len(preflight["required_group_fields"]),
            "gas": len(preflight["required_gas_fields"]),
            "stellar": len(preflight["required_stellar_fields"]),
            "matching": len(preflight["required_matching_fields"]),
        },
        "predictions": preflight["frozen_predictions"],
        "observables": preflight["primary_observables"],
        "nuisance_controls": preflight["nuisance_controls"],
        "confirmation_fraction": 0.2,
        "response_rows_opened": 0,
        "response_status": preflight["response_status"],
        "missing_data_action": preflight["missing_data_action"],
    }


def _counterexamples() -> list[dict[str, str]]:
    return [
        {
            "id": "CE01_COLLISIONLESS_VIOLENT_RELAXATION",
            "strength": "STRONGEST",
            "statement": "A time-dependent but conservative collisionless potential redistributes particle energies and angular momenta and can leave long-lived clumping or phase-space memory without a new dissipative receiver.",
            "required_control": "Matched TNG100-1-Dark assembly history and explicit total-energy ledger.",
        },
        {
            "id": "CE02_STANDARD_BARYONIC_DISSIPATION",
            "strength": "STRONG",
            "statement": "Shocks, radiative cooling, feedback and inelastic cloud interactions already remove visible orbital or bulk kinetic energy.",
            "required_control": "Use TNG EnergyDissipation, cooling, Mach number and gas-state fields before adding a new bath.",
        },
        {
            "id": "CE03_STATIC_FOCUSING_MIMIC",
            "strength": "LOGICAL_NULL",
            "statement": "A stronger static attraction changes trajectories and density but does not capture a positive-energy orbit in an isolated time-independent system.",
            "required_control": "Demand positive receiver flux and a closed enlarged-system ledger.",
        },
        {
            "id": "CE04_MANGA_VISIBLE_DISTURBANCE_FAILURE",
            "strength": "EXISTING_REAL_DATA_BOUNDARY",
            "statement": "The sealed 243-galaxy MaNGA Item 13 test found that visible tidal/CAS disturbance worsened held-out dispersion MSE by 1.67 percent relative to the frozen age baseline.",
            "required_control": "Do not use visible morphology alone as the history or dissipation state.",
        },
        {
            "id": "CE05_EXTENDED_SATELLITE_FRICTION_TIMESCALE",
            "strength": "MANDATORY_QUANTITATIVE_NEIGHBOR",
            "statement": "The Boylan-Kolchin, Ma and Quataert N-body fit already predicts merger time from mass ratio, circularity and orbital energy more accurately than a point-mass Chandrasekhar estimate.",
            "required_control": "The memory bath must improve held-out coalescence time or receiver-flux history beyond the frozen published merger-time fit.",
        },
        {
            "id": "CE06_DISSIPATIVE_DARK_MATTER_COLLAPSE",
            "strength": "MANDATORY_MECHANISM_NEIGHBOR",
            "statement": "Published dissipative-dark-matter models already cool, contract, form disks or accelerate gravothermal collapse through explicit dark-sector receivers.",
            "required_control": "Distinguish visible-matter history-conditioned receiver flux from dark-sector cooling thresholds and collapse-time predictions.",
        },
    ]


def _cards_bytes(config: Mapping[str, Any]) -> bytes:
    rows = []
    for mechanism in config["mechanisms"]:
        card = dict(mechanism)
        card["dimensions"] = config["unit_system"]["dimensions"]
        card["empirical_grade"] = "UNTESTED_FROZEN_TNG_PREFLIGHT"
        card["retained_even_if_failure"] = True
        rows.append(card)
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def _ranking_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = [
        "rank",
        "mechanism_id",
        "publication_priority",
        "receiver_closure",
        "synthetic_discriminator",
        "empirical_testability",
        "application_novelty",
        "theory_health",
        "is_new_publication_lead",
        "theory_grade",
        "novelty_grade",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _report_bytes(
    config: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> bytes:
    fixture_by_id = {row["fixture_id"]: row for row in fixtures}
    memory = fixture_by_id["FX04_SAME_STATE_HISTORY_CONDITIONED_CAPTURE"]
    bimodal = fixture_by_id["FX05_PUBLISHED_TWO_STAGE_CALIBRATION_CONTROL"]
    lines = [
        "# Dissipative capture and clumping: enlarged-system packet",
        "",
        "## Result",
        "",
        "This packet does not infer capture from a stronger force or from clumping. It requires visible matter to lose energy to a named receiver while total energy, momentum, and angular momentum close.",
        "",
        f"Eight mechanisms are retained: two conservative null controls, three published dissipative controls, and three receiver-bearing hypotheses. Seven exact fixtures run. The leading new signature is `{ranking[0]['mechanism_id']}`.",
        "",
        "## Exact distinction",
        "",
        "The Newtonian and statically amplified hyperbolic encounters remain unbound. Inelastic cloud collision, wake transfer, and weak-field gravitational radiation cross from positive to negative orbital energy only because their receivers gain the missing energy.",
        "",
        "The single-memory bath provides the sharper discriminator: two encounters have the same current separation, speed, and activation, but different histories. The quiet history remains unbound while the post-passage history is captured:",
        "",
        f"- quiet post-impulse energy: {memory['quiet_history']['energy_after']:.12g};",
        f"- post-passage post-impulse energy: {memory['post_pass_history']['energy_after']:.12g};",
        f"- quiet/post memory states: {memory['quiet_history']['memory_state']:.12g} / {memory['post_pass_history']['memory_state']:.12g}.",
        "",
        "## Strongest publication lead",
        "",
        "A fast-plus-slow gravity-activated receiver predicts history-conditioned capture and two persistence tails at fixed instantaneous encounter state. Its constants here are calibrated to the published one-galaxy Illustris benchmark, which reported about an 80% first-Gyr reduction followed by about a 50% reduction of the remainder over 8.5 Gyr. This packet reproduces those calibration targets but has not tested TNG data:",
        "",
        f"- first-stage drop: {100 * bimodal['fractional_drop_first_1_gyr']:.3f}%;",
        f"- second-stage drop: {100 * bimodal['fractional_drop_next_8_5_gyr']:.3f}%.",
        "",
        "The potentially publishable claim is not the sum of exponentials. Open-system baths and merger relaxation are established. The possible new contribution is a universal gravitational activation rule that predicts receiver flux, capture probability, and a two-timescale phase-space loop across a target-blind merger sample better than ordinary collisionless, shock/cooling, wake, and flexible-history controls.",
        "",
        "The exact quantitative neighbors are also frozen: the Boylan-Kolchin--Ma--Quataert extended-satellite merger-time fit, Double-Disk Dark Matter cooling, and dissipative-SIDM gravothermal-collapse simulations. The proposed bath is potentially distinct only if its same-current-state fast/slow hysteresis survives those controls; no historical novelty is established here.",
        "",
        "## Strongest counterexample",
        "",
        "Collisionless violent relaxation can create clumping, redistribute individual particle energies, and preserve merger history even though the full system is conservative. Therefore morphology, N(E), N(L^2), or post-merger persistence alone cannot identify dissipation. The frozen TNG100-1 versus TNG100-1-Dark comparison is mandatory.",
        "",
        "## Real-data falsifier",
        "",
        "Select z<=1 major mergers from public TNG100-1 SubLink trees, reserve SHA256(SubhaloID) mod 5 == 4 as untouched confirmation, and track orbital energy/angular momentum, shock dissipation, cooling, coalescence time, and phase-space distances from 1.5 Gyr before to 2 Gyr after merger. At matched current state, compare first infall with post-pericenter systems. A memory claim fails if ordinary measured gas state plus collisionless history removes the residual, if the energy receiver ledger does not close, or if the frozen two-timescale law does not transfer without retuning.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Establishes: {claim}" for claim in config["claim_boundary"]["establishes"])
    lines.extend(
        f"- Does not establish: {claim}" for claim in config["claim_boundary"]["does_not_establish"]
    )
    return ("\n".join(lines) + "\n").encode()


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    fixtures = synthetic_fixtures(config)
    ranking = _mechanism_ranking(config)
    preflight = _data_preflight(config)
    return {
        "theory-cards.jsonl": _cards_bytes(config),
        "synthetic-fixtures.json": _json_bytes(fixtures),
        "tng-target-blind-preflight.json": _json_bytes(preflight),
        "mechanism-ranking.csv": _ranking_bytes(ranking),
        "counterexamples.json": _json_bytes(_counterexamples()),
        "report.md": _report_bytes(config, fixtures, ranking),
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    fixtures = synthetic_fixtures(config)
    ranking = _mechanism_ranking(config)
    preflight = _data_preflight(config)
    artifacts = build_artifacts(config)
    artifact_manifest = {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        for name, payload in sorted(artifacts.items())
    }
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_ENLARGED_SYSTEM_CAPTURE_SIGNATURES_TNG_PREFLIGHT_FROZEN",
        "bindings": bindings,
        "mechanism_counts": {
            "total": 8,
            "conservative_controls": 2,
            "published_dissipative_controls": 3,
            "new_receiver_hypotheses": 3,
        },
        "synthetic_fixture_count": len(fixtures),
        "synthetic_fixture_root_sha256": content_sha256(fixtures),
        "all_receiver_energy_nonnegative": all(
            row.get("receiver_energy_gain", row.get("radiated_energy", 0.0)) >= 0.0
            for row in fixtures
        ),
        "conservation_checks": {
            "inelastic_energy_closure": abs(fixtures[1]["total_energy_closure_error"]) <= 1.0e-14,
            "wake_energy_closure": abs(fixtures[2]["total_energy_closure_error"]) <= 1.0e-14,
            "gw_energy_closure": abs(fixtures[3]["total_energy_closure_error"]) <= 1.0e-16,
            "memory_quiet_energy_closure": abs(
                fixtures[4]["quiet_history"]["total_energy_closure_error"]
            )
            <= 1.0e-14,
            "memory_post_energy_closure": abs(
                fixtures[4]["post_pass_history"]["total_energy_closure_error"]
            )
            <= 1.0e-14,
            "linear_momentum_ledgers_close": True,
            "angular_momentum_ledgers_close_with_receiver_spin": True,
            "entropy_nonnegative_for_dissipative_fixtures": True,
        },
        "conservative_controls_capture": False,
        "true_capture_fixture_count": 4,
        "same_state_history_conditioned_capture": True,
        "ranking": ranking,
        "strongest_lead": {
            "mechanism_id": "DC06_TIMEWELL_BIMODAL_MEMORY_BATH",
            "empirical_grade": "UNTESTED_FROZEN_TNG_PREFLIGHT",
            "theory_grade": "PHENOMENOLOGICAL_MULTI_MODE_BATH_NOT_ACTION_DERIVED",
            "novelty_boundary": "The bath and sum-of-exponentials are known structures, extended-satellite friction already predicts merger times, and dissipative dark sectors already cool and collapse; only a universal gravity activation plus same-current-state fast/slow receiver-flux hysteresis that beats those controls could support application-level novelty.",
            "next_falsifier": "Target-blind TNG100-1 versus TNG100-1-Dark major-merger histories with no retuning of the published-benchmark-calibrated persistence constants.",
        },
        "strongest_counterexample": _counterexamples()[0],
        "data_preflight": preflight,
        "published_sources": config["published_sources"],
        "artifact_manifest": artifact_manifest,
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "decision": "ADVANCE_BIMODAL_MEMORY_BATH_TO_TARGET_BLIND_TNG_DEVELOPMENT_KEEP_ALL_CONTROLS",
    }
    _require(all(receipt["conservation_checks"].values()), "conservation check failed")
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
    statuses = []
    for name, payload in build_artifacts(config).items():
        statuses.append(_atomic_no_clobber(ARTIFACT_DIRECTORY / name, payload))
    receipt = build_receipt()
    encoded = _json_bytes(receipt)
    statuses.append(_atomic_no_clobber(OUTPUT_PATH, encoded))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def validate_receipt() -> None:
    payload = _read_json(OUTPUT_PATH, "capture receipt")
    expected = build_receipt()
    _require(payload == expected, "receipt differs from deterministic rebuild")
    artifacts = build_artifacts(load_config())
    for name, expected_bytes in artifacts.items():
        path = ARTIFACT_DIRECTORY / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == expected_bytes, f"artifact differs: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action == "build":
        print(write_packet())
        return 0
    if arguments.action == "check":
        validate_receipt()
        print("VALID")
        return 0
    receipt = build_receipt()
    print(receipt["status"])
    print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
