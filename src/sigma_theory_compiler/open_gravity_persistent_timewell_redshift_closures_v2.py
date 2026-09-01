"""Strict-audit persistent-time-well redshift closure successor."""

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

CONFIG_PATH = Path("configs/open_gravity_persistent_timewell_redshift_closures_v2.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_persistent_timewell_redshift_closures_v2.py"
)
TEST_PATH = Path("tests/test_open_gravity_persistent_timewell_redshift_closures_v2.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-persistent-timewell-redshift-closures-v2/receipt.json"
)
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "79b4f988409c8ee53968fdf57fb8c3f001d139d5e34667e44226aec9a00fd9cd"
_CONFIG_CONTENT_SHA256 = "7a4ed84915bdc081d21c5f23f7eed25bc021e63761febaaf5be7d893d90ab000"
_MODULE_SEMANTIC_SHA256 = "d3ca6da3a267db1638aeef3b76291f70147e40e9a540ee927413ed56324ba68c"
_TEST_RAW_SHA256 = "baf35c897a4f25f542058453770d2b92705e0ff409207e36061f57d54de94f83"
_SCHEMA = "invariant-open-gravity-persistent-timewell-redshift-closures-2.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-persistent-timewell-redshift-receipt-2.0"
_CLOSURE_IDS = (
    "C00_GR_ENDPOINT",
    "C01_EXACT_GRADIENT_PATH",
    "C02_POTENTIAL_COLUMN",
    "C03_TIDAL_CURVATURE_COLUMN",
    "C04_DISPERSIVE_PERSISTENT_MEDIUM",
    "C05_ENDPOINT_MEMORY",
    "C06_PATH_MEMORY_OPACITY",
    "C07_GENERAL_TIME_VARYING_METRIC_PATH",
    "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY",
)
_CANDIDATES = _CLOSURE_IDS[3:7] + (_CLOSURE_IDS[8],)
_TAU_FAST = 0.5
_TAU_SLOW = 1.5


class PersistentRedshiftV2Error(RuntimeError):
    """Raised when a strict closure or source invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistentRedshiftV2Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    marker = '_MODULE_SEMANTIC_SHA256 = "'
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    normalized = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(normalized.encode()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersistentRedshiftV2Error(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-persistent-timewell-redshift-closures-v2",
        "package ID changed",
    )
    _require(config["status"] == "FROZEN_RESPONSE_FREE_STRICT_AUDIT_SUCCESSOR", "status changed")
    _require(tuple(row["id"] for row in config["closures"]) == _CLOSURE_IDS, "closure set changed")
    _require(
        tuple(config["structural_triage"]["candidate_ids"]) == _CANDIDATES, "candidate set changed"
    )
    _require(
        config["claim_boundary"]["closure_taxonomy_complete"] is False, "taxonomy claim widened"
    )
    _require(
        config["claim_boundary"]["global_identifiability_established"] is False,
        "identifiability claim widened",
    )
    _require(config["claim_boundary"]["C08_independent_of_C07"] is False, "nesting claim changed")
    _require(config["claim_boundary"]["real_data_scored"] is False, "response claim widened")
    _require(
        config["geometric_contract"]["physical_path_measure"]
        == "dell=-uhat_a dx^a for a future-directed null ray",
        "physical path measure changed",
    )
    _require(
        "affine ds" in config["geometric_contract"]["forbidden_measure"],
        "affine-path prohibition removed",
    )
    _require(
        config["state_family"]["nesting_statement"].startswith("C08 is"),
        "nesting statement removed",
    )
    _require(
        len(config["galileo_metadata_manifest"]["directory_entries"]) == 14,
        "Galileo manifest count changed",
    )
    _require(config["galileo_metadata_manifest"]["payload_rows_opened"] == 0, "payload rows opened")
    _require(
        config["galileo_metadata_manifest"]["payload_sha256_status"] == "NOT_DOWNLOADED_NOT_HASHED",
        "payload status changed",
    )
    names = {row["name"] for row in config["galileo_metadata_manifest"]["directory_entries"]}
    expected = {f"esoc2044{day}.{suffix}" for day in range(7) for suffix in ("clk", "sp3")}
    _require(names == expected, "Galileo manifest names changed")
    _require(
        config["galileo_frozen_analysis"]["tau_seconds"]
        == [300.0, 1800.0, 7200.0, 21600.0, 43200.0],
        "Galileo tau grid changed",
    )
    _require(len(config["mandatory_controls"]) == 12, "control count changed")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "receipt path changed")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIR.as_posix(), "artifact path changed"
    )
    access = config["access_contract"]
    _require(access["observational_payload_files_downloaded"] == 0, "payload downloaded")
    _require(access["observational_payload_rows_opened"] == 0, "payload row opened")
    _require(access["scientific_response_rows_scored"] == 0, "response scored")


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic hash changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH, "v2 config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["local_bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing local binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"changed local binding: {row['role']}")
        observed[row["role"]] = digest
    old = config["supersedes"]
    old_path = Path(old["receipt_path"])
    _require(old_path.is_file(), "missing superseded receipt")
    _require(file_sha256(old_path) == old["receipt_sha256"], "superseded receipt changed")
    _require(Path(old["blocked_audit_record"]).is_file(), "missing blocked-audit preservation")
    blocked_v2 = Path(old["blocked_v2_crossvalidation_wording_receipt"])
    _require(blocked_v2.is_file(), "missing blocked v2 wording receipt")
    _require(
        file_sha256(blocked_v2) == old["blocked_v2_crossvalidation_wording_receipt_sha256"],
        "blocked v2 wording receipt changed",
    )
    return observed


def physical_path_measure(dx: Sequence[float], rapidity: float) -> float:
    """Compute -u_hat_a dx^a in 1+1 Minkowski coordinates (c=1)."""
    _require(len(dx) == 2, "dx must have two components")
    dt, spatial_dx = (float(value) for value in dx)
    _require(dt > 0.0 and abs(dt - abs(spatial_dx)) <= 1.0e-12, "displacement is not future null")
    u_t = math.cosh(rapidity)
    u_x = math.sinh(rapidity)
    normalization = -(u_t**2) + u_x**2
    _require(abs(normalization + 1.0) <= 1.0e-12, "observer is not unit timelike")
    dell = u_t * dt - u_x * spatial_dx
    _require(dell > 0.0, "physical path measure is not positive")
    return dell


def _integrate_segments(
    segments: Sequence[Mapping[str, float]], field: str, rapidity: float
) -> float:
    return sum(
        float(segment[field])
        * physical_path_measure((float(segment["dt"]), float(segment["dx"])), rapidity)
        for segment in segments
    )


def _subdivide(segments: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    fractions = (0.2, 0.3, 0.5)
    return [
        {
            **{key: float(value) for key, value in segment.items()},
            "dt": float(segment["dt"]) * fraction,
            "dx": float(segment["dx"]) * fraction,
        }
        for segment in segments
        for fraction in fractions
    ]


def _source_value(history: str, time: float) -> float:
    if history == "H_STEP":
        return 0.0 if time < 1.0 else 1.0
    if history == "H_PULSE":
        return 1.0 if 1.0 <= time < 3.0 else 0.0
    if history == "H_SINE":
        return 0.5 + 0.4 * math.sin(2.0 * math.pi * time / 4.0)
    raise PersistentRedshiftV2Error(f"unknown history: {history}")


def _source_derivative(history: str, time: float) -> float:
    if history in {"H_STEP", "H_PULSE"}:
        _require(
            all(abs(time - edge) > 1.0e-10 for edge in (1.0, 3.0)), "derivative requested at jump"
        )
        return 0.0
    if history == "H_SINE":
        return 0.4 * (2.0 * math.pi / 4.0) * math.cos(2.0 * math.pi * time / 4.0)
    raise PersistentRedshiftV2Error(f"unknown history: {history}")


def _memory_series(history: str, tau: float) -> tuple[list[float], list[float], list[float]]:
    _require(tau > 0.0, "tau must be positive")
    dt = 0.05
    times = [index * dt for index in range(121)]
    source = [_source_value(history, time) for time in times]
    state = [0.0] * len(times)
    decay = math.exp(-dt / tau)
    for index in range(1, len(times)):
        state[index] = source[index] + (state[index - 1] - source[index]) * decay
    derivative = [
        (source_value - state_value) / tau
        for source_value, state_value in zip(source, state, strict=True)
    ]
    return times, state, derivative


def _segments(
    lengths: Sequence[float],
    u_values: Sequence[float],
    k_values: Sequence[float],
    source_scales: Sequence[float],
    *,
    direction: float = 1.0,
) -> list[dict[str, float]]:
    _require(
        len(lengths) == len(u_values) == len(k_values) == len(source_scales),
        "segment arrays differ",
    )
    return [
        {
            "dt": float(length),
            "dx": direction * float(length),
            "u": float(u_value),
            "sqrt_k": float(k_value),
            "source_scale": float(source_scale),
        }
        for length, u_value, k_value, source_scale in zip(
            lengths, u_values, k_values, source_scales, strict=True
        )
    ]


def _fixture_rows() -> list[dict[str, Any]]:
    common = _segments((0.4, 0.6), (0.45, 0.25), (0.3, 0.5), (0.7, 0.4))
    reverse = list(
        reversed(_segments((0.4, 0.6), (0.45, 0.25), (0.3, 0.5), (0.7, 0.4), direction=-1.0))
    )
    short = _segments((0.5,), (0.3,), (0.2,), (0.35,))
    long = _segments((0.5, 0.5, 0.5), (0.3, 0.35, 0.25), (0.3, 0.8, 0.4), (0.35, 0.9, 0.5))
    gauge = _segments((0.5, 0.5), (0.4, 0.2), (0.3, 0.2), (0.5, 0.3))
    gauge_shift = [{**segment, "u": segment["u"] + 5.0} for segment in gauge]
    lens_a = _segments((0.4, 0.6), (0.4, 0.3), (0.3, 0.4), (0.4, 0.5))
    lens_b = _segments((0.5, 0.7, 0.6), (0.5, 0.6, 0.35), (0.7, 1.0, 0.8), (0.8, 0.9, 0.7))
    zero = _segments((1.0,), (0.0,), (0.0,), (0.0,))

    def row(
        fixture: str,
        case: str,
        path: list[dict[str, float]],
        emit_u: float,
        observe_u: float,
        emit_scale: float,
        observe_scale: float,
        history: str,
        evaluation_time: float,
        frequency: float = 1.0,
        stretch: float = 1.0,
    ) -> dict[str, Any]:
        return {
            "fixture_id": fixture,
            "case_id": case,
            "segments": path,
            "emit_u": emit_u,
            "observe_u": observe_u,
            "emit_source_scale": emit_scale,
            "observe_source_scale": observe_scale,
            "history": history,
            "evaluation_time": evaluation_time,
            "frequency": frequency,
            "time_stretch": stretch,
        }

    return [
        row("F01_ENDPOINT_SWAP", "forward", common, 0.6, 0.0, 0.8, 0.2, "H_SINE", 2.0),
        row("F01_ENDPOINT_SWAP", "reverse", reverse, 0.0, 0.6, 0.2, 0.8, "H_SINE", 2.0),
        row("F02_EQUAL_ENDPOINT_TWO_PATHS", "short", short, 0.2, 0.1, 0.4, 0.3, "H_SINE", 2.0),
        row("F02_EQUAL_ENDPOINT_TWO_PATHS", "long", long, 0.2, 0.1, 0.4, 0.3, "H_SINE", 2.0),
        row("F03_POTENTIAL_ZERO_SHIFT", "base", gauge, 0.3, 0.1, 0.5, 0.2, "H_SINE", 2.0),
        row(
            "F03_POTENTIAL_ZERO_SHIFT",
            "shift_plus_5",
            gauge_shift,
            5.3,
            5.1,
            0.5,
            0.2,
            "H_SINE",
            2.0,
        ),
        row(
            "F04_COMMON_SOURCE_HISTORY", "pulse_active", common, 0.2, 0.0, 0.8, 0.2, "H_PULSE", 2.0
        ),
        row(
            "F04_COMMON_SOURCE_HISTORY", "pulse_memory", common, 0.2, 0.0, 0.8, 0.2, "H_PULSE", 4.0
        ),
        row("F05_TWO_FREQUENCIES", "nu_1", common, 0.15, 0.0, 0.6, 0.2, "H_SINE", 2.0, 1.0),
        row("F05_TWO_FREQUENCIES", "nu_2", common, 0.15, 0.0, 0.6, 0.2, "H_SINE", 2.0, 2.0),
        row("F06_LENS_TWO_IMAGES", "image_A", lens_a, 0.4, 0.0, 0.6, 0.2, "H_SINE", 2.0),
        row("F06_LENS_TWO_IMAGES", "image_B", lens_b, 0.4, 0.0, 0.6, 0.2, "H_SINE", 2.0),
        row("F07_ROUND_TRIP", "outbound", common, 0.25, 0.0, 0.6, 0.2, "H_SINE", 2.0),
        row("F07_ROUND_TRIP", "return", reverse, 0.0, 0.25, 0.2, 0.6, "H_SINE", 2.0),
        row(
            "F08_EXPANSION_TIME_DILATION",
            "z_0p5",
            zero,
            0.0,
            0.0,
            0.0,
            0.0,
            "H_SINE",
            2.0,
            1.0,
            1.5,
        ),
    ]


def _time_index(time: float) -> int:
    index = round(time / 0.05)
    _require(abs(index * 0.05 - time) <= 1.0e-12, "time is off grid")
    return index


def _state_at(history: str, tau: float, time: float, scale: float) -> tuple[float, float]:
    _, state, derivative = _memory_series(history, tau)
    index = _time_index(time)
    return scale * state[index], scale * derivative[index]


def _path_state_integral(row: Mapping[str, Any], tau: float, *, derivative: bool) -> float:
    history = str(row["history"])
    time = float(row["evaluation_time"])
    field_name = "state_probe"
    segments = []
    for segment in row["segments"]:
        state, state_derivative = _state_at(history, tau, time, float(segment["source_scale"]))
        segments.append({**segment, field_name: state_derivative if derivative else state})
    return _integrate_segments(segments, field_name, rapidity=0.0)


def _instantaneous_derivative_integral(row: Mapping[str, Any]) -> float:
    derivative = _source_derivative(str(row["history"]), float(row["evaluation_time"]))
    segments = [
        {**segment, "instant_derivative": float(segment["source_scale"]) * derivative}
        for segment in row["segments"]
    ]
    return _integrate_segments(segments, "instant_derivative", rapidity=0.0)


def _predict_members(closure_id: str, row: Mapping[str, Any]) -> dict[str, float]:
    endpoint = float(row["emit_u"]) - float(row["observe_u"])
    if closure_id == "C00_GR_ENDPOINT":
        return {"BASE": endpoint}
    if closure_id == "C01_EXACT_GRADIENT_PATH":
        exact_line_integral = float(row["observe_u"]) - float(row["emit_u"])
        return {"BASE": -exact_line_integral}
    if closure_id == "C02_POTENTIAL_COLUMN":
        return {"BASE": -_integrate_segments(row["segments"], "u", rapidity=0.0)}
    if closure_id == "C03_TIDAL_CURVATURE_COLUMN":
        return {"BASE": -_integrate_segments(row["segments"], "sqrt_k", rapidity=0.0)}
    if closure_id == "C04_DISPERSIVE_PERSISTENT_MEDIUM":
        value = (
            -_path_state_integral(row, _TAU_FAST, derivative=False) / float(row["frequency"]) ** 2
        )
        return {"TAU_FAST": value}
    if closure_id == "C05_ENDPOINT_MEMORY":
        emit_state, _ = _state_at(
            str(row["history"]),
            _TAU_FAST,
            float(row["evaluation_time"]),
            float(row["emit_source_scale"]),
        )
        observe_state, _ = _state_at(
            str(row["history"]),
            _TAU_FAST,
            float(row["evaluation_time"]),
            float(row["observe_source_scale"]),
        )
        return {"TAU_FAST": endpoint + emit_state - observe_state}
    if closure_id == "C06_PATH_MEMORY_OPACITY":
        return {"TAU_FAST": -_path_state_integral(row, _TAU_FAST, derivative=False)}
    if closure_id == "C07_GENERAL_TIME_VARYING_METRIC_PATH":
        return {
            "G_INST": endpoint + _instantaneous_derivative_integral(row),
            "G_MEM_FAST": endpoint + _path_state_integral(row, _TAU_FAST, derivative=True),
            "G_MEM_SLOW": endpoint + _path_state_integral(row, _TAU_SLOW, derivative=True),
        }
    if closure_id == "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY":
        return {
            "G_MEM_FAST": endpoint + _path_state_integral(row, _TAU_FAST, derivative=True),
            "G_MEM_SLOW": endpoint + _path_state_integral(row, _TAU_SLOW, derivative=True),
        }
    raise PersistentRedshiftV2Error(f"unknown closure: {closure_id}")


def _signature_vectors() -> tuple[list[dict[str, Any]], dict[str, dict[str, tuple[float, ...]]]]:
    fixtures = _fixture_rows()
    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, list[float]]] = {closure_id: {} for closure_id in _CLOSURE_IDS}
    for fixture in fixtures:
        for closure_id in _CLOSURE_IDS:
            for member, prediction in _predict_members(closure_id, fixture).items():
                vectors[closure_id].setdefault(member, []).extend(
                    (prediction, float(fixture["time_stretch"]))
                )
                rows.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "case_id": fixture["case_id"],
                        "closure_id": closure_id,
                        "member_id": member,
                        "delta_log_frequency_unit_probe": format(prediction, ".12e"),
                        "expansion_time_stretch": format(float(fixture["time_stretch"]), ".12e"),
                    }
                )
    frozen = {
        closure_id: {
            member: tuple(round(value, 12) for value in vector)
            for member, vector in members.items()
        }
        for closure_id, members in vectors.items()
    }
    return rows, frozen


def _relation_rows(vectors: Mapping[str, Mapping[str, tuple[float, ...]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(_CLOSURE_IDS):
        for right in _CLOSURE_IDS[left_index + 1 :]:
            left_set = set(vectors[left].values())
            right_set = set(vectors[right].values())
            overlap = left_set & right_set
            if left_set == right_set:
                relation = "EQUIVALENT_ON_EXECUTED_FAMILY"
            elif left_set < right_set:
                relation = "LEFT_CONSTRAINED_SUBFAMILY_OF_RIGHT"
            elif right_set < left_set:
                relation = "RIGHT_CONSTRAINED_SUBFAMILY_OF_LEFT"
            elif overlap:
                relation = "PARTIAL_MEMBER_OVERLAP_ON_EXECUTED_FAMILY"
            else:
                relation = "FINITE_FIXTURE_MEMBER_SEPARATION_ONLY"
            minimum = min(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(left_vector, right_vector, strict=True)))
                for left_vector in left_set
                for right_vector in right_set
            )
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "relation": relation,
                    "shared_member_vectors": len(overlap),
                    "minimum_fixture_vector_distance": format(minimum, ".12e"),
                    "global_identifiability_claimed": False,
                }
            )
    _require(
        next(
            row
            for row in rows
            if row["left"] == _CLOSURE_IDS[0] and row["right"] == _CLOSURE_IDS[1]
        )["relation"]
        == "EQUIVALENT_ON_EXECUTED_FAMILY",
        "C00/C01 equivalence missing",
    )
    _require(
        next(
            row
            for row in rows
            if row["left"] == _CLOSURE_IDS[7] and row["right"] == _CLOSURE_IDS[8]
        )["relation"]
        == "RIGHT_CONSTRAINED_SUBFAMILY_OF_LEFT",
        "C08/C07 nesting missing",
    )
    return rows


def _path_measure_audit() -> list[dict[str, Any]]:
    rows = []
    congruence_differences = 0
    for fixture in _fixture_rows():
        for rapidity in (0.0, 0.4):
            original = _integrate_segments(fixture["segments"], "source_scale", rapidity)
            subdivided = _integrate_segments(
                _subdivide(fixture["segments"]), "source_scale", rapidity
            )
            error = abs(original - subdivided)
            _require(error <= 1.0e-12, "reparameterization/subdivision invariance failed")
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "case_id": fixture["case_id"],
                    "observer_rapidity": rapidity,
                    "original_scalar_integral": format(original, ".12e"),
                    "subdivided_scalar_integral": format(subdivided, ".12e"),
                    "absolute_error": format(error, ".12e"),
                    "reparameterization_invariant": True,
                    "congruence_independent_claimed": False,
                }
            )
        stationary = float(rows[-2]["original_scalar_integral"])
        boosted = float(rows[-1]["original_scalar_integral"])
        congruence_differences += int(abs(stationary - boosted) > 1.0e-12)
    _require(congruence_differences > 0, "congruence dependence was not exposed")
    return rows


def _fixture_channel_count(
    closure_id: str, vectors: Mapping[str, Mapping[str, tuple[float, ...]]]
) -> int:
    base = next(iter(vectors["C00_GR_ENDPOINT"].values()))
    fixture_ids = [row["fixture_id"] for row in _fixture_rows()]
    member_vectors = list(vectors[closure_id].values())
    count = 0
    for fixture_id in dict.fromkeys(fixture_ids):
        case_indices = [index for index, value in enumerate(fixture_ids) if value == fixture_id]
        positions = [position for index in case_indices for position in (2 * index, 2 * index + 1)]
        if any(
            any(abs(vector[position] - base[position]) > 1.0e-12 for position in positions)
            for vector in member_vectors
        ):
            count += 1
    return count


def _triage_rows(
    config: Mapping[str, Any], vectors: Mapping[str, Mapping[str, tuple[float, ...]]]
) -> list[dict[str, Any]]:
    energy_unclosed = {
        "C03_TIDAL_CURVATURE_COLUMN",
        "C04_DISPERSIVE_PERSISTENT_MEDIUM",
        "C05_ENDPOINT_MEMORY",
        "C06_PATH_MEMORY_OPACITY",
        "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY",
    }
    source_route = {
        "C03_TIDAL_CURVATURE_COLUMN": 1,
        "C04_DISPERSIVE_PERSISTENT_MEDIUM": 0,
        "C05_ENDPOINT_MEMORY": 1,
        "C06_PATH_MEMORY_OPACITY": 1,
        "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY": 1,
    }
    closure_map = {row["id"]: row for row in config["closures"]}
    rows = []
    for closure_id in _CANDIDATES:
        closure = closure_map[closure_id]
        metrics = {
            "dimensionally_closed": 1,
            "reparameterization_safe_or_no_path_integral": 1,
            "potential_zero_gauge_safe": int(bool(closure["gauge_invariant"])),
            "executable_state_equation_or_no_state": 1,
            "causal_when_stateful": 1,
            "has_frozen_empirical_source_route": source_route[closure_id],
        }
        fixture_channels = _fixture_channel_count(closure_id, vectors)
        penalties = {
            "exact_equivalence_to_known_comparator": 0,
            "nested_in_general_neighbor": int(closure_id == "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY"),
            "unclosed_energy_momentum": int(closure_id in energy_unclosed),
        }
        score = sum(metrics.values()) + fixture_channels - sum(penalties.values())
        rows.append(
            {
                "closure_id": closure_id,
                **metrics,
                "finite_fixture_channel_count": fixture_channels,
                **penalties,
                "structural_preflight_score": score,
                "score_kind": config["structural_triage"]["label"],
                "empirical_rows_used": 0,
            }
        )
    ordered_scores = sorted({int(row["structural_preflight_score"]) for row in rows}, reverse=True)
    tier_by_score = {score: index + 1 for index, score in enumerate(ordered_scores)}
    for row in rows:
        row["triage_tier"] = tier_by_score[int(row["structural_preflight_score"])]
    return sorted(rows, key=lambda row: (int(row["triage_tier"]), str(row["closure_id"])))


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "cannot write empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().replace("\r\n", "\n").encode()


def _report(
    config: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    triage: Sequence[Mapping[str, Any]],
) -> bytes:
    equivalent = sum(row["relation"] == "EQUIVALENT_ON_EXECUTED_FAMILY" for row in relations)
    nested = sum("SUBFAMILY" in str(row["relation"]) for row in relations)
    tier_lines = "\n".join(
        f"- Tier {row['triage_tier']}: {row['closure_id']} (structural preflight {row['structural_preflight_score']}; {row['finite_fixture_channel_count']} finite-fixture channels)"
        for row in triage
    )
    text = f"""# Persistent time-well redshift closures — strict-audit successor

## Result

**PASS** for a bounded, response-free mathematical preflight. **SOURCE_BLOCKED** for the Galileo development run until all fourteen frozen ESOC `repro4` files have payload hashes, format/header receipts, and source-only circular-control PRNs. No observational payload row was opened and no empirical score was computed.

This successor makes no complete-taxonomy or global-identifiability claim. It executes nine declared closures on common source histories and common zero initial conditions. Across {len(relations)} pair relations it finds {equivalent} finite-family equivalence and {nested} explicit nesting relation(s), while every remaining difference is labeled only finite-fixture member separation or overlap.

## Two corrected structural results

1. `C00_GR_ENDPOINT` and `C01_EXACT_GRADIENT_PATH` are the same executed observable family. The exact one-form integral is endpoint redshift, not accumulated new physics.
2. `C08_CAUSAL_METRIC_MEMORY_SUBFAMILY` is exactly nested inside `C07_GENERAL_TIME_VARYING_METRIC_PATH`: its two causal-memory member vectors are a strict subset of the three declared C07 member vectors. C08 may be a more predictive physical restriction, but it is not independently distinguishable from a general C07 family that permits the same state.

## Physical path measure

Every cumulative closure now uses `dell=-uhat_a dx^a`, where `uhat` is a future-directed unit timelike observer congruence. Because `dx^a` is a geometric one-form displacement, the integral is unchanged by orientation-preserving ray reparameterization; the executable audit also obtains identical values after nonuniform segment subdivision. The measure is intentionally **congruence-dependent**: changing observers changes locally measured photon frequency and path length. No unspecified affine `ds` remains.

## Computed structural triage

The following tiers come from the frozen equal-weight binary metrics, finite-fixture channel count, and declared penalties. They are not data scores, model fits, truth probabilities, or novelty rankings.

{tier_lines}

## Retained failures and controls

- C02 still fails additive-potential-zero invariance and has no energy sink.
- Same-sign cumulative C03/C04/C06 laws still owe energy-momentum and round-trip consequences.
- C04 remains plasma-adjacent and needs raw multi-frequency measurements; ionosphere-free CLK products cannot establish chromaticity.
- All closures retain cosmological expansion/time dilation rather than replacing it. Distance duality/photon number, dust, plasma, kinematics, source evolution, moving-lens, delay, orbit, thermal, magnetic, and processing controls remain mandatory.

## Frozen next falsifier

The exact public one-week manifest contains seven `.clk` and seven `.sp3` files at `ftp://gssc.esa.int/esa/great/repro4/`, with E14/GSAT0202 development and E18/GSAT0201 untouched holdout. The epoch intersection, five-tau grid, nuisance design, AR(1) GLS likelihood, E14-only selection, E18 transfer statistic, circular controls, and phase-scrambled controls are frozen in the config. The source remains blocked because listing metadata is not a payload receipt and the public clock products lack temperature, magnetic, and individual-clock identity needed for a discovery claim.

## Publication boundary

A bounded methods/theory note could report the invariant-path formulation, the endpoint equivalence, and the C08-within-C07 nesting result. Historical novelty, covariant completion, energy closure, empirical support, and a gravity discovery are not established.
"""
    return text.encode()


def _artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    signature_rows, vectors = _signature_vectors()
    relations = _relation_rows(vectors)
    path_audit = _path_measure_audit()
    triage = _triage_rows(config, vectors)
    source_preflight = {
        "schema": "invariant-open-gravity-galileo-memory-preflight-2.0",
        "metadata_manifest": config["galileo_metadata_manifest"],
        "frozen_analysis": config["galileo_frozen_analysis"],
        "mandatory_controls": config["mandatory_controls"],
        "observational_payload_rows_opened": 0,
        "empirical_scores_computed": 0,
        "content_sha256": "",
    }
    source_preflight["content_sha256"] = content_sha256({**source_preflight, "content_sha256": ""})
    return {
        "finite-fixture-outputs.csv": _csv_bytes(signature_rows),
        "structural-relations.csv": _csv_bytes(relations),
        "physical-path-measure-audit.csv": _csv_bytes(path_audit),
        "structural-triage.csv": _csv_bytes(triage),
        "galileo-source-and-likelihood-preflight.json": _canonical(source_preflight),
        "strict-audit-report.md": _report(config, relations, triage),
    }


def _artifact_index(payloads: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "path": (ARTIFACT_DIR / name).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]


def build_receipt() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    bindings = _validate_bindings(config)
    signatures, vectors = _signature_vectors()
    relations = _relation_rows(vectors)
    triage = _triage_rows(config, vectors)
    path_audit = _path_measure_audit()
    payloads = _artifact_payloads(config)
    relation_counts: dict[str, int] = {}
    for row in relations:
        relation_counts[str(row["relation"])] = relation_counts.get(str(row["relation"]), 0) + 1
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_STRICT_RESPONSE_FREE_CLOSURES_GALILEO_SOURCE_BLOCKED_ON_PAYLOAD_RECEIPTS",
        "decision": "RETAIN_C05_AND_C08_CAUSAL_RESTRICTION_C08_NESTED_IN_C07_ADVANCE_ONLY_AFTER_EXACT_SOURCE_RECEIPT",
        "input_sha256": bindings,
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "summary": {
            "declared_closures": len(config["closures"]),
            "closure_taxonomy_complete": False,
            "finite_fixture_output_rows": len(signatures),
            "structural_pair_relations": len(relations),
            "relation_counts": relation_counts,
            "global_identifiability_established": False,
            "physical_path_audit_rows": len(path_audit),
            "maximum_reparameterization_error": max(
                float(row["absolute_error"]) for row in path_audit
            ),
            "congruence_independence_claimed": False,
            "galileo_manifest_files": len(config["galileo_metadata_manifest"]["directory_entries"]),
            "galileo_manifest_bytes": sum(
                int(row["bytes"])
                for row in config["galileo_metadata_manifest"]["directory_entries"]
            ),
            "observational_payload_rows_opened": 0,
            "empirical_scores_computed": 0,
            "structural_triage": triage,
            "artifact_index": _artifact_index(payloads),
        },
        "retained_failures": [
            {
                "closure_id": "C02_POTENTIAL_COLUMN",
                "reason": "additive-potential gauge failure plus unclosed energy and same-sign round trip",
            },
            {
                "closure_ids": [
                    "C03_TIDAL_CURVATURE_COLUMN",
                    "C04_DISPERSIVE_PERSISTENT_MEDIUM",
                    "C06_PATH_MEMORY_OPACITY",
                ],
                "reason": "genuine physical-path accumulation remains conditional on energy-momentum completion",
            },
        ],
        "source_status": config["galileo_metadata_manifest"]["source_status"],
        "claim_boundary": config["claim_boundary"],
        "access_accounting": config["access_contract"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt, payloads


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_package(*, overwrite: bool = False) -> dict[str, Any]:
    receipt, payloads = build_receipt()
    targets = [OUTPUT_PATH, *(ARTIFACT_DIR / name for name in payloads)]
    if not overwrite:
        _require(not any(path.exists() for path in targets), "output already exists")
    for name, payload in payloads.items():
        _atomic_write(ARTIFACT_DIR / name, payload)
    _atomic_write(OUTPUT_PATH, _canonical(receipt))
    return receipt


def check_package() -> dict[str, Any]:
    observed = _read_json(OUTPUT_PATH, "v2 receipt")
    _require(type(observed) is dict, "receipt is not an object")
    rebuilt, payloads = build_receipt()
    _require(observed == rebuilt, "receipt differs from deterministic rebuild")
    _require(
        observed["content_sha256"] == content_sha256({**observed, "content_sha256": ""}),
        "receipt content hash changed",
    )
    for name, payload in payloads.items():
        path = ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact changed: {name}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        receipt = write_package(overwrite=arguments.overwrite)
    else:
        receipt = check_package()
    print(
        json.dumps({"status": receipt["status"], "decision": receipt["decision"]}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
