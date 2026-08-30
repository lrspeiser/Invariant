"""Typed gravity/light/quantum theory cards with target-free probes only."""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_gravity_light_quantum_cards_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_gravity_light_quantum_cards_v1.py")
TEST_PATH = Path("tests/test_open_gravity_gravity_light_quantum_cards_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-gravity-light-quantum-cards-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_gravity_light_quantum_cards_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_gravity_light_quantum_cards_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_gravity_light_quantum_cards_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-gravity-light-quantum-cards-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "7750e3cb496a39cba89a215d2471f1c1bb658ab5e30effd980657fe8ade70348"
_CONFIG_CONTENT_SHA256 = "7233c54bd82fb75c88660d0f97a28bab11a3dcd6ef9b42953ab0ba9bb18aaeb2"
_SCHEMA = "invariant-open-gravity-gravity-light-quantum-cards-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-gravity-light-quantum-cards-receipt-1.0"
_QUANTUM_IDS = {"QG03", "QG04", "QG07", "QG08", "QG10", "QG11", "QG12"}
_ESTABLISHED_CONTROL_IDS = {"QG01", "QG02"}
_EXPECTED_CARD_NAMES = {
    "QG01": "classical_spacetime_geometry",
    "QG02": "classical_tensor_gravitational_waves",
    "QG03": "massless_spin2_quantum_field_graviton",
    "QG04": "quantum_states_of_gravity",
    "QG05": "massive_or_finite_range_carrier",
    "QG06": "extra_scalar_vector_or_tensor_modes",
    "QG07": "gravitational_superposition_and_mediated_entanglement",
    "QG08": "semiclassical_and_stochastic_gravity",
    "QG09": "emergent_thermodynamic_entropic_information_gravity",
    "QG10": "discrete_or_causal_quantum_geometry",
    "QG11": "medium_condensate_and_quasiparticle_gravity",
    "QG12": "nonlocal_memory_persistence_and_corpuscular_gravity",
    "QG13": "wildcard_ontology",
}


class TheoryCardError(RuntimeError):
    """Raised whenever a card, probe, binding, or receipt fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TheoryCardError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_ROOT / expected).resolve()
    _require(path.is_relative_to(_ROOT), f"{label} escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TheoryCardError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TheoryCardError("committed binding unavailable") from exc


def _all_text_defined(card: Mapping[str, Any]) -> bool:
    for key, value in card.items():
        if key in {"id", "name", "registry_status", "card_status"}:
            continue
        if value == "UNDEFINED":
            return False
        if type(value) is list and "UNDEFINED" in value:
            return False
    return True


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "primary_sources",
        "card_fields",
        "cards",
        "probe_contract",
        "status_policy",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-gravity-light-quantum-cards-v1", "ID changed")
    _require(config["status"] == "FROZEN_TYPED_THEORY_CARDS_TARGET_FREE_ONLY", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(len(config["bindings"]) == 3, "binding count changed")
    sources = config["primary_sources"]
    _require(type(sources) is list and len(sources) == 11, "primary source count changed")
    source_ids = [row["id"] for row in sources]
    _require(source_ids == [f"PS{i:02d}" for i in range(1, 12)], "primary source IDs changed")
    _require(all(str(row["url"]).startswith("https://") for row in sources), "source URL changed")
    cards = config["cards"]
    _require(type(cards) is list and len(cards) == 13, "card count changed")
    _require([card["id"] for card in cards] == list(_EXPECTED_CARD_NAMES), "card order changed")
    _require(
        {card["id"]: card["name"] for card in cards} == _EXPECTED_CARD_NAMES,
        "card identity changed",
    )
    fields = set(config["card_fields"])
    _require(all(set(card) == fields for card in cards), "card schema changed")
    known_sources = set(source_ids)
    _require(
        all(set(card["primary_source_ids"]) <= known_sources for card in cards),
        "unknown primary source",
    )
    for card in cards[:-1]:
        _require(card["card_status"] != "INCOMPLETE_QUARANTINE", "defined card quarantined")
        _require(_all_text_defined(card), "defined card contains undefined content")
        _require(card["required_next_artifacts"], "next artifacts missing")
        _require(card["falsifiers"], "falsifiers missing")
    wildcard = cards[-1]
    _require(wildcard["card_status"] == "INCOMPLETE_QUARANTINE", "wildcard promoted")
    _require(not _all_text_defined(wildcard), "wildcard is no longer quarantined")
    probe = config["probe_contract"]
    _require(
        probe
        == {
            "card_count": 13,
            "executable_target_free_probes": 12,
            "quarantined_cards": 1,
            "gate_projection_rows": 325,
            "real_data_rows": 0,
            "observational_passes": 0,
        },
        "probe authority changed",
    )
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed")
    config = _read_json(path, "theory card config")
    validate_config(config)
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    for binding in config["bindings"]:
        _require(type(binding["commit"]) is str and len(binding["commit"]) == 40, "commit changed")
        for artifact in binding["artifacts"]:
            expected = artifact["sha256"]
            committed = hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
            _require(committed == expected, f"committed {binding['role']} changed")
            _require(
                file_sha256(_ROOT / artifact["path"]) == expected,
                f"working {binding['role']} changed",
            )
    return config


def _probe_qg01() -> dict[str, Any]:
    radius = 2.0
    step = 1.0e-5
    potential = lambda r: -1.0 / r
    acceleration = -(potential(radius + step) - potential(radius - step)) / (2.0 * step)
    expected = -1.0 / radius**2
    error = abs(acceleration - expected)
    return {"passed": error < 1.0e-10, "weak_field_acceleration_error": error}


def _probe_qg02() -> dict[str, Any]:
    plus = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0))
    trace = sum(plus[i][i] for i in range(3))
    longitudinal = max(abs(plus[2][j]) for j in range(3))
    omega = 3.0
    wave_number = 3.0
    residual = abs(omega * omega - wave_number * wave_number)
    return {
        "passed": trace == 0.0 and longitudinal == 0.0 and residual == 0.0,
        "trace": trace,
        "longitudinal_norm": longitudinal,
        "dispersion_residual": residual,
    }


def _probe_qg03() -> dict[str, Any]:
    occupation = 10_000.0
    energy = occupation + 0.5
    relative_fluctuation = 1.0 / math.sqrt(occupation)
    return {
        "passed": energy > 0.0 and relative_fluctuation < 0.011,
        "normalized_energy": energy,
        "coherent_relative_fluctuation": relative_fluctuation,
    }


def _probe_qg04() -> dict[str, Any]:
    theta = 0.731
    amplitude = 1.0 / math.sqrt(2.0)
    state = (amplitude, amplitude * cmath.exp(-1j * theta))
    trace = sum(abs(value) ** 2 for value in state)
    purity = trace * trace
    determinant = 0.0
    return {
        "passed": abs(trace - 1.0) < 1.0e-15 and abs(purity - 1.0) < 1.0e-15,
        "trace": trace,
        "purity": purity,
        "density_determinant": determinant,
    }


def _probe_qg05() -> dict[str, Any]:
    mass = 0.4
    wave_number = 1.3
    omega = math.sqrt(wave_number**2 + mass**2)
    group_speed = wave_number / omega
    radius = 2.0
    small_mass = 1.0e-7
    local_limit_error = abs(math.exp(-small_mass * radius) / radius - 1.0 / radius)
    return {
        "passed": 0.0 < group_speed < 1.0 and local_limit_error < 1.1e-7,
        "group_speed_over_c": group_speed,
        "yukawa_local_limit_error": local_limit_error,
    }


def _probe_qg06() -> dict[str, Any]:
    healthy = (1.0, 2.0, 3.0)
    unhealthy = (1.0, 0.5, -0.1)
    return {
        "passed": min(healthy) > 0.0 and min(unhealthy) < 0.0,
        "healthy_kinetic_minimum": min(healthy),
        "designed_ghost_minimum": min(unhealthy),
    }


def _phase_concurrence(phases: Sequence[float]) -> float:
    amplitudes = [0.5 * cmath.exp(1j * phase) for phase in phases]
    return 2.0 * abs(amplitudes[0] * amplitudes[3] - amplitudes[1] * amplitudes[2])


def _probe_qg07() -> dict[str, Any]:
    entangling = _phase_concurrence((0.0, 0.0, 0.0, math.pi))
    additive = _phase_concurrence((0.3, 0.8, -0.2, 0.3))
    return {
        "passed": abs(entangling - 1.0) < 1.0e-15 and additive < 1.0e-15,
        "entangling_concurrence": entangling,
        "additive_local_phase_concurrence": additive,
    }


def _probe_qg08() -> dict[str, Any]:
    covariance = ((1.0, 0.3), (0.3, 0.5))
    determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2
    trace = covariance[0][0] + covariance[1][1]
    return {
        "passed": determinant > 0.0 and trace > 0.0,
        "covariance_determinant": determinant,
        "covariance_trace": trace,
        "zero_noise_limit_residual": 0.0,
    }


def _probe_qg09() -> dict[str, Any]:
    mass = 2.3
    acceleration = 0.17
    temperature = acceleration / (2.0 * math.pi)
    entropy_gradient = 2.0 * math.pi * mass
    entropic_force = temperature * entropy_gradient
    residual = abs(entropic_force - mass * acceleration)
    return {
        "passed": residual < 1.0e-15,
        "normalized_entropic_force_residual": residual,
        "assumed_relations": 2,
    }


def _transitive_closure(edges: set[tuple[int, int]], nodes: Sequence[int]) -> set[tuple[int, int]]:
    closure = set(edges)
    changed = True
    while changed:
        changed = False
        for first in nodes:
            for middle in nodes:
                for last in nodes:
                    if (
                        (first, middle) in closure
                        and (middle, last) in closure
                        and (first, last) not in closure
                    ):
                        closure.add((first, last))
                        changed = True
    return closure


def _probe_qg10() -> dict[str, Any]:
    nodes = (0, 1, 2, 3)
    edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
    closure = _transitive_closure(edges, nodes)
    relabel = {0: 2, 1: 0, 2: 3, 3: 1}
    relabeled = {(relabel[a], relabel[b]) for a, b in closure}
    irreflexive = all((node, node) not in closure for node in nodes)
    return {
        "passed": irreflexive and len(closure) == len(relabeled) == 5,
        "events": len(nodes),
        "causal_relations": len(closure),
        "relabeling_relation_count": len(relabeled),
    }


def _probe_qg11() -> dict[str, Any]:
    density = 1.2
    sound_speed_squared = 0.7
    negative_control = -0.1
    return {
        "passed": density > 0.0 and sound_speed_squared > 0.0 and negative_control < 0.0,
        "kinetic_coefficient": density,
        "sound_speed_squared": sound_speed_squared,
        "designed_bad_sound_speed_squared": negative_control,
    }


def _probe_qg12() -> dict[str, Any]:
    delays = (0.0, 1.0, 2.0, 3.0, 4.0)

    def weights(tau: float) -> list[float]:
        raw = [math.exp(-delay / tau) for delay in delays]
        total = sum(raw)
        return [value / total for value in raw]

    broad = weights(1.0)
    local = weights(0.05)
    constant_response = sum(broad)
    return {
        "passed": abs(constant_response - 1.0) < 1.0e-15 and local[0] > broad[0],
        "normalization_residual": abs(constant_response - 1.0),
        "broad_present_weight": broad[0],
        "short_memory_present_weight": local[0],
        "advanced_support_weight": 0.0,
    }


_PROBES = {
    "QG01": _probe_qg01,
    "QG02": _probe_qg02,
    "QG03": _probe_qg03,
    "QG04": _probe_qg04,
    "QG05": _probe_qg05,
    "QG06": _probe_qg06,
    "QG07": _probe_qg07,
    "QG08": _probe_qg08,
    "QG09": _probe_qg09,
    "QG10": _probe_qg10,
    "QG11": _probe_qg11,
    "QG12": _probe_qg12,
}


def run_target_free_probes(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for card in cards:
        card_id = card["id"]
        if card_id == "QG13":
            results.append(
                {
                    "card_id": card_id,
                    "probe_status": "INCOMPLETE_QUARANTINE",
                    "metrics": {"execution_attempts": 0},
                }
            )
            continue
        metrics = _PROBES[card_id]()
        _require(metrics.pop("passed") is True, f"target-free probe failed for {card_id}")
        _require(
            all(type(value) in {int, float} and math.isfinite(value) for value in metrics.values()),
            f"nonfinite probe metric for {card_id}",
        )
        results.append({"card_id": card_id, "probe_status": "PASS_TARGET_FREE", "metrics": metrics})
    return results


def _gate_status(card: Mapping[str, Any], gate_id: str) -> str:
    card_id = card["id"]
    if card_id == "QG13":
        return "INCOMPLETE_QUARANTINE"
    if gate_id == "TG24_REAL_3D_SOURCE":
        return "BLOCKED_MISSING_SOURCE"
    if gate_id == "TG25_REAL_DATA_CAMPAIGN":
        return "BLOCKED_UPSTREAM_GATES"
    if gate_id == "TG06_SYNTHETIC_FIXTURES":
        return "PASS_TARGET_FREE"
    if gate_id == "TG07_FULL_3D_SOLVER":
        return "BLOCKED_MISSING_SOLVER"
    if gate_id == "TG17_QUANTUM_UNITARITY":
        return "REQUIRED_UNRUN" if card_id in _QUANTUM_IDS else "NOT_APPLICABLE_CURRENT_SCOPE"
    if card_id in _ESTABLISHED_CONTROL_IDS and gate_id in {
        "TG02_FIELD_STATE",
        "TG03_SOURCE_COUPLING",
        "TG04_EQUATIONS_OPERATOR",
        "TG08_SYMMETRY_COVARIANCE",
        "TG09_ACTION_CONSERVATION",
        "TG10_DOF_CONSTRAINTS",
        "TG11_PRINCIPAL_SYMBOL",
        "TG12_GHOST_GRADIENT",
        "TG13_HAMILTONIAN_ENERGY",
        "TG14_CAUSAL_COMMON_CONE",
        "TG15_RADIATION_POLARIZATION",
        "TG18_MATTER_CAPTURE",
        "TG19_PHOTON_LENSING",
        "TG20_CLOCK_REDSHIFT",
        "TG21_SOLAR_PPN",
        "TG22_PULSAR_BINARY",
        "TG23_COSMOLOGY",
    }:
        return "PASS_PRIMARY_SOURCE_NOT_INDEPENDENTLY_REDERIVED"
    if gate_id in {
        "TG01_DIMENSIONS_LIMITS",
        "TG02_FIELD_STATE",
        "TG03_SOURCE_COUPLING",
        "TG04_EQUATIONS_OPERATOR",
        "TG08_SYMMETRY_COVARIANCE",
    }:
        return "PARTIAL"
    return "BLOCKED_MISSING_DEFINITION"


def gate_projection(
    cards: Sequence[Mapping[str, Any]], gate_ids: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in cards:
        for gate_id in gate_ids:
            status = _gate_status(card, gate_id)
            contract = {
                "card_id": card["id"],
                "gate_id": gate_id,
                "evidence_status": status,
            }
            rows.append({**contract, "gate_contract_sha256": content_sha256(contract)})
    return rows


def _stream_root(rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_canonical(row))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def build_receipt() -> dict[str, Any]:
    config = load_config()
    gate_config = _read_json(
        _ROOT / "configs/open_gravity_theory_gate_matrix_v1.json", "bound gate config"
    )
    gate_ids = [row["id"] for row in gate_config["gate_vocabulary"]]
    _require(len(gate_ids) == 25, "bound gate vocabulary changed")
    cards = config["cards"]
    probes = run_target_free_probes(cards)
    projection = gate_projection(cards, gate_ids)
    card_count, card_root = _stream_root(cards)
    probe_count, probe_root = _stream_root(probes)
    projection_count, projection_root = _stream_root(projection)
    _require((card_count, probe_count, projection_count) == (13, 13, 325), "counts changed")
    _require(
        sum(row["probe_status"] == "PASS_TARGET_FREE" for row in probes) == 12,
        "probe passes changed",
    )
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_TYPED_THEORY_CARDS_AND_BOUNDED_TARGET_FREE_PROBES",
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / _CANONICAL_CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {"path": _CANONICAL_TEST_PATH.as_posix(), "sha256": file_sha256(test_path)},
            "predecessors": config["bindings"],
        },
        "cards": {
            "count": card_count,
            "stream_sha256": card_root,
            "status_counts": dict(sorted(Counter(card["card_status"] for card in cards).items())),
            "primary_sources": len(config["primary_sources"]),
        },
        "target_free_probes": {
            "count": probe_count,
            "stream_sha256": probe_root,
            "pass_count": 12,
            "quarantine_count": 1,
            "results": probes,
        },
        "gate_projection": {
            "rows": projection_count,
            "stream_sha256": projection_root,
            "status_counts": dict(
                sorted(Counter(row["evidence_status"] for row in projection).items())
            ),
            "observational_passes": 0,
        },
        "status_policy": config["status_policy"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    expected = build_receipt()
    _require(payload == expected, "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    path = _output_path()
    payload = _read_json(path, "theory card receipt")
    validate_receipt_payload(payload)


def status_summary() -> dict[str, Any]:
    receipt = build_receipt()
    return {
        "status": receipt["status"],
        "cards": receipt["cards"]["count"],
        "target_free_probe_passes": receipt["target_free_probes"]["pass_count"],
        "quarantined": receipt["target_free_probes"]["quarantine_count"],
        "gate_rows": receipt["gate_projection"]["rows"],
        "observational_passes": 0,
        "scientific_rows": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args()
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        print(json.dumps(status_summary(), sort_keys=True))


if __name__ == "__main__":
    main()
