"""Independent read-only audit for the frozen Lane 4 hydro/DMO matrix.

This script deliberately does not import the subject matrix module.  It reads
only its JSON/JSONL/synthetic-NPZ artifacts and independently reimplements the
six frozen dynamical adapters and scorer.  It never opens an HDF5 file.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1"
OUT = ROOT / "runs" / "gravity" / PACKAGE
CONFIG = (
    ROOT
    / "configs"
    / ("open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.json")
)
PARAMETERS = (
    ROOT
    / "configs"
    / (
        "open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1."
        "parameters.schema.json"
    )
)
MODULE = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / ("open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.py")
)
TEST = (
    ROOT
    / "tests"
    / (
        "test_open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.py"
    )
)

EXPECTED_RAW = {
    CONFIG: "ce756b0f095f758b2c92af36e0bb1d1c962c8c52e42fc1f2dbc10705fbac9e00",
    PARAMETERS: "918503b7f16cef4414a56767916cdf5a5628884df4f30befdfe6d80ea7467021",
    MODULE: "aca83d89ae69bc24ce2070c70087ff5e8d5effcd9b4f2050b056808a65a75c05",
    TEST: "d0b4550184c88451df909f08ae2f7bed29cfc2742f106f212528a145158b8cf3",
    OUT / "receipt.json": "904e285f7f93c477a8696877f7868452b342dfd7107578ce6053b260be2600b9",
}

EXECUTABLE = (
    "CM01_CONSERVATIVE_THREE_BODY_CAPTURE",
    "DC00_NEWTONIAN_FOCUSING_CONTROL",
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL",
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH",
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH",
    "DC07_COMPRESSION_GATED_BATH",
)
PREDICTIONS = (
    "prediction.vector.entropy",
    "prediction.vector.receiver-energy",
    "prediction.vector.separation",
    "prediction.vector.visible-pair-energy",
)
RESPONSES = {value: value.replace("prediction.", "response.synthetic-", 1) for value in PREDICTIONS}
SIGMA_KEY = {
    "prediction.vector.entropy": "entropy_sigma",
    "prediction.vector.receiver-energy": "receiver_energy_sigma",
    "prediction.vector.separation": "separation_sigma",
    "prediction.vector.visible-pair-energy": "energy_sigma",
}
MECHANISM_CODE = {formula_id: index for index, formula_id in enumerate(EXECUTABLE)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def raw_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.name.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def array_key(kind: str, scenario_id: str, element_id: str) -> str:
    return "__".join(
        [kind, scenario_id.replace(".", "_"), element_id.replace(".", "_").replace("-", "_")]
    )


def scalar(features: dict[str, np.ndarray], element_id: str) -> float:
    value = np.asarray(features[element_id], dtype=np.float64)
    require(value.shape == (1,) and np.all(np.isfinite(value)), f"bad scalar {element_id}")
    return float(value[0])


def two_body_derivative(
    state: np.ndarray,
    *,
    m1: float,
    m2: float,
    activation_scale: float,
    temperature: float,
    force_scale: float,
    gamma: float,
    mode: str,
    tau: tuple[float, ...],
    weights: tuple[float, ...],
    compression_speed_scale: float,
) -> np.ndarray:
    r, radial_momentum, angular_momentum, _receiver, _entropy, *memory = state
    require(r > 0.05 and np.all(np.isfinite(state)), "invalid two-body state")
    mu = m1 * m2 / (m1 + m2)
    radial_velocity = radial_momentum / mu
    activation = math.exp(-((r / activation_scale) ** 2))
    if mode in {"DC00", "DC01"}:
        effective_memory = 0.0
    elif mode in {"DC05", "DC06"}:
        effective_memory = sum(weight * value for weight, value in zip(weights, memory))
    elif mode == "DC07":
        effective_memory = max(0.0, -radial_velocity / compression_speed_scale) * activation
    else:
        raise AssertionError(f"unknown mode {mode}")
    heat_rate = gamma * effective_memory * radial_velocity * radial_velocity
    return np.asarray(
        [
            radial_velocity,
            angular_momentum * angular_momentum / (mu * r**3)
            - force_scale * m1 * m2 / r**2
            - gamma * effective_memory * radial_velocity,
            0.0,
            heat_rate,
            heat_rate / temperature,
            *((activation - value) / scale for value, scale in zip(memory, tau)),
        ],
        dtype=np.float64,
    )


def visible_energy(state: np.ndarray, m1: float, m2: float, force_scale: float) -> float:
    r, radial_momentum, angular_momentum = state[:3]
    mu = m1 * m2 / (m1 + m2)
    return float(
        radial_momentum**2 / (2.0 * mu)
        + angular_momentum**2 / (2.0 * mu * r**2)
        - force_scale * m1 * m2 / r
    )


def simulate_two_body(
    features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    times = np.asarray(features["source.vector.encounter-time"], dtype=np.float64)
    mode = parameters["mechanism"]
    mass_ratio = scalar(features, "source.scalar.mass-ratio")
    total_mass = scalar(features, "source.scalar.total-mass")
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    mu = m1 * m2 / total_mass
    initial_radial_velocity = scalar(features, "source.scalar.initial-radial-velocity")
    tau = tuple(float.fromhex(value) for value in parameters.get("tau", ()))
    weights = tuple(float.fromhex(value) for value in parameters.get("weights", ()))
    state = np.asarray(
        [
            scalar(features, "source.scalar.initial-separation"),
            mu * initial_radial_velocity,
            mu * scalar(features, "source.scalar.impact-parameter") * abs(initial_radial_velocity),
            0.0,
            0.0,
            *(scalar(features, "source.scalar.history-memory") for _ in tau),
        ],
        dtype=np.float64,
    )
    force_scale = float.fromhex(parameters["force_scale"])
    kwargs = {
        "m1": m1,
        "m2": m2,
        "activation_scale": scalar(features, "source.scalar.activation-scale"),
        "temperature": scalar(features, "source.scalar.temperature"),
        "force_scale": force_scale,
        "gamma": float.fromhex(parameters["gamma"]),
        "mode": mode,
        "tau": tau,
        "weights": weights,
        "compression_speed_scale": float.fromhex(
            parameters.get("compression_speed_scale", "0x1.0000000000000p+0")
        ),
    }
    energy: list[float] = []
    separation: list[float] = []
    receiver: list[float] = []
    entropy: list[float] = []
    angular_initial = float(state[2])
    max_angular_drift = 0.0
    min_separation = float(state[0])
    current_time = 0.0
    dt = 0.01
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(dt, float(target) - current_time)
            k1 = two_body_derivative(state, **kwargs)
            k2 = two_body_derivative(state + 0.5 * step * k1, **kwargs)
            k3 = two_body_derivative(state + 0.5 * step * k2, **kwargs)
            k4 = two_body_derivative(state + step * k3, **kwargs)
            state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            current_time += step
            min_separation = min(min_separation, float(state[0]))
            max_angular_drift = max(max_angular_drift, abs(float(state[2]) - angular_initial))
        energy.append(visible_energy(state, m1, m2, force_scale))
        separation.append(float(state[0]))
        receiver.append(float(state[3]))
        entropy.append(float(state[4]))
    result = {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(
            PREDICTIONS, (entropy, receiver, separation, energy), strict=True
        )
    }
    total = (
        result["prediction.vector.visible-pair-energy"]
        + result["prediction.vector.receiver-energy"]
    )
    diagnostics = {
        "state_derived_energy_residual": float(np.max(np.abs(total - total[0]))),
        "angular_momentum_residual": max_angular_drift,
        "minimum_separation": min_separation,
        "minimum_receiver_energy": float(np.min(result["prediction.vector.receiver-energy"])),
        "minimum_entropy": float(np.min(result["prediction.vector.entropy"])),
    }
    return result, diagnostics


def three_body_acceleration(
    positions: np.ndarray, masses: np.ndarray, softening: float
) -> np.ndarray:
    acceleration = np.zeros_like(positions)
    for first in range(3):
        for second in range(first + 1, 3):
            displacement = positions[second] - positions[first]
            denominator = (float(displacement @ displacement) + softening**2) ** 1.5
            acceleration[first] += masses[second] * displacement / denominator
            acceleration[second] -= masses[first] * displacement / denominator
    return acceleration


def three_body_total_energy(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, softening: float
) -> float:
    energy = 0.5 * float(np.sum(masses[:, None] * velocities * velocities))
    for first in range(3):
        for second in range(first + 1, 3):
            displacement = positions[first] - positions[second]
            energy -= (
                masses[first]
                * masses[second]
                / math.sqrt(float(displacement @ displacement) + softening**2)
            )
    return energy


def three_body_invariants(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, softening: float
) -> tuple[float, np.ndarray, float]:
    energy = three_body_total_energy(positions, velocities, masses, softening)
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    angular = float(
        np.sum(masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0]))
    )
    return energy, momentum, angular


def simulate_three_body(
    features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    times = np.asarray(features["source.vector.encounter-time"], dtype=np.float64)
    mass_ratio = scalar(features, "source.scalar.mass-ratio")
    total_mass = scalar(features, "source.scalar.total-mass")
    initial_separation = scalar(features, "source.scalar.initial-separation")
    initial_radial_velocity = scalar(features, "source.scalar.initial-radial-velocity")
    impact_parameter = scalar(features, "source.scalar.impact-parameter")
    softening = float.fromhex(parameters["softening"])
    third_mass_fraction = float.fromhex(parameters["third_mass_fraction"])
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    m3 = third_mass_fraction * total_mass
    masses = np.asarray([m1, m2, m3], dtype=np.float64)
    pair_mass = m1 + m2
    positions = np.asarray(
        [
            [-initial_separation * m2 / pair_mass, 0.0],
            [initial_separation * m1 / pair_mass, 0.0],
            [0.25 * initial_separation, -1.5 * initial_separation],
        ],
        dtype=np.float64,
    )
    transverse = impact_parameter * abs(initial_radial_velocity) / initial_separation
    relative_velocity = np.asarray([-initial_radial_velocity, transverse], dtype=np.float64)
    velocities = np.asarray(
        [
            relative_velocity * (m2 / pair_mass),
            -relative_velocity * (m1 / pair_mass),
            [0.2 * abs(initial_radial_velocity), 0.3 * abs(initial_radial_velocity)],
        ],
        dtype=np.float64,
    )
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / float(np.sum(masses))
    initial_energy, initial_momentum, initial_angular = three_body_invariants(
        positions, velocities, masses, softening
    )
    acceleration = three_body_acceleration(positions, masses, softening)
    energy: list[float] = []
    separation: list[float] = []
    receiver: list[float] = []
    entropy: list[float] = []
    max_energy_drift = 0.0
    max_momentum_drift = 0.0
    max_angular_drift = 0.0
    min_separation = math.inf
    current_time = 0.0
    dt = 0.01
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(dt, float(target) - current_time)
            velocities += 0.5 * step * acceleration
            positions += step * velocities
            acceleration = three_body_acceleration(positions, masses, softening)
            velocities += 0.5 * step * acceleration
            current_time += step
            actual_energy, momentum, angular = three_body_invariants(
                positions, velocities, masses, softening
            )
            max_energy_drift = max(max_energy_drift, abs(actual_energy - initial_energy))
            max_momentum_drift = max(
                max_momentum_drift, float(np.linalg.norm(momentum - initial_momentum))
            )
            max_angular_drift = max(max_angular_drift, abs(angular - initial_angular))
        displacement = positions[0] - positions[1]
        relative_velocity = velocities[0] - velocities[1]
        pair_separation = math.sqrt(float(displacement @ displacement) + softening**2)
        reduced_mass = masses[0] * masses[1] / (masses[0] + masses[1])
        pair_energy = 0.5 * reduced_mass * float(relative_velocity @ relative_velocity) - (
            masses[0] * masses[1] / pair_separation
        )
        energy.append(pair_energy)
        separation.append(pair_separation)
        receiver.append(initial_energy - pair_energy)
        entropy.append(0.0)
        min_separation = min(min_separation, pair_separation)
    result = {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(
            PREDICTIONS, (entropy, receiver, separation, energy), strict=True
        )
    }
    diagnostics = {
        "actual_state_energy_residual": max_energy_drift,
        "linear_momentum_residual": max_momentum_drift,
        "angular_momentum_residual": max_angular_drift,
        "minimum_separation": min_separation,
        "reported_complement_residual": float(
            np.max(
                np.abs(
                    result["prediction.vector.visible-pair-energy"]
                    + result["prediction.vector.receiver-energy"]
                    - initial_energy
                )
            )
        ),
    }
    return result, diagnostics


def simulate(
    formula_id: str, features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if formula_id == "CM01_CONSERVATIVE_THREE_BODY_CAPTURE":
        return simulate_three_body(features, parameters)
    return simulate_two_body(features, parameters)


def output_sha(predictions: dict[str, np.ndarray], scenario: dict[str, Any]) -> str:
    specs = {row["element_id"]: row for row in scenario["expected_predictions"]}
    artifacts = {}
    for element_id in sorted(predictions):
        spec = specs[element_id]
        artifacts[element_id] = {
            "element_id": element_id,
            "artifact_path": spec["artifact_path"],
            "value_sha256": array_sha(predictions[element_id]),
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "axes": spec["axes"],
            "unit": spec["unit"],
            "frame": spec["frame"],
        }
    return canonical_sha(artifacts)


def score(
    predictions: dict[str, np.ndarray],
    responses: dict[str, np.ndarray],
    variances: dict[str, np.ndarray],
) -> float:
    residuals: list[np.ndarray] = []
    for prediction in PREDICTIONS:
        delta = predictions[prediction].reshape(-1) - responses[RESPONSES[prediction]].reshape(-1)
        variance = variances[prediction].reshape(-1)
        require(np.all(variance > 0), "nonpositive variance")
        residuals.append(np.square(delta) / variance)
    squared = np.concatenate(residuals)
    return math.sqrt(math.fsum(float(value) for value in squared) / squared.size)


def main() -> None:
    for path, expected in EXPECTED_RAW.items():
        require(raw_sha(path) == expected, f"raw hash mismatch: {path}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "receipt.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    require(canonical_sha(body) == receipt["content_sha256"], "receipt content hash mismatch")
    require(
        receipt["content_sha256"]
        == "f93f09b9926949c72ec1917970ccded0e45c03b0137ee285c83cd9ff071a4df0",
        "unexpected receipt content identity",
    )
    for row in config["upstream_bindings"]:
        require(raw_sha(ROOT / row["path"]) == row["sha256"], f"upstream mismatch {row['role']}")
    for source in config["source_families"].values():
        require(
            raw_sha(ROOT / source["source_manifest_path"]) == source["source_manifest_sha256"],
            "source manifest mismatch",
        )
    for row in receipt["artifacts"]:
        artifact = ROOT / row["path"]
        require(artifact.stat().st_size == row["bytes"], f"artifact size mismatch {artifact}")
        require(raw_sha(artifact) == row["sha256"], f"artifact hash mismatch {artifact}")

    camels = json.loads(
        (ROOT / config["source_families"]["camels"]["source_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    tng100 = json.loads(
        (ROOT / config["source_families"]["tng100"]["source_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    require(
        [row["scale_factor"] for row in camels["candidate"]["snapshots"]]
        == config["source_families"]["camels"]["cadence_scale_factors"],
        "CAMELS cadence is not manifest-exact",
    )
    require(
        [row["age_gyr"] for row in tng100["snapshot_grid"]]
        == config["source_families"]["tng100"]["cadence_age_gyr"],
        "TNG cadence is not manifest-exact",
    )
    require(camels["scientific_rows_opened"] == 0, "CAMELS rows were opened")
    require(
        not camels["eligibility_gates"]["direct_nbody_groups"]
        and not camels["eligibility_gates"]["documented_cross_tree_object_matching"],
        "CAMELS source blocker changed",
    )
    require(
        tng100["status"] == "SOURCE_BLOCKED_API_AUTH_AND_PAYLOAD_CHECKSUMS_UNAVAILABLE",
        "TNG blocker changed",
    )
    require(
        all(value == 0 for value in config["access_contract"].values()), "access boundary changed"
    )

    records = [json.loads(line) for line in (OUT / "scenarios.jsonl").read_text().splitlines()]
    matrix = json.loads((OUT / "matrix-result.json").read_text())
    ledger = json.loads((OUT / "ledger.json").read_text())
    diagnostics = json.loads((OUT / "invariance-identifiability-and-blocks.json").read_text())
    require(len(records) == 384, "scenario count")
    require(
        [row["scenario"]["scenario_id"] for row in records]
        == sorted(row["scenario"]["scenario_id"] for row in records),
        "scenarios not sorted",
    )
    require(matrix["scenario_count"] == 384 and len(matrix["cells"]) == 6144, "matrix counts")
    require(matrix["scored_cell_count"] == 2304, "scored count")
    require(len(ledger["entries"]) == 8448, "ledger count")
    require(
        Counter(cell["eligibility"] for cell in matrix["cells"])
        == {"ELIGIBLE": 2304, "SOURCE_BLOCKED": 768, "UNADAPTED": 3072},
        "eligibility Cartesian counts",
    )
    require(
        Counter(row["status"] for row in config["nonexecutable_formulas"].values())
        == {"SOURCE_BLOCKED": 2, "UNADAPTED": 8},
        "formula block inventory",
    )
    require(
        {
            "DC02_INELASTIC_CLOUD_SHOCK",
            "DC03_CHANDRASEKHAR_WAKE_TRANSFER",
            "DC04_GRAVITATIONAL_WAVE_CAPTURE",
            "ORDINARY_GAS_SHOCK_COOLING",
            "ESTABLISHED_DYNAMICAL_FRICTION",
            "DISSIPATIVE_DARK_MATTER",
            "COLLISIONLESS_VIOLENT_RELAXATION",
            "COVARIANT_RECEIVER_COMPLETION",
        }
        <= set(config["nonexecutable_formulas"]),
        "required blocks missing",
    )

    require(canonical_sha(ledger) == matrix["ledger_sha256"], "ledger root hash mismatch")
    matrix_body = {key: value for key, value in matrix.items() if key != "content_sha256"}
    require(canonical_sha(matrix_body) == matrix["content_sha256"], "matrix root hash mismatch")
    prior = None
    ledger_by_hash: dict[str, dict[str, Any]] = {}
    for sequence, entry in enumerate(ledger["entries"]):
        require(entry["sequence"] == sequence, "ledger sequence mismatch")
        require(entry["prior_entry_sha256"] == prior, "ledger chain mismatch")
        entry_body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        require(canonical_sha(entry_body) == entry["entry_sha256"], "entry hash mismatch")
        ledger_by_hash[entry["entry_sha256"]] = entry
        prior = entry["entry_sha256"]
    require(len(ledger_by_hash) == 8448, "ledger hashes not unique")

    cells_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in matrix["cells"]:
        require(cell["ledger_entry_sha256"] in ledger_by_hash, "cell ledger link missing")
        entry = ledger_by_hash[cell["ledger_entry_sha256"]]
        require(entry["formula_id"] == cell["formula_id"], "cell formula ledger mismatch")
        require(entry["status"] == cell["discovery_status"], "cell status ledger mismatch")
        require(
            entry["parameter_cell_id"] == cell["parameter_cell_id"], "parameter lineage mismatch"
        )
        cells_by_scenario[cell["scenario_id"]].append(cell)
    require(
        set(cells_by_scenario) == {row["scenario"]["scenario_id"] for row in records},
        "cell scenarios differ",
    )
    require(
        all(len(cells) == 16 for cells in cells_by_scenario.values()), "candidate Cartesian width"
    )

    record_by_id = {row["scenario"]["scenario_id"]: row for row in records}
    cartesian = Counter()
    paired = Counter()
    cadence_lengths = Counter()
    prediction_cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    full_energy_residuals: dict[str, float] = defaultdict(float)
    max_momentum_residual = 0.0
    max_angular_residual = 0.0
    minimum_separation = math.inf
    max_inactive_role_residual = 0.0
    nonrecoveries: list[dict[str, Any]] = []
    nondistinct_recoveries: list[str] = []
    expected_npz_keys: set[str] = set()

    with np.load(OUT / "values.npz", allow_pickle=False) as values:
        require(len(values.files) == 10752, "synthetic NPZ array count")
        for record in records:
            scenario = record["scenario"]
            scenario_id = scenario["scenario_id"]
            require(canonical_sha(scenario) == record["scenario_sha256"], "scenario hash mismatch")
            require(
                record["public_object_id_claimed"] is False
                and record["real_response_used"] is False,
                "response blindness record changed",
            )
            require(scenario["seed_lineage"]["scenario_id"] == scenario_id, "seed scenario lineage")
            require(
                scenario["seed_lineage"]["object_id"] == scenario["object_id"],
                "seed object lineage",
            )
            require(
                scenario["seed_lineage"]["truth_world_id"]
                == f"truth.{record['truth_formula_id'].lower()}",
                "seed truth lineage",
            )
            require(
                scenario["seed_lineage"]["nuisance_draw"]
                == (0 if record["noise_family"] == "analytic-diagonal" else 1),
                "noise family lineage",
            )
            cartesian[
                (
                    record["source_family"],
                    record["mass_ratio"],
                    record["pericenter_proxy"],
                    record["history"],
                    record["pair_role"],
                    record["truth_formula_id"],
                    record["noise_family"],
                )
            ] += 1
            paired[
                (
                    record["source_family"],
                    record["mass_ratio"],
                    record["pericenter_proxy"],
                    record["history"],
                    record["truth_formula_id"],
                    record["noise_family"],
                )
            ] += 1

            features: dict[str, np.ndarray] = {}
            for reference in scenario["formula_features"]:
                key = array_key("feature", scenario_id, reference["element_id"])
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(
                    value.dtype.name == reference["dtype"]
                    and list(value.shape) == reference["shape"],
                    "feature shape/type",
                )
                require(array_sha(value) == reference["value_sha256"], "feature hash")
                features[reference["element_id"]] = value
            responses: dict[str, np.ndarray] = {}
            for reference in scenario["scoring_responses"]:
                key = array_key("response", scenario_id, reference["element_id"])
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(array_sha(value) == reference["value_sha256"], "response hash")
                responses[reference["element_id"]] = value
            truth_reference = scenario["hidden_truth"][0]
            truth_key = array_key("truth", scenario_id, "formula-code")
            expected_npz_keys.add(truth_key)
            truth_value = np.asarray(values[truth_key])
            require(array_sha(truth_value) == truth_reference["value_sha256"], "truth hash")
            require(
                int(truth_value[0]) == MECHANISM_CODE[record["truth_formula_id"]],
                "truth code mismatch",
            )
            variances: dict[str, np.ndarray] = {}
            inverse_response = {response: prediction for prediction, response in RESPONSES.items()}
            for reference in scenario["uncertainties"]:
                prediction = inverse_response[reference["applies_to_element_id"]]
                key = array_key("variance", scenario_id, prediction)
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(array_sha(value) == reference["artifact_sha256"], "variance hash")
                variances[prediction] = value

            times = features["source.vector.encounter-time"]
            lower = features["source.vector.cadence-interval-lower"]
            upper = features["source.vector.cadence-interval-upper"]
            coordinate = features["source.vector.cadence-coordinate"]
            cadence_lengths[len(times)] += 1
            require(
                np.all(np.diff(times) > 0) and np.all(lower <= times) and np.all(times <= upper),
                "interval censoring",
            )
            require(
                times[0] == lower[0] == coordinate[0] == 0.0
                and times[-1] == upper[-1] == 20.0
                and coordinate[-1] == 1.0,
                "cadence endpoints",
            )
            require(
                np.array_equal(lower[1:], 0.5 * (times[:-1] + times[1:])),
                "lower midpoint intervals",
            )
            require(
                np.array_equal(upper[:-1], 0.5 * (times[:-1] + times[1:])),
                "upper midpoint intervals",
            )

            dynamic_hash = canonical_sha(
                {
                    key: array_sha(features[key])
                    for key in sorted(features)
                    if key not in diagnostics["invariants_and_limits"]["inactive_formula_features"]
                }
            )
            per_formula: dict[str, dict[str, np.ndarray]] = {}
            for formula_id in EXECUTABLE:
                cache_key = (dynamic_hash, formula_id)
                if cache_key not in prediction_cache:
                    prediction_cache[cache_key] = simulate(
                        formula_id, features, config["executable_formulas"][formula_id]
                    )
                predictions, state_diagnostics = prediction_cache[cache_key]
                per_formula[formula_id] = predictions
                if formula_id == "CM01_CONSERVATIVE_THREE_BODY_CAPTURE":
                    full_energy_residuals[formula_id] = max(
                        full_energy_residuals[formula_id],
                        state_diagnostics["actual_state_energy_residual"],
                    )
                    max_momentum_residual = max(
                        max_momentum_residual, state_diagnostics["linear_momentum_residual"]
                    )
                else:
                    full_energy_residuals[formula_id] = max(
                        full_energy_residuals[formula_id],
                        state_diagnostics["state_derived_energy_residual"],
                    )
                max_angular_residual = max(
                    max_angular_residual, state_diagnostics["angular_momentum_residual"]
                )
                minimum_separation = min(
                    minimum_separation, state_diagnostics["minimum_separation"]
                )

            truth_predictions = per_formula[record["truth_formula_id"]]
            seed_hash = canonical_sha(scenario["seed_lineage"])
            rng = np.random.default_rng(int(seed_hash[:16], 16))
            for prediction in PREDICTIONS:
                sigma = float(config["noise"][SIGMA_KEY[prediction]])
                draw = (
                    np.zeros(len(times), dtype=np.float64)
                    if record["noise_family"] == "zero-draw"
                    else rng.normal(0.0, sigma, size=len(times)).astype(np.float64)
                )
                expected_response = truth_predictions[prediction] + draw
                require(
                    np.array_equal(expected_response, responses[RESPONSES[prediction]]),
                    "direct truth/noise injection mismatch",
                )
                require(
                    np.array_equal(variances[prediction], np.full(len(times), sigma * sigma)),
                    "variance mismatch",
                )
            require(
                record["noise_draws_per_response_vector"]
                == int(record["noise_family"] != "zero-draw"),
                "draw count marker",
            )

            seed_lineage_hash = canonical_sha(scenario["seed_lineage"])
            eligible_cells = [
                cell for cell in cells_by_scenario[scenario_id] if cell["eligibility"] == "ELIGIBLE"
            ]
            require(len(eligible_cells) == 6, "eligible width")
            distances = {}
            for cell in eligible_cells:
                formula_id = cell["formula_id"]
                predictions = per_formula[formula_id]
                independent_output_sha = output_sha(predictions, scenario)
                require(
                    independent_output_sha == cell["result_sha256"], "formula output hash mismatch"
                )
                distance = score(predictions, responses, variances)
                require(distance.hex() == cell["whitened_rmse_hex"], "full score mismatch")
                distances[formula_id] = distance
                entry = ledger_by_hash[cell["ledger_entry_sha256"]]
                require(
                    entry["scenario_id"] == scenario_id
                    and entry["object_id"] == scenario["object_id"],
                    "eligible ledger scenario/object",
                )
                require(
                    entry["truth_world_id"] == scenario["seed_lineage"]["truth_world_id"],
                    "eligible ledger truth",
                )
                require(entry["seed_lineage_sha256"] == seed_lineage_hash, "eligible ledger seed")
                require(
                    entry["nuisance_draw"] == scenario["seed_lineage"]["nuisance_draw"],
                    "eligible ledger nuisance",
                )
                require(
                    entry["result_sha256"] == cell["result_sha256"]
                    and entry["metrics_sha256"] == cell["metrics_sha256"]
                    and entry["diagnostics_sha256"] == cell["diagnostics_sha256"],
                    "eligible ledger result lineage",
                )
                precursor = ledger["entries"][entry["sequence"] - 1]
                require(
                    precursor["status"] == "ELIGIBLE_NOT_RUN"
                    and entry["prior_entry_sha256"] == precursor["entry_sha256"],
                    "eligible precursor lineage",
                )

            minimum = min(distances.values())
            winners = sorted(
                formula for formula, distance in distances.items() if distance == minimum
            )
            larger = sorted(distance for distance in distances.values() if distance > minimum)
            gap = larger[0] - minimum if larger else None
            distinct = len(winners) == 1 and gap is not None and gap >= 0.1 and gap > 0
            recovered = record["truth_formula_id"] in winners
            require(
                [cell["formula_id"] for cell in eligible_cells if cell["winner"]] == winners,
                "winner recomputation",
            )
            require(
                all(
                    cell["truth_recovered"] == recovered and cell["distinct"] == distinct
                    for cell in eligible_cells
                ),
                "truth/distinct recomputation",
            )
            if not recovered:
                nonrecoveries.append(
                    {
                        "scenario_id": scenario_id,
                        "truth_formula_id": record["truth_formula_id"],
                        "winner_formula_ids": winners,
                        "scores_hex": {
                            key: value.hex() for key, value in sorted(distances.items())
                        },
                    }
                )
            elif not distinct:
                nondistinct_recoveries.append(scenario_id)

        require(
            set(values.files) == expected_npz_keys, "unreferenced or missing synthetic NPZ arrays"
        )

    require(len(cartesian) == 384 and set(cartesian.values()) == {1}, "scenario Cartesian product")
    require(len(paired) == 192 and set(paired.values()) == {2}, "hydro/DMO pair product")
    require(cadence_lengths == {8: 192, 15: 192}, "8/15 cadence counts")
    require(len(prediction_cache) == 96, "dynamic formula evaluation grid")

    # Independent hydro/DMO inactive-feature equality across all 16 analytic pair designs.
    dynamic_pairs: dict[tuple[Any, ...], dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for record in records:
        if record["truth_formula_id"] != EXECUTABLE[0] or record["noise_family"] != "zero-draw":
            continue
        scenario_id = record["scenario"]["scenario_id"]
        with np.load(OUT / "values.npz", allow_pickle=False) as values:
            features = {
                ref["element_id"]: np.asarray(
                    values[array_key("feature", scenario_id, ref["element_id"])]
                )
                for ref in record["scenario"]["formula_features"]
            }
        identity = (
            record["source_family"],
            record["mass_ratio"],
            record["pericenter_proxy"],
            record["history"],
        )
        dynamic_hash = canonical_sha(
            {
                key: array_sha(features[key])
                for key in sorted(features)
                if key not in diagnostics["invariants_and_limits"]["inactive_formula_features"]
            }
        )
        dynamic_pairs[identity][record["pair_role"]] = {
            formula: prediction_cache[(dynamic_hash, formula)][0][prediction]
            for formula in EXECUTABLE
            for prediction in PREDICTIONS
        }
    for roles in dynamic_pairs.values():
        require(set(roles) == {"dmo", "hydro"}, "role pairing missing")
        for key in roles["dmo"]:
            max_inactive_role_residual = max(
                max_inactive_role_residual,
                float(np.max(np.abs(roles["dmo"][key] - roles["hydro"][key]))),
            )
    require(max_inactive_role_residual == 0.0, "inactive hydro/DMO features affect dynamics")

    require(len(nonrecoveries) == 1, "sole nonrecovery not retained")
    require(
        nonrecoveries[0]["scenario_id"]
        == "capture.tng100-q075-p16-quiet.hydro.dc06_timewell_bimodal_memory_bath.analytic-diagonal.v1",
        "nonrecovery identity changed",
    )
    require(matrix["truth_recovery_count"] == 383, "truth recovery total")
    require(
        matrix["distinct_truth_recovery_count"] == 377 and len(nondistinct_recoveries) == 6,
        "distinct recovery total",
    )

    # Exact limits on one independent source-shaped base state.
    representative = record_by_id[
        "capture.camels-q025-p08-quiet.hydro.cm01_conservative_three_body_capture.zero-draw.v1"
    ]
    scenario_id = representative["scenario"]["scenario_id"]
    with np.load(OUT / "values.npz", allow_pickle=False) as values:
        features = {
            ref["element_id"]: np.asarray(
                values[array_key("feature", scenario_id, ref["element_id"])]
            )
            for ref in representative["scenario"]["formula_features"]
        }
    newton, _ = simulate_two_body(
        features, {"mechanism": "DC00", "force_scale": (1.0).hex(), "gamma": (0.0).hex()}
    )
    limits = [
        simulate_two_body(
            features, {"mechanism": "DC01", "force_scale": (1.0).hex(), "gamma": (0.0).hex()}
        )[0],
        simulate_two_body(
            features,
            {
                "mechanism": "DC05",
                "force_scale": (1.0).hex(),
                "gamma": (0.0).hex(),
                "tau": [(3.0).hex()],
                "weights": [(1.0).hex()],
            },
        )[0],
        simulate_two_body(
            features,
            {
                "mechanism": "DC06",
                "force_scale": (1.0).hex(),
                "gamma": (0.0).hex(),
                "tau": [(0.5).hex(), (12.0).hex()],
                "weights": [(0.7).hex(), (0.3).hex()],
            },
        )[0],
        simulate_two_body(
            features,
            {
                "mechanism": "DC07",
                "force_scale": (1.0).hex(),
                "gamma": (0.0).hex(),
                "compression_speed_scale": (1.0).hex(),
            },
        )[0],
    ]
    require(
        max(
            float(np.max(np.abs(limit[key] - newton[key])))
            for limit in limits
            for key in PREDICTIONS
        )
        == 0.0,
        "zero-gamma/unit-force limit",
    )
    dc06_one_mode = simulate_two_body(
        features,
        {
            "mechanism": "DC06",
            "force_scale": (1.0).hex(),
            "gamma": (0.02).hex(),
            "tau": [(0.5).hex(), (12.0).hex()],
            "weights": [(1.0).hex(), (0.0).hex()],
        },
    )[0]
    dc05_same_mode = simulate_two_body(
        features,
        {
            "mechanism": "DC05",
            "force_scale": (1.0).hex(),
            "gamma": (0.02).hex(),
            "tau": [(0.5).hex()],
            "weights": [(1.0).hex()],
        },
    )[0]
    require(
        max(float(np.max(np.abs(dc06_one_mode[key] - dc05_same_mode[key]))) for key in PREDICTIONS)
        == 0.0,
        "bimodal zero-weight reduction",
    )

    summary = {
        "subject_raw_sha256": {
            path.relative_to(ROOT).as_posix(): expected for path, expected in EXPECTED_RAW.items()
        },
        "receipt_content_sha256": receipt["content_sha256"],
        "artifact_sha256": {row["path"]: row["sha256"] for row in receipt["artifacts"]},
        "upstream_binding_count": len(config["upstream_bindings"]),
        "scenario_count": len(records),
        "attempted_cell_count": len(matrix["cells"]),
        "scored_cell_count": matrix["scored_cell_count"],
        "source_blocked_cell_count": 768,
        "unadapted_cell_count": 3072,
        "replay_entry_count": len(ledger["entries"]),
        "truth_recovery_count": matrix["truth_recovery_count"],
        "distinct_truth_recovery_count": matrix["distinct_truth_recovery_count"],
        "nondistinct_recovered_scenario_count": len(nondistinct_recoveries),
        "sole_nonrecovery": nonrecoveries[0],
        "cadence_scenario_counts": {
            str(key): value for key, value in sorted(cadence_lengths.items())
        },
        "independent_formula_dynamic_grid_size": len(prediction_cache),
        "full_grid_state_derived_energy_residual_by_formula": dict(
            sorted(full_energy_residuals.items())
        ),
        "full_grid_maximum_linear_momentum_residual": max_momentum_residual,
        "full_grid_maximum_angular_momentum_residual": max_angular_residual,
        "full_grid_minimum_separation": minimum_separation,
        "maximum_inactive_hydro_dmo_role_residual": max_inactive_role_residual,
        "zero_gamma_and_alias_limit_residual": 0.0,
        "bimodal_zero_weight_limit_residual": 0.0,
    }
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
