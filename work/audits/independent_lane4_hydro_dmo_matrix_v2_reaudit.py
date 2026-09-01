"""Independent full re-audit of the Lane 4 v2 synthetic repair.

The subject v2 module is intentionally not imported.  The audit reconstructs
all six laws, synthetic injections, scores, and hashes from the frozen JSON,
JSONL, and synthetic NPZ only.  It never opens an HDF5 file.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import independent_lane4_hydro_dmo_matrix_audit as core
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v2"
OUT = ROOT / "runs" / "gravity" / PACKAGE
CONFIG = (
    ROOT
    / "configs/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2.json"
)
SCHEMA = (
    ROOT
    / "configs/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v1.parameters.schema.json"
)
MODULE = (
    ROOT
    / "src/sigma_theory_compiler/open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2.py"
)
TEST = (
    ROOT
    / "tests/test_open_gravity_hydro_dmo_capture_clumping_source_shaped_synthetic_injection_matrix_v2.py"
)
RECEIPT = OUT / "receipt.json"

EXPECTED_RAW = {
    CONFIG: "36b730073ada1db30611b62ddb9e0e0055b482de4c0f2589d3a14c53f3bdf706",
    SCHEMA: "918503b7f16cef4414a56767916cdf5a5628884df4f30befdfe6d80ea7467021",
    MODULE: "06a81d7846382957733485dcc49bf89d997161ecd0ecd0b9f09ac772d6279184",
    TEST: "5b7159bca51e306bd0f5d199d64b1e5cdef8aed8546badd2cc170541345f6384",
    RECEIPT: "82aa6230f7cc9c5fc905ec08e4383ee2cb40d3a4170434c5c559f373d364aca9",
}

EXPECTED_ARTIFACTS = {
    "confusion-matrix.json": "38cf56de81bdb94bcda5b9de15cd7b7a2e73c34e14785f87d84b7922ad0819c5",
    "invariance-identifiability-and-blocks.json": "d1c1e7a5c1ed3c913db3912d34ed6291380beba7f4fff0e900e83fd1013b8d10",
    "ledger.json": "ec2339b3c5320c408618d251d6911d6ecb1684dc461fc5faedab421d4f697204",
    "matrix-result.json": "79482acb74539e0fc291ae5c0d7066b8844c86767812883dfbdb157ad1a36dd5",
    "scenarios.jsonl": "19b02f47aafbe35b6ba970285d1ca42d302ce243d67a4a1c14f0f798fa235d5f",
    "values.npz": "bd7fb6686691bd8157e3b7daec06cc1dbf826baab25af5b222bd1dc4d365517a",
}

EXPECTED_ENERGY = {
    "CM01_CONSERVATIVE_THREE_BODY_CAPTURE": 1.7827990161922713e-7,
    "DC00_NEWTONIAN_FOCUSING_CONTROL": 6.02205441069259e-9,
    "DC01_STATIC_FORCE_AMPLIFICATION_CONTROL": 1.6037091832288297e-7,
    "DC05_TIMEWELL_SINGLE_MEMORY_BATH": 5.893123378175602e-9,
    "DC06_TIMEWELL_BIMODAL_MEMORY_BATH": 5.724883123114921e-9,
    "DC07_COMPRESSION_GATED_BATH": 4.915867712895761e-9,
}
INACTIVE_FEATURES = {
    "source.scalar.cooling-control",
    "source.scalar.gas-fraction",
    "source.scalar.pericenter-proxy",
    "source.scalar.relaxation-time",
    "source.scalar.role-code",
    "source.scalar.shock-mach-control",
    "source.scalar.wake-coulomb-log",
    "source.vector.cadence-coordinate",
    "source.vector.cadence-interval-lower",
    "source.vector.cadence-interval-upper",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def two_body(
    features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    times = np.asarray(features["source.vector.encounter-time"], dtype=np.float64)
    mode = parameters["mechanism"]
    mass_ratio = core.scalar(features, "source.scalar.mass-ratio")
    total_mass = core.scalar(features, "source.scalar.total-mass")
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    mu = m1 * m2 / total_mass
    initial_velocity = core.scalar(features, "source.scalar.initial-radial-velocity")
    tau = tuple(float.fromhex(value) for value in parameters.get("tau", ()))
    weights = tuple(float.fromhex(value) for value in parameters.get("weights", ()))
    state = np.asarray(
        [
            core.scalar(features, "source.scalar.initial-separation"),
            mu * initial_velocity,
            mu * core.scalar(features, "source.scalar.impact-parameter") * abs(initial_velocity),
            0.0,
            0.0,
            *(core.scalar(features, "source.scalar.history-memory") for _ in tau),
        ],
        dtype=np.float64,
    )
    force_scale = float.fromhex(parameters["force_scale"])
    kwargs = {
        "m1": m1,
        "m2": m2,
        "activation_scale": core.scalar(features, "source.scalar.activation-scale"),
        "temperature": core.scalar(features, "source.scalar.temperature"),
        "force_scale": force_scale,
        "gamma": float.fromhex(parameters["gamma"]),
        "mode": mode,
        "tau": tau,
        "weights": weights,
        "compression_speed_scale": float.fromhex(
            parameters.get("compression_speed_scale", "0x1.0000000000000p+0")
        ),
    }
    entropy: list[float] = []
    receiver: list[float] = []
    separation: list[float] = []
    visible: list[float] = []
    initial_angular = float(state[2])
    max_angular = 0.0
    min_separation = float(state[0])
    current_time = 0.0
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(0.005, float(target) - current_time)
            k1 = core.two_body_derivative(state, **kwargs)
            k2 = core.two_body_derivative(state + 0.5 * step * k1, **kwargs)
            k3 = core.two_body_derivative(state + 0.5 * step * k2, **kwargs)
            k4 = core.two_body_derivative(state + step * k3, **kwargs)
            state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            current_time += step
            require(np.all(np.isfinite(state)) and state[0] > 0.05, "two-body boundary")
            min_separation = min(min_separation, float(state[0]))
            max_angular = max(max_angular, abs(float(state[2]) - initial_angular))
        entropy.append(float(state[4]))
        receiver.append(float(state[3]))
        separation.append(float(state[0]))
        visible.append(core.visible_energy(state, m1, m2, force_scale))
    predictions = {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(
            core.PREDICTIONS, (entropy, receiver, separation, visible), strict=True
        )
    }
    total = (
        predictions["prediction.vector.visible-pair-energy"]
        + predictions["prediction.vector.receiver-energy"]
    )
    return predictions, {
        "cadence_energy_residual": float(np.max(np.abs(total - total[0]))),
        "all_step_energy_residual": float(np.max(np.abs(total - total[0]))),
        "linear_momentum_residual": 0.0,
        "angular_momentum_residual": max_angular,
        "minimum_separation": min_separation,
        "receiver_entropy_identity": float(
            np.max(
                np.abs(
                    predictions["prediction.vector.receiver-energy"]
                    - predictions["prediction.vector.entropy"]
                )
            )
        ),
        "minimum_receiver": float(np.min(predictions["prediction.vector.receiver-energy"])),
        "minimum_entropy": float(np.min(predictions["prediction.vector.entropy"])),
    }


def three_body(
    features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    times = np.asarray(features["source.vector.encounter-time"], dtype=np.float64)
    mass_ratio = core.scalar(features, "source.scalar.mass-ratio")
    total_mass = core.scalar(features, "source.scalar.total-mass")
    initial_separation = core.scalar(features, "source.scalar.initial-separation")
    initial_velocity = core.scalar(features, "source.scalar.initial-radial-velocity")
    impact = core.scalar(features, "source.scalar.impact-parameter")
    softening = float.fromhex(parameters["softening"])
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = total_mass - m1
    masses = np.asarray(
        [m1, m2, float.fromhex(parameters["third_mass_fraction"]) * total_mass],
        dtype=np.float64,
    )
    pair_mass = m1 + m2
    positions = np.asarray(
        [
            [-initial_separation * m2 / pair_mass, 0.0],
            [initial_separation * m1 / pair_mass, 0.0],
            [0.25 * initial_separation, -1.5 * initial_separation],
        ],
        dtype=np.float64,
    )
    transverse = impact * abs(initial_velocity) / initial_separation
    relative_velocity = np.asarray([-initial_velocity, transverse], dtype=np.float64)
    velocities = np.asarray(
        [
            relative_velocity * (m2 / pair_mass),
            -relative_velocity * (m1 / pair_mass),
            [0.2 * abs(initial_velocity), 0.3 * abs(initial_velocity)],
        ],
        dtype=np.float64,
    )
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / float(np.sum(masses))
    initial_total, initial_momentum, initial_angular = core.three_body_invariants(
        positions, velocities, masses, softening
    )
    acceleration = core.three_body_acceleration(positions, masses, softening)
    entropy: list[float] = []
    receiver: list[float] = []
    separation: list[float] = []
    visible: list[float] = []
    current_totals: list[float] = []
    max_step_energy = 0.0
    max_momentum = 0.0
    max_angular = 0.0
    min_separation = math.inf
    current_time = 0.0
    for target in times:
        while current_time + 1.0e-14 < target:
            step = min(0.0004, float(target) - current_time)
            velocities += 0.5 * step * acceleration
            positions += step * velocities
            acceleration = core.three_body_acceleration(positions, masses, softening)
            velocities += 0.5 * step * acceleration
            current_time += step
            require(
                np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities)),
                "three-body finite boundary",
            )
            current_total, momentum, angular = core.three_body_invariants(
                positions, velocities, masses, softening
            )
            max_step_energy = max(max_step_energy, abs(current_total - initial_total))
            max_momentum = max(max_momentum, float(np.linalg.norm(momentum - initial_momentum)))
            max_angular = max(max_angular, abs(angular - initial_angular))
        current_total = core.three_body_total_energy(positions, velocities, masses, softening)
        displacement = positions[0] - positions[1]
        relative_velocity = velocities[0] - velocities[1]
        pair_separation = math.sqrt(float(displacement @ displacement) + softening**2)
        reduced_mass = masses[0] * masses[1] / (masses[0] + masses[1])
        pair_energy = 0.5 * reduced_mass * float(relative_velocity @ relative_velocity) - (
            masses[0] * masses[1] / pair_separation
        )
        visible.append(pair_energy)
        separation.append(pair_separation)
        receiver.append(current_total - pair_energy)
        entropy.append(0.0)
        current_totals.append(current_total)
        min_separation = min(min_separation, pair_separation)
    predictions = {
        prediction: np.asarray(value, dtype=np.float64)
        for prediction, value in zip(
            core.PREDICTIONS, (entropy, receiver, separation, visible), strict=True
        )
    }
    recombined = (
        predictions["prediction.vector.visible-pair-energy"]
        + predictions["prediction.vector.receiver-energy"]
    )
    current_totals_array = np.asarray(current_totals, dtype=np.float64)
    current_recombination_residual = float(np.max(np.abs(recombined - current_totals_array)))
    require(
        current_recombination_residual <= 1.0e-12,
        "CM01 not current-Hamiltonian-derived",
    )
    initial_deficit_receiver = initial_total - predictions["prediction.vector.visible-pair-energy"]
    current_vs_initial = float(
        np.max(np.abs(predictions["prediction.vector.receiver-energy"] - initial_deficit_receiver))
    )
    require(current_vs_initial > 0.0, "CM01 repair still equals initial-total bookkeeping")
    return predictions, {
        "cadence_energy_residual": float(
            np.max(np.abs(current_totals_array - current_totals_array[0]))
        ),
        "all_step_energy_residual": max_step_energy,
        "linear_momentum_residual": max_momentum,
        "angular_momentum_residual": max_angular,
        "minimum_separation": min_separation,
        "current_vs_initial_complement": current_vs_initial,
        "current_hamiltonian_recombination_residual": current_recombination_residual,
        "receiver_entropy_identity": 0.0,
        "minimum_receiver": float(np.min(predictions["prediction.vector.receiver-energy"])),
        "minimum_entropy": 0.0,
    }


def simulate(
    formula_id: str, features: dict[str, np.ndarray], parameters: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if formula_id == "CM01_CONSERVATIVE_THREE_BODY_CAPTURE":
        return three_body(features, parameters)
    return two_body(features, parameters)


def main() -> None:
    for path, expected in EXPECTED_RAW.items():
        require(core.raw_sha(path) == expected, f"subject hash mismatch {path}")
    for name, expected in EXPECTED_ARTIFACTS.items():
        require(core.raw_sha(OUT / name) == expected, f"artifact hash mismatch {name}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    require(core.canonical_sha(receipt_body) == receipt["content_sha256"], "receipt self hash")
    require(
        receipt["content_sha256"]
        == "53843d0318f46ffa10181d81893702df7b3b78e522316e17dbb3df670c073c06",
        "receipt content identity",
    )
    require(all(value == 0 for value in config["access_contract"].values()), "access contract")
    require(config["repair"]["retain_changed_recovery_without_tuning"] is True, "retuning boundary")
    for row in receipt["artifacts"]:
        path = ROOT / row["path"]
        require(
            path.stat().st_size == row["bytes"] and core.raw_sha(path) == row["sha256"],
            "receipt artifact binding",
        )

    predecessor = config["predecessor"]
    for prefix in ("config", "parameter_schema", "module", "test", "receipt"):
        require(
            core.raw_sha(ROOT / predecessor[f"{prefix}_path"])
            == predecessor[f"{prefix}_raw_sha256"],
            f"v1 {prefix} drift",
        )
    for name, expected in predecessor["artifact_sha256"].items():
        require(
            core.raw_sha(
                ROOT
                / "runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v1"
                / name
            )
            == expected,
            f"v1 artifact drift {name}",
        )
    v1_receipt = json.loads((ROOT / predecessor["receipt_path"]).read_text())
    require(
        v1_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "v1 receipt content",
    )
    blocked = config["blocked_audit"]
    require(core.raw_sha(ROOT / blocked["path"]) == blocked["raw_sha256"], "v1 BLOCK raw")
    blocked_body = json.loads((ROOT / blocked["path"]).read_text())
    blocked_content = blocked_body.pop("content_sha256")
    require(
        blocked_content == "d09beea2b8ea20b0436189aa59f5af993d223a48bf67d9c052d8516f1f34ddf7"
        and core.canonical_sha(blocked_body) == blocked_content,
        "v1 BLOCK self hash",
    )

    inherited = json.loads((ROOT / predecessor["config_path"]).read_text())
    for row in inherited["upstream_bindings"]:
        require(core.raw_sha(ROOT / row["path"]) == row["sha256"], f"upstream {row['role']}")
    for source in inherited["source_families"].values():
        require(
            core.raw_sha(ROOT / source["source_manifest_path"]) == source["source_manifest_sha256"],
            "source manifest",
        )
    camels = json.loads(
        (ROOT / inherited["source_families"]["camels"]["source_manifest_path"]).read_text()
    )
    tng = json.loads(
        (ROOT / inherited["source_families"]["tng100"]["source_manifest_path"]).read_text()
    )
    require(
        [row["scale_factor"] for row in camels["candidate"]["snapshots"]]
        == inherited["source_families"]["camels"]["cadence_scale_factors"],
        "CAMELS cadence",
    )
    require(
        [row["age_gyr"] for row in tng["snapshot_grid"]]
        == inherited["source_families"]["tng100"]["cadence_age_gyr"],
        "TNG cadence",
    )
    require(
        camels["scientific_rows_opened"] == 0
        and not camels["eligibility_gates"]["direct_nbody_groups"]
        and not camels["eligibility_gates"]["documented_cross_tree_object_matching"],
        "CAMELS blocker",
    )
    require(
        tng["status"] == "SOURCE_BLOCKED_API_AUTH_AND_PAYLOAD_CHECKSUMS_UNAVAILABLE",
        "TNG blocker",
    )

    schema = json.loads(SCHEMA.read_text())
    allowed_parameters = {
        branch["properties"]["mechanism"]["const"]: {
            key: value["const"] for key, value in branch["properties"].items()
        }
        for branch in schema["oneOf"]
    }
    for parameters in inherited["executable_formulas"].values():
        require(parameters == allowed_parameters[parameters["mechanism"]], "parameter schema")
    require(
        Counter(row["status"] for row in inherited["nonexecutable_formulas"].values())
        == {"SOURCE_BLOCKED": 2, "UNADAPTED": 8},
        "block inventory",
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
        <= set(inherited["nonexecutable_formulas"]),
        "explicit receiver/ordinary/covariant blocks",
    )

    records = [json.loads(line) for line in (OUT / "scenarios.jsonl").read_text().splitlines()]
    matrix = json.loads((OUT / "matrix-result.json").read_text())
    ledger = json.loads((OUT / "ledger.json").read_text())
    diagnostics = json.loads((OUT / "invariance-identifiability-and-blocks.json").read_text())
    require(len(records) == matrix["scenario_count"] == 384, "scenario count")
    require(len(matrix["cells"]) == matrix["attempted_cell_count"] == 6144, "cell count")
    require(
        matrix["scored_cell_count"] == 2304 and len(ledger["entries"]) == 8448, "score/ledger count"
    )
    require(
        Counter(cell["eligibility"] for cell in matrix["cells"])
        == {"ELIGIBLE": 2304, "SOURCE_BLOCKED": 768, "UNADAPTED": 3072},
        "eligibility counts",
    )
    require(core.canonical_sha(ledger) == matrix["ledger_sha256"], "ledger root")
    matrix_body = {key: value for key, value in matrix.items() if key != "content_sha256"}
    require(core.canonical_sha(matrix_body) == matrix["content_sha256"], "matrix root")

    prior = None
    entries: dict[str, dict[str, Any]] = {}
    for sequence, entry in enumerate(ledger["entries"]):
        require(
            entry["sequence"] == sequence and entry["prior_entry_sha256"] == prior, "ledger chain"
        )
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        require(core.canonical_sha(body) == entry["entry_sha256"], "ledger entry hash")
        entries[entry["entry_sha256"]] = entry
        prior = entry["entry_sha256"]
    require(len(entries) == 8448, "ledger entry uniqueness")
    cells_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in matrix["cells"]:
        entry = entries[cell["ledger_entry_sha256"]]
        require(
            entry["formula_id"] == cell["formula_id"]
            and entry["status"] == cell["discovery_status"]
            and entry["parameter_cell_id"] == cell["parameter_cell_id"],
            "cell ledger lineage",
        )
        cells_by_scenario[cell["scenario_id"]].append(cell)

    cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    profiles: dict[tuple[Any, ...], dict[str, dict[str, dict[str, np.ndarray]]]] = defaultdict(dict)
    cadence_counts = Counter()
    cartesian = Counter()
    paired = Counter()
    expected_npz_keys: set[str] = set()
    max_energy = {formula: 0.0 for formula in core.EXECUTABLE}
    max_step_energy = {formula: 0.0 for formula in core.EXECUTABLE}
    max_momentum = 0.0
    max_angular = 0.0
    min_separation = math.inf
    max_entropy_identity = 0.0
    min_receiver = math.inf
    min_entropy = math.inf
    cm01_current_vs_initial = 0.0
    nondistinct: list[str] = []
    nonrecovered: list[str] = []

    with np.load(OUT / "values.npz", allow_pickle=False) as values:
        require(len(values.files) == 10752, "NPZ array count")
        for record in records:
            scenario = record["scenario"]
            scenario_id = scenario["scenario_id"]
            require(core.canonical_sha(scenario) == record["scenario_sha256"], "scenario hash")
            require(
                record["real_response_used"] is False
                and record["public_object_id_claimed"] is False
                and record["v1_recovery_status_not_presumed"] is True,
                "scenario claim boundary",
            )
            seed = scenario["seed_lineage"]
            require(
                seed["scenario_id"] == scenario_id
                and seed["object_id"] == scenario["object_id"]
                and seed["truth_world_id"] == f"truth.{record['truth_formula_id'].lower()}.v2",
                "seed identity",
            )
            expected_draw = 0 if record["noise_family"] == "analytic-diagonal" else 1
            require(seed["nuisance_draw"] == expected_draw, "seed nuisance")
            dimensions = (
                record["source_family"],
                record["mass_ratio"],
                record["pericenter_proxy"],
                record["history"],
                record["pair_role"],
                record["truth_formula_id"],
                record["noise_family"],
            )
            cartesian[dimensions] += 1
            paired[dimensions[:4] + dimensions[5:]] += 1

            features: dict[str, np.ndarray] = {}
            for reference in scenario["formula_features"]:
                key = core.array_key("feature", scenario_id, reference["element_id"])
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(core.array_sha(value) == reference["value_sha256"], "feature hash")
                require(
                    value.dtype.name == reference["dtype"]
                    and list(value.shape) == reference["shape"],
                    "feature metadata",
                )
                features[reference["element_id"]] = value
            responses: dict[str, np.ndarray] = {}
            for reference in scenario["scoring_responses"]:
                key = core.array_key("response", scenario_id, reference["element_id"])
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(core.array_sha(value) == reference["value_sha256"], "response hash")
                responses[reference["element_id"]] = value
            truth_key = core.array_key("truth", scenario_id, "formula-code")
            expected_npz_keys.add(truth_key)
            truth = np.asarray(values[truth_key])
            require(
                core.array_sha(truth) == scenario["hidden_truth"][0]["value_sha256"]
                and int(truth[0]) == core.MECHANISM_CODE[record["truth_formula_id"]],
                "hidden truth",
            )
            inverse = {response: prediction for prediction, response in core.RESPONSES.items()}
            variances: dict[str, np.ndarray] = {}
            for reference in scenario["uncertainties"]:
                prediction = inverse[reference["applies_to_element_id"]]
                key = core.array_key("variance", scenario_id, prediction)
                expected_npz_keys.add(key)
                value = np.asarray(values[key])
                require(core.array_sha(value) == reference["artifact_sha256"], "variance hash")
                variances[prediction] = value

            times = features["source.vector.encounter-time"]
            lower = features["source.vector.cadence-interval-lower"]
            upper = features["source.vector.cadence-interval-upper"]
            coordinate = features["source.vector.cadence-coordinate"]
            cadence_counts[len(times)] += 1
            require(
                np.all(np.diff(times) > 0)
                and np.all(lower <= times)
                and np.all(times <= upper)
                and times[0] == lower[0] == coordinate[0] == 0.0
                and times[-1] == upper[-1] == 20.0
                and coordinate[-1] == 1.0,
                "interval-censored cadence",
            )
            require(
                np.array_equal(lower[1:], 0.5 * (times[:-1] + times[1:]))
                and np.array_equal(upper[:-1], 0.5 * (times[:-1] + times[1:])),
                "cadence midpoints",
            )

            dynamic_hash = core.canonical_sha(
                {
                    key: core.array_sha(features[key])
                    for key in sorted(features)
                    if key not in INACTIVE_FEATURES
                }
            )
            predictions_by_formula: dict[str, dict[str, np.ndarray]] = {}
            for formula_id in core.EXECUTABLE:
                cache_key = (dynamic_hash, formula_id)
                if cache_key not in cache:
                    cache[cache_key] = simulate(
                        formula_id, features, inherited["executable_formulas"][formula_id]
                    )
                prediction, invariant = cache[cache_key]
                predictions_by_formula[formula_id] = prediction
                max_energy[formula_id] = max(
                    max_energy[formula_id], invariant["cadence_energy_residual"]
                )
                max_step_energy[formula_id] = max(
                    max_step_energy[formula_id], invariant["all_step_energy_residual"]
                )
                max_momentum = max(max_momentum, invariant["linear_momentum_residual"])
                max_angular = max(max_angular, invariant["angular_momentum_residual"])
                min_separation = min(min_separation, invariant["minimum_separation"])
                if formula_id.startswith(("DC05", "DC06", "DC07")):
                    max_entropy_identity = max(
                        max_entropy_identity, invariant["receiver_entropy_identity"]
                    )
                    min_receiver = min(min_receiver, invariant["minimum_receiver"])
                    min_entropy = min(min_entropy, invariant["minimum_entropy"])
                if formula_id == "CM01_CONSERVATIVE_THREE_BODY_CAPTURE":
                    cm01_current_vs_initial = max(
                        cm01_current_vs_initial, invariant["current_vs_initial_complement"]
                    )
            profile_id = (
                record["source_family"],
                record["mass_ratio"],
                record["pericenter_proxy"],
                record["history"],
            )
            profiles[profile_id][record["pair_role"]] = predictions_by_formula

            truth_predictions = predictions_by_formula[record["truth_formula_id"]]
            rng = np.random.default_rng(int(core.canonical_sha(seed)[:16], 16))
            for prediction in core.PREDICTIONS:
                sigma = float(inherited["noise"][core.SIGMA_KEY[prediction]])
                draw = (
                    np.zeros(len(times), dtype=np.float64)
                    if record["noise_family"] == "zero-draw"
                    else rng.normal(0.0, sigma, size=len(times)).astype(np.float64)
                )
                require(
                    np.array_equal(
                        truth_predictions[prediction] + draw,
                        responses[core.RESPONSES[prediction]],
                    ),
                    "direct injection or noise draw",
                )
                require(
                    np.array_equal(variances[prediction], np.full(len(times), sigma * sigma)),
                    "frozen analytic variance",
                )
            require(
                record["noise_draws_per_response_vector"]
                == int(record["noise_family"] != "zero-draw"),
                "draw count marker",
            )

            scenario_cells = cells_by_scenario[scenario_id]
            require(len(scenario_cells) == 16, "candidate width")
            eligible = [cell for cell in scenario_cells if cell["eligibility"] == "ELIGIBLE"]
            require(len(eligible) == 6, "eligible width")
            distances: dict[str, float] = {}
            seed_hash = core.canonical_sha(seed)
            for cell in eligible:
                formula_id = cell["formula_id"]
                prediction = predictions_by_formula[formula_id]
                require(
                    core.output_sha(prediction, scenario) == cell["result_sha256"],
                    "formula result hash",
                )
                distance = core.score(prediction, responses, variances)
                require(distance.hex() == cell["whitened_rmse_hex"], "whitened score")
                distances[formula_id] = distance
                entry = entries[cell["ledger_entry_sha256"]]
                require(
                    entry["scenario_id"] == scenario_id
                    and entry["object_id"] == scenario["object_id"]
                    and entry["truth_world_id"] == seed["truth_world_id"]
                    and entry["seed_lineage_sha256"] == seed_hash
                    and entry["nuisance_draw"] == seed["nuisance_draw"]
                    and entry["result_sha256"] == cell["result_sha256"]
                    and entry["metrics_sha256"] == cell["metrics_sha256"]
                    and entry["diagnostics_sha256"] == cell["diagnostics_sha256"],
                    "eligible replay lineage",
                )
                precursor = ledger["entries"][entry["sequence"] - 1]
                require(
                    precursor["status"] == "ELIGIBLE_NOT_RUN"
                    and entry["prior_entry_sha256"] == precursor["entry_sha256"],
                    "eligible replay adjacency",
                )
            minimum = min(distances.values())
            winners = sorted(
                formula for formula, distance in distances.items() if distance == minimum
            )
            larger = sorted(distance for distance in distances.values() if distance > minimum)
            gap = larger[0] - minimum if larger else None
            distinct = len(winners) == 1 and gap is not None and gap > 0 and gap >= 0.1
            recovered = record["truth_formula_id"] in winners
            require(
                [cell["formula_id"] for cell in eligible if cell["winner"]] == winners
                and all(
                    cell["truth_recovered"] == recovered and cell["distinct"] == distinct
                    for cell in eligible
                ),
                "winner/recovery/distinct",
            )
            if not recovered:
                nonrecovered.append(scenario_id)
            elif not distinct:
                nondistinct.append(scenario_id)
        require(set(values.files) == expected_npz_keys, "NPZ reference closure")

    require(len(cartesian) == 384 and set(cartesian.values()) == {1}, "Cartesian scenarios")
    require(len(paired) == 192 and set(paired.values()) == {2}, "paired roles")
    require(cadence_counts == {8: 192, 15: 192}, "cadence counts")
    require(len(cache) == 96 and len(profiles) == 16, "independent dynamic grid")
    require(max_energy == EXPECTED_ENERGY, f"energy residual mismatch: {max_energy}")
    require(max(max_step_energy.values()) < 5e-6, "all-step energy tolerance")
    require(
        max_entropy_identity == 0.0 and min_receiver >= 0.0 and min_entropy >= 0.0,
        "receiver entropy",
    )
    require(
        cm01_current_vs_initial == EXPECTED_ENERGY["CM01_CONSERVATIVE_THREE_BODY_CAPTURE"],
        "CM01 current-state repair magnitude",
    )
    require(nonrecovered == [] and len(nondistinct) == 2, "recovery counts")
    require(
        sorted(nondistinct) == diagnostics["truth_recovered_but_nondistinct_scenarios"],
        "nondistinct identities",
    )
    require(
        matrix["truth_recovery_count"] == 384 and matrix["distinct_truth_recovery_count"] == 382,
        "matrix recovery totals",
    )

    max_role_residual = 0.0
    for roles in profiles.values():
        require(set(roles) == {"dmo", "hydro"}, "role profile missing")
        for formula in core.EXECUTABLE:
            for prediction in core.PREDICTIONS:
                max_role_residual = max(
                    max_role_residual,
                    float(
                        np.max(
                            np.abs(
                                roles["dmo"][formula][prediction]
                                - roles["hydro"][formula][prediction]
                            )
                        )
                    ),
                )
    require(max_role_residual == 0.0, "inactive hydro/DMO role features")

    representative = records[0]
    scenario_id = representative["scenario"]["scenario_id"]
    with np.load(OUT / "values.npz", allow_pickle=False) as values:
        features = {
            reference["element_id"]: np.asarray(
                values[core.array_key("feature", scenario_id, reference["element_id"])]
            )
            for reference in representative["scenario"]["formula_features"]
        }
    newton = two_body(
        features,
        {"mechanism": "DC00", "force_scale": (1.0).hex(), "gamma": (0.0).hex()},
    )[0]
    limits = [
        two_body(
            features,
            {"mechanism": "DC01", "force_scale": (1.0).hex(), "gamma": (0.0).hex()},
        )[0],
        *[
            two_body(
                features,
                {
                    "mechanism": mode,
                    "force_scale": (1.0).hex(),
                    "gamma": (0.0).hex(),
                    **(
                        {
                            "tau": [(0.5).hex(), (12.0).hex()],
                            "weights": [(0.7).hex(), (0.3).hex()],
                        }
                        if mode == "DC06"
                        else (
                            {"tau": [(3.0).hex()], "weights": [(1.0).hex()]}
                            if mode == "DC05"
                            else {"compression_speed_scale": (1.0).hex()}
                        )
                    ),
                },
            )[0]
            for mode in ("DC05", "DC06", "DC07")
        ],
    ]
    zero_limit = max(
        float(np.max(np.abs(candidate[prediction] - newton[prediction])))
        for candidate in limits
        for prediction in core.PREDICTIONS
    )
    require(zero_limit == 0.0, "zero-gamma/unit-force limits")
    one_bimodal = two_body(
        features,
        {
            "mechanism": "DC06",
            "force_scale": (1.0).hex(),
            "gamma": (0.02).hex(),
            "tau": [(0.5).hex(), (12.0).hex()],
            "weights": [(1.0).hex(), (0.0).hex()],
        },
    )[0]
    one_single = two_body(
        features,
        {
            "mechanism": "DC05",
            "force_scale": (1.0).hex(),
            "gamma": (0.02).hex(),
            "tau": [(0.5).hex()],
            "weights": [(1.0).hex()],
        },
    )[0]
    bimodal_limit = max(
        float(np.max(np.abs(one_bimodal[key] - one_single[key]))) for key in core.PREDICTIONS
    )
    require(bimodal_limit == 0.0, "bimodal zero-weight limit")

    summary = {
        "subject_raw_sha256": {
            path.relative_to(ROOT).as_posix(): expected for path, expected in EXPECTED_RAW.items()
        },
        "receipt_content_sha256": receipt["content_sha256"],
        "artifact_sha256": EXPECTED_ARTIFACTS,
        "scenario_count": 384,
        "attempted_cell_count": 6144,
        "scored_cell_count": 2304,
        "source_blocked_cell_count": 768,
        "unadapted_cell_count": 3072,
        "replay_entry_count": 8448,
        "truth_recovery_count": 384,
        "distinct_truth_recovery_count": 382,
        "nonrecovery_count": 0,
        "nondistinct_scenarios": sorted(nondistinct),
        "cadence_scenario_counts": {
            str(key): value for key, value in sorted(cadence_counts.items())
        },
        "current_state_energy_residual_by_formula": max_energy,
        "all_step_maximum_energy_residual_by_formula": max_step_energy,
        "maximum_linear_momentum_residual": max_momentum,
        "maximum_angular_momentum_residual": max_angular,
        "minimum_separation": min_separation,
        "cm01_current_receiver_difference_from_initial_total_bookkeeping": cm01_current_vs_initial,
        "maximum_receiver_entropy_identity_residual": max_entropy_identity,
        "minimum_receiver_energy": min_receiver,
        "minimum_entropy": min_entropy,
        "maximum_inactive_hydro_dmo_role_residual": max_role_residual,
        "zero_gamma_and_unit_force_limit_residual": zero_limit,
        "bimodal_zero_weight_limit_residual": bimodal_limit,
        "v1_block_evidence_raw_sha256": config["blocked_audit"]["raw_sha256"],
        "v1_block_evidence_content_sha256": blocked_content,
        "upstream_binding_count": len(inherited["upstream_bindings"]),
    }
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
