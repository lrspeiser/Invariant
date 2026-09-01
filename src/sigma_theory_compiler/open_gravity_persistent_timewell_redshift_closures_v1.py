"""Persistent-time-well redshift closures and response-free identifiability tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_persistent_timewell_redshift_closures_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_persistent_timewell_redshift_closures_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_persistent_timewell_redshift_closures_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-persistent-timewell-redshift-closures-v1/receipt.json"
)
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "5ccc17ba31fae700ee0fa51cbafdc0cc0af431205ca7d8806fa91b512a4c03cd"
_CONFIG_CONTENT_SHA256 = "d02cc0b02b897e6bdadfff506714db3f2e555b1cdd5e9abca88a384eef687fd1"
_MODULE_SEMANTIC_SHA256 = "8311d509b73fae0ccdc85c42a71442c634f04010fcd2ad64a5bcc8b27ee8ac2d"
_TEST_RAW_SHA256 = "49583cf1aa7705efd927c89b40bf275c1e267da4a21cd10a6ee62fdc7cb8a9e1"
_SCHEMA = "invariant-open-gravity-persistent-timewell-redshift-closures-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-persistent-timewell-redshift-receipt-1.0"
_CLOSURE_IDS = tuple(
    f"C{i:02d}_{name}"
    for i, name in enumerate(
        (
            "GR_ENDPOINT",
            "EXACT_GRADIENT_PATH",
            "POTENTIAL_COLUMN",
            "TIDAL_CURVATURE_COLUMN",
            "DISPERSIVE_PERSISTENT_MEDIUM",
            "ENDPOINT_MEMORY",
            "PATH_MEMORY_OPACITY",
            "TIME_VARYING_METRIC_PATH",
            "CAUSAL_METRIC_MEMORY_PATH",
        )
    )
)


class PersistentRedshiftError(RuntimeError):
    """Raised when a frozen closure or package invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistentRedshiftError(message)


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
        raise PersistentRedshiftError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-persistent-timewell-redshift-closures-v1",
        "package ID changed",
    )
    _require(
        config["status"] == "FROZEN_RESPONSE_FREE_CLOSURE_AND_REAL_DATA_PREFLIGHT", "status changed"
    )
    _require(tuple(row["id"] for row in config["closures"]) == _CLOSURE_IDS, "closures changed")
    _require(len(config["published_neighbors"]) == 7, "published-neighbor count changed")
    _require(len(config["mandatory_controls"]) == 12, "control count changed")
    _require(len(config["synthetic_fixture_ids"]) == 8, "fixture count changed")
    _require(set(config["access_contract"].values()) == {0}, "access contract changed")
    _require(config["claim_boundary"]["real_data_scored"] is False, "claim widened")
    _require(config["claim_boundary"]["historical_novelty_established"] is False, "claim widened")
    _require(
        config["real_data_preflight"]["response_status"] == "NOT_OPENED_NOT_SCORED",
        "response gate changed",
    )
    _require(
        config["astrophysical_falsifier_preflight"]["response_status"] == "NOT_OPENED_NOT_SCORED",
        "response gate changed",
    )
    _require(
        config["real_data_preflight"]["frozen_tau_seconds"]
        == [300.0, 1800.0, 7200.0, 21600.0, 86400.0, 604800.0],
        "tau grid changed",
    )
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIR.as_posix(), "artifact path changed"
    )
    dimensions = config["dimensions"]
    _require(dimensions["tidal_sqrtK"] == "L^-2", "curvature units changed")
    _require(dimensions["curvature_length_ellK"] == "L", "curvature length units changed")
    _require(dimensions["partial_t_psi_dt"] == "1", "integrated time derivative units changed")


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic hash changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH, "persistent-redshift config")
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
    _require(len(observed) == 3, "local binding count changed")
    return observed


def _fixtures() -> list[dict[str, Any]]:
    """Target-free, dimensionless design rows; values are not observations."""
    return [
        {
            "fixture": "F01_ENDPOINT_SWAP",
            "case": "forward",
            "u": 0.6,
            "psi": 0.3,
            "col_u": 0.5,
            "col_k": 0.4,
            "col_psi": 0.7,
            "nu": 1.0,
            "metric_dt": 0.0,
            "causal_dt": 0.1,
        },
        {
            "fixture": "F01_ENDPOINT_SWAP",
            "case": "reverse",
            "u": -0.6,
            "psi": -0.3,
            "col_u": 0.5,
            "col_k": 0.4,
            "col_psi": 0.7,
            "nu": 1.0,
            "metric_dt": 0.0,
            "causal_dt": -0.1,
        },
        {
            "fixture": "F02_EQUAL_ENDPOINT_TWO_PATHS",
            "case": "short",
            "u": 0.1,
            "psi": 0.05,
            "col_u": 0.2,
            "col_k": 0.1,
            "col_psi": 0.2,
            "nu": 1.0,
            "metric_dt": 0.02,
            "causal_dt": 0.03,
        },
        {
            "fixture": "F02_EQUAL_ENDPOINT_TWO_PATHS",
            "case": "long",
            "u": 0.1,
            "psi": 0.05,
            "col_u": 0.6,
            "col_k": 0.8,
            "col_psi": 0.9,
            "nu": 1.0,
            "metric_dt": 0.08,
            "causal_dt": 0.12,
        },
        {
            "fixture": "F03_POTENTIAL_ZERO_SHIFT",
            "case": "base",
            "u": 0.2,
            "psi": 0.1,
            "col_u": 0.4,
            "col_k": 0.25,
            "col_psi": 0.3,
            "nu": 1.0,
            "metric_dt": 0.01,
            "causal_dt": 0.02,
        },
        {
            "fixture": "F03_POTENTIAL_ZERO_SHIFT",
            "case": "add_constant",
            "u": 0.2,
            "psi": 0.1,
            "col_u": 2.4,
            "col_k": 0.25,
            "col_psi": 0.3,
            "nu": 1.0,
            "metric_dt": 0.01,
            "causal_dt": 0.02,
        },
        {
            "fixture": "F04_SOURCE_SWITCH_OFF_HISTORY",
            "case": "immediate",
            "u": 0.2,
            "psi": 0.8,
            "col_u": 0.3,
            "col_k": 0.2,
            "col_psi": 0.8,
            "nu": 1.0,
            "metric_dt": -0.05,
            "causal_dt": -0.8,
        },
        {
            "fixture": "F04_SOURCE_SWITCH_OFF_HISTORY",
            "case": "late",
            "u": 0.2,
            "psi": 0.1,
            "col_u": 0.3,
            "col_k": 0.2,
            "col_psi": 0.1,
            "nu": 1.0,
            "metric_dt": -0.01,
            "causal_dt": -0.1,
        },
        {
            "fixture": "F05_TWO_FREQUENCIES",
            "case": "low",
            "u": 0.15,
            "psi": 0.2,
            "col_u": 0.35,
            "col_k": 0.3,
            "col_psi": 0.7,
            "nu": 0.5,
            "metric_dt": 0.03,
            "causal_dt": 0.04,
        },
        {
            "fixture": "F05_TWO_FREQUENCIES",
            "case": "high",
            "u": 0.15,
            "psi": 0.2,
            "col_u": 0.35,
            "col_k": 0.3,
            "col_psi": 0.7,
            "nu": 2.0,
            "metric_dt": 0.03,
            "causal_dt": 0.04,
        },
        {
            "fixture": "F06_LENS_TWO_IMAGES",
            "case": "image_A",
            "u": 0.4,
            "psi": 0.15,
            "col_u": 0.5,
            "col_k": 0.45,
            "col_psi": 0.5,
            "nu": 1.0,
            "metric_dt": 0.02,
            "causal_dt": 0.06,
        },
        {
            "fixture": "F06_LENS_TWO_IMAGES",
            "case": "image_B",
            "u": 0.4,
            "psi": 0.15,
            "col_u": 0.8,
            "col_k": 1.1,
            "col_psi": 1.0,
            "nu": 1.0,
            "metric_dt": 0.09,
            "causal_dt": 0.2,
        },
        {
            "fixture": "F07_ROUND_TRIP",
            "case": "outbound",
            "u": 0.25,
            "psi": 0.2,
            "col_u": 0.4,
            "col_k": 0.35,
            "col_psi": 0.45,
            "nu": 1.0,
            "metric_dt": 0.01,
            "causal_dt": 0.03,
        },
        {
            "fixture": "F07_ROUND_TRIP",
            "case": "return",
            "u": -0.25,
            "psi": -0.2,
            "col_u": 0.4,
            "col_k": 0.35,
            "col_psi": 0.45,
            "nu": 1.0,
            "metric_dt": -0.01,
            "causal_dt": -0.03,
        },
        {
            "fixture": "F08_EXPANSION_TIME_DILATION",
            "case": "z_0p5",
            "u": 0.0,
            "psi": 0.0,
            "col_u": 0.0,
            "col_k": 0.0,
            "col_psi": 0.0,
            "nu": 1.0,
            "metric_dt": 0.0,
            "causal_dt": 0.0,
        },
    ]


def _predict(closure_id: str, row: Mapping[str, Any]) -> tuple[float, float]:
    """Return residual Delta ln(nu) and expansion stretch; all couplings are unit probes."""
    u = float(row["u"])
    psi = float(row["psi"])
    col_u = float(row["col_u"])
    col_k = float(row["col_k"])
    col_psi = float(row["col_psi"])
    nu = float(row["nu"])
    if closure_id in {"C00_GR_ENDPOINT", "C01_EXACT_GRADIENT_PATH"}:
        value = u
    elif closure_id == "C02_POTENTIAL_COLUMN":
        value = -col_u
    elif closure_id == "C03_TIDAL_CURVATURE_COLUMN":
        value = -col_k
    elif closure_id == "C04_DISPERSIVE_PERSISTENT_MEDIUM":
        value = -col_psi / nu**2
    elif closure_id == "C05_ENDPOINT_MEMORY":
        value = u + psi
    elif closure_id == "C06_PATH_MEMORY_OPACITY":
        value = -col_psi
    elif closure_id == "C07_TIME_VARYING_METRIC_PATH":
        value = u + float(row["metric_dt"])
    elif closure_id == "C08_CAUSAL_METRIC_MEMORY_PATH":
        value = u + float(row["causal_dt"])
    else:  # pragma: no cover - protected by config validation
        raise PersistentRedshiftError(f"unknown closure: {closure_id}")
    stretch = 1.5 if row["fixture"] == "F08_EXPANSION_TIME_DILATION" else 1.0
    return value, stretch


def _synthetic_signatures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixtures = _fixtures()
    signature_rows: list[dict[str, Any]] = []
    vectors: dict[str, list[float]] = {closure_id: [] for closure_id in _CLOSURE_IDS}
    for row in fixtures:
        for closure_id in _CLOSURE_IDS:
            value, stretch = _predict(closure_id, row)
            vectors[closure_id].extend((value, stretch))
            signature_rows.append(
                {
                    "fixture_id": row["fixture"],
                    "case_id": row["case"],
                    "closure_id": closure_id,
                    "delta_log_frequency_unit_couplings": format(value, ".12e"),
                    "time_stretch_factor": format(stretch, ".12e"),
                }
            )
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(_CLOSURE_IDS):
        for right in _CLOSURE_IDS[left_index + 1 :]:
            differences = [abs(a - b) for a, b in zip(vectors[left], vectors[right], strict=True)]
            max_difference = max(differences)
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "max_abs_signature_difference": format(max_difference, ".12e"),
                    "distinguishable": max_difference > 1.0e-12,
                }
            )
    equivalent = [row for row in pair_rows if not row["distinguishable"]]
    _require(len(equivalent) == 1, "unexpected synthetic degeneracy count")
    _require(
        {equivalent[0]["left"], equivalent[0]["right"]}
        == {"C00_GR_ENDPOINT", "C01_EXACT_GRADIENT_PATH"},
        "endpoint equivalence not isolated",
    )
    return signature_rows, pair_rows


def _closure_ledger(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dispositions = {
        "C00_GR_ENDPOINT": (None, "KNOWN_COMPARATOR", "GR endpoint reference"),
        "C01_EXACT_GRADIENT_PATH": (
            None,
            "FAIL_NOT_DISTINCT",
            "exact differential collapses to endpoints",
        ),
        "C02_POTENTIAL_COLUMN": (
            None,
            "FAIL_GAUGE_ENERGY_RECIPROCITY",
            "additive-potential zero changes prediction",
        ),
        "C03_TIDAL_CURVATURE_COLUMN": (
            3,
            "OPEN_ACTION_REQUIRED",
            "gauge-safe path dependence but an energy sink/action is absent",
        ),
        "C04_DISPERSIVE_PERSISTENT_MEDIUM": (
            5,
            "OPEN_PLASMA_ADJACENT",
            "testable chromatic control; low uniqueness before plasma rejection",
        ),
        "C05_ENDPOINT_MEMORY": (
            2,
            "OPEN_EMPIRICAL_LEAD",
            "cleanest eccentric-clock hysteresis test",
        ),
        "C06_PATH_MEMORY_OPACITY": (
            4,
            "OPEN_ACTION_REQUIRED",
            "strong multiple-path signal but dissipative bookkeeping debt",
        ),
        "C07_TIME_VARYING_METRIC_PATH": (
            None,
            "KNOWN_COMPARATOR",
            "ISW/Rees-Sciama-adjacent time-varying metric control",
        ),
        "C08_CAUSAL_METRIC_MEMORY_PATH": (
            1,
            "LEAD_CONDITIONAL",
            "joins causal state history to achromatic metric photon propagation",
        ),
    }
    rows = []
    for closure in config["closures"]:
        rank, disposition, reason = dispositions[closure["id"]]
        rows.append(
            {
                "closure_id": closure["id"],
                "candidate_rank": "" if rank is None else rank,
                "disposition": disposition,
                "class": closure["class"],
                "chromaticity": closure["chromaticity"],
                "reciprocity": closure["reciprocity"],
                "dimension_check": closure["dimension_check"],
                "energy_accounting": closure["energy_accounting"],
                "reason": reason,
                "equation": closure["equation"],
                "novelty_boundary": closure["novelty_boundary"],
            }
        )
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "cannot write empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().replace("\r\n", "\n").encode()


def _report(config: Mapping[str, Any], pair_rows: Sequence[Mapping[str, Any]]) -> bytes:
    distinct_pairs = sum(bool(row["distinguishable"]) for row in pair_rows)
    text = f"""# Persistent time-well redshift closure result

## Outcome

**PASS** for a response-free closure taxonomy, dimensional audit, and synthetic identifiability benchmark. **BLOCKED for an empirical gravity claim** until exact public clock/orbit payloads are downloaded, hashed, and scored under the frozen preflight.

The package evaluates nine explicit closures over eight target-free fixtures. Of the {len(pair_rows)} pairwise comparisons, {distinct_pairs} are distinguishable. The sole exact degeneracy is C00 versus C01: integrating an exact potential gradient is merely endpoint redshift written as a path integral.

## Ranked open candidates

1. **C08 causal metric-memory path.** A causal state obeys `tau Dpsi/Dt + psi = S`, while photons receive an achromatic integrated response to `partial_t psi`. This is the best synthesis because it predicts orbit-phase hysteresis and multiple-image differences while remaining compatible in principle with metric reciprocity. It is not yet a theory: one action must derive the state, metric, photon propagation, and stress energy.
2. **C05 endpoint memory.** The cleanest immediate falsifier. It predicts a quadrature/hysteresis component in eccentric Galileo clock residuals but is not a genuinely accumulated path effect.
3. **C03 tidal-curvature column.** It avoids the arbitrary additive zero of Newtonian potential and predicts different shifts for equal endpoints along different curvature columns. It owes an energy-momentum sink and round-trip prediction.
4. **C06 path-memory opacity.** It directly realizes persistent cumulative loss, but same-sign forward/reverse accumulation is dissipative and must predict field heating and distance-duality consequences.
5. **C04 dispersive persistent medium.** Highly testable through its `nu^-2` signature, but it is presumptively ordinary plasma until multi-frequency controls fail.

## Retained failures and boundaries

- C01 is not new; it is endpoint redshift by the fundamental theorem for line integrals.
- C02 is genuinely cumulative but changes when a constant is added to the potential. It also lacks an energy sink and accumulates with the same sign on a return trip.
- C07 is the required known time-varying-metric comparator; a time-dependent potential can already shift photons along a path.
- Supernova expansion time dilation, Etherington reciprocity/photon number, dust, plasma, kinematics, source evolution, moving lenses, lens delays, and instrument/orbit errors are mandatory controls. The proposed laws are residual additions, not replacements for cosmological expansion.

## Next empirical falsifier

Use the public ESA/GSSC GREAT clock and orbit products for GSAT0201 and GSAT0202. For the frozen tau grid, test whether a causal filtered-potential quadrature transfers across clocks and both eccentric satellites while remaining absent in near-circular satellites and phase-scrambled controls. A `nu^-2` residual that disappears in the ionosphere-free combination supports plasma, not an achromatic time well. Exact payload URLs and SHA-256 receipts are required before scoring.

## Novelty boundary

The taxonomy and the specific C08 synthesis may support a methods/theory note after a deeper historical audit. This package does not establish historical novelty, a covariant completion, a fit to real data, or a gravity discovery. The closest bound primary neighbors are the two independent 2018 Galileo redshift tests, cluster gravitational redshift, supernova time dilation, distance duality, continuous GNSS-clock processing, and lensed-image redshift-difference work.
"""
    return text.encode()


def _artifact_payloads(config: Mapping[str, Any]) -> dict[str, bytes]:
    signatures, pairs = _synthetic_signatures()
    preflight = {
        "schema": "invariant-open-gravity-persistent-timewell-redshift-preflight-1.0",
        "primary": config["real_data_preflight"],
        "astrophysical": config["astrophysical_falsifier_preflight"],
        "mandatory_controls": config["mandatory_controls"],
        "response_opened": False,
        "scores_computed": 0,
        "content_sha256": "",
    }
    preflight["content_sha256"] = content_sha256({**preflight, "content_sha256": ""})
    return {
        "closure-ledger.csv": _csv_bytes(_closure_ledger(config)),
        "synthetic-signatures.csv": _csv_bytes(signatures),
        "pairwise-distinguishability.csv": _csv_bytes(pairs),
        "real-data-preflight.json": _canonical(preflight),
        "report.md": _report(config, pairs),
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
    signatures, pairs = _synthetic_signatures()
    payloads = _artifact_payloads(config)
    ledger = _closure_ledger(config)
    result: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_FREE_PERSISTENT_REDSHIFT_CLOSURES_EMPIRICAL_SCORE_BLOCKED_ON_PAYLOAD_RECEIPT",
        "decision": "ADVANCE_C08_CAUSAL_METRIC_MEMORY_AND_C05_ENDPOINT_MEMORY_TO_FROZEN_GALILEO_FALSIFIER_RETAIN_ALL_FAILURES",
        "input_sha256": bindings,
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "summary": {
            "closures": len(config["closures"]),
            "target_free_fixture_families": len(config["synthetic_fixture_ids"]),
            "synthetic_signature_rows": len(signatures),
            "pairwise_comparisons": len(pairs),
            "distinguishable_pairs": sum(bool(row["distinguishable"]) for row in pairs),
            "exact_degeneracies": sum(not bool(row["distinguishable"]) for row in pairs),
            "ranked_open_candidates": sum(row["candidate_rank"] != "" for row in ledger),
            "published_neighbors": len(config["published_neighbors"]),
            "mandatory_controls": len(config["mandatory_controls"]),
            "real_response_rows_scored": 0,
            "artifact_index": _artifact_index(payloads),
        },
        "candidate_ranking": [
            {
                "rank": int(row["candidate_rank"]),
                "closure_id": row["closure_id"],
                "disposition": row["disposition"],
            }
            for row in ledger
            if row["candidate_rank"] != ""
        ],
        "retained_failures": [
            {
                "closure_id": row["closure_id"],
                "disposition": row["disposition"],
                "reason": row["reason"],
            }
            for row in ledger
            if str(row["disposition"]).startswith("FAIL")
        ],
        "next_empirical_falsifier": "ESA/GSSC eccentric-Galileo cross-clock filtered-potential quadrature with near-circular, phase-scrambled, plasma, orbit, and environment controls",
        "source_status": config["real_data_preflight"]["source_status"],
        "claim_boundary": config["claim_boundary"],
        "access_accounting": config["access_contract"],
        "content_sha256": "",
    }
    result["content_sha256"] = content_sha256({**result, "content_sha256": ""})
    return result, payloads


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
    observed = _read_json(OUTPUT_PATH, "receipt")
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
