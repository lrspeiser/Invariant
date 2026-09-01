"""Same-field matter, photon, lensing, redshift, delay, and tensor closure compiler."""

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

CONFIG_PATH = Path("configs/open_gravity_same_law_matter_photon_closures_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_same_law_matter_photon_closures_v1.py")
TEST_PATH = Path("tests/test_open_gravity_same_law_matter_photon_closures_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-same-law-matter-photon-closures-v1/receipt.json")
ARTIFACT_DIRECTORY = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "b022a8f862bb86b332462c2ea556a5f535d0c218c959fe4f71c68bc8b85cd204"
_CONFIG_CONTENT_SHA256 = "c68412d10197a1adef081727f294fe8d5bb26ad23a8746b97a761bb5253c3e68"
_MODULE_SEMANTIC_SHA256 = "b2b09a8815cd909ace043c47c9315d24f28f07147ae743a3c627a9f58665419e"
_TEST_RAW_SHA256 = "502bd815b09be651e187d9ed9d2954f33df77b2677aa12e5084f375bb3ee21c7"
_SCHEMA = "invariant-open-gravity-same-law-matter-photon-closures-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-same-law-matter-photon-receipt-1.0"
_CLOSURE_IDS = tuple(
    f"L{i:02d}_{name}"
    for i, name in enumerate(
        (
            "GR_BARYON_METRIC",
            "GR_STARS_NFW_DARK_MATTER",
            "UNIVERSAL_CONFORMAL_SCALAR",
            "SCALAR_SLIP_METRIC",
            "TEVES_DISFORMAL_VECTOR_SCALAR",
            "TIED_REFRACTIVE_CONSTITUTIVE",
            "SPATIAL_NONLOCAL_SINGLE_METRIC",
            "TIED_PATH_MEMORY_CHARACTERISTIC",
            "CAUSAL_METRIC_MEMORY",
            "MASSIVE_CONFORMAL_SCALAR",
            "MASSIVE_VECTOR_SINGLE_METRIC",
            "MASSIVE_SPIN2_FIXED_SLIP",
            "PURE_PHOTON_REFRACTION_FAILURE",
            "MASSIVE_ONLY_ACCELERATION_FAILURE",
            "EXACT_GRADIENT_PATH_REWRITE",
            "NONRELATIVISTIC_AQUAL_ONLY",
        )
    )
)
_CHANNELS = (
    "massive_particle_acceleration",
    "photon_characteristic",
    "deflection",
    "shapiro_delay",
    "gravitational_redshift",
    "image_time_delay",
    "distance_duality_and_chromaticity",
    "tensor_cone",
)


class SameLawClosureError(RuntimeError):
    """Raised when a frozen closure, source, or output changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SameLawClosureError(message)


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SameLawClosureError(f"invalid {label}") from error
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-same-law-matter-photon-closures-v1",
        "package ID changed",
    )
    _require(
        config["status"] == "FROZEN_RESPONSE_BLIND_CLOSURE_COMPILER_AND_DATA_PREFLIGHT",
        "status changed",
    )
    _require(tuple(config["channel_names"]) == _CHANNELS, "channel registry changed")
    closures = config["closures"]
    _require(tuple(row["id"] for row in closures) == _CLOSURE_IDS, "closure registry changed")
    _require(len(closures) == 16, "closure count changed")
    _require(len({row["family"] for row in closures}) >= 14, "taxonomy breadth changed")
    _require(len(config["published_sources"]) == 10, "published source ledger changed")
    _require(len(config["mandatory_controls"]) == 12, "mandatory control ledger changed")
    _require(
        config["weak_field_convention"]["no_independent_lensing_multiplier"] is True,
        "same-law rule changed",
    )
    _require(config["claim_boundary"]["real_data_scored"] is False, "claim widened")
    _require(
        config["claim_boundary"]["historical_novelty_established"] is False, "novelty claim widened"
    )
    for stage in ("stage_1", "stage_2", "stage_3", "stage_4"):
        _require(
            config["real_data_preflight"][stage]["response_status"] == "NOT_DOWNLOADED_NOT_SCORED",
            f"{stage} response gate changed",
        )
    access = config["access_contract"]
    for key in (
        "raw_scientific_payloads_downloaded",
        "scientific_response_rows_opened",
        "scientific_response_rows_scored",
        "network_calls_by_builder",
        "external_model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(access[key] == 0, f"access boundary changed: {key}")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "output changed")
    _require(
        config["outputs"]["artifact_directory"] == ARTIFACT_DIRECTORY.as_posix(),
        "artifact directory changed",
    )


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic hash changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH, "same-law config")
    validate_config(config)
    return config


def _validate_local_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["local_bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing local binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"changed local binding: {row['role']}")
        observed[row["role"]] = digest
    _require(len(observed) == 4, "local binding count changed")
    return observed


def synthetic_fixtures() -> list[dict[str, Any]]:
    """Dimensionless source-side fixtures; no value is an observation."""
    base = {
        "u_grad": 1.0,
        "extra_grad": 0.6,
        "halo_grad": 0.8,
        "u_column": -0.9,
        "extra_column": -0.4,
        "halo_column": -0.7,
        "phi_endpoint": 0.2,
        "extra_endpoint": 0.1,
        "geometric_delay": 0.5,
        "frequency_ratio": 1.0,
        "memory_dt": 0.0,
        "range_factor": 1.0,
        "tensor_k_ratio": 1.0,
    }
    cases: list[tuple[str, str, dict[str, float]]] = [
        ("F01_STATIC_BASE", "base", {}),
        ("F02_EXTRA_DOMINATED", "extra", {"u_grad": 0.2, "extra_grad": 1.0}),
        ("F03_HALO_DOMINATED", "halo", {"u_grad": 0.2, "halo_grad": 1.1}),
        ("F04_EQUAL_ENDPOINT_TWO_PATHS", "short", {"extra_column": -0.2}),
        ("F04_EQUAL_ENDPOINT_TWO_PATHS", "long", {"extra_column": -1.1}),
        ("F05_SAME_STATE_MEMORY_PHASE", "rising", {"memory_dt": 0.3}),
        ("F05_SAME_STATE_MEMORY_PHASE", "falling", {"memory_dt": -0.3}),
        ("F06_TWO_FREQUENCIES", "low", {"frequency_ratio": 0.5}),
        ("F06_TWO_FREQUENCIES", "high", {"frequency_ratio": 2.0}),
        ("F07_MEDIATOR_RANGE", "near", {"range_factor": 0.9}),
        ("F07_MEDIATOR_RANGE", "far", {"range_factor": 0.1}),
        ("F08_TENSOR_DISPERSION", "low_k", {"tensor_k_ratio": 0.5}),
        ("F08_TENSOR_DISPERSION", "high_k", {"tensor_k_ratio": 4.0}),
    ]
    fixtures = []
    for fixture_id, case, updates in cases:
        row: dict[str, Any] = {"fixture_id": fixture_id, "case": case, **base}
        row.update(updates)
        fixtures.append(row)
    return fixtures


def _coverage(closure: Mapping[str, Any]) -> dict[str, str]:
    mode = closure["coverage_mode"]
    if mode == "ALL_EIGHT_DERIVED":
        return {name: "DERIVED" for name in _CHANNELS}
    if mode == "SEVEN_DERIVED_TENSOR_BLOCKED":
        return {name: ("BLOCKED" if name == "tensor_cone" else "DERIVED") for name in _CHANNELS}
    if mode == "SIX_DERIVED_TENSOR_BLOCKED_MASSIVE_DISCONNECTED":
        return {
            name: (
                "BLOCKED"
                if name == "tensor_cone"
                else "DERIVED_BUT_DISCONNECTED_FAIL"
                if name == "massive_particle_acceleration"
                else "DERIVED"
            )
            for name in _CHANNELS
        }
    if mode == "MASSIVE_ONLY_OTHER_SEVEN_BLOCKED":
        return {
            name: "DERIVED" if name == "massive_particle_acceleration" else "BLOCKED"
            for name in _CHANNELS
        }
    raise SameLawClosureError(f"unknown coverage mode: {mode}")


def _massive_range_factor(closure: Mapping[str, Any], fixture: Mapping[str, Any]) -> float:
    return (
        float(fixture["range_factor"])
        if closure["family"]
        in {
            "massive_scalar_mediator",
            "massive_vector_mediator",
            "massive_tensor_mediator",
        }
        else 1.0
    )


def predict(closure: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    coefficients = closure["coefficients"]
    range_factor = _massive_range_factor(closure, fixture)
    extra_grad = float(fixture["extra_grad"]) * range_factor
    extra_column = float(fixture["extra_column"]) * range_factor
    extra_endpoint = float(fixture["extra_endpoint"]) * range_factor
    phi_grad = (
        float(fixture["u_grad"])
        + float(coefficients["halo"]) * float(fixture["halo_grad"])
        + float(coefficients["phi_extra"]) * extra_grad
    )
    phi_column = (
        float(fixture["u_column"])
        + float(coefficients["halo"]) * float(fixture["halo_column"])
        + float(coefficients["phi_extra"]) * extra_column
    )
    phi_endpoint = (
        float(fixture["phi_endpoint"]) + float(coefficients["phi_extra"]) * extra_endpoint
    )
    coverage = _coverage(closure)
    result: dict[str, Any] = {
        "closure_id": closure["id"],
        "fixture_id": fixture["fixture_id"],
        "case": fixture["case"],
        "matter_acceleration": phi_grad,
        "same_law_gate": "PASS" if closure["same_constants"] else "FAIL_RETAINED",
    }
    if coverage["photon_characteristic"] == "BLOCKED":
        result.update(
            {
                "psi_gradient": None,
                "lensing_acceleration": None,
                "deflection": None,
                "shapiro_delay": None,
                "gravitational_redshift": None,
                "image_time_delay": None,
                "distance_duality_eta": None,
                "chromatic_log_slope": None,
            }
        )
    else:
        psi_grad = (
            float(fixture["u_grad"])
            + float(coefficients["halo"]) * float(fixture["halo_grad"])
            + float(coefficients["psi_extra"]) * extra_grad
        )
        psi_column = (
            float(fixture["u_column"])
            + float(coefficients["halo"]) * float(fixture["halo_column"])
            + float(coefficients["psi_extra"]) * extra_column
        )
        frequency_factor = (
            float(fixture["frequency_ratio"]) ** float(coefficients["dispersion_power"])
            if float(coefficients["photon_direct"]) != 0.0
            else 1.0
        )
        direct_gradient = float(coefficients["photon_direct"]) * extra_grad * frequency_factor
        direct_column = float(coefficients["photon_direct"]) * extra_column * frequency_factor
        lensing_acceleration = 0.5 * (phi_grad + psi_grad) + direct_gradient
        optical_column = phi_column + psi_column + 2.0 * direct_column
        result.update(
            {
                "psi_gradient": psi_grad,
                "lensing_acceleration": lensing_acceleration,
                "deflection": 2.0 * lensing_acceleration,
                "shapiro_delay": -optical_column,
                "gravitational_redshift": (
                    phi_endpoint + float(coefficients["memory_dt"]) * float(fixture["memory_dt"])
                ),
                "image_time_delay": float(fixture["geometric_delay"]) - optical_column,
                "distance_duality_eta": math.exp(
                    float(coefficients["opacity"]) * abs(extra_column)
                ),
                "chromatic_log_slope": (
                    float(coefficients["dispersion_power"])
                    if float(coefficients["photon_direct"]) != 0.0
                    else 0.0
                ),
            }
        )
    if coverage["tensor_cone"] == "BLOCKED":
        result["tensor_characteristic_speed_over_c"] = None
        result["tensor_group_speed_over_c"] = None
    else:
        mu_over_k = float(coefficients["tensor_mu_over_k"]) / float(fixture["tensor_k_ratio"])
        result["tensor_characteristic_speed_over_c"] = 1.0
        result["tensor_group_speed_over_c"] = 1.0 / math.sqrt(1.0 + mu_over_k**2)
    return result


def reconstruct_spatial_potential_gradient(
    matter_acceleration: float, lensing_acceleration: float
) -> float:
    """Exact weak-field same-metric inversion: Psi'=2*g_lens-Phi'."""
    return 2.0 * lensing_acceleration - matter_acceleration


def _predictions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        predict(closure, fixture)
        for closure in config["closures"]
        for fixture in synthetic_fixtures()
    ]


def _signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    fields = (
        "matter_acceleration",
        "psi_gradient",
        "lensing_acceleration",
        "deflection",
        "shapiro_delay",
        "gravitational_redshift",
        "image_time_delay",
        "distance_duality_eta",
        "chromatic_log_slope",
        "tensor_characteristic_speed_over_c",
        "tensor_group_speed_over_c",
    )
    return tuple(None if row[name] is None else round(float(row[name]), 12) for name in fields)


def equivalence_classes(config: Mapping[str, Any]) -> list[list[str]]:
    predictions = _predictions(config)
    signatures: dict[str, tuple[Any, ...]] = {}
    for closure in config["closures"]:
        rows = [row for row in predictions if row["closure_id"] == closure["id"]]
        signatures[closure["id"]] = tuple(_signature(row) for row in rows)
    groups: dict[tuple[Any, ...], list[str]] = {}
    for closure_id, signature in signatures.items():
        groups.setdefault(signature, []).append(closure_id)
    return sorted((sorted(group) for group in groups.values()), key=lambda row: row[0])


def _cross_channel_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    predictions = _predictions(config)
    closure_by_id = {row["id"]: row for row in config["closures"]}
    single_metric_ids = {
        row["id"]
        for row in config["closures"]
        if row["photon_mode"].startswith("SINGLE_")
        and _coverage(row)["photon_characteristic"] != "BLOCKED"
    }
    metric_rows = [row for row in predictions if row["closure_id"] in single_metric_ids]
    inversion_errors = [
        abs(
            reconstruct_spatial_potential_gradient(
                float(row["matter_acceleration"]), float(row["lensing_acceleration"])
            )
            - float(row["psi_gradient"])
        )
        for row in metric_rows
    ]
    deflection_errors = [
        abs(float(row["deflection"]) - 2.0 * float(row["lensing_acceleration"]))
        for row in metric_rows
    ]
    extra_fixture = next(
        row
        for row in predictions
        if row["closure_id"] == "L02_UNIVERSAL_CONFORMAL_SCALAR"
        and row["fixture_id"] == "F02_EXTRA_DOMINATED"
    )
    gr_fixture = next(
        row
        for row in predictions
        if row["closure_id"] == "L00_GR_BARYON_METRIC"
        and row["fixture_id"] == "F02_EXTRA_DOMINATED"
    )
    memory_rows = {
        row["case"]: row
        for row in predictions
        if row["closure_id"] == "L08_CAUSAL_METRIC_MEMORY"
        and row["fixture_id"] == "F05_SAME_STATE_MEMORY_PHASE"
    }
    frequency_rows = {
        row["case"]: row
        for row in predictions
        if row["closure_id"] == "L05_TIED_REFRACTIVE_CONSTITUTIVE"
        and row["fixture_id"] == "F06_TWO_FREQUENCIES"
    }
    base_slip = 0.5 * (
        float(frequency_rows["low"]["matter_acceleration"])
        + float(frequency_rows["low"]["psi_gradient"])
    )
    low_direct = float(frequency_rows["low"]["lensing_acceleration"]) - base_slip
    high_direct = float(frequency_rows["high"]["lensing_acceleration"]) - base_slip
    _require(
        closure_by_id["L12_PURE_PHOTON_REFRACTION_FAILURE"]["same_constants"] is False,
        "negative same-law control changed",
    )
    return {
        "single_metric_closures": len(single_metric_ids),
        "single_metric_rows": len(metric_rows),
        "maximum_slip_inversion_error": max(inversion_errors),
        "maximum_deflection_identity_error": max(deflection_errors),
        "conformal_scalar_changes_matter": (
            extra_fixture["matter_acceleration"] != gr_fixture["matter_acceleration"]
        ),
        "conformal_scalar_direct_lensing_cancels": math.isclose(
            float(extra_fixture["lensing_acceleration"]),
            float(gr_fixture["lensing_acceleration"]),
            abs_tol=1.0e-14,
        ),
        "causal_memory_same_instantaneous_dynamics": math.isclose(
            float(memory_rows["rising"]["matter_acceleration"]),
            float(memory_rows["falling"]["matter_acceleration"]),
            abs_tol=1.0e-14,
        ),
        "causal_memory_opposite_redshift_phase": (
            float(memory_rows["rising"]["gravitational_redshift"])
            > float(memory_rows["falling"]["gravitational_redshift"])
        ),
        "constitutive_direct_low_to_high_frequency_ratio": low_direct / high_direct,
        "forbidden_separate_photon_controls_retained": 1,
        "massive_only_underdefined_controls_retained": 2,
    }


def _channel_ledger(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for closure in config["closures"]:
        coverage = _coverage(closure)
        rows.append(
            {
                "closure_id": closure["id"],
                "family": closure["family"],
                "status": closure["status"],
                "same_law_gate": "PASS" if closure["same_constants"] else "FAIL_RETAINED",
                **coverage,
                "health_flags": ";".join(closure["health_flags"]),
            }
        )
    return rows


def _counterexamples() -> list[dict[str, str]]:
    return [
        {
            "id": "CE01_ACCELERATION_DOES_NOT_FIX_LENSING",
            "finding": "The same massive-particle acceleration admits conformal, slip, disformal, refractive, and no-photon completions with different lensing.",
            "consequence": "A rotation curve or cluster acceleration fit is not a lensing prediction.",
        },
        {
            "id": "CE02_CONFORMAL_NULL_CANCELLATION",
            "finding": "A universal conformal scalar changes massive motion while its direct weak-field contribution cancels from Phi+Psi.",
            "consequence": "Extra attraction does not automatically mean extra light bending.",
        },
        {
            "id": "CE03_SEPARATE_PHOTON_KNOB",
            "finding": "A photon-only refractive multiplier can fit lensing independently by construction.",
            "consequence": "It is retained as a failure and cannot count as same-law evidence.",
        },
        {
            "id": "CE04_MASS_SHEET_AND_ANISOTROPY",
            "finding": "Lens mass-sheet/source-position degeneracies and stellar anisotropy can mimic gravitational slip.",
            "consequence": "Joint resolved kinematics, line-of-sight structure, and shared nuisance contracts are mandatory.",
        },
        {
            "id": "CE05_PLASMA_AND_DUST",
            "finding": "Ordinary media generate dispersion, extinction, scattering, and differential delays.",
            "consequence": "A constitutive gravity claim must transfer achromatically or beat explicit nu^-2 and extinction controls.",
        },
        {
            "id": "CE06_TENSOR_CONE",
            "finding": "A static metric closure can match dynamics and lensing while its propagating tensor sector fails multimessenger speed bounds.",
            "consequence": "Empirical static interest is recorded before a separate causal/health disposition.",
        },
    ]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _prediction_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _report_bytes(config: Mapping[str, Any], checks: Mapping[str, Any]) -> bytes:
    classes = equivalence_classes(config)
    nontrivial = [group for group in classes if len(group) > 1]
    lines = [
        "# Same-law matter and photon closure audit",
        "",
        "## What was built",
        "",
        "Sixteen closures now turn one source-side field prescription into massive motion and, wherever defined, the photon characteristic, bending, Shapiro delay, gravitational redshift, image time delay, distance-duality/chromaticity behavior, and tensor cone. A separate lens multiplier is forbidden; two deliberately incomplete controls are retained to prove the gate catches it.",
        "",
        "The exact weak-field identity is `Psi' = 2 g_lens - g_dyn`. Once dynamics fixes `Phi'` and lensing fixes `(Phi'+Psi')/2`, slip is reconstructed rather than fitted. The same `Phi+Psi` must then normalize deflection, Shapiro delay, and the potential part of image time delay. This makes the law overconstrained by additional channels.",
        "",
        "## What the synthetic audit learned",
        "",
        f"- {checks['single_metric_closures']} closures use one metric and satisfy the slip inversion on {checks['single_metric_rows']} fixture rows with maximum error {checks['maximum_slip_inversion_error']:.3g}.",
        f"- The conformal-scalar control changes matter motion but leaves direct lensing equal to the baryonic GR control: {checks['conformal_scalar_direct_lensing_cancels']}.",
        f"- The tied refractive term changes by a factor {checks['constitutive_direct_low_to_high_frequency_ratio']:.1f} between the frozen low/high frequencies, giving a sharp plasma-like falsifier.",
        f"- The causal metric-memory fixture has the same instantaneous dynamics in rising and falling phases but opposite redshift phase ordering: {checks['causal_memory_opposite_redshift_phase']}.",
        f"- Nontrivial observational equivalence classes on this fixture suite: {len(nontrivial)}. They are retained, not counted as independent wins.",
        "",
        "## Strongest potentially publishable result",
        "",
        "The strongest present result is methodological and exact, not an empirical discovery: a same-law closure compiler plus a cross-channel consistency triangle. Joint dynamics and lensing reconstruct the two weak-field potentials; the reconstructed field then predicts Shapiro/image delays and static redshift without a new photon coefficient. A failure localizes what must be added: nonmetric photon propagation, opacity/dispersion, time dependence, or an underived source sector. Similar ingredients are established in PPN/slip analyses; historical novelty of this combined audit and staged falsifier has not been established.",
        "",
        "Among physical hypotheses, `L08_CAUSAL_METRIC_MEMORY` ranks first because one causal state produces a response in every static matter/photon channel plus a phase-sensitive redshift term. `L06_SPATIAL_NONLOCAL_SINGLE_METRIC` ranks second because one 3D convolution changes dynamics and lensing together. Both remain action/health incomplete.",
        "",
        "## Exact next falsifier",
        "",
        "First reproduce ESO 325-G004 from HST arcs and MUSE resolved stellar kinematics under one nuisance contract and reconstruct slip with no photon multiplier. Then transfer the unchanged closure to whole-galaxy SLACS holdouts selected by the frozen name hash. Only a survivor may enter the third-channel SN Refsdal time-delay test or CLASH cluster geometry. Exact payload hashes must be frozen before response rows are opened.",
        "",
        "The published ESO 325-G004 abstract reports gamma=0.97 +/- 0.09, which is a benchmark value only; no payload was downloaded and no closure was scored here.",
        "",
        "## Limitations",
        "",
        "The coefficient fixtures are dimensionless identifiability probes, not fitted galaxies. Most new closures lack a covariant action, stress-energy ledger, or derived tensor principal symbol. Stellar mass-to-light ratio, anisotropy, mass-sheet/source-position transformations, line-of-sight structure, triaxiality, plasma, dust, microlensing, and cosmology can all imitate pieces of the signal. Static empirical interest would be retained even if a later theory-health gate fails.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Establishes: {item}" for item in config["claim_boundary"]["establishes"])
    lines.extend(
        f"- Does not establish: {item}" for item in config["claim_boundary"]["does_not_establish"]
    )
    return ("\n".join(lines) + "\n").encode()


def build_artifacts(config: Mapping[str, Any]) -> dict[str, bytes]:
    predictions = _predictions(config)
    checks = _cross_channel_checks(config)
    closure_cards = [
        {
            **closure,
            "channel_coverage": _coverage(closure),
            "retained_even_if_failure": True,
            "empirical_grade": "UNTESTED_RESPONSE_BLIND_PREFLIGHT",
        }
        for closure in config["closures"]
    ]
    return {
        "closure-cards.jsonl": _jsonl_bytes(closure_cards),
        "channel-ledger.csv": _csv_bytes(_channel_ledger(config)),
        "synthetic-predictions.csv": _prediction_csv_bytes(predictions),
        "cross-channel-checks.json": _json_bytes(checks),
        "equivalence-classes.json": _json_bytes(equivalence_classes(config)),
        "counterexamples.json": _json_bytes(_counterexamples()),
        "response-blind-data-preflight.json": _json_bytes(config["real_data_preflight"]),
        "report.md": _report_bytes(config, checks),
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_local_bindings(config)
    predictions = _predictions(config)
    checks = _cross_channel_checks(config)
    ledgers = _channel_ledger(config)
    classes = equivalence_classes(config)
    artifacts = build_artifacts(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_SAME_LAW_CROSS_CHANNEL_COMPILER_REAL_RESPONSES_UNOPENED",
        "bindings": bindings,
        "summary": {
            "closures": len(config["closures"]),
            "families": len({row["family"] for row in config["closures"]}),
            "channels_per_closure": len(_CHANNELS),
            "synthetic_fixtures": len(synthetic_fixtures()),
            "synthetic_prediction_rows": len(predictions),
            "same_law_pass_closures": sum(row["same_law_gate"] == "PASS" for row in ledgers),
            "retained_same_law_failures": sum(
                row["same_law_gate"] == "FAIL_RETAINED" for row in ledgers
            ),
            "fully_derived_eight_channel_closures": sum(
                all(row[name] == "DERIVED" for name in _CHANNELS) for row in ledgers
            ),
            "equivalence_classes": len(classes),
            "nontrivial_equivalence_classes": sum(len(group) > 1 for group in classes),
            "real_response_rows_opened": 0,
            "real_response_rows_scored": 0,
        },
        "exact_results": checks,
        "ranking": config["ranking"],
        "strongest_potentially_publishable_result": {
            "class": "METHODS_AND_IDENTIFIABILITY_RESULT_NOT_EMPIRICAL_DISCOVERY",
            "statement": "Joint dynamics and lensing reconstruct Phi and Psi; deflection, Shapiro/image delay, and static redshift are then cross-channel predictions with no independent photon multiplier.",
            "historical_novelty": "NOT_ESTABLISHED_PPN_SLIP_AND_RELATIVISTIC_MOND_NEIGHBORS_EXIST",
            "leading_physical_hypothesis": "L08_CAUSAL_METRIC_MEMORY",
        },
        "exact_next_falsifier": {
            "first": "ESO325_G004_JOINT_SLIP_REPRODUCTION",
            "transfer": "SLACS_WHOLE_GALAXY_TRANSFER",
            "third_channel": "SN_REFSDAL_TIME_DELAY",
            "rule": "no object-specific photon coefficient and no retuning between stages",
            "response_status": "NOT_DOWNLOADED_NOT_SCORED",
        },
        "counterexamples": _counterexamples(),
        "artifact_manifest": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(artifacts.items())
        },
        "artifact_bindings": {
            "config": {"path": CONFIG_PATH.as_posix(), "sha256": file_sha256(CONFIG_PATH)},
            "module": {"path": MODULE_PATH.as_posix(), "sha256": file_sha256(MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": file_sha256(TEST_PATH)},
        },
        "published_sources": config["published_sources"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "decision": "ADVANCE_CAUSAL_MEMORY_AND_SPATIAL_NONLOCAL_SAME_METRIC_TO_FROZEN_LENS_DYNAMICS_FALSIFIER_RETAIN_ALL_FAILURES",
    }
    _require(checks["maximum_slip_inversion_error"] <= 1.0e-14, "same-metric slip inversion failed")
    _require(checks["maximum_deflection_identity_error"] <= 1.0e-14, "deflection identity failed")
    _require(
        math.isclose(
            checks["constitutive_direct_low_to_high_frequency_ratio"],
            16.0,
            abs_tol=1.0e-12,
        ),
        "chromatic falsifier changed",
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
    observed = _read_json(OUTPUT_PATH, "same-law receipt")
    _require(observed == build_receipt(), "receipt differs from deterministic rebuild")
    for name, expected in build_artifacts(load_config()).items():
        path = ARTIFACT_DIRECTORY / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == expected, f"artifact differs: {name}")


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
