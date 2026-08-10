from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-g4-real-sun-interval-source-audit-1.0"
FACT_CLASSES = {"raw", "calibrated", "model_dependent"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_content = descriptor.get("content_sha256")
    if expected_content is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected_content or _sha(body) != expected_content:
            raise ValueError(f"bound content hash mismatch: {descriptor['path']}")
    return value


def _validate_predecessor(value: dict[str, Any], descriptor: dict[str, Any]) -> None:
    if value.get("schema_version") != "sigma-g4-source-class-scalar-uniqueness-audit-1.0":
        raise ValueError("source-class predecessor schema mismatch")
    records = value.get("candidate_records", [])
    if len(records) != 1:
        raise ValueError("source-class predecessor target is not unique")
    record = records[0]
    required = descriptor["required_certificates"]
    if (
        record.get("seed_id") != descriptor["seed_id"]
        or record.get("action_sha256") != descriptor["action_sha256"]
        or record.get("source_class_theorem_decision") != "pass"
        or record.get("decision") != "blocked"
    ):
        raise ValueError("source-class predecessor target mismatch")
    actual = {
        "candidate_binding_sha256": record["provenance"]["binding_sha256"],
        "global_coupling_sha256": record["global_coupling_lipschitz_certificate"][
            "content_sha256"
        ],
        "coercivity_sha256": record["source_class_coercivity_certificate"][
            "content_sha256"
        ],
        "instantiation_contract_sha256": record[
            "minimal_real_source_instantiation_contract"
        ]["content_sha256"],
    }
    if actual != required:
        raise ValueError("source-class predecessor certificate mismatch")


def _validate_seals(
    protocol: dict[str, Any], audit: dict[str, Any], registration: dict[str, Any]
) -> None:
    if protocol.get("status") != "sealed" or protocol.get("data_opened") is not False:
        raise ValueError("Solar protocol is not sealed")
    if (
        audit.get("status") != "pass"
        or audit.get("observational_dataset_opened") is not False
        or audit.get("formula_search_authorized") is not False
    ):
        raise ValueError("Solar protocol audit does not remain fail-closed")
    if (
        registration.get("status") != "metadata_registered_data_sealed"
        or registration.get("data_opened") is not False
        or registration.get("candidate_use_authorized") is not False
        or registration.get("readiness", {}).get("primary_files_downloaded") is not False
    ):
        raise ValueError("Solar source registration does not remain sealed")


def _validate_fact_registry(registry: list[dict[str, Any]]) -> None:
    expected = {
        "iau_nominal_solar_radius": "calibrated",
        "iau_nominal_solar_mass_parameter": "calibrated",
        "nist_speed_of_light": "calibrated",
        "nist_gravitational_constant": "calibrated",
        "nasa_photospheric_mean_radius": "calibrated",
        "nasa_solar_wind_noncompact_extent": "calibrated",
        "nasa_compiled_solar_mass": "model_dependent",
        "nasa_model_central_density": "model_dependent",
        "nasa_model_central_pressure": "model_dependent",
        "helioseismic_density_inversion": "model_dependent",
    }
    actual = {item.get("fact_id"): item.get("classification") for item in registry}
    if len(actual) != len(registry) or actual != expected:
        raise ValueError("authoritative fact registry or classifications changed")
    for item in registry:
        if item["classification"] not in FACT_CLASSES:
            raise ValueError("fact has an unsupported evidence classification")
        if item.get("used_to_instantiate_theorem") is not False:
            raise ValueError("unqualified source fact was promoted to theorem evidence")
        if not item.get("authority") or not item.get("source_url"):
            raise ValueError("fact lacks source provenance")
    pdf_hashes = {
        item["fact_id"]: item.get("remote_document_sha256") for item in registry
    }
    if pdf_hashes["iau_nominal_solar_radius"] != (
        "21644e136a12fe301fc29398433a60dcf1e810b85b14934f8b8499956fd8e4a1"
    ):
        raise ValueError("IAU primary-document fingerprint changed")
    if pdf_hashes["nist_speed_of_light"] != (
        "5c4a2c8127a80698d05f4d682bd4cac200dabcd9c2e9dd783661a0aa60f4bb1c"
    ):
        raise ValueError("NIST primary-document fingerprint changed")


def _diagnostics() -> dict[str, Any]:
    radius = Fraction(695_700_000)
    mu = Fraction(13_271_244) * 10**13
    c = Fraction(299_792_458)
    nominal_compactness = 2 * mu / (radius * c**2)
    if nominal_compactness != Fraction(31_598_200_000_000, 7_443_618_783_895_286_097):
        raise ValueError("nominal compactness arithmetic changed")

    gravity = Fraction(667_430, 10**16)
    central_density = Fraction(162_200)
    density_radius_ratio = gravity * central_density * radius**2 / c**2
    central_pressure = Fraction(24_770_000_000_000_000)
    pressure_trace_ratio = 3 * central_pressure / (central_density * c**2)
    body = {
        "nominal_compactness_calibration": {
            "formula": "2*(GM_sun)^N/(R_sun^N*c^2)",
            "exact_fraction": str(nominal_compactness),
            "decimal": float(nominal_compactness),
            "status": "pass_calibration_only",
            "theorem_evidence": False,
            "reason": "IAU B3 defines nominal conversion factors, not true Solar intervals",
        },
        "central_density_model_diagnostic": {
            "formula": "G_CODATA*rho_center_model*(R_sun^N)^2/c^2",
            "exact_fraction": str(density_radius_ratio),
            "decimal": float(density_radius_ratio),
            "below_source_class_threshold_1_over_1000": density_radius_ratio
            < Fraction(1, 1000),
            "status": "model_dependent_counterfactual_only",
            "theorem_evidence": False,
        },
        "central_pressure_trace_model_diagnostic": {
            "formula": "3*p_center_model/(rho_center_model*c^2)",
            "exact_fraction": str(pressure_trace_ratio),
            "decimal": float(pressure_trace_ratio),
            "status": "model_dependent_single_point_only",
            "theorem_evidence": False,
        },
    }
    return {**body, "content_sha256": _sha(body)}


def _interval_assessments() -> list[dict[str, Any]]:
    return [
        {
            "requirement_id": "source_support_radius_upper",
            "status": "blocked",
            "available": "nominal and compiled photospheric radii; mission evidence for an extended atmosphere and solar wind",
            "decisive_negative_control": "photospheric_radius_is_not_total_trace_compact_support",
            "missing": "registered finite trace-support radius with uncertainty, or a resolved exterior-tail profile proving the Kato/Birman-Schwinger bound",
        },
        {
            "requirement_id": "total_mass_and_compactness",
            "status": "blocked",
            "available": "exact nominal compactness calibration and a compiled gravity-model Solar mass",
            "missing": "candidate-independent true mass/compactness interval, or an independently audited weak-geometry solution for the declared source",
        },
        {
            "requirement_id": "trace_density_or_concentration_upper",
            "status": "blocked",
            "available": "standard-solar-model central density and helioseismic model inversions",
            "missing": "rigorous global upper interval for tau=(-T_E)/c^2 or a hash-bound resolved trace profile including atmosphere/wind and uncertainty",
        },
        {
            "requirement_id": "pressure_trace_sign",
            "status": "blocked",
            "available": "one standard-solar-model central pressure value",
            "missing": "global material certificate that 0<=epsilon-sum_i(p_i), including anisotropy, atmosphere, and wind",
        },
        {
            "requirement_id": "static_geometry_intervals",
            "status": "blocked",
            "available": "weak nominal compactness diagnostic only",
            "missing": "non-circular candidate-independent bounds on lapse, inverse-metric ellipticity, and volume density over the full source/tail domain",
        },
        {
            "requirement_id": "scalar_boundary_and_topology",
            "status": "blocked",
            "available": "none; these are theory-domain assumptions, while the real Sun rotates, oscillates, evolves, and drives an outflow",
            "missing": "registered static approximation, one-end/no-inner-boundary topology, chi_infinity=0, D1,2 falloff, and vanishing boundary-flux contract",
        },
    ]


def build_g4_real_sun_interval_source_audit(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("campaign eligibility is not fail-closed")
    if config.get("observational_authorization") is not False:
        raise ValueError("observational authorization must remain false")
    registry = config["authoritative_fact_registry"]
    _validate_fact_registry(registry)

    predecessor = _load_bound(root, config["source_bindings"]["predecessor"])
    _validate_predecessor(predecessor, config["source_bindings"]["predecessor"])
    protocol = _load_bound(root, config["source_bindings"]["solar_protocol"])
    audit = _load_bound(root, config["source_bindings"]["solar_protocol_audit"])
    registration = _load_bound(root, config["source_bindings"]["source_registration"])
    _validate_seals(protocol, audit, registration)

    assessments = _interval_assessments()
    if any(item["status"] != "blocked" for item in assessments):
        raise ValueError("real-Sun theorem premise was opened without exact evidence")
    diagnostics = _diagnostics()
    class_counts = {
        name: sum(item["classification"] == name for item in registry)
        for name in sorted(FACT_CLASSES)
    }
    registry_body = {
        "retrieved_utc_date": config["retrieved_utc_date"],
        "facts": registry,
        "classification_counts": class_counts,
        "raw_values_opened": 0,
        "tracking_target_values_opened": 0,
    }
    fact_registry = {**registry_body, "content_sha256": _sha(registry_body)}
    circularity = {
        "GR_fitted_ephemeris_residual_as_truth": "reject",
        "fitted_PPN_as_truth": "reject",
        "standard_solar_model_as_candidate_evidence": "reject",
        "nominal_conversion_constants_as_true_intervals": "reject",
        "photosphere_as_total_trace_support": "reject",
    }
    provenance_body = {
        "predecessor_content_sha256": config["source_bindings"]["predecessor"][
            "content_sha256"
        ],
        "predecessor_candidate_binding_sha256": config["source_bindings"][
            "predecessor"
        ]["required_certificates"]["candidate_binding_sha256"],
        "fact_registry_sha256": fact_registry["content_sha256"],
        "diagnostics_sha256": diagnostics["content_sha256"],
        "assessment_sha256": _sha(assessments),
        "sealed_protocol_file_sha256": config["source_bindings"]["solar_protocol"][
            "file_sha256"
        ],
        "sealed_registration_file_sha256": config["source_bindings"][
            "source_registration"
        ]["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "source_bindings": config["source_bindings"],
        "decision": "blocked",
        "first_missing_premise": "registered_finite_trace_support_or_resolved_exterior_tail_Kato_bound",
        "candidate_independent_source_audit": True,
        "candidate_rejection_found": False,
        "authoritative_fact_registry": fact_registry,
        "calibration_and_model_diagnostics": diagnostics,
        "interval_assessments": assessments,
        "theorem_requirement_counts": {"pass": 0, "blocked": 6},
        "calibration_diagnostic_pass_count": 1,
        "no_circularity_ledger": circularity,
        "real_source_interval_certificate_admissible": False,
        "real_solar_bundle_admissible": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "tracking_target_values_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Authoritative metadata supports calibration diagnostics but no rigorous real-Sun "
            "interval required by the source-class theorem. IAU nominal constants are conversion "
            "factors, Solar interior density and pressure are model-dependent, and the atmosphere "
            "and wind prevent treating the photospheric radius as exact compact trace support. "
            "The source certificate and every downstream observational opening remain blocked."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
