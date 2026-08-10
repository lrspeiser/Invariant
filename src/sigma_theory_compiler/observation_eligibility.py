from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REQUIRED_FORMAL_GATES = (
    "field_contract",
    "static_dictionary_derivation",
    "higher_jet_regularity",
    "nonlinear_x_regularity",
    "adm_decomposition",
    "legendre_map",
    "generated_dirac_closure",
    "parameter_domain",
    "covariant_variation",
    "covariant_identity",
    "adm_dirac",
    "hamiltonian_stability",
    "principal_symbol",
)


def _allowed_https_host(value: str, allowed_hosts: set[str]) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname in allowed_hosts
        and parsed.username is None
        and parsed.password is None
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolved_artifact_path(health_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (health_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _portable_path(path: Path) -> str:
    """Return a repository-relative provenance path when possible."""

    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def audit_solar_observable_protocol(
    protocol_path: str | Path, evidence_policy_path: str | Path
) -> dict[str, Any]:
    """Validate the sealed direct-observable Solar-System test contract.

    Passing this audit never opens data. It only proves that the future dataset
    contract separates direct signals from calibrated, derived, model-dependent,
    and latent quantities and prevents target leakage into theory selection.
    """

    protocol_path = Path(protocol_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-solar-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load Solar observable protocol: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-solar-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load observational evidence policy: {error}"],
            "observational_dataset_opened": False,
        }

    if protocol.get("schema_version") != "sigma-solar-observable-protocol-1.0":
        errors.append("unsupported Solar observable protocol schema")
    if protocol.get("status") != "sealed":
        errors.append("Solar observable protocol must remain sealed")
    if protocol.get("data_opened") is not False:
        errors.append("Solar observable protocol cannot claim opened data")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("unsupported observational evidence policy schema")
    if policy.get("status") != "frozen":
        errors.append("observational evidence policy is not frozen")

    classes = protocol.get("quantity_classes", {})
    required_classes = {"raw", "calibrated", "derived", "model_dependent", "latent"}
    if set(classes) != required_classes:
        errors.append("quantity classes must be exactly raw/calibrated/derived/model-dependent/latent")
    for name in required_classes:
        if not isinstance(classes.get(name, {}).get("examples"), list) or not classes.get(
            name, {}
        ).get("examples"):
            errors.append(f"quantity class {name} lacks examples")
    if classes.get("model_dependent", {}).get("allowed_as_prediction_truth") is not False:
        errors.append("model-dependent quantities cannot be prediction truth")
    if classes.get("latent", {}).get("allowed_as_input_or_target") is not False:
        errors.append("latent quantities cannot be inputs or targets")
    derived_rule = str(classes.get("derived", {}).get("rule", "")).casefold()
    if not all(token in derived_rule for token in ("raw inputs", "transformation", "covariance")):
        errors.append("derived quantities lack raw-input, transformation, and covariance provenance")

    channel = protocol.get("measurement_channel", {})
    inputs = channel.get("inputs", [])
    targets = channel.get("prediction_targets", [])
    allowed_inputs = channel.get("allowed_formula_inputs", [])
    forbidden_inputs = set(channel.get("forbidden_formula_inputs", []))
    if not isinstance(inputs, list) or len(inputs) < 4:
        errors.append("Solar measurement channel lacks direct signal classes")
    target_text = " ".join(str(item) for item in targets).casefold()
    if "held-out" not in target_text or not any(
        token in target_text for token in ("light-time", "frequency", "angular")
    ):
        errors.append("Solar targets must be held-out direct timing/frequency/angular signals")
    contaminated = " ".join(
        [*(str(item) for item in inputs), *(str(item) for item in allowed_inputs), *targets]
    ).casefold()
    for prohibited in (
        "fitted ppn gamma",
        "fitted ppn beta",
        "precomputed shapiro residual",
        "perihelion anomaly label",
        "ephemeris residual as truth",
        "gr-derived correction",
    ):
        if prohibited in contaminated:
            errors.append(f"allowed Solar channel contains prohibited input: {prohibited}")
    required_forbidden = {
        "fitted PPN gamma or beta",
        "precomputed Shapiro, perihelion, or light-deflection residual labeled by a gravity model",
        "ephemeris state or residual estimated using held-out target records",
        "GR-derived correction or post-fit anomaly treated as truth",
        "object-specific gravitational coupling or screening parameter",
        "cosmological redshift-derived distance",
    }
    missing_forbidden = sorted(required_forbidden - forbidden_inputs)
    if missing_forbidden:
        errors.append("missing forbidden Solar inputs: " + ", ".join(missing_forbidden))
    initial_state_rule = str(channel.get("initial_state_rule", "")).casefold()
    if not all(
        token in initial_state_rule
        for token in ("training", "frozen", "held-out", "uncertainty")
    ):
        errors.append("initial-state estimation is not frozen before held-out prediction")

    split = protocol.get("split_contract", {})
    if split.get("unit") != "tracking pass or observing session":
        errors.append("Solar split unit must be a whole tracking pass or observing session")
    if split.get("group_leakage_forbidden") is not True:
        errors.append("Solar tracking-session leakage is not forbidden")
    roles = set(split.get("roles", []))
    if "untouched target-blind test sessions" not in roles:
        errors.append("target-blind Solar test-session role is missing")
    sealed_rule = str(split.get("sealed_test_rule", "")).casefold()
    if not all(
        token in sealed_rule
        for token in ("action hash", "initial state", "stopping rule", "inaccessible")
    ):
        errors.append("Solar sealed-test rule does not freeze theory, state, and stopping decisions")

    scoring = protocol.get("scoring_contract", {})
    if scoring.get("object_specific_gravity_parameters") != 0:
        errors.append("object-specific Solar gravitational parameters must equal zero")
    if "covariance" not in str(scoring.get("fit_term", "")).casefold():
        errors.append("Solar fit score must carry measurement and calibration covariance")
    selection_rule = str(scoring.get("selection_rule", "")).casefold()
    if not all(token in selection_rule for token in ("validation", "no test", "freeze")):
        errors.append("Solar selection rule does not isolate the sealed test channel")
    if not scoring.get("required_baselines"):
        errors.append("Solar reference and instrument-null baselines are missing")

    prohibited_truth = {
        str(item).casefold() for item in protocol.get("prohibited_truth_or_rescue", [])
    }
    for required in (
        "fitted ppn parameter",
        "gravity-model-derived ephemeris residual",
        "dark matter or invisible halo",
        "redshift-derived distance",
        "supernova distance modulus",
    ):
        if required not in prohibited_truth:
            errors.append(f"missing prohibited Solar truth/rescue quantity: {required}")
    if policy.get("supernovae", {}).get("default_status") != "excluded":
        errors.append("supernova distances are not excluded by the evidence policy")
    if "treating redshift as a distance" not in policy.get("redshift", {}).get(
        "not_allowed_by_default", []
    ):
        errors.append("redshift-distance inference is not excluded by default")

    opening_requirements = protocol.get("opening_requirements", [])
    if not isinstance(opening_requirements, list) or len(opening_requirements) < 6:
        errors.append("Solar dataset-opening requirements are incomplete")
    passed = not errors
    return {
        "schema_version": "sigma-solar-observable-audit-1.0",
        "status": "pass" if passed else "fail",
        "protocol": _portable_path(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "policy": _portable_path(policy_path),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "errors": errors,
        "quantity_classes": sorted(classes),
        "prediction_targets": targets,
        "split_unit": split.get("unit"),
        "object_specific_gravity_parameters": scoring.get(
            "object_specific_gravity_parameters"
        ),
        "redshift_distance_allowed_by_default": False,
        "supernova_default_status": policy.get("supernovae", {}).get("default_status"),
        "observational_dataset_opened": False,
        "formula_search_authorized": False,
        "next_required_binding": (
            "an eligible exact action hash and an independently audited, hash-committed "
            "Solar direct-observation dataset manifest"
        ),
    }


def audit_solar_source_registration(
    manifest_path: str | Path,
    protocol_path: str | Path,
    evidence_policy_path: str | Path,
) -> dict[str, Any]:
    """Audit metadata registration for a still-sealed Solar data source.

    This gate verifies authoritative-source identity and remote catalog
    fingerprints. It intentionally does not claim that primary records have been
    downloaded, parsed, calibrated, split, or made available to formula search.
    """

    manifest_path = Path(manifest_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-solar-source-registration-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load Solar source registration: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-solar-source-registration-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load bound protocol or policy: {error}"],
            "observational_dataset_opened": False,
        }

    if manifest.get("schema_version") != "sigma-observation-source-registration-1.0":
        errors.append("unsupported observation-source registration schema")
    if manifest.get("status") != "metadata_registered_data_sealed":
        errors.append("source registration must remain metadata-only and sealed")
    if manifest.get("data_opened") is not False:
        errors.append("source registration cannot claim opened primary data")
    if manifest.get("candidate_use_authorized") is not False:
        errors.append("source registration cannot authorize candidate use")
    if protocol.get("schema_version") != "sigma-solar-observable-protocol-1.0":
        errors.append("bound Solar protocol schema is unsupported")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("bound observational policy schema is unsupported")
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    bindings = manifest.get("bindings", {})
    if bindings.get("solar_protocol_sha256") != protocol_sha:
        errors.append("source registration Solar protocol hash mismatch")
    if bindings.get("evidence_policy_sha256") != policy_sha:
        errors.append("source registration evidence-policy hash mismatch")

    source = manifest.get("source", {})
    if source.get("authority") != "NASA Planetary Data System":
        errors.append("Solar source authority must be NASA Planetary Data System")
    if source.get("dataset_id") != "CO-SS-RSS-1-SCE1-V1.0":
        errors.append("unexpected Cassini SCE1 dataset identifier")
    for key in ("profile_url", "archive_url"):
        value = str(source.get(key, ""))
        if not _allowed_https_host(value, {"pds.nasa.gov", "atmos.nmsu.edu"}):
            errors.append(f"source {key} is not an allowlisted authoritative HTTPS URL")
    if source.get("volume_count") != 8:
        errors.append("Cassini SCE1 registration must retain all eight archive volumes")
    if source.get("start_time") != "2002-06-06T12:00:00Z" or source.get(
        "stop_time"
    ) != "2002-07-05T12:00:00Z":
        errors.append("Cassini SCE1 source coverage differs from the PDS profile")

    indexes = manifest.get("remote_catalog_fingerprints", [])
    expected_catalog_pairs = {
        (f"cors_{volume:04d}", name)
        for volume in range(21, 29)
        for name in ("cumindex.lbl", "cumindex.tab")
    }
    actual_catalog_pairs = (
        {(item.get("volume"), item.get("name")) for item in indexes}
        if isinstance(indexes, list)
        else set()
    )
    if actual_catalog_pairs != expected_catalog_pairs or len(indexes) != len(
        expected_catalog_pairs
    ):
        errors.append("all eight Cassini catalog label/table fingerprint pairs are required")
    for item in indexes if isinstance(indexes, list) else []:
        sha = str(item.get("sha256", ""))
        url = str(item.get("url", ""))
        if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
            errors.append(f"remote catalog fingerprint is invalid for {item.get('name')}")
        if not isinstance(item.get("byte_length"), int) or item.get("byte_length", 0) <= 0:
            errors.append(f"remote catalog byte length is invalid for {item.get('name')}")
        expected_suffix = f"/{item.get('volume')}/index/{item.get('name')}"
        if not _allowed_https_host(url, {"atmos.nmsu.edu"}) or not urlsplit(
            url
        ).path.endswith(expected_suffix):
            errors.append(f"remote catalog URL is not allowlisted for {item.get('name')}")

    classification = manifest.get("record_classification", {})
    direct = set(classification.get("direct_signal_records", []))
    if not {"ATDF/TDF closed-loop tracking records", "RSR open-loop receiver samples"} <= direct:
        errors.append("direct Cassini tracking/receiver record classes are incomplete")
    model_dependent = classification.get("model_dependent_records", {})
    if model_dependent.get("allowed_as_prediction_truth") is not False:
        errors.append("Cassini model-dependent ODF/SPK records cannot be prediction truth")
    if not {"ODF navigation products", "SPK ephemeris kernels"} <= set(
        model_dependent.get("examples", [])
    ):
        errors.append("Cassini ODF/SPK model-dependent classification is incomplete")
    if classification.get("latent_quantities", {}).get("allowed_as_input_or_target") is not False:
        errors.append("latent Cassini quantities cannot be inputs or targets")

    targets_text = " ".join(manifest.get("future_prediction_targets", [])).casefold()
    for prohibited in ("ppn gamma", "ppn beta", "shapiro residual", "dark matter", "halo"):
        if prohibited in targets_text:
            errors.append(f"future Cassini targets contain a prohibited label: {prohibited}")
    forbidden = {str(item).casefold() for item in manifest.get("forbidden_uses", [])}
    for required in (
        "odf or spk product treated as direct observation truth",
        "fitted ppn parameter used as a target",
        "held-out tracking record used to estimate its own initial state",
        "dark matter or invisible halo used as truth or rescue",
        "redshift-derived distance or supernova distance modulus",
    ):
        if required not in forbidden:
            errors.append(f"missing forbidden Cassini use: {required}")

    split = manifest.get("future_split_contract", {})
    if split.get("unit") != protocol.get("split_contract", {}).get("unit"):
        errors.append("Cassini split unit differs from the frozen Solar protocol")
    if split.get("group_leakage_forbidden") is not True:
        errors.append("Cassini future session grouping does not forbid leakage")
    if "before primary files download" not in str(split.get("freeze_rule", "")).casefold():
        errors.append("Cassini split is not frozen before primary-file download")

    readiness = manifest.get("readiness", {})
    if readiness.get("dataset_ready") is not False:
        errors.append("metadata registration cannot claim dataset readiness")
    blockers = readiness.get("blockers", [])
    if not isinstance(blockers, list) or len(blockers) < 6:
        errors.append("Cassini data-readiness blockers are incomplete")
    if readiness.get("primary_files_downloaded") is not False:
        errors.append("metadata registration cannot claim primary files downloaded")
    if readiness.get("raw_parser_verified") is not False:
        errors.append("metadata registration cannot claim a verified raw parser")

    passed = not errors
    return {
        "schema_version": "sigma-solar-source-registration-audit-1.0",
        "status": "pass_metadata_registration" if passed else "fail",
        "manifest": _portable_path(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "protocol": _portable_path(protocol_path),
        "protocol_sha256": protocol_sha,
        "policy": _portable_path(policy_path),
        "policy_sha256": policy_sha,
        "dataset_id": source.get("dataset_id"),
        "registered_catalog_files": len(indexes) if isinstance(indexes, list) else 0,
        "errors": errors,
        "dataset_ready": False,
        "candidate_use_authorized": False,
        "observational_dataset_opened": False,
        "next_required_work": blockers,
    }


def audit_galaxy_observable_protocol(
    protocol_path: str | Path, evidence_policy_path: str | Path
) -> dict[str, Any]:
    """Validate the sealed observable-to-observable galaxy discovery contract.

    A pass validates only the protocol. It does not open data: an exact candidate
    action and a hash-committed dataset manifest must still pass separate audits.
    """

    protocol_path = Path(protocol_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-galaxy-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load galaxy observable protocol: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-galaxy-observable-audit-1.0",
            "status": "fail",
            "errors": [f"cannot load observational evidence policy: {error}"],
            "observational_dataset_opened": False,
        }

    if protocol.get("schema_version") != "sigma-galaxy-observable-protocol-1.0":
        errors.append("unsupported galaxy observable protocol schema")
    if protocol.get("status") != "sealed":
        errors.append("galaxy observable protocol must remain sealed")
    if protocol.get("data_opened") is not False:
        errors.append("galaxy observable protocol cannot claim opened data")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("unsupported observational evidence policy schema")
    if policy.get("status") != "frozen":
        errors.append("observational evidence policy is not frozen")

    discovery = protocol.get("discovery_channel", {})
    inputs = discovery.get("inputs", [])
    target = discovery.get("prediction_target", "")
    allowed_formula_inputs = discovery.get("allowed_formula_inputs", [])
    forbidden_formula_inputs = set(discovery.get("forbidden_formula_inputs", []))
    if not isinstance(inputs, list) or len(inputs) < 4:
        errors.append("discovery channel lacks the required measured input classes")
    if not isinstance(target, str) or not all(
        token in target.casefold() for token in ("held-out", "doppler")
    ):
        errors.append("prediction target must be held-out Doppler/spectral kinematics")
    contaminated_allowed_text = " ".join(
        [*(str(item) for item in inputs), *(str(item) for item in allowed_formula_inputs), target]
    ).casefold()
    for prohibited in (
        "halo mass",
        "halo concentration",
        "nfw",
        "redshift-derived distance",
        "lensing-derived mass",
        "per-galaxy acceleration",
    ):
        if prohibited in contaminated_allowed_text:
            errors.append(f"allowed discovery channel contains prohibited input: {prohibited}")
    required_forbidden_inputs = {
        "dark-matter halo mass, concentration, radius, profile, or abundance-matching label",
        "mass discrepancy inferred from the target rotation curve",
        "redshift-derived distance or physical size",
        "lensing-derived mass, convergence, or acceleration map",
        "per-galaxy acceleration scale or gravitational coupling",
    }
    missing_forbidden_inputs = sorted(required_forbidden_inputs - forbidden_formula_inputs)
    if missing_forbidden_inputs:
        errors.append(
            "missing forbidden discovery inputs: " + ", ".join(missing_forbidden_inputs)
        )
    conversion_rule = str(discovery.get("baryonic_conversion_rule", "")).casefold()
    if "not direct observations" not in conversion_rule or "each galaxy" not in conversion_rule:
        errors.append("baryonic conversion nuisances are not constrained against target fitting")

    split = protocol.get("split_contract", {})
    if split.get("unit") != "whole galaxy":
        errors.append("train/validation/test splitting must occur by whole galaxy")
    if split.get("group_leakage_forbidden") is not True:
        errors.append("galaxy group leakage is not forbidden")
    roles = set(split.get("roles", []))
    if "untouched target-blind test galaxies" not in roles:
        errors.append("target-blind whole-galaxy test role is missing")
    sealed_rule = str(split.get("sealed_test_rule", "")).casefold()
    if not all(token in sealed_rule for token in ("formula hash", "stopping rule", "inaccessible")):
        errors.append("sealed test rule does not freeze formula and stopping decisions")

    scoring = protocol.get("scoring_contract", {})
    if scoring.get("object_specific_gravity_parameters") != 0:
        errors.append("object-specific gravitational parameters must equal zero")
    selection_rule = str(scoring.get("selection_rule", "")).casefold()
    if not all(token in selection_rule for token in ("validation", "no test", "lensing")):
        errors.append("selection rule does not isolate test and lensing channels")
    if "covariance" not in str(scoring.get("fit_term", "")).casefold():
        errors.append("fit score must carry measurement and calibration covariance")
    if not scoring.get("required_baselines"):
        errors.append("baryons-only and empirical baselines are missing")

    lensing = protocol.get("independent_lensing_falsification", {})
    if lensing.get("status") != "sealed_until_after_kinematic_formula_freeze":
        errors.append("lensing channel is not sealed until after formula freeze")
    if lensing.get("formula_selection_use") != "prohibited":
        errors.append("lensing is not prohibited from formula selection")
    same_law = str(lensing.get("same_law_rule", "")).casefold()
    if not all(token in same_law for token in ("frozen covariant action", "without any lensing-only")):
        errors.append("lensing does not require the same frozen law without private parameters")
    lensing_forbidden = " ".join(lensing.get("forbidden_targets", [])).casefold()
    if "dark-matter map" not in lensing_forbidden or "gr-derived" not in lensing_forbidden:
        errors.append("lensing-derived invisible-mass targets are not fully prohibited")

    prohibited_truth = {
        str(item).casefold() for item in protocol.get("prohibited_truth_or_rescue", [])
    }
    for required in (
        "dark matter",
        "halo mass",
        "halo concentration",
        "redshift-derived distance",
        "supernova distance modulus",
    ):
        if required not in prohibited_truth:
            errors.append(f"missing prohibited truth/rescue quantity: {required}")
    policy_uses = set(policy.get("unobserved_components", {}).get("prohibited_uses", []))
    if not {"target labels", "formula selection", "object-specific halo fitting"} <= policy_uses:
        errors.append("evidence policy no longer prohibits inferred-component target leakage")
    if policy.get("supernovae", {}).get("default_status") != "excluded":
        errors.append("supernova distances are not excluded by the evidence policy")
    if "treating redshift as a distance" not in policy.get("redshift", {}).get(
        "not_allowed_by_default", []
    ):
        errors.append("redshift-distance inference is not excluded by default")

    opening_requirements = protocol.get("opening_requirements", [])
    if not isinstance(opening_requirements, list) or len(opening_requirements) < 5:
        errors.append("dataset-opening requirements are incomplete")
    passed = not errors
    return {
        "schema_version": "sigma-galaxy-observable-audit-1.0",
        "status": "pass" if passed else "fail",
        "protocol": _portable_path(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "policy": _portable_path(policy_path),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "errors": errors,
        "discovery_target": target,
        "split_unit": split.get("unit"),
        "object_specific_gravity_parameters": scoring.get(
            "object_specific_gravity_parameters"
        ),
        "lensing_formula_selection_use": lensing.get("formula_selection_use"),
        "redshift_distance_allowed_by_default": False,
        "supernova_default_status": policy.get("supernovae", {}).get("default_status"),
        "observational_dataset_opened": False,
        "formula_search_authorized": False,
        "next_required_binding": (
            "an eligible exact action hash and an independently audited, hash-committed dataset manifest"
        ),
    }


def audit_theory_observation_eligibility(
    health_report_path: str | Path,
    evidence_policy_path: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Fail closed before Solar references or any candidate dataset can be opened."""

    if mode not in {"known_answer_reference", "candidate_data"}:
        raise ValueError("mode must be known_answer_reference or candidate_data")
    health_path = Path(health_report_path).resolve()
    policy_path = Path(evidence_policy_path).resolve()
    errors: list[str] = []
    try:
        health = _load_json(health_path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-observation-eligibility-1.0",
            "status": "ineligible",
            "mode": mode,
            "errors": [f"cannot load action-health report: {error}"],
            "observational_dataset_opened": False,
        }
    try:
        policy_bytes = policy_path.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": "sigma-observation-eligibility-1.0",
            "status": "ineligible",
            "mode": mode,
            "errors": [f"cannot load observational evidence policy: {error}"],
            "observational_dataset_opened": False,
        }

    if health.get("schema_version") != "sigma-action-health-1.0":
        errors.append("unsupported action-health schema")
    if policy.get("schema_version") != "sigma-observational-evidence-policy-1.0":
        errors.append("unsupported observational evidence policy schema")
    if policy.get("status") != "frozen":
        errors.append("observational evidence policy is not frozen")
    gates = health.get("gates", {})
    missing_gates = [name for name in REQUIRED_FORMAL_GATES if name not in gates]
    failed_gates = [
        name for name in REQUIRED_FORMAL_GATES if gates.get(name, {}).get("status") != "pass"
    ]
    if missing_gates:
        errors.append("missing formal gates: " + ", ".join(missing_gates))
    if failed_gates:
        errors.append("formal gates not passed: " + ", ".join(failed_gates))

    action_path_value = health.get("action_ir")
    action_ir: dict[str, Any] = {}
    if not action_path_value:
        errors.append("action-health report does not name its action IR")
    else:
        try:
            action_ir = _load_json(_resolved_artifact_path(health_path, action_path_value))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load bound action IR: {error}")
    action_hash = action_ir.get("content_sha256")
    if not action_ir.get("valid"):
        errors.append("bound covariant action IR is invalid")
    if health.get("input_action_sha256") != action_hash:
        errors.append("action-health report belongs to a different action hash")

    artifact_records: dict[str, Any] = {}
    for key, expected_schema, hash_field in (
        ("generated_q_operator_ir", "sigma-q-operator-ir-1.0", "input_action_sha256"),
        ("generated_q_variation_ir", "sigma-q-variation-ir-1.0", "input_action_sha256"),
        ("generated_x_operator_ir", "sigma-x-operator-ir-1.0", "input_action_sha256"),
        ("generated_principal_ir", "sigma-physical-principal-ir-1.0", "input_action_sha256"),
        (
            "generated_hamiltonian_ir",
            "sigma-physical-hamiltonian-ir-1.0",
            "input_action_sha256",
        ),
    ):
        record = health.get(key, {})
        if not record.get("path"):
            errors.append(f"action-health report does not name {key}")
            continue
        try:
            artifact = _load_json(_resolved_artifact_path(health_path, record["path"]))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load {key}: {error}")
            continue
        artifact_records[key] = artifact
        if artifact.get("schema_version") != expected_schema:
            errors.append(f"unsupported {key} schema")
        if artifact.get("status") != "pass":
            errors.append(f"{key} has not passed")
        if artifact.get(hash_field) != action_hash:
            errors.append(f"{key} belongs to a different action hash")
        if record.get("content_sha256") != artifact.get("content_sha256"):
            errors.append(f"{key} content hash differs from the health report")

    source_role = action_ir.get("canonical", {}).get("source_role")
    family = health.get("family")
    if mode == "known_answer_reference":
        if source_role != "known_answer_control":
            errors.append("Solar reference mode requires a known-answer control action")
        if family != "einstein_hilbert":
            errors.append("Solar reference mode requires the Einstein-Hilbert family")
        if not health.get("promotion_allowed") or health.get("status") != "pass":
            errors.append("Einstein-Hilbert control health packet is not a full pass")
    else:
        if source_role != "candidate":
            errors.append("candidate-data mode requires a generated candidate action")
        if not health.get("promotion_allowed") or health.get("status") != "pass":
            errors.append("candidate has not passed every formal promotion gate")
        if health.get("discovery_blockers"):
            errors.append("candidate still has discovery blockers")

    eligible = not errors
    return {
        "schema_version": "sigma-observation-eligibility-1.0",
        "status": "eligible" if eligible else "ineligible",
        "mode": mode,
        "health_report": str(health_path),
        "input_action_sha256": action_hash,
        "source_role": source_role,
        "family": family,
        "formal_gate_statuses": {
            name: gates.get(name, {}).get("status") for name in REQUIRED_FORMAL_GATES
        },
        "policy": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "policy_status": policy.get("status"),
        "errors": errors,
        "reference_controls_allowed": eligible and mode == "known_answer_reference",
        "candidate_dataset_manifest_may_be_audited": eligible and mode == "candidate_data",
        "observational_dataset_opened": False,
        "dataset_opening_blocker": (
            "an independently audited dataset manifest is still required"
            if eligible and mode == "candidate_data"
            else "theory-side eligibility has not passed"
            if not eligible
            else "known-answer reference mode does not authorize opening candidate datasets"
        ),
        "prohibited_targets_preserved": policy.get("unobserved_components", {}).get(
            "prohibited_uses", []
        ),
        "supernova_default_status": policy.get("supernovae", {}).get("default_status"),
        "redshift_distance_allowed_by_default": False,
    }


def write_observation_eligibility(
    eligibility: dict[str, Any], output: str | Path
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
