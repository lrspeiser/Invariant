"""Acquire and evaluate the frozen Item 39 SWELLS lensing transfer."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammainc, gammaln

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    CONFIG_PATH,
    POLICY_PATH,
    GravityItem39Error,
    _source_path,
    fixed_control_multiplier,
    generate_raw_candidates,
    load_config,
    predict_multiplier,
)

MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item39_holographic_boundary_lensing.py")


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=text)
    return result.stdout.strip() if text else result.stdout


def _normalized_transfer(config: Mapping[str, Any]) -> dict[str, Any]:
    transfer = json.loads(json.dumps(config["lensing_transfer"]))
    transfer["freeze_commit"] = "<BOUND_COMMIT>"
    return transfer


def verify_transfer_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["lensing_transfer"]["freeze_commit"])
    if commit.startswith("PENDING_"):
        raise GravityItem39Error("Item 39 lensing transfer freeze is not bound")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise GravityItem39Error("Item 39 lensing transfer freeze is not an ancestor")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _normalized_transfer(frozen_config) != _normalized_transfer(config):
        raise GravityItem39Error("Item 39 lensing transfer contract differs from freeze")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem39Error("could not read Item 39 frozen lensing module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem39Error("Item 39 lensing evaluator differs from freeze")


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Invariant-Item39-SWELLS-Transfer/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem39Error(f"empty SWELLS source response: {url}")
    return body, headers


def _normalize_id(value: object) -> str:
    text = str(value).strip().replace("SDSS", "").replace(" ", "")
    text = text.replace("−", "-").replace("–", "-").replace("++", "+")
    text = text.replace("-+", "-").replace("--", "-")
    text = text.replace("$", "").replace("\\", "")
    return text


def _nominal_error(value: object) -> tuple[float, float]:
    text = str(value).replace(" ", "")
    nominal = re.search(r"([-+]?\d+(?:\.\d+)?)±", text)
    if nominal is None:
        raise GravityItem39Error(f"could not parse published value and error: {value}")
    error_text = text.rsplit("\\pm", 1)[-1] if "\\pm" in text else text.split("±", 1)[1]
    error = re.search(r"[-+]?\d+(?:\.\d+)?", error_text)
    if error is None:
        raise GravityItem39Error(f"could not parse published uncertainty: {value}")
    return float(nominal.group(1)), float(error.group(0))


def _plain_value(value: object) -> float:
    text = str(value).strip().replace(" ", "")
    text = re.sub(r"\\phantom\{[^}]*\}", "", text)
    for split in range(1, len(text)):
        if text[:split] == text[split:]:
            try:
                return float(text[:split])
            except ValueError:
                pass
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        raise GravityItem39Error(f"could not parse published scalar: {value}")
    return float(match.group(0))


def _table_rows(frame: Any) -> list[Any]:
    rows = []
    for _, row in frame.iterrows():
        name = _normalize_id(row.iloc[0])
        if re.fullmatch(r"J\d{4}[+-]\d{4}", name):
            rows.append(row)
    return rows


def _parse_source_tables(
    paper3_payload: bytes, paper5_payload: bytes, expected: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - acquisition-only dependency
        raise GravityItem39Error("pandas is required for SWELLS HTML table acquisition") from exc

    tables3 = pd.read_html(io.BytesIO(paper3_payload))
    tables5 = pd.read_html(io.BytesIO(paper5_payload))
    if len(tables3) < 4 or len(tables5) < 1:
        raise GravityItem39Error("published SWELLS table layout changed")

    lens_rows: dict[str, Any] = {}
    for row in _table_rows(tables3[0]):
        lens_rows[_normalize_id(row.iloc[0])] = row
    structure_rows: dict[str, Any] = {}
    for row in _table_rows(tables3[2]):
        structure_rows[_normalize_id(row.iloc[0])] = row
    mass_rows: dict[str, Any] = {}
    for row in _table_rows(tables3[3]):
        mass_rows[_normalize_id(row.iloc[0])] = row
    paper5_rows: dict[str, Any] = {}
    for row in _table_rows(tables5[0]):
        paper5_rows[_normalize_id(row.iloc[0])] = row

    records = []
    for name in expected:
        if name not in lens_rows:
            raise GravityItem39Error(f"missing frozen SWELLS lens response: {name}")
        lens = lens_rows[name]
        einstein, einstein_error = _nominal_error(lens.iloc[5])
        lens_mass, lens_mass_error = _nominal_error(lens.iloc[8])
        if name in paper5_rows:
            predictor = paper5_rows[name]
            bulge_radius = _plain_value(predictor.iloc[3])
            disk_radius = _plain_value(predictor.iloc[4])
            bulge_mass, bulge_mass_error = _nominal_error(predictor.iloc[5])
            disk_mass, disk_mass_error = _nominal_error(predictor.iloc[6])
            radius_unit = "kpc"
            predictor_source = "SWELLS_V_TABLE_1"
        else:
            if name not in structure_rows or name not in mass_rows:
                raise GravityItem39Error(f"missing frozen SWELLS predictor: {name}")
            structure = structure_rows[name]
            mass = mass_rows[name]
            bulge_radius = _plain_value(structure.iloc[9])
            disk_radius = _plain_value(structure.iloc[11])
            bulge_mass, bulge_mass_error = _nominal_error(mass.iloc[1])
            disk_mass, disk_mass_error = _nominal_error(mass.iloc[2])
            radius_unit = "arcsec"
            predictor_source = "SWELLS_III_TABLES_3_4"
        records.append(
            {
                "name": name,
                "predictor_source": predictor_source,
                "lens_redshift": _plain_value(lens.iloc[3]),
                "bulge_half_light_radius": bulge_radius,
                "disk_half_light_radius": disk_radius,
                "radius_unit": radius_unit,
                "log10_chabrier_bulge_mass_msun": bulge_mass,
                "log10_chabrier_bulge_mass_error_dex": bulge_mass_error,
                "log10_chabrier_disk_mass_msun": disk_mass,
                "log10_chabrier_disk_mass_error_dex": disk_mass_error,
                "einstein_radius_arcsec": einstein,
                "einstein_radius_error_arcsec": einstein_error,
                "projected_lens_mass_1e10_msun": lens_mass,
                "projected_lens_mass_error_1e10_msun": lens_mass_error,
            }
        )
    if [row["name"] for row in records] != expected:
        raise GravityItem39Error("SWELLS source order or identity set changed")
    counts = {
        "paper3_lens_rows": len(lens_rows),
        "paper3_structure_rows": len(structure_rows),
        "paper3_mass_rows": len(mass_rows),
        "paper5_predictor_rows": len(paper5_rows),
        "eligible_records": len(records),
    }
    return records, counts


def acquire(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_transfer_freeze(root, config)
    transfer = config["lensing_transfer"]
    urls = list(transfer["machine_sources"])
    paper3, headers3 = _download(str(urls[0]))
    paper5, headers5 = _download(str(urls[1]))
    records, counts = _parse_source_tables(
        paper3, paper5, list(transfer["expected_eligible_names"])
    )
    source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-swells-source-1.0",
            "item": 39,
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "transfer_freeze_commit": transfer["freeze_commit"],
            "dynamics_result_commit": transfer["dynamics_result_commit"],
            "sources": [
                {
                    "url": urls[0],
                    "sha256": _sha256_bytes(paper3),
                    "etag": headers3.get("etag"),
                    "last_modified": headers3.get("last-modified"),
                },
                {
                    "url": urls[1],
                    "sha256": _sha256_bytes(paper5),
                    "etag": headers5.get("etag"),
                    "last_modified": headers5.get("last-modified"),
                },
            ],
            "counts": counts,
            "selection_audit": {
                "expected_eligible_names": transfer["expected_eligible_names"],
                "prefreeze_excluded_names": transfer["prefreeze_web_exposed_or_predecessor_names"],
                "no_failed_identity_replacement": True,
                "formula_or_coefficient_retuning": False,
                "post_selection_candidate_cells": 0,
                "published_fstar_columns_used": False,
                "unblinded_diagnostic": True,
            },
            "records": records,
        }
    )
    path = _source_path(root, config, "lensing_transfer_source")
    _write_json(path, source)
    return path


def _kpc_per_arcsecond(z: float, config: Mapping[str, Any]) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(96)
    points = 0.5 * z * (nodes + 1.0)
    expansion = np.sqrt(0.3 * (1.0 + points) ** 3 + 0.7)
    integral = 0.5 * z * float(np.sum(weights / expansion))
    distance_kpc = (
        float(config["constants"]["speed_of_light_km_s"])
        / float(config["constants"]["hubble_constant_km_s_mpc"])
        * integral
        * 1000.0
        / (1.0 + z)
    )
    return distance_kpc / float(config["constants"]["arcseconds_per_radian"])


def _sersic_fraction_and_growth(
    radius: float, half_light_radius: float, n: float, b_n: float
) -> tuple[float, float]:
    if radius <= 0.0 or half_light_radius <= 0.0:
        raise GravityItem39Error("SWELLS projected radii must be positive")
    z = b_n * (radius / half_light_radius) ** (1.0 / n)
    shape = 2.0 * n
    fraction = float(gammainc(shape, z))
    derivative = math.exp(shape * math.log(z) - z - math.log(n) - float(gammaln(shape)))
    return fraction, derivative


def _candidate_row(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    candidate_id = int(config["lensing_transfer"]["selected_candidate"]["candidate_id"])
    raw = generate_raw_candidates(config)
    return {key: np.asarray(value[candidate_id : candidate_id + 1]) for key, value in raw.items()}


def _predictions(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    mass_factor: float = 1.0,
    screen_factor: float = 4.0,
) -> dict[str, np.ndarray]:
    enclosed = []
    total = []
    radius = []
    fraction = []
    radius_over_screen = []
    slope = []
    observed = []
    observed_error = []
    source_mass_error_fraction = []
    for row in records:
        kpc_per_arcsec = _kpc_per_arcsecond(float(row["lens_redshift"]), config)
        r_einstein = float(row["einstein_radius_arcsec"]) * kpc_per_arcsec
        if row["radius_unit"] == "arcsec":
            r_bulge = float(row["bulge_half_light_radius"]) * kpc_per_arcsec
            r_disk = float(row["disk_half_light_radius"]) * kpc_per_arcsec
        else:
            r_bulge = float(row["bulge_half_light_radius"])
            r_disk = float(row["disk_half_light_radius"])
        m_bulge = mass_factor * 10.0 ** float(row["log10_chabrier_bulge_mass_msun"])
        m_disk = mass_factor * 10.0 ** float(row["log10_chabrier_disk_mass_msun"])
        f_bulge, d_bulge = _sersic_fraction_and_growth(r_einstein, r_bulge, 4.0, 7.66924944)
        f_disk, d_disk = _sersic_fraction_and_growth(r_einstein, r_disk, 1.0, 1.67834699)
        m_enclosed = m_bulge * f_bulge + m_disk * f_disk
        m_total = m_bulge + m_disk
        growth = (m_bulge * d_bulge + m_disk * d_disk) / m_enclosed
        enclosed.append(m_enclosed)
        total.append(m_total)
        radius.append(r_einstein)
        fraction.append(m_enclosed / m_total)
        radius_over_screen.append(r_einstein / (screen_factor * r_disk))
        slope.append(growth)
        observed.append(float(row["projected_lens_mass_1e10_msun"]) * 1e10)
        observed_error.append(float(row["projected_lens_mass_error_1e10_msun"]) * 1e10)
        bulge_sigma = (
            math.log(10.0) * m_bulge * f_bulge * float(row["log10_chabrier_bulge_mass_error_dex"])
        )
        disk_sigma = (
            math.log(10.0) * m_disk * f_disk * float(row["log10_chabrier_disk_mass_error_dex"])
        )
        source_mass_error_fraction.append(math.hypot(bulge_sigma, disk_sigma) / m_enclosed)

    enclosed_array = np.asarray(enclosed)
    radius_array = np.asarray(radius)
    native_acceleration = (
        float(config["constants"]["gravitational_constant_kpc_km2_s2_msun"])
        * enclosed_array
        / np.square(radius_array)
    )
    acceleration_m_s2 = native_acceleration * 1e6 / 3.085677581491367e19
    u = acceleration_m_s2 / float(config["constants"]["acceleration_scale_m_s2"])
    fraction_array = np.asarray(fraction)
    screen_array = np.asarray(radius_over_screen)
    slope_array = np.asarray(slope)
    candidate_nu = predict_multiplier(
        _candidate_row(config), u, fraction_array, screen_array, slope_array, config
    )[0]
    result = {
        "u": u,
        "enclosed_baryonic_mass_msun": enclosed_array,
        "total_baryonic_mass_msun": np.asarray(total),
        "einstein_radius_kpc": radius_array,
        "enclosed_fraction": fraction_array,
        "radius_over_screen": screen_array,
        "enclosed_log_slope": slope_array,
        "observed_lens_mass_msun": np.asarray(observed),
        "observed_lens_mass_error_msun": np.asarray(observed_error),
        "source_mass_error_fraction": np.asarray(source_mass_error_fraction),
        "candidate_multiplier": candidate_nu,
        "candidate": candidate_nu * enclosed_array,
    }
    for name in ("baryonic_newton", "mond_RAR", "item38_selected"):
        result[name] = fixed_control_multiplier(name, u) * enclosed_array
    return result


def _loss(prediction: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(np.square(np.log10(prediction / observed))))


def _losses(predictions: Mapping[str, np.ndarray]) -> dict[str, float]:
    observed = predictions["observed_lens_mass_msun"]
    return {
        name: _loss(predictions[name], observed)
        for name in ("candidate", "baryonic_newton", "mond_RAR", "item38_selected")
    }


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    verify_transfer_freeze(root, config)
    source_path = _source_path(root, config, "lensing_transfer_source")
    source = _read_json(source_path)
    _verify_content_hash(source, "Item 39 SWELLS source")
    records = list(source["records"])
    expected = list(config["lensing_transfer"]["expected_eligible_names"])
    if [row["name"] for row in records] != expected:
        raise GravityItem39Error("Item 39 SWELLS source identity drifted")
    if len(records) < int(config["lensing_transfer"]["minimum_evaluable_lenses"]):
        raise GravityItem39Error("too few Item 39 SWELLS lenses")

    primary = _predictions(records, config)
    losses = _losses(primary)
    control_names = ("baryonic_newton", "mond_RAR", "item38_selected")
    strongest = min(control_names, key=lambda name: losses[name])
    improvement = 100.0 * (losses[strongest] - losses["candidate"]) / losses[strongest]

    audit_specs = {
        "stellar_mass_minus_0.25_dex": (10.0**-0.25, 4.0),
        "stellar_mass_plus_0.25_dex": (10.0**0.25, 4.0),
        "missing_gas_plus_0.10_dex": (10.0**0.10, 4.0),
        "screen_radius_factor_3": (1.0, 3.0),
        "screen_radius_factor_5": (1.0, 5.0),
    }
    audit_predictions = {
        name: _predictions(records, config, mass_factor=mass, screen_factor=screen)
        for name, (mass, screen) in audit_specs.items()
    }
    audits = {}
    for name, values in audit_predictions.items():
        audit_losses = _losses(values)
        audit_strongest = min(control_names, key=lambda key: audit_losses[key])
        audits[name] = {
            "losses": audit_losses,
            "strongest_control": audit_strongest,
            "improvement_vs_strongest_percent": 100.0
            * (audit_losses[audit_strongest] - audit_losses["candidate"])
            / audit_losses[audit_strongest],
        }

    observed = primary["observed_lens_mass_msun"]
    candidate_object = np.square(np.log10(primary["candidate"] / observed))
    reference_object = np.square(np.log10(primary[strongest] / observed))
    differences = candidate_object - reference_object
    stable_counterexample = differences > 0.0
    for values in audit_predictions.values():
        audit_losses = _losses(values)
        audit_strongest = min(control_names, key=lambda key: audit_losses[key])
        comparison = np.square(np.log10(values["candidate"] / observed)) - np.square(
            np.log10(values[audit_strongest] / observed)
        )
        stable_counterexample &= comparison > 0.0

    full_sign = float(np.mean(differences)) > 0.0
    leave_one_signs = [
        float(np.mean(np.delete(differences, index))) > 0.0 for index in range(len(differences))
    ]
    influential = int(np.argmax(np.abs(differences - np.median(differences))))
    trimmed_sign = float(np.mean(np.delete(differences, influential))) > 0.0
    quality_passed = len(records) >= int(config["lensing_transfer"]["minimum_evaluable_lenses"])
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(records),
        "raw_counterexample_count": int(np.sum(differences > 0.0)),
        "quality_verified_counterexample_count": int(np.sum(differences > 0.0)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_counterexample)),
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": quality_passed,
        "strongest_baseline_failed": losses["candidate"] >= losses[strongest],
        "leave_one_changes_sign": any(value != full_sign for value in leave_one_signs),
        "trim_changes_sign": trimmed_sign != full_sign,
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": False,
    }
    policy = load_counterexample_policy(root / POLICY_PATH)
    assessment = assess_counterexample_evidence(report, policy)
    object_records = []
    for index, row in enumerate(records):
        combined_fractional_error = math.hypot(
            float(primary["observed_lens_mass_error_msun"][index] / observed[index]),
            float(primary["source_mass_error_fraction"][index]),
        )
        object_records.append(
            {
                "name": row["name"],
                "u": float(primary["u"][index]),
                "einstein_radius_kpc": float(primary["einstein_radius_kpc"][index]),
                "enclosed_fraction": float(primary["enclosed_fraction"][index]),
                "radius_over_screen": float(primary["radius_over_screen"][index]),
                "enclosed_log_slope": float(primary["enclosed_log_slope"][index]),
                "published_lens_mass_msun": float(observed[index]),
                "enclosed_baryonic_mass_msun": float(primary["enclosed_baryonic_mass_msun"][index]),
                "candidate_multiplier": float(primary["candidate_multiplier"][index]),
                "candidate_lens_mass_msun": float(primary["candidate"][index]),
                "strongest_control_lens_mass_msun": float(primary[strongest][index]),
                "combined_fractional_measurement_error": combined_fractional_error,
                "raw_counterexample": bool(differences[index] > 0.0),
                "systematics_stable_counterexample": bool(stable_counterexample[index]),
            }
        )

    improves_all = all(losses["candidate"] < losses[name] for name in control_names)
    if improves_all:
        decision = "ITEM39_UNCHANGED_SWELLS_TRANSFER_IMPROVES_DIAGNOSTIC"
    else:
        decision = "ITEM39_UNCHANGED_SWELLS_TRANSFER_DOES_NOT_IMPROVE_RETAINED"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-swells-result-1.0",
            "item": 39,
            "decision": decision,
            "protocol": {
                "transfer_freeze_commit": config["lensing_transfer"]["freeze_commit"],
                "dynamics_result_commit": config["lensing_transfer"]["dynamics_result_commit"],
                "source_file_sha256": _sha256_file(source_path),
                "source_content_sha256": source["content_sha256"],
                "candidate_id": config["lensing_transfer"]["selected_candidate"]["candidate_id"],
                "post_selection_candidate_cells": 0,
                "formula_or_coefficient_retuning": False,
                "paid_model_calls": 0,
                "unblinded_diagnostic": True,
            },
            "sample": {
                "evaluable_lenses": len(records),
                "names": expected,
                "minimum_gate": config["lensing_transfer"]["minimum_evaluable_lenses"],
                "quality_gate_passed": quality_passed,
            },
            "primary": {
                "losses": losses,
                "strongest_fixed_control": strongest,
                "improvement_vs_strongest_percent": improvement,
                "candidate_improves_over_every_fixed_control": improves_all,
                "object_level": object_records,
                "counterexample_policy_report": report,
                "counterexample_assessment": assessment,
            },
            "systematic_audits": audits,
            "gates": {
                "minimum_sample_passes": quality_passed,
                "candidate_beats_baryonic": losses["candidate"] < losses["baryonic_newton"],
                "candidate_beats_mond_RAR": losses["candidate"] < losses["mond_RAR"],
                "candidate_beats_item38_selected": losses["candidate"] < losses["item38_selected"],
                "unchanged_formula": True,
                "one_metric_motion_light_contract_preserved": True,
                "no_lensing_retuning": True,
            },
            "claim_boundaries": {
                "blinded_confirmation": False,
                "direct_image_likelihood": False,
                "complete_relativistic_lensing_theory": False,
                "dark_matter_excluded": False,
                "historical_novelty_established": False,
                "formula_pruned": False,
                "formula_family_pruned": False,
                "one_counterexample_is_veto": False,
            },
        }
    )
    if write:
        _write_json(_source_path(root, config, "lensing_transfer_result"), result)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "lensing_transfer_result")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 39 SWELLS result")
    replay = evaluate(root, write=False)
    if existing != replay:
        raise GravityItem39Error("Item 39 SWELLS transfer replay drifted")
    return {
        "status": "ITEM39_SWELLS_TRANSFER_REPLAY_VALID",
        "decision": existing["decision"],
        "result_file_sha256": _sha256_file(path),
        "post_selection_candidate_cells": 0,
        "paid_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "evaluate", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "acquire":
        print(acquire(args.root))
    elif args.command == "evaluate":
        value = evaluate(args.root)
        print(json.dumps({"decision": value["decision"]}, sort_keys=True))
    else:
        print(json.dumps(check(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
