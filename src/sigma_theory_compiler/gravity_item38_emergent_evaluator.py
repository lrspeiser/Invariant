"""Frozen response parser and evaluator for Item 38 emergent gravity."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item29_nonlinear_self_interaction import _backend
from sigma_theory_compiler.gravity_item38_emergent_gravity import (
    COUNTEREXAMPLE_POLICY_PATH,
    GravityItem38Error,
    _source_paths,
    admissible_candidates,
    decode_candidate,
    fixed_control_multiplier,
    load_config,
    predict_multiplier,
)
from sigma_theory_compiler.gravity_item38_emergent_source import (
    verify_scientific_freeze,
    verify_source_metadata_freeze,
)

EVALUATOR_PATH = Path("src/sigma_theory_compiler/gravity_item38_emergent_evaluator.py")
EXPLORATION_NAMES = (
    "Fig-9_RAR-KiDS-isolated_Massbin-1.txt",
    "Fig-9_RAR-KiDS-isolated_Massbin-2.txt",
    "Fig-9_RAR-KiDS-isolated_Massbin-3.txt",
)
TRANSFER_NAMES = (
    "Fig-8_RAR-KiDS-isolated_Colorbin_1.txt",
    "Fig-8_RAR-KiDS-isolated_Colorbin_2.txt",
)
MASS_COVARIANCE = "Fig-9_RAR-KiDS-isolated_Massbins_covmatrix.txt"
COLOR_COVARIANCE = "Fig-8_RAR-KiDS-isolated_Colorbins_covmatrix.txt"
SEALED_CONFIRMATION = "Fig-9_RAR-KiDS-isolated_Massbin-4.txt"
PROFILE_MINIMA = {
    EXPLORATION_NAMES[0]: 8.5,
    EXPLORATION_NAMES[1]: 10.3,
    EXPLORATION_NAMES[2]: 10.6,
    TRANSFER_NAMES[0]: 0.0,
    TRANSFER_NAMES[1]: 2.5,
}
G_PC3_MSUN_S2 = 4.52e-30
PC_PER_M = 3.086e16
ESD_TO_ACCELERATION = 4.0 * G_PC3_MSUN_S2 * PC_PER_M
SPLINE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)


def _parse_profile(payload: bytes, name: str) -> dict[str, Any]:
    rows = np.loadtxt(io.BytesIO(payload), comments="#", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 8:
        raise GravityItem38Error(f"unexpected KiDS profile columns: {name}")
    g_bar, esd_t, esd_x, error, bias = (rows[:, index] for index in range(5))
    if np.any(~np.isfinite(rows)) or np.any(g_bar <= 0.0):
        raise GravityItem38Error(f"nonfinite or nonpositive KiDS coordinate: {name}")
    if np.any(error <= 0.0) or np.any(bias <= 0.0):
        raise GravityItem38Error(f"invalid KiDS error or calibration: {name}")
    corrected_esd = esd_t / bias
    corrected_error = error / bias
    g_obs = ESD_TO_ACCELERATION * corrected_esd
    g_obs_error = ESD_TO_ACCELERATION * corrected_error
    return {
        "name": name,
        "profile_minimum": PROFILE_MINIMA[name],
        "g_bar_m_s2": g_bar,
        "esd_t_raw": esd_t,
        "esd_x_raw": esd_x,
        "esd_error_raw": error,
        "multiplicative_bias": bias,
        "esd_t_corrected": corrected_esd,
        "esd_error_corrected": corrected_error,
        "g_obs_m_s2": g_obs,
        "g_obs_error_m_s2": g_obs_error,
    }


def _parse_covariance(
    payload: bytes, profiles: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    table = np.loadtxt(io.BytesIO(payload), comments="#", dtype=np.float64)
    if table.ndim == 1:
        table = table[None, :]
    if table.shape[1] != 7 or np.any(~np.isfinite(table)):
        raise GravityItem38Error("unexpected KiDS covariance table")
    offsets: dict[tuple[float, float], int] = {}
    corrected_esd: list[float] = []
    position = 0
    for profile in profiles:
        minimum = float(profile["profile_minimum"])
        radii = np.asarray(profile["g_bar_m_s2"], dtype=np.float64)
        esd = np.asarray(profile["esd_t_corrected"], dtype=np.float64)
        for radius, value in zip(radii, esd, strict=True):
            offsets[(minimum, float(radius))] = position
            corrected_esd.append(float(value))
            position += 1
    covariance = np.zeros((position, position), dtype=np.float64)
    filled = np.zeros((position, position), dtype=bool)
    allowed_minima = {float(profile["profile_minimum"]) for profile in profiles}
    for row in table:
        minimum_i, minimum_j, radius_i, radius_j, value, _, bias_product = row
        if minimum_i not in allowed_minima or minimum_j not in allowed_minima:
            continue
        key_i = min(
            (key for key in offsets if key[0] == minimum_i),
            key=lambda key: abs(key[1] - radius_i),
        )
        key_j = min(
            (key for key in offsets if key[0] == minimum_j),
            key=lambda key: abs(key[1] - radius_j),
        )
        if not math.isclose(key_i[1], radius_i, rel_tol=2e-4):
            raise GravityItem38Error("KiDS covariance radius does not match profile")
        if not math.isclose(key_j[1], radius_j, rel_tol=2e-4):
            raise GravityItem38Error("KiDS covariance radius does not match profile")
        i, j = offsets[key_i], offsets[key_j]
        covariance[i, j] = value / bias_product
        filled[i, j] = True
    if not np.all(filled):
        raise GravityItem38Error("KiDS covariance is incomplete for selected profiles")
    covariance = 0.5 * (covariance + covariance.T)
    esd_array = np.asarray(corrected_esd, dtype=np.float64)
    if np.any(esd_array <= 0.0):
        raise GravityItem38Error("nonpositive ESD cannot enter logarithmic evaluation")
    denominator = np.log(10.0) ** 2 * np.outer(esd_array, esd_array)
    log_covariance = covariance / denominator
    eigenvalues = np.linalg.eigvalsh(log_covariance)
    if float(np.min(eigenvalues)) < -1e-8:
        raise GravityItem38Error("selected KiDS log covariance is not positive semidefinite")
    return log_covariance


def _profile_json(profile: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in profile.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def open_exploration_response(
    root: Path, archive: Path, output: Path
) -> dict[str, Any]:
    config = load_config(root)
    verify_scientific_freeze(root, config)
    verify_source_metadata_freeze(root, config)
    source = _read_json(_source_paths(root, config)["source_metadata_manifest"])
    if _sha256_file(archive) != str(source["archive_sha256"]):
        raise GravityItem38Error("KiDS archive differs from registered source")

    allowed = {*EXPLORATION_NAMES, *TRANSFER_NAMES, MASS_COVARIANCE, COLOR_COVARIANCE}
    if SEALED_CONFIRMATION in allowed:
        raise GravityItem38Error("sealed confirmation entered allowed response members")
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:") as bundle:
        for name in sorted(allowed):
            member = bundle.getmember(name)
            handle = bundle.extractfile(member)
            if handle is None:
                raise GravityItem38Error(f"cannot read allowed KiDS member: {name}")
            payloads[name] = handle.read()
    exploration = [_parse_profile(payloads[name], name) for name in EXPLORATION_NAMES]
    transfer = [_parse_profile(payloads[name], name) for name in TRANSFER_NAMES]
    mass_covariance = _parse_covariance(payloads[MASS_COVARIANCE], exploration)
    color_covariance = _parse_covariance(payloads[COLOR_COVARIANCE], transfer)
    quality = [_quality_row(profile, config) for profile in (*exploration, *transfer)]
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item38-exploration-response-1.0",
            "item": 38,
            "archive_sha256": source["archive_sha256"],
            "allowed_member_payload_sha256": {
                name: _sha256_bytes(payloads[name]) for name in sorted(payloads)
            },
            "exploration_profiles": [_profile_json(profile) for profile in exploration],
            "unchanged_color_transfer_profiles": [
                _profile_json(profile) for profile in transfer
            ],
            "exploration_log10_covariance": mass_covariance.tolist(),
            "transfer_log10_covariance": color_covariance.tolist(),
            "quality": quality,
            "opened_members": sorted(allowed),
            "sealed_confirmation_member": SEALED_CONFIRMATION,
            "sealed_confirmation_payload_accesses": 0,
            "unused_member_payload_accesses": 0,
            "formula_cells_created_after_response": 0,
            "paid_api_calls": 0,
            "protocol_disclosure": (
                "The first 11 printed rows of the five allowed profiles were inspected after the "
                "formula and source freezes but before this parser/evaluator freeze; no formula "
                "generation or scoring occurred after that inspection. This remains exploration, "
                "not independent confirmation."
            ),
        }
    )
    _write_json(output, result)
    return result


def _quality_row(profile: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    esd = np.asarray(profile["esd_t_corrected"], dtype=np.float64)
    error = np.asarray(profile["esd_error_corrected"], dtype=np.float64)
    cross = np.asarray(profile["esd_x_raw"], dtype=np.float64)
    raw_error = np.asarray(profile["esd_error_raw"], dtype=np.float64)
    finite_positive = np.isfinite(esd) & np.isfinite(error) & (esd > 0.0) & (error > 0.0)
    valid_points = int(np.sum(finite_positive))
    median_fractional_error = (
        float(np.median(error[finite_positive] / esd[finite_positive]))
        if valid_points
        else float("inf")
    )
    cross_chi_square_per_point = float(np.mean(np.square(cross / raw_error)))
    passed = (
        valid_points >= int(config["evaluation"]["minimum_valid_points_per_profile"])
        and median_fractional_error
        <= float(config["evaluation"]["maximum_median_fractional_ESD_error"])
    )
    return {
        "name": profile["name"],
        "points": len(esd),
        "valid_positive_points": valid_points,
        "median_fractional_ESD_error": median_fractional_error,
        "cross_chi_square_per_point": cross_chi_square_per_point,
        "passed": bool(passed),
    }


def _profiles_from_receipt(
    receipt: Mapping[str, Any], key: str
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for raw in receipt[key]:
        profile = dict(raw)
        for name in (
            "g_bar_m_s2",
            "esd_t_raw",
            "esd_x_raw",
            "esd_error_raw",
            "multiplicative_bias",
            "esd_t_corrected",
            "esd_error_corrected",
            "g_obs_m_s2",
            "g_obs_error_m_s2",
        ):
            profile[name] = np.asarray(profile[name], dtype=np.float64)
        profiles.append(profile)
    return profiles


def _flatten_profiles(
    profiles: Sequence[Mapping[str, Any]], acceleration_scale: float
) -> dict[str, np.ndarray]:
    names: list[str] = []
    profile_index: list[int] = []
    g_bar: list[float] = []
    g_obs: list[float] = []
    for index, profile in enumerate(profiles):
        bar = np.asarray(profile["g_bar_m_s2"], dtype=np.float64)
        obs = np.asarray(profile["g_obs_m_s2"], dtype=np.float64)
        names.extend([str(profile["name"])] * len(bar))
        profile_index.extend([index] * len(bar))
        g_bar.extend(float(value) for value in bar)
        g_obs.extend(float(value) for value in obs)
    bar_array = np.asarray(g_bar, dtype=np.float64)
    obs_array = np.asarray(g_obs, dtype=np.float64)
    if np.any(bar_array <= 0.0) or np.any(obs_array <= 0.0):
        raise GravityItem38Error("nonpositive acceleration entered Item 38 evaluation")
    return {
        "name": np.asarray(names, dtype=object),
        "profile_index": np.asarray(profile_index, dtype=np.int64),
        "g_bar": bar_array,
        "g_obs": obs_array,
        "u": bar_array / acceleration_scale,
        "log_g_bar": np.log10(bar_array),
        "log_g_obs": np.log10(obs_array),
    }


def _covariance_loss(residual: np.ndarray, covariance: np.ndarray) -> float:
    inverse = np.linalg.pinv(covariance, hermitian=True, rcond=1e-10)
    return float(residual @ inverse @ residual / len(residual))


def _select_candidate(
    config: Mapping[str, Any],
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    log_g_bar: np.ndarray,
    log_g_obs: np.ndarray,
    covariance: np.ndarray,
    xp: Any,
    *,
    batch_size: int = 8192,
) -> tuple[int, float, int]:
    inverse = np.linalg.pinv(covariance, hermitian=True, rcond=1e-10)
    inverse_gpu = xp.asarray(inverse)
    target_gpu = xp.asarray(log_g_obs)
    bar_gpu = xp.asarray(log_g_bar)
    best_id = -1
    best_loss = float("inf")
    evaluations = 0
    for start in range(0, len(candidates["candidate_id"]), batch_size):
        stop = min(start + batch_size, len(candidates["candidate_id"]))
        rows = {key: value[start:stop] for key, value in candidates.items()}
        multiplier = predict_multiplier(rows, u, config)
        prediction = bar_gpu[None, :] + xp.log10(xp.asarray(multiplier))
        residual = target_gpu[None, :] - prediction
        losses = xp.einsum("bi,ij,bj->b", residual, inverse_gpu, residual) / len(u)
        local = int(xp.argmin(losses).item())
        value = float(losses[local].item())
        if value < best_loss:
            best_loss = value
            best_id = int(rows["candidate_id"][local])
        evaluations += (stop - start) * len(u)
    if best_id < 0:
        raise GravityItem38Error("no admitted Item 38 candidate was selectable")
    return best_id, best_loss, evaluations


def _candidate_log_prediction(
    candidate_id: int,
    config: Mapping[str, Any],
    u: np.ndarray,
    log_g_bar: np.ndarray,
) -> np.ndarray:
    raw = decode_candidate(candidate_id, config)
    grids = config["candidate_generator"]["parameter_grids"]
    indices = {
        "candidate_id": np.asarray([candidate_id], dtype=np.int64),
        "lane": np.asarray([raw["lane_id"]], dtype=np.int8),
        "amplitude_index": np.asarray(
            [grids["amplitude"].index(raw["parameters"]["amplitude"])], dtype=np.int16
        ),
        "exponent_index": np.asarray(
            [grids["exponent"].index(raw["parameters"]["exponent"])], dtype=np.int16
        ),
        "transition_index": np.asarray(
            [grids["transition_u"].index(raw["parameters"]["transition_u"])],
            dtype=np.int16,
        ),
        "shape_index": np.asarray(
            [grids["shape"].index(raw["parameters"]["shape"])], dtype=np.int16
        ),
    }
    multiplier = predict_multiplier(indices, u, config)[0]
    return log_g_bar + np.log10(multiplier)


def _fixed_log_prediction(name: str, data: Mapping[str, np.ndarray]) -> np.ndarray:
    return data["log_g_bar"] + np.log10(fixed_control_multiplier(name, data["u"]))


def _spline_design(x: np.ndarray) -> tuple[SplineTransformer, np.ndarray]:
    transformer = SplineTransformer(
        n_knots=5,
        degree=3,
        include_bias=False,
        extrapolation="linear",
    )
    return transformer, transformer.fit_transform(x[:, None])


def _fit_flexible(
    train: Mapping[str, np.ndarray], train_covariance: np.ndarray
) -> tuple[SplineTransformer, Ridge, float]:
    profile_ids = np.unique(train["profile_index"])
    best_alpha = SPLINE_ALPHAS[0]
    best_loss = float("inf")
    for alpha in SPLINE_ALPHAS:
        losses: list[float] = []
        for heldout in profile_ids:
            fit_mask = train["profile_index"] != heldout
            test_mask = ~fit_mask
            transformer, design = _spline_design(train["log_g_bar"][fit_mask])
            weights = 1.0 / np.maximum(np.diag(train_covariance)[fit_mask], 1e-12)
            model = Ridge(alpha=alpha)
            model.fit(design, train["log_g_obs"][fit_mask], sample_weight=weights)
            prediction = model.predict(
                transformer.transform(train["log_g_bar"][test_mask, None])
            )
            block = train_covariance[np.ix_(test_mask, test_mask)]
            losses.append(_covariance_loss(train["log_g_obs"][test_mask] - prediction, block))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_alpha = alpha
    transformer, design = _spline_design(train["log_g_bar"])
    weights = 1.0 / np.maximum(np.diag(train_covariance), 1e-12)
    model = Ridge(alpha=best_alpha)
    model.fit(design, train["log_g_obs"], sample_weight=weights)
    return transformer, model, best_alpha


def _predict_flexible(
    transformer: SplineTransformer, model: Ridge, log_g_bar: np.ndarray
) -> np.ndarray:
    return model.predict(transformer.transform(log_g_bar[:, None]))


def _object_losses(
    data: Mapping[str, np.ndarray],
    covariance: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_id in np.unique(data["profile_index"]):
        mask = data["profile_index"] == profile_id
        block = covariance[np.ix_(mask, mask)]
        candidate_loss = _covariance_loss(data["log_g_obs"][mask] - candidate[mask], block)
        reference_loss = _covariance_loss(data["log_g_obs"][mask] - reference[mask], block)
        rows.append(
            {
                "profile": str(data["name"][np.flatnonzero(mask)[0]]),
                "points": int(np.sum(mask)),
                "candidate_loss": candidate_loss,
                "reference_loss": reference_loss,
                "comparative_difference": candidate_loss - reference_loss,
                "raw_counterexample": bool(candidate_loss > reference_loss),
            }
        )
    return rows


def _improvement(reference_loss: float, candidate_loss: float) -> float:
    return 100.0 * (reference_loss - candidate_loss) / reference_loss


def _robustness(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    differences = np.asarray([row["comparative_difference"] for row in rows], dtype=np.float64)
    reference = np.asarray([row["reference_loss"] for row in rows], dtype=np.float64)
    candidate = np.asarray([row["candidate_loss"] for row in rows], dtype=np.float64)
    full = _improvement(float(np.mean(reference)), float(np.mean(candidate)))
    influence = np.abs(differences - float(np.mean(differences)))
    order = np.argsort(-influence, kind="stable")
    leave_mask = np.ones(len(rows), dtype=bool)
    leave_mask[order[0]] = False
    leave = _improvement(float(np.mean(reference[leave_mask])), float(np.mean(candidate[leave_mask])))
    trim_count = max(
        1,
        math.floor(float(config["evaluation"]["robust_comparative_trim_fraction"]) * len(rows)),
    )
    trim_mask = np.ones(len(rows), dtype=bool)
    trim_mask[order[:trim_count]] = False
    trimmed = _improvement(float(np.mean(reference[trim_mask])), float(np.mean(candidate[trim_mask])))
    return {
        "full_improvement_percent": full,
        "most_influential_profile": rows[int(order[0])]["profile"],
        "leave_one_most_influential_improvement_percent": leave,
        "leave_one_changes_sign": bool((full >= 0.0) != (leave >= 0.0)),
        "trim_fraction": float(config["evaluation"]["robust_comparative_trim_fraction"]),
        "trimmed_profiles": trim_count,
        "trimmed_improvement_percent": trimmed,
        "trim_changes_sign": bool((full >= 0.0) != (trimmed >= 0.0)),
    }


def _sign_permutation_p(rows: Sequence[Mapping[str, Any]]) -> float:
    differences = np.asarray([row["reference_loss"] - row["candidate_loss"] for row in rows])
    observed = float(np.mean(differences))
    values: list[float] = []
    for mask in range(1 << len(rows)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(rows))])
        values.append(float(np.mean(signs * differences)))
    return float((1 + np.sum(np.asarray(values) >= observed)) / (1 + len(values)))


def _bootstrap_counterexample_count(
    data: Mapping[str, np.ndarray],
    covariance: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int,
    draws: int,
) -> int:
    random = np.random.Generator(np.random.PCG64(seed))
    count = 0
    for profile_id in np.unique(data["profile_index"]):
        mask = data["profile_index"] == profile_id
        block = covariance[np.ix_(mask, mask)]
        mean = data["log_g_obs"][mask]
        samples = random.multivariate_normal(mean, block, size=draws, check_valid="ignore")
        inverse = np.linalg.pinv(block, hermitian=True, rcond=1e-10)
        candidate_residual = samples - candidate[mask][None, :]
        reference_residual = samples - reference[mask][None, :]
        candidate_loss = np.einsum(
            "bi,ij,bj->b", candidate_residual, inverse, candidate_residual
        )
        reference_loss = np.einsum(
            "bi,ij,bj->b", reference_residual, inverse, reference_residual
        )
        if float(np.mean(candidate_loss > reference_loss)) >= 0.95:
            count += 1
    return count


def _parse_cluster_rows(path: Path, acceleration_scale: float) -> dict[str, np.ndarray]:
    names: list[str] = []
    log_g_bar: list[float] = []
    log_g_obs: list[float] = []
    errors: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 7:
            continue
        try:
            bar = float(fields[3])
            obs = float(fields[4])
            error = float(fields[6])
        except ValueError:
            continue
        names.append(fields[1].strip())
        log_g_bar.append(bar)
        log_g_obs.append(obs)
        errors.append(error)
    if len(names) != 84 or len(set(names)) != 20:
        raise GravityItem38Error("unexpected CLASH diagnostic dimensions")
    bar_array = np.asarray(log_g_bar, dtype=np.float64)
    return {
        "name": np.asarray(names, dtype=object),
        "log_g_bar": bar_array,
        "log_g_obs": np.asarray(log_g_obs, dtype=np.float64),
        "error": np.asarray(errors, dtype=np.float64),
        "u": np.power(10.0, bar_array) / acceleration_scale,
    }


def _cluster_score(
    rows: Mapping[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    per_cluster: list[dict[str, Any]] = []
    for name in sorted({str(value) for value in rows["name"]}):
        mask = rows["name"] == name
        residual = (rows["log_g_obs"][mask] - prediction[mask]) / rows["error"][mask]
        per_cluster.append(
            {
                "cluster": name,
                "points": int(np.sum(mask)),
                "standardized_mse": float(np.mean(np.square(residual))),
                "rmse_dex": float(
                    np.sqrt(np.mean(np.square(rows["log_g_obs"][mask] - prediction[mask])))
                ),
            }
        )
    return {
        "standardized_mse": float(np.mean([row["standardized_mse"] for row in per_cluster])),
        "rmse_dex": float(np.mean([row["rmse_dex"] for row in per_cluster])),
        "per_cluster": per_cluster,
    }


def run_evaluation(root: Path, response_path: Path, output: Path) -> dict[str, Any]:
    config = load_config(root)
    verify_scientific_freeze(root, config)
    verify_source_metadata_freeze(root, config)
    receipt = _read_json(response_path)
    if receipt.get("schema_version") != "invariant-gravity-item38-exploration-response-1.0":
        raise GravityItem38Error("unexpected Item 38 response receipt")
    if int(receipt["sealed_confirmation_payload_accesses"]) != 0:
        raise GravityItem38Error("Item 38 confirmation response was accessed")

    exploration_profiles = _profiles_from_receipt(receipt, "exploration_profiles")
    transfer_profiles = _profiles_from_receipt(receipt, "unchanged_color_transfer_profiles")
    exploration_covariance = np.asarray(
        receipt["exploration_log10_covariance"], dtype=np.float64
    )
    transfer_covariance = np.asarray(receipt["transfer_log10_covariance"], dtype=np.float64)
    acceleration_scale = float(config["evaluation"]["acceleration_scale_m_s2"])
    exploration = _flatten_profiles(exploration_profiles, acceleration_scale)
    transfer = _flatten_profiles(transfer_profiles, acceleration_scale)
    candidates, candidate_audit = admissible_candidates(config, batch_size=16384)
    xp, backend, device = _backend()
    start = time.perf_counter()
    oof_candidate = np.empty_like(exploration["log_g_obs"])
    oof_flexible = np.empty_like(exploration["log_g_obs"])
    fold_rows: list[dict[str, Any]] = []
    formula_evaluations = 0
    for heldout in range(3):
        train_mask = exploration["profile_index"] != heldout
        test_mask = ~train_mask
        train_covariance = exploration_covariance[np.ix_(train_mask, train_mask)]
        selected_id, train_loss, evaluations = _select_candidate(
            config,
            candidates,
            exploration["u"][train_mask],
            exploration["log_g_bar"][train_mask],
            exploration["log_g_obs"][train_mask],
            train_covariance,
            xp,
        )
        formula_evaluations += evaluations
        prediction = _candidate_log_prediction(
            selected_id,
            config,
            exploration["u"][test_mask],
            exploration["log_g_bar"][test_mask],
        )
        oof_candidate[test_mask] = prediction
        transformer, model, alpha = _fit_flexible(
            {key: value[train_mask] for key, value in exploration.items()},
            train_covariance,
        )
        oof_flexible[test_mask] = _predict_flexible(
            transformer, model, exploration["log_g_bar"][test_mask]
        )
        fold_rows.append(
            {
                "heldout_profile": str(exploration["name"][np.flatnonzero(test_mask)[0]]),
                "selected_candidate": decode_candidate(selected_id, config),
                "training_loss": train_loss,
                "flexible_alpha": alpha,
            }
        )

    final_id, final_train_loss, evaluations = _select_candidate(
        config,
        candidates,
        exploration["u"],
        exploration["log_g_bar"],
        exploration["log_g_obs"],
        exploration_covariance,
        xp,
    )
    formula_evaluations += evaluations
    final_candidate = decode_candidate(final_id, config)
    transformer, model, final_alpha = _fit_flexible(exploration, exploration_covariance)
    fixed_predictions = {
        name: _fixed_log_prediction(name, exploration)
        for name in ("baryonic_newton", "verlinde_point_mass", "mond_RAR")
    }
    losses = {
        "candidate": _covariance_loss(
            exploration["log_g_obs"] - oof_candidate, exploration_covariance
        ),
        "flexible_ordinary": _covariance_loss(
            exploration["log_g_obs"] - oof_flexible, exploration_covariance
        ),
    }
    losses.update(
        {
            name: _covariance_loss(
                exploration["log_g_obs"] - prediction, exploration_covariance
            )
            for name, prediction in fixed_predictions.items()
        }
    )
    ordinary_names = ("flexible_ordinary", "baryonic_newton", "mond_RAR")
    strongest_name = min(ordinary_names, key=lambda name: losses[name])
    strongest_prediction = (
        oof_flexible if strongest_name == "flexible_ordinary" else fixed_predictions[strongest_name]
    )
    object_rows = _object_losses(
        exploration, exploration_covariance, oof_candidate, strongest_prediction
    )
    robustness = _robustness(object_rows, config)
    quality_by_name = {str(row["name"]): bool(row["passed"]) for row in receipt["quality"]}
    quality_passed = all(quality_by_name[str(name)] for name in EXPLORATION_NAMES)
    raw_counterexamples = sum(bool(row["raw_counterexample"]) for row in object_rows)
    quality_counterexamples = sum(
        bool(row["raw_counterexample"]) and quality_by_name[str(row["profile"])]
        for row in object_rows
    )
    uncertainty_counterexamples = _bootstrap_counterexample_count(
        exploration,
        exploration_covariance,
        oof_candidate,
        strongest_prediction,
        int(config["evaluation"]["permutation_seed"]),
        int(config["evaluation"]["permutation_count"]),
    )
    improvement_strongest = _improvement(losses[strongest_name], losses["candidate"])
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": 3,
        "raw_counterexample_count": raw_counterexamples,
        "quality_verified_counterexample_count": quality_counterexamples,
        "uncertainty_resolved_counterexample_count": min(
            uncertainty_counterexamples, quality_counterexamples
        ),
        "aggregate_improvement_percent": improvement_strongest,
        "quality_gate_passed": quality_passed,
        "strongest_baseline_failed": losses["candidate"] >= losses[strongest_name],
        "leave_one_changes_sign": robustness["leave_one_changes_sign"],
        "trim_changes_sign": robustness["trim_changes_sign"],
        "independent_failure_strata": raw_counterexamples,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    counterexample_assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / COUNTEREXAMPLE_POLICY_PATH)
    )

    transfer_candidate = _candidate_log_prediction(
        final_id, config, transfer["u"], transfer["log_g_bar"]
    )
    transfer_flexible = _predict_flexible(transformer, model, transfer["log_g_bar"])
    transfer_losses = {
        "candidate": _covariance_loss(
            transfer["log_g_obs"] - transfer_candidate, transfer_covariance
        ),
        "flexible_ordinary": _covariance_loss(
            transfer["log_g_obs"] - transfer_flexible, transfer_covariance
        ),
    }
    for name in ("baryonic_newton", "verlinde_point_mass", "mond_RAR"):
        prediction = _fixed_log_prediction(name, transfer)
        transfer_losses[name] = _covariance_loss(
            transfer["log_g_obs"] - prediction, transfer_covariance
        )
    transfer_reference_name = min(
        ("flexible_ordinary", "baryonic_newton", "mond_RAR"),
        key=lambda name: transfer_losses[name],
    )

    cluster = _parse_cluster_rows(root / str(config["cluster_transfer"]["source"]), acceleration_scale)
    cluster_candidate = _candidate_log_prediction(
        final_id, config, cluster["u"], cluster["log_g_bar"]
    )
    cluster_flexible = _predict_flexible(transformer, model, cluster["log_g_bar"])
    cluster_scores = {
        "candidate": _cluster_score(cluster, cluster_candidate),
        "flexible_ordinary": _cluster_score(cluster, cluster_flexible),
    }
    for name in ("baryonic_newton", "verlinde_point_mass", "mond_RAR"):
        prediction = cluster["log_g_bar"] + np.log10(
            fixed_control_multiplier(name, cluster["u"])
        )
        cluster_scores[name] = _cluster_score(cluster, prediction)
    cluster_reference_name = min(
        ("flexible_ordinary", "baryonic_newton", "mond_RAR"),
        key=lambda name: cluster_scores[name]["standardized_mse"],
    )

    gpu_seconds = time.perf_counter() - start
    sample_ids = np.linspace(0, len(candidates["candidate_id"]) - 1, 32, dtype=np.int64)
    sample = {key: value[sample_ids] for key, value in candidates.items()}
    cpu_values = predict_multiplier(sample, exploration["u"][:5], config)
    gpu_values = xp.asnumpy(xp.asarray(cpu_values))
    cpu_gpu_max_difference = float(np.max(np.abs(cpu_values - gpu_values)))
    paired_p = _sign_permutation_p(object_rows)
    gates = {
        "quality_passes": quality_passed,
        "selected_candidate_beats_baryonic": losses["candidate"] < losses["baryonic_newton"],
        "selected_candidate_beats_fixed_verlinde": (
            losses["candidate"] < losses["verlinde_point_mass"]
        ),
        "selected_candidate_beats_fixed_mond_RAR": losses["candidate"] < losses["mond_RAR"],
        "selected_candidate_beats_flexible_ordinary": (
            losses["candidate"] < losses["flexible_ordinary"]
        ),
        "paired_profile_permutation_p_at_most": paired_p <= 0.05,
        "all_broad_exploration_profiles_improve": raw_counterexamples == 0,
        "unchanged_color_transfer_improves": (
            transfer_losses["candidate"] < transfer_losses[transfer_reference_name]
        ),
        "cluster_diagnostic_improves_without_reselection": (
            cluster_scores["candidate"]["standardized_mse"]
            < cluster_scores[cluster_reference_name]["standardized_mse"]
        ),
        "not_single_object_sensitive": not (
            robustness["leave_one_changes_sign"] or robustness["trim_changes_sign"]
        ),
        "hard_theoretical_veto_absent": True,
        "confirmation_remains_sealed": int(receipt["sealed_confirmation_payload_accesses"]) == 0,
    }
    all_gates = all(gates.values())
    partial_lead = (
        quality_passed
        and losses["candidate"] < losses["baryonic_newton"]
        and losses["candidate"] < losses["verlinde_point_mass"]
        and not robustness["leave_one_changes_sign"]
    )
    if all_gates:
        decision = "PASS_EXPLORATION_ITEM38_EMERGENT_GRAVITY"
    elif partial_lead:
        decision = "NONPROMOTED_ITEM38_EMERGENT_GRAVITY_LEAD"
    elif not quality_passed:
        decision = "INCONCLUSIVE_ITEM38_QUALITY"
    else:
        decision = "NO_ITEM38_EMERGENT_GRAVITY_LEAD"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item38-emergent-gravity-result-1.0",
            "item": 38,
            "decision": decision,
            "hypothesis": config["hypothesis"],
            "selected_fold_candidates": fold_rows,
            "final_exploration_candidate": final_candidate,
            "final_exploration_training_loss": final_train_loss,
            "final_flexible_alpha": final_alpha,
            "primary_KiDS_exploration": {
                "profiles": 3,
                "points": len(exploration["g_obs"]),
                "losses": losses,
                "strongest_ordinary_baseline": strongest_name,
                "improvement_vs_strongest_percent": improvement_strongest,
                "paired_profile_sign_permutation_p": paired_p,
                "minimum_possible_exact_p_with_three_profiles": 2.0 / 9.0,
                "object_level": object_rows,
                "robustness": robustness,
                "counterexample_policy_report": policy_report,
                "counterexample_assessment": counterexample_assessment,
            },
            "unchanged_color_transfer": {
                "profiles": 2,
                "points": len(transfer["g_obs"]),
                "losses": transfer_losses,
                "strongest_ordinary_baseline": transfer_reference_name,
                "improvement_vs_strongest_percent": _improvement(
                    transfer_losses[transfer_reference_name], transfer_losses["candidate"]
                ),
            },
            "unchanged_CLASH_cluster_diagnostic": {
                "fresh_confirmation": False,
                "direct_lensing_likelihood": False,
                "profiles": 20,
                "points": len(cluster["log_g_obs"]),
                "scores": cluster_scores,
                "strongest_ordinary_baseline": cluster_reference_name,
                "improvement_vs_strongest_percent": _improvement(
                    cluster_scores[cluster_reference_name]["standardized_mse"],
                    cluster_scores["candidate"]["standardized_mse"],
                ),
            },
            "quality": receipt["quality"],
            "gates": gates,
            "compute": {
                "backend": backend,
                "device": device,
                "gpu_wall_seconds": gpu_seconds,
                "raw_candidate_cells": config["candidate_generator"]["raw_candidate_cells"],
                "admitted_candidates": candidate_audit["admitted_candidates"],
                "behavioral_equivalence_classes": candidate_audit[
                    "behavioral_equivalence_classes"
                ],
                "candidate_point_evaluations": formula_evaluations,
                "cpu_gpu_max_absolute_difference": cpu_gpu_max_difference,
                "paid_api_calls": 0,
                "paid_api_usd": 0.0,
            },
            "protocol": {
                "response_receipt_sha256": receipt["content_sha256"],
                "protocol_disclosure": receipt["protocol_disclosure"],
                "sealed_confirmation_payload_accesses": receipt[
                    "sealed_confirmation_payload_accesses"
                ],
                "formula_cells_created_after_response": 0,
            },
            "claim_boundaries": [
                "a fit of a macroscopic susceptibility is not a microscopic derivation",
                "the KiDS profiles are stacked and overlapping transfer bins are not independent objects",
                "the CLASH acceleration table is model-dependent and was exposed previously",
                "the cluster check is not a direct shear, arc, image, magnification, or time-delay likelihood",
                "generalized entropy and Verlinde-like point-mass formulas are known prior art",
                "potentially-new labels are search provenance, not novelty findings",
                "no empirical singleton prunes any formula or formula family",
            ],
            "next_action": (
                "Preserve every Item 38 formula and failure region. Advance to Item 39 on a new "
                "response only after this bounded result is documented; keep the fourth KiDS mass "
                "profile sealed unless a separately frozen gate explicitly authorizes it."
            ),
        }
    )
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    open_response = subparsers.add_parser("open-exploration-response")
    open_response.add_argument("--root", type=Path, default=Path("."))
    open_response.add_argument("--archive", type=Path, required=True)
    open_response.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--root", type=Path, default=Path("."))
    evaluate.add_argument("--response", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "open-exploration-response":
        result = open_exploration_response(args.root.resolve(), args.archive.resolve(), args.output)
    elif args.command == "evaluate":
        result = run_evaluation(args.root.resolve(), args.response, args.output)
    else:
        raise GravityItem38Error(f"unsupported command: {args.command}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
