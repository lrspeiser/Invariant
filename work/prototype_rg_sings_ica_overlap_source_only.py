"""Source-only SINGS two-band FastICA benchmark against S4G P5 products.

This is a work prototype, not a campaign artifact.  It never opens rotation,
pressure, temperature, lensing, or other response data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
IRAC1_ROOT = ROOT / "work/private/open-gravity-rg-sings-s4g-overlap-source-only-v1"
IRAC2_ROOT = ROOT / "work/private/open-gravity-rg-sings-irac2-source-only-v1"
S4G_ROOT = ROOT / "work/private/open-gravity-rg-12gal-source-only-v1"
OUTPUT = ROOT / "work/rg-sings-fastica-overlap-source-comparison-prototype-v1.json"
OBJECTS = ("NGC2976", "NGC3198", "NGC3521")
IRAC1_ZERO_JY = 280.9
IRAC2_ZERO_JY = 179.7
S4G_RMS_MJY_SR = 0.0072


def _read(path: Path) -> tuple[np.ndarray, fits.Header]:
    return np.asarray(fits.getdata(path, memmap=False), dtype=np.float64), fits.getheader(path)


def _wcs(header: fits.Header) -> WCS:
    value = WCS(header, relax=True)
    value.sip = None
    return value


def _color_from_ratio(f36_over_f45: float) -> float:
    return -2.5 * math.log10(f36_over_f45 * IRAC2_ZERO_JY / IRAC1_ZERO_JY)


def _ratio45_over36(color: float) -> float:
    return (IRAC2_ZERO_JY / IRAC1_ZERO_JY) * 10.0 ** (0.4 * color)


def _symmetric_decorrelation(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix @ matrix.T)
    if np.min(values) <= 0.0:
        raise RuntimeError("singular FastICA decorrelation")
    return (vectors @ np.diag(values ** -0.5) @ vectors.T) @ matrix


def _fastica_two_source(
    samples: np.ndarray,
    *,
    stellar_seed_color: float,
    dust_seed_color: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, int, float]:
    """Return a two-column mixing matrix, initialized by physical colors.

    The fixed-point nonlinearity is the standard FastICA ``gauss`` choice,
    g(u)=u exp(-u^2/2).  This is a transparent reimplementation, not a claim
    of byte equivalence with the historical IT++ implementation.
    """

    if samples.shape[0] != 2:
        raise RuntimeError("expected two feature rows")
    centered = samples - np.mean(samples, axis=1, keepdims=True)
    covariance = centered @ centered.T / centered.shape[1]
    values, vectors = np.linalg.eigh(covariance)
    if np.min(values) <= 1.0e-18:
        raise RuntimeError("degenerate two-band covariance")
    whitening = np.diag(values ** -0.5) @ vectors.T
    white = whitening @ centered
    mixing_seed = np.array(
        [
            [1.0, 1.0],
            [_ratio45_over36(stellar_seed_color), _ratio45_over36(dust_seed_color)],
        ],
        dtype=np.float64,
    )
    unmixing = _symmetric_decorrelation(np.linalg.inv(whitening @ mixing_seed))
    convergence = math.inf
    for iteration in range(1, max_iterations + 1):
        projected = unmixing @ white
        exponential = np.exp(-0.5 * projected**2)
        g = projected * exponential
        gp_mean = np.mean((1.0 - projected**2) * exponential, axis=1)
        update = g @ white.T / white.shape[1] - gp_mean[:, None] * unmixing
        update = _symmetric_decorrelation(update)
        convergence = float(
            np.max(np.abs(np.abs(np.diag(update @ unmixing.T)) - 1.0))
        )
        unmixing = update
        if convergence <= tolerance:
            break
    else:
        raise RuntimeError("FastICA failed to converge")
    original_unmixing = unmixing @ whitening
    mixing = np.linalg.inv(original_unmixing)
    return mixing, iteration, convergence


def _largest_center_source(
    data: np.ndarray,
    finite: np.ndarray,
    sigma: float,
    *,
    center_yx: tuple[int, int],
) -> np.ndarray:
    smooth = ndimage.gaussian_filter(np.where(finite, data, 0.0), sigma=2.0)
    detected = finite & (smooth > 3.0 * sigma)
    detected = ndimage.binary_closing(detected, iterations=3)
    labels, count = ndimage.label(detected)
    if count == 0:
        raise RuntimeError("no central source detected")
    cy, cx = center_yx
    central = int(labels[cy, cx])
    if central == 0:
        centers = ndimage.center_of_mass(detected, labels, range(1, count + 1))
        central = min(
            range(1, count + 1),
            key=lambda index: (centers[index - 1][0] - cy) ** 2
            + (centers[index - 1][1] - cx) ** 2,
        )
    return ndimage.binary_dilation(labels == central, iterations=8)


def _robust_sigma(data: np.ndarray, finite: np.ndarray) -> float:
    values = data[finite]
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("invalid robust noise")
    return sigma


def _component_colors(mixing: np.ndarray) -> list[float]:
    colors = []
    for index in range(2):
        column = mixing[:, index]
        if column[0] == 0.0 or column[1] / column[0] <= 0.0:
            colors.append(math.nan)
        else:
            colors.append(_color_from_ratio(column[0] / column[1]))
    return colors


def _estimate_colors(f36: np.ndarray, f45: np.ndarray, training: np.ndarray) -> dict[str, object]:
    samples = np.vstack([f36[training], f45[training]])
    if samples.shape[1] > 250_000:
        indices = np.linspace(0, samples.shape[1] - 1, 250_000, dtype=np.int64)
        samples = samples[:, indices]
    solutions: list[dict[str, float | int]] = []
    stellar_seeds = np.arange(-0.20, 0.0001, 0.04)
    dust_seeds = np.arange(0.0, 1.5001, 0.30)
    for stellar_seed in stellar_seeds:
        for dust_seed in dust_seeds:
            if dust_seed <= stellar_seed:
                continue
            try:
                mixing, iterations, convergence = _fastica_two_source(
                    samples,
                    stellar_seed_color=float(stellar_seed),
                    dust_seed_color=float(dust_seed),
                )
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            colors = _component_colors(mixing)
            if not all(math.isfinite(value) for value in colors):
                continue
            stellar_index = int(np.argmin(colors))
            dust_index = 1 - stellar_index
            stellar_color = float(colors[stellar_index])
            dust_color = float(colors[dust_index])
            if dust_color <= stellar_color:
                continue
            solutions.append(
                {
                    "stellar_color": stellar_color,
                    "dust_color": dust_color,
                    "iterations": iterations,
                    "convergence": convergence,
                }
            )
    if not solutions:
        raise RuntimeError("no physical FastICA solutions")
    stellar_values = np.array([row["stellar_color"] for row in solutions], dtype=np.float64)
    dust_values = np.array([row["dust_color"] for row in solutions], dtype=np.float64)
    return {
        "documented_seed_grid_count": int(stellar_seeds.size * dust_seeds.size),
        "converged_solution_count": len(solutions),
        "stellar_color_mean": float(np.mean(stellar_values)),
        "stellar_color_std": float(np.std(stellar_values)),
        "dust_color_mean": float(np.mean(dust_values)),
        "dust_color_std": float(np.std(dust_values)),
        "max_iterations": max(int(row["iterations"]) for row in solutions),
        "max_convergence_residual": max(float(row["convergence"]) for row in solutions),
    }


def _decompose(f36: np.ndarray, f45: np.ndarray, star_color: float, dust_color: float) -> tuple[np.ndarray, np.ndarray]:
    mixing = np.array(
        [
            [1.0, 1.0],
            [_ratio45_over36(star_color), _ratio45_over36(dust_color)],
        ],
        dtype=np.float64,
    )
    sources = np.linalg.solve(mixing, np.vstack([f36.ravel(), f45.ravel()]))
    star = sources[0].reshape(f36.shape)
    dust = sources[1].reshape(f36.shape)
    return star, dust


def _metrics(reference: np.ndarray, candidate: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    x = reference[valid]
    y = candidate[valid]
    return {
        "pixels": int(x.size),
        "integrated_candidate_over_reference": float(np.sum(y) / np.sum(x)),
        "through_origin_slope": float(np.dot(x, y) / np.dot(x, x)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
        "median_absolute_error_mjy_sr": float(np.median(np.abs(y - x))),
    }


def analyze(object_id: str) -> dict[str, object]:
    f36, h36 = _read(IRAC1_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_FLUX.fits")
    w36, hw36 = _read(IRAC1_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_WEIGHT.fits")
    f45_raw, h45 = _read(IRAC2_ROOT / f"{object_id}__SINGS_IRAC2__STELLAR_IRAC2_FLUX.fits")
    w45_raw, hw45 = _read(IRAC2_ROOT / f"{object_id}__SINGS_IRAC2__STELLAR_IRAC2_WEIGHT.fits")
    target36 = _wcs(h36)
    f45, p45 = reproject_interp((f45_raw, _wcs(h45)), target36, shape_out=f36.shape, order="bilinear")
    w45, pw45 = reproject_interp((w45_raw, _wcs(hw45)), target36, shape_out=f36.shape, order="nearest-neighbor")
    finite = (
        np.isfinite(f36)
        & np.isfinite(f45)
        & np.isfinite(w36)
        & np.isfinite(w45)
        & (w36 > 0.0)
        & (w45 > 0.0)
        & (p45 > 0.999)
        & (pw45 > 0.999)
    )
    sigma36 = _robust_sigma(f36, finite)
    sigma45 = _robust_sigma(f45, finite)
    galaxy = _largest_center_source(
        f36,
        finite,
        sigma36,
        center_yx=(int(round(float(h36["CRPIX2"]) - 1.0)), int(round(float(h36["CRPIX1"]) - 1.0))),
    )
    positive = galaxy & finite & (f36 > 0.0) & (f45 > 0.0)
    color = np.full(f36.shape, np.nan, dtype=np.float64)
    color[positive] = -2.5 * np.log10(
        (f36[positive] / f45[positive]) * (IRAC2_ZERO_JY / IRAC1_ZERO_JY)
    )
    training = (
        positive
        & (f36 > 10.0 * sigma36)
        & (f45 > 10.0 * sigma45)
        & (color > -0.3)
        & (color < 1.5)
    )
    if np.count_nonzero(training) < 1_000:
        raise RuntimeError(f"too few training pixels for {object_id}")
    first_solution = _estimate_colors(f36, f45, training)
    first_star, first_dust = _decompose(
        f36,
        f45,
        float(first_solution["stellar_color_mean"]),
        float(first_solution["dust_color_mean"]),
    )
    red_outliers = positive & (
        color > float(first_solution["dust_color_mean"])
    )
    red_outliers = ndimage.binary_dilation(red_outliers, iterations=1)
    dust_training = first_dust[training]
    dust_five_sigma = float(np.mean(dust_training) + 5.0 * np.std(dust_training))
    high_dust = (
        training
        & (first_dust > dust_five_sigma)
        & (color > min(float(first_solution["dust_color_mean"]), 0.1))
    )
    second_training = training & ~red_outliers & ~high_dust
    second_solution: dict[str, object] | None = None
    if np.count_nonzero(second_training) >= 1_000:
        second_solution = _estimate_colors(f36, f45, second_training)
    use_second = bool(
        second_solution is not None
        and -0.2 <= float(second_solution["stellar_color_mean"]) <= 0.0
        and 0.0 <= float(second_solution["dust_color_mean"]) <= 1.5
        and float(second_solution["dust_color_mean"])
        < float(first_solution["dust_color_mean"])
    )
    solution = second_solution if use_second else first_solution
    assert solution is not None
    star, dust = _decompose(
        f36,
        f45,
        float(solution["stellar_color_mean"]),
        float(solution["dust_color_mean"]),
    )

    reference_star, hs = _read(S4G_ROOT / f"{object_id}__S4G_P5__STELLAR_MASS_MAP.fits")
    reference_dust, hd = _read(IRAC1_ROOT / f"{object_id}__S4G_P5__NONSTELLAR_MAP.fits")
    reference_mask, hm = _read(S4G_ROOT / f"{object_id}__S4G_P5__STELLAR_ICA_MASK.fits")
    target = _wcs(hs)
    candidate_star, ps = reproject_interp((star, target36), target, shape_out=reference_star.shape, order="bilinear")
    candidate_dust, pd = reproject_interp((dust, target36), target, shape_out=reference_star.shape, order="bilinear")
    mask, pm = reproject_interp((reference_mask, _wcs(hm)), target, shape_out=reference_star.shape, order="nearest-neighbor")
    valid = (
        np.isfinite(reference_star)
        & np.isfinite(reference_dust)
        & np.isfinite(candidate_star)
        & np.isfinite(candidate_dust)
        & (ps > 0.999)
        & (pd > 0.999)
        & (pm > 0.999)
        & (mask == 0.0)
        & (reference_star > 5.0 * S4G_RMS_MJY_SR)
    )
    if np.count_nonzero(valid) < 1_000:
        raise RuntimeError(f"too few validation pixels for {object_id}")
    return {
        "object_id": object_id,
        "noise_mjy_sr": {"irac1": sigma36, "irac2": sigma45},
        "training_pixels": int(np.count_nonzero(training)),
        "second_training_pixels": int(np.count_nonzero(second_training)),
        "first_solution": first_solution,
        "second_solution": second_solution,
        "selected_iteration": 2 if use_second else 1,
        "solution": solution,
        "physical_color_gates": {
            "stellar_in_minus0p2_to_0": -0.2 <= float(solution["stellar_color_mean"]) <= 0.0,
            "dust_in_0_to_1p5": 0.0 <= float(solution["dust_color_mean"]) <= 1.5,
        },
        "exact_reconstruction_max_abs_mjy_sr": float(np.nanmax(np.abs(star + dust - f36))),
        "stellar_benchmark": _metrics(reference_star, candidate_star, valid),
        "dust_benchmark": _metrics(reference_dust, candidate_dust, valid),
    }


def main() -> None:
    records = [analyze(object_id) for object_id in OBJECTS]
    payload = {
        "schema": "invariant-work-rg-sings-fastica-overlap-source-comparison-prototype-1.0",
        "status": "SOURCE_ONLY_PROTOTYPE_NOT_ADMITTED",
        "method": "Two-band fixed-point FastICA using the documented stellar/dust color seed lattice, followed by raw-intensity reconstruction from the mean recovered component colors.",
        "implementation_boundary": "The papers document 48 pipeline perturbations but the stated 0.04-mag stellar and 0.3-mag dust steps over the stated closed color intervals define 36 Cartesian seeds. This prototype uses those 36 reproducible cells and makes no byte-equivalence claim to IT++ or S4G Pipeline 5.",
        "paper_bindings": {
            "meidt_2012": "https://arxiv.org/abs/1110.2683",
            "querejeta_2015": "https://arxiv.org/abs/1410.0009",
            "sings_dr5": "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/doc/sings_fifth_delivery_v2.pdf",
        },
        "records": records,
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
