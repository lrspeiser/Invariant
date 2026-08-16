"""Spectral (Fourier) bias scanning over integer sequences.

The Ulam-sequence progress receipt names its own blocker: *statement_kinds_too_weak* —
"the declared statement kinds cannot express the empirical quasi-periodic signal
(cos(alpha a_n) < 0, Steinerberger 2017)".  This module is that missing statement
kind's engine: a brute-force-backwards scan that recovers exact structure from a
massive numeric sweep.

The scan computes ``S(lambda) = |mean_n exp(i * lambda * a_n)|`` on a declared dense
grid of frequencies, on the **prefix only**.  Peaks of `S` are candidate hidden
frequencies.  For each peak the conjecture takes the Steinerberger form:

    cos(lambda* * a_n) < 0 for at least fraction f of terms

with `lambda*` refined by golden-section search in mpmath and then quantized to the
declared reporting tolerance, so the lambda printed in the statement is *exactly* the
lambda that is tested.  The held-out suffix is then confronted with exact cos-sign
counts at that stated lambda.

Honesty rules:

**A peak is not a proof.**  Every result carries ``peak_is_not_proof: true``.  A large
`S` value on a finite prefix establishes nothing beyond that prefix.

**The float layer is a pre-filter, never an adjudicator.**  The fp64 grid sweep (GPU
cupy or CPU numpy) only nominates a bounded candidate pool.  Ranking, peak
confirmation, refinement, quantization, and every count that reaches the receipt are
recomputed in mpmath at declared precision, so the claim surface is deterministic and
exactly recheckable without the original device.

**Thresholds come from a declared ladder.**  The claimed fraction `f` is the largest
entry of a frozen ladder that the prefix supports — never a number tuned to the
suffix.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-spectral-signal-scan-result-1.0"

#: Grid profiles.  Frequencies are exact rationals `k / grid_denominator` for
#: `k` in [start_k, stop_k]; the dense profile is the declared ~314k-point sweep
#: over [0.01, pi), step 1e-5.
PROFILES: dict[str, dict[str, int]] = {
    "engine": {
        "grid_denominator": 1000,
        "start_k": 10,
        "stop_k": 3141,
        "top_k": 3,
        "candidate_pool": 12,
        "min_separation_k": 50,
    },
    "dense": {
        "grid_denominator": 100000,
        "start_k": 1000,
        "stop_k": 314159,
        "top_k": 5,
        "candidate_pool": 20,
        "min_separation_k": 5000,
    },
}

SYSTEM_CAPS = {
    "max_rows": 8192,
    "min_prefix_rows": 16,
    "min_holdout_rows": 8,
    "prefix_numerator": 3,
    "prefix_denominator": 5,
    "noise_floor_numerator": 7,
    "noise_floor_denominator": 2,
    "quantize_denominator": 10**12,
    "base_dps": 40,
}

#: Declared fraction ladder for the "at least f of terms" claim.  The proposer picks
#: the largest entry at or below the measured prefix fraction; below 11/20 no bias
#: is claimed at all (1/2 is the unbiased expectation).
THRESHOLD_LADDER = (
    Fraction(49, 50),
    Fraction(19, 20),
    Fraction(9, 10),
    Fraction(4, 5),
    Fraction(3, 4),
    Fraction(2, 3),
    Fraction(3, 5),
    Fraction(11, 20),
)

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "float_layer_is_prefilter_only": True,
    "grid_is_declared_and_finite": True,
    "peak_is_not_proof": True,
    "proposed_from_prefix_only": True,
    "survival_on_holdout_establishes_truth": False,
}

REFINEMENT_TOLERANCE = "1e-12"


class SpectralScanError(ValueError):
    """Raised on malformed input, cap violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact-side helpers (mpmath at declared precision)
# ---------------------------------------------------------------------------


def _working_dps(values: Sequence[int]) -> int:
    """Declared precision: base digits plus the width of the largest term."""

    magnitude = max((abs(int(value)) for value in values), default=1)
    return SYSTEM_CAPS["base_dps"] + len(str(magnitude))


def _mp_magnitude(values: Sequence[int], lam: Any, dps: int) -> Any:
    """`|mean_n exp(i*lam*a_n)|` in mpmath at the declared precision."""

    with mpmath.workdps(dps):
        real = mpmath.mpf(0)
        imag = mpmath.mpf(0)
        for value in values:
            phase = lam * value
            real += mpmath.cos(phase)
            imag += mpmath.sin(phase)
        count = len(values)
        return mpmath.sqrt((real / count) ** 2 + (imag / count) ** 2)


def _fraction_mpf(value: Fraction, dps: int) -> Any:
    with mpmath.workdps(dps):
        return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)


def cos_sign(lam: Fraction, value: Fraction, dps: int) -> int:
    """Sign of cos(lam * value): -1, +1, or 0 when numerically ambiguous.

    lam and value are exact rationals and pi is irrational, so cos(lam*value) is
    never exactly zero for nonzero arguments; 0 here means "too close to the sign
    boundary to certify at this precision" and callers must fail closed on it.
    """

    with mpmath.workdps(dps):
        phase = (
            mpmath.mpf(lam.numerator)
            / lam.denominator
            * mpmath.mpf(value.numerator)
            / value.denominator
        )
        cosine = mpmath.cos(phase)
        if abs(cosine) < mpmath.mpf(10) ** (-(dps // 2)):
            return 0
        return -1 if cosine < 0 else 1


def _negative_counts(
    lam: Fraction, values: Sequence[Fraction], dps: int
) -> tuple[int, int]:
    """(negative_count, ambiguous_count) of cos(lam * value) over `values`."""

    negative = 0
    ambiguous = 0
    for value in values:
        sign = cos_sign(lam, value, dps)
        if sign < 0:
            negative += 1
        elif sign == 0:
            ambiguous += 1
    return negative, ambiguous


def _quantize(lam: Any, dps: int) -> Fraction:
    """Round an mpmath lambda to the declared reporting grid, exactly."""

    denominator = SYSTEM_CAPS["quantize_denominator"]
    with mpmath.workdps(dps):
        numerator = int(mpmath.nint(lam * denominator))
    return Fraction(numerator, denominator)


def lambda_decimal(lam: Fraction) -> str:
    """Exact fixed-point decimal rendering of a quantized lambda."""

    denominator = SYSTEM_CAPS["quantize_denominator"]
    if lam.denominator != 1 and denominator % lam.denominator != 0:
        raise SpectralScanError("lambda is not on the declared reporting grid")
    scaled = lam.numerator * (denominator // lam.denominator)
    sign = "-" if scaled < 0 else ""
    scaled = abs(scaled)
    return f"{sign}{scaled // denominator}.{scaled % denominator:012d}"


def _decimal_fraction(text: str) -> Fraction:
    return Fraction(text)


# ---------------------------------------------------------------------------
# fp64 pre-filter (GPU when requested and available, else numpy)
# ---------------------------------------------------------------------------


def _grid_magnitudes_fp64(
    values: Sequence[int], profile: Mapping[str, int], use_gpu: bool
) -> tuple[np.ndarray, str]:
    """|S| over the declared grid in fp64.  Pre-filter only: nominates candidates."""

    ks = np.arange(profile["start_k"], profile["stop_k"] + 1, dtype=np.int64)
    lambdas = ks.astype(np.float64) / float(profile["grid_denominator"])
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    device = "cpu-numpy-fp64"
    xp = np
    if use_gpu:
        try:
            import cupy

            cupy.cuda.runtime.getDeviceCount()
            xp = cupy
            device = "gpu-cupy-fp64"
        except (ImportError, RuntimeError):  # pragma: no cover - GPU absence
            xp = np
    lambdas_x = xp.asarray(lambdas)
    data_x = xp.asarray(data)
    chunk = 20000
    out = xp.empty(lambdas_x.shape[0], dtype=xp.float64)
    for start in range(0, lambdas_x.shape[0], chunk):
        stop = min(start + chunk, lambdas_x.shape[0])
        phases = data_x[:, None] * lambdas_x[None, start:stop]
        real = xp.cos(phases).mean(axis=0)
        imag = xp.sin(phases).mean(axis=0)
        out[start:stop] = xp.sqrt(real * real + imag * imag)
    if xp is not np:
        out = xp.asnumpy(out)
    return np.asarray(out), device


def _candidate_pool(
    magnitudes: np.ndarray, profile: Mapping[str, int]
) -> list[int]:
    """Interior local maxima, ranked by fp64 magnitude, separation-limited."""

    interior = np.arange(1, magnitudes.shape[0] - 1)
    is_peak = (magnitudes[interior] >= magnitudes[interior - 1]) & (
        magnitudes[interior] > magnitudes[interior + 1]
    )
    peak_offsets = interior[is_peak]
    order = sorted(
        (int(offset) for offset in peak_offsets),
        key=lambda offset: (-float(magnitudes[offset]), offset),
    )
    chosen: list[int] = []
    separation = int(profile["min_separation_k"])
    for offset in order:
        if all(abs(offset - kept) >= separation for kept in chosen):
            chosen.append(offset)
        if len(chosen) >= int(profile["candidate_pool"]):
            break
    return chosen


# ---------------------------------------------------------------------------
# Peak adjudication and refinement (mpmath)
# ---------------------------------------------------------------------------


def _golden_refine(
    values: Sequence[int], low: Fraction, high: Fraction, dps: int
) -> tuple[Any, int]:
    """Golden-section maximization of |S| on [low, high] to the declared tolerance."""

    with mpmath.workdps(dps):
        inv_phi = (mpmath.sqrt(5) - 1) / 2
        left = _fraction_mpf(low, dps)
        right = _fraction_mpf(high, dps)
        tolerance = mpmath.mpf(1) / SYSTEM_CAPS["quantize_denominator"]
        inner_left = right - inv_phi * (right - left)
        inner_right = left + inv_phi * (right - left)
        value_left = _mp_magnitude(values, inner_left, dps)
        value_right = _mp_magnitude(values, inner_right, dps)
        iterations = 0
        while (right - left) > tolerance:
            iterations += 1
            if value_left > value_right:
                right, inner_right, value_right = inner_right, inner_left, value_left
                inner_left = right - inv_phi * (right - left)
                value_left = _mp_magnitude(values, inner_left, dps)
            else:
                left, inner_left, value_left = inner_left, inner_right, value_right
                inner_right = left + inv_phi * (right - left)
                value_right = _mp_magnitude(values, inner_right, dps)
        return (left + right) / 2, iterations


def find_peaks(
    values: Sequence[int], profile_name: str = "engine", use_gpu: bool = False
) -> list[dict[str, Any]]:
    """Adjudicated spectral peaks of the prefix, largest first.

    Each peak carries the refined and quantized lambda (exact rational plus decimal
    string), the mpmath magnitudes, and the fp64 grid index that nominated it.
    """

    profile = PROFILES[profile_name]
    integers = [int(value) for value in values]
    dps = _working_dps(integers)
    magnitudes, device = _grid_magnitudes_fp64(integers, profile, use_gpu)
    pool = _candidate_pool(magnitudes, profile)
    denominator = profile["grid_denominator"]
    start_k = profile["start_k"]

    with mpmath.workdps(dps):
        floor = (
            mpmath.mpf(SYSTEM_CAPS["noise_floor_numerator"])
            / SYSTEM_CAPS["noise_floor_denominator"]
            / mpmath.sqrt(len(integers))
        )
    adjudicated: list[tuple[Any, int]] = []
    for offset in pool:
        k = start_k + offset
        center = _mp_magnitude(integers, _fraction_mpf(Fraction(k, denominator), dps), dps)
        left = _mp_magnitude(
            integers, _fraction_mpf(Fraction(k - 1, denominator), dps), dps
        )
        right = _mp_magnitude(
            integers, _fraction_mpf(Fraction(k + 1, denominator), dps), dps
        )
        if center < floor or center < left or center <= right:
            continue
        adjudicated.append((center, k))
    adjudicated.sort(key=lambda item: (-item[0], item[1]))

    peaks: list[dict[str, Any]] = []
    for center, k in adjudicated[: int(profile["top_k"])]:
        refined, iterations = _golden_refine(
            integers,
            Fraction(k - 1, denominator),
            Fraction(k + 1, denominator),
            dps,
        )
        quantized = _quantize(refined, dps)
        magnitude = _mp_magnitude(integers, _fraction_mpf(quantized, dps), dps)
        with mpmath.workdps(dps):
            magnitude_text = mpmath.nstr(magnitude, 12)
            grid_text = mpmath.nstr(center, 12)
        peaks.append(
            {
                "grid_k": k,
                "grid_lambda": lambda_decimal(Fraction(k, denominator)),
                "grid_magnitude": grid_text,
                "lambda_fraction": quantized,
                "lambda": lambda_decimal(quantized),
                "magnitude_at_lambda": magnitude_text,
                "refinement": {
                    "bracket": [
                        lambda_decimal(Fraction(k - 1, denominator)),
                        lambda_decimal(Fraction(k + 1, denominator)),
                    ],
                    "iterations": iterations,
                    "tolerance": REFINEMENT_TOLERANCE,
                },
                "device": device,
                "working_dps": dps,
            }
        )
    return peaks


def propose_bias_from_prefix(
    values: Sequence[int], profile_name: str = "engine", use_gpu: bool = False
) -> list[dict[str, Any]]:
    """Peaks plus the Steinerberger-form bias proposal each one supports, if any.

    The proposal threshold is the largest declared ladder fraction at or below the
    prefix's own cos-negative fraction *at the stated quantized lambda*; peaks whose
    prefix fraction does not exceed 1/2 propose nothing.
    """

    integers = [int(value) for value in values]
    dps = _working_dps(integers)
    fractions = [Fraction(value) for value in integers]
    proposals: list[dict[str, Any]] = []
    for peak in find_peaks(values, profile_name, use_gpu):
        negative, ambiguous = _negative_counts(peak["lambda_fraction"], fractions, dps)
        measured = Fraction(negative, len(integers))
        threshold = next(
            (entry for entry in THRESHOLD_LADDER if measured >= entry), None
        )
        entry = {
            **peak,
            "prefix_negative_terms": negative,
            "prefix_ambiguous_terms": ambiguous,
            "prefix_terms": len(integers),
            "threshold": None
            if threshold is None
            else {"numerator": threshold.numerator, "denominator": threshold.denominator},
            "statement": None
            if threshold is None
            else (
                f"cos({peak['lambda']} * a(n)) < 0 for at least "
                f"{threshold.numerator}/{threshold.denominator} of terms"
            ),
        }
        entry["threshold_fraction"] = threshold
        proposals.append(entry)
    return proposals


# ---------------------------------------------------------------------------
# Standalone sealed scan receipts (prefix scan + suffix confrontation)
# ---------------------------------------------------------------------------


def _parse_rows(rows: Any) -> list[tuple[int, int]]:
    if not isinstance(rows, list) or not rows:
        raise SpectralScanError("rows must be a non-empty list")
    if len(rows) > SYSTEM_CAPS["max_rows"]:
        raise SpectralScanError("row count exceeds cap")
    parsed: list[tuple[int, int]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"point", "value"}:
            raise SpectralScanError("each row needs exactly point and value")
        point = row["point"]
        value = row["value"]
        if isinstance(value, Mapping):
            if set(value) != {"numerator", "denominator"} or value["denominator"] != 1:
                raise SpectralScanError("spectral scan rows must be exact integers")
            value = value["numerator"]
        if (
            not isinstance(point, int)
            or isinstance(point, bool)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise SpectralScanError("point and value must be integers")
        if point in seen:
            raise SpectralScanError("duplicate point")
        seen.add(point)
        parsed.append((point, value))
    parsed.sort(key=lambda item: item[0])
    return parsed


def _split(rows: Sequence[tuple[int, int]]) -> tuple[list, list]:
    count = len(rows)
    cut = max(
        SYSTEM_CAPS["min_prefix_rows"],
        count * SYSTEM_CAPS["prefix_numerator"] // SYSTEM_CAPS["prefix_denominator"],
    )
    cut = min(cut, count - SYSTEM_CAPS["min_holdout_rows"])
    if cut < SYSTEM_CAPS["min_prefix_rows"]:
        raise SpectralScanError("insufficient rows for a prefix/holdout split")
    return list(rows[:cut]), list(rows[cut:])


def scan_receipt(
    rows: Any,
    sequence_label: str = "",
    profile_name: str = "dense",
    use_gpu: bool = False,
    builtin_knowledge: str | None = None,
) -> dict[str, Any]:
    """Full sealed scan: prefix-only peak discovery, suffix confrontation."""

    parsed = _parse_rows(rows)
    prefix, holdout = _split(parsed)
    prefix_values = [value for _, value in prefix]
    holdout_values = [Fraction(value) for _, value in holdout]
    dps = _working_dps([value for _, value in parsed])
    profile = PROFILES[profile_name]

    peaks: list[dict[str, Any]] = []
    survived = 0
    refuted = 0
    for entry in propose_bias_from_prefix(prefix_values, profile_name, use_gpu):
        threshold = entry.pop("threshold_fraction")
        lam = entry.pop("lambda_fraction")
        public = dict(entry)
        if threshold is None:
            public["holdout"] = None
            public["status"] = "NO_BIAS_PROPOSED"
        else:
            negative, ambiguous = _negative_counts(lam, holdout_values, dps)
            ok = Fraction(negative, len(holdout_values)) >= threshold
            public["holdout"] = {
                "rows": len(holdout_values),
                "negative_terms": negative,
                "ambiguous_terms": ambiguous,
                "required_fraction": {
                    "numerator": threshold.numerator,
                    "denominator": threshold.denominator,
                },
            }
            public["status"] = "SURVIVED" if ok else "REFUTED"
            survived += 1 if ok else 0
            refuted += 0 if ok else 1
        peaks.append(public)

    if any(peak["status"] == "SURVIVED" for peak in peaks):
        decision = "SPECTRAL_BIAS_SURVIVED"
    elif any(peak["status"] == "REFUTED" for peak in peaks):
        decision = "NONE_SURVIVED"
    else:
        decision = "NONE_PROPOSED"

    body: dict[str, Any] = {
        "builtin_knowledge": builtin_knowledge,
        "claims": CLAIMS,
        "counts": {
            "grid_points": profile["stop_k"] - profile["start_k"] + 1,
            "holdout_rows": len(holdout),
            "peaks_reported": len(peaks),
            "prefix_rows": len(prefix),
            "proposals_refuted": refuted,
            "proposals_survived": survived,
        },
        "decision": decision,
        "novelty": {
            "claimed": False,
            "note": (
                "a surviving bias here is a candidate signal in this declared range "
                "only; absence from this repository's builtin knowledge establishes "
                "nothing about the literature"
            ),
        },
        "peaks": peaks,
        "profile": {"name": profile_name, **profile},
        "public_rows": [{"point": point, "value": value} for point, value in parsed],
        "refinement_tolerance": REFINEMENT_TOLERANCE,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Spectral bias scan over a declared finite frequency grid, proposed from a "
            "prefix and confronted with a held-out suffix by exact cos-sign counts at "
            "the stated quantized lambda. The fp64 grid sweep is a pre-filter only; "
            "every reported number is recomputed in mpmath at the declared precision. "
            "A peak is not a proof, survival is not truth, and no novelty is claimed."
        ),
        "sequence_label": sequence_label,
        "system_caps": SYSTEM_CAPS,
        "threshold_ladder": [
            {"numerator": entry.numerator, "denominator": entry.denominator}
            for entry in THRESHOLD_LADDER
        ],
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal check plus exact recheck of every claimed count at the stated lambdas.

    The fp64 grid sweep is device-dependent pre-filtering and is *not* replayed; what
    is rechecked is the claim surface: every prefix and holdout cos-sign count at the
    stated quantized lambda, every threshold comparison, and every statement string.
    """

    if value.get("schema_version") != RESULT_SCHEMA:
        raise SpectralScanError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SpectralScanError("result seal changed")
    parsed = _parse_rows(
        [{"point": row["point"], "value": row["value"]} for row in value["public_rows"]]
    )
    prefix, holdout = _split(parsed)
    if len(prefix) != value["counts"]["prefix_rows"]:
        raise SpectralScanError("prefix split changed")
    if len(holdout) != value["counts"]["holdout_rows"]:
        raise SpectralScanError("holdout split changed")
    # Mirror generation exactly: prefix counts were made inside the prefix-only
    # proposal (prefix-derived dps); holdout counts at the full-data dps.
    prefix_dps = _working_dps([item for _, item in prefix])
    dps = _working_dps([item for _, item in parsed])
    prefix_values = [Fraction(item) for _, item in prefix]
    holdout_values = [Fraction(item) for _, item in holdout]
    for peak in value["peaks"]:
        lam = _decimal_fraction(peak["lambda"])
        negative, ambiguous = _negative_counts(lam, prefix_values, prefix_dps)
        if (
            negative != peak["prefix_negative_terms"]
            or ambiguous != peak["prefix_ambiguous_terms"]
            or len(prefix_values) != peak["prefix_terms"]
        ):
            raise SpectralScanError("prefix cos-sign counts do not recheck")
        threshold = peak["threshold"]
        measured = Fraction(negative, len(prefix_values))
        expected = next(
            (entry for entry in THRESHOLD_LADDER if measured >= entry), None
        )
        if threshold is None:
            if expected is not None:
                raise SpectralScanError("threshold ladder selection does not recheck")
            if peak["status"] != "NO_BIAS_PROPOSED" or peak["holdout"] is not None:
                raise SpectralScanError("unproposed peak status does not recheck")
            continue
        if expected is None or Fraction(
            threshold["numerator"], threshold["denominator"]
        ) != expected:
            raise SpectralScanError("threshold ladder selection does not recheck")
        expected_statement = (
            f"cos({peak['lambda']} * a(n)) < 0 for at least "
            f"{threshold['numerator']}/{threshold['denominator']} of terms"
        )
        if peak["statement"] != expected_statement:
            raise SpectralScanError("statement string does not recheck")
        negative, ambiguous = _negative_counts(lam, holdout_values, dps)
        block = peak["holdout"]
        if (
            block["rows"] != len(holdout_values)
            or block["negative_terms"] != negative
            or block["ambiguous_terms"] != ambiguous
        ):
            raise SpectralScanError("holdout cos-sign counts do not recheck")
        ok = Fraction(negative, len(holdout_values)) >= expected
        if peak["status"] != ("SURVIVED" if ok else "REFUTED"):
            raise SpectralScanError("holdout verdict does not recheck")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise SpectralScanError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Spectral bias scan.")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--profile", default="dense", choices=sorted(PROFILES))
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--builtin-knowledge", default=None)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    result = scan_receipt(
        rows,
        args.label,
        args.profile,
        use_gpu=args.gpu,
        builtin_knowledge=args.builtin_knowledge,
    )
    if args.output:
        _write_immutable(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "SPECTRAL_BIAS_SURVIVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
