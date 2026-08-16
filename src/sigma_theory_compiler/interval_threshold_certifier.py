"""Outward-rounded interval certification of the billion-scale screen's verdicts.

The GPU screen decides pass/fail by comparing fp64 numbers against thresholds.  A
decision whose margin is smaller than the accumulated floating-point error is not a
decision — it is luck with a hash on it.  This module removes that failure mode for
every reported Pareto candidate: each screening criterion is re-evaluated with
``mpmath.iv`` interval arithmetic at ``iv.dps = 60`` (outward rounding on every
operation), and the criterion is classified three ways instead of two:

* ``certified_pass``   — the whole interval sits strictly inside the pass region;
* ``certified_fail``   — the whole interval sits strictly outside it;
* ``unresolved_straddle`` — the interval contains the threshold, so no finite-precision
  evaluation at this working precision is entitled to a verdict either way.

A candidate is ``certified`` only if EVERY criterion is ``certified_pass``.  The
criteria mirror :func:`~.gpu_baryonic_interpolation_screen.verify_candidate_exact`
exactly — definedness/positivity at the Newton probes, Newtonian recovery near and
far, monotone g_obs over the full acceleration grid, per-disk outer-curve flatness,
and the baryonic Tully-Fisher slope — with the same fp64 thresholds parsed from
``SCREEN_CONFIG`` as exact rationals and enclosed outward into intervals.

Disk-grid path (stated in the receipt as ``disk_grid_path``): mpmath 1.3's ``iv``
context cannot evaluate Bessel functions (``iv.besseli`` raises at call time), so the
candidate-independent Freeman g_bar values are computed with the screen's own
:func:`~.gpu_baryonic_interpolation_screen._disk_gbar` at ``mp.dps = 80`` and widened
outward by a relative pad of ``2**(3 - prec)`` at the certification precision (four
ulp at dps 60, ~1e18 times the dps-80 evaluation error), so the true Freeman value is
strictly inside every probe interval.  Every subsequent operation runs in ``iv``
arithmetic and stays outward.

Claim boundary: certification proves only that the screen's threshold comparisons are
decided by real margins rather than fp64 rounding.  It does not validate any physics,
does not open observational data, and does not shortcut the sealed validation ladder.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
from mpmath import iv
from mpmath.libmp import to_str

from .gpu_baryonic_interpolation_screen import (
    RESULT_SCHEMA as SCREEN_RESULT_SCHEMA,
)
from .gpu_baryonic_interpolation_screen import (
    SCREEN_CONFIG,
    _disk_gbar,
    decode_ordinal,
    encode_candidate,
    render_candidate,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-interval-threshold-certifier-result-1.0"

#: Working precision of every interval operation (and of the outward pad below).
IV_DPS = 60
#: Precision of the scalar mpmath pass that seeds the disk-grid intervals.
GRID_MP_DPS = 80
#: Stated in the receipt: which of the two mandated grid paths this build uses.
DISK_GRID_PATH = "mp-dps80-freeman-bessel-widened-outward-4ulp-at-dps60"

CERTIFIED_PASS = "certified_pass"
CERTIFIED_FAIL = "certified_fail"
UNRESOLVED = "unresolved_straddle"

MARGIN_DIGITS = 12

CLAIMS = {
    "certification_is_not_physical_validation": True,
    "corpus_absence_establishes_novelty": False,
    "fp64_luck_can_decide_a_verdict": False,
    "interval_rounding_is_outward": True,
}

SCOPE = (
    "Outward-rounded interval re-evaluation (mpmath.iv, dps 60) of every screening "
    "criterion for every Pareto candidate reported by the sealed billion-scale "
    "baryonic screen receipt. Each criterion is classified certified_pass, "
    "certified_fail, or unresolved_straddle; a candidate is certified only if every "
    "criterion is certified_pass, so no verdict in this receipt can be an artifact "
    "of fp64 rounding luck. Thresholds are the screen's fp64 thresholds parsed as "
    "exact rationals; disk probes are Freeman-disk values recomputed on the stated "
    "disk_grid_path. min_margin fields are decimal renderings of rigorous "
    "worst-case signed margins (positive means safely certified); certification "
    "decisions never consume the rendered strings. Certification is arithmetic "
    "hygiene only: it validates no physics, opens no observational data, and "
    "bypasses no sealed validation ladder."
)


class IntervalCertifierError(ValueError):
    """Raised on malformed input, seal mismatch, binding drift, or failed replay."""


# ---------------------------------------------------------------------------
# Exact-rational parsing and interval helpers
# ---------------------------------------------------------------------------


def _iv_rational(text: str | int) -> Any:
    """Parse a decimal/fraction string (or int) into a tight outward interval."""

    fraction = Fraction(str(text))
    return iv.mpf(fraction.numerator) / iv.mpf(fraction.denominator)


def _decimal(value: Any) -> str:
    """Render one interval endpoint (a degenerate ivmpf) as a decimal string."""

    lower, upper = value._mpi_
    if lower != upper:
        raise IntervalCertifierError("margin endpoint is not degenerate")
    return to_str(lower, MARGIN_DIGITS)


def _combine(verdicts: Sequence[str]) -> str:
    """All-of combination: any certain failure fails; all certain passes pass."""

    if any(verdict == CERTIFIED_FAIL for verdict in verdicts):
        return CERTIFIED_FAIL
    if all(verdict == CERTIFIED_PASS for verdict in verdicts):
        return CERTIFIED_PASS
    return UNRESOLVED


def _le_verdict(measured: Any, threshold: Any) -> tuple[str, Any]:
    """Certify ``true(measured) <= true(threshold)``; margin is (threshold-measured).a."""

    margin = (threshold - measured).a
    if measured.b <= threshold.a:
        return CERTIFIED_PASS, margin
    if measured.a > threshold.b:
        return CERTIFIED_FAIL, margin
    return UNRESOLVED, margin


def _gt_verdict(left: Any, right: Any) -> tuple[str, Any]:
    """Certify ``true(left) > true(right)``; margin is (left-right).a."""

    margin = (left - right).a
    if left.a > right.b:
        return CERTIFIED_PASS, margin
    if left.b <= right.a:
        return CERTIFIED_FAIL, margin
    return UNRESOLVED, margin


def _interval_max(values: Sequence[Any]) -> Any:
    lower = max(value.a for value in values)
    upper = max(value.b for value in values)
    return iv.mpf([lower, upper])


def _interval_min(values: Sequence[Any]) -> Any:
    lower = min(value.a for value in values)
    upper = min(value.b for value in values)
    return iv.mpf([lower, upper])


# ---------------------------------------------------------------------------
# Candidate-independent probe grid, recomputed as outward intervals
# ---------------------------------------------------------------------------


def build_interval_grid() -> dict[str, Any]:
    """Rebuild every candidate-independent probe as a rigorous outward interval.

    Probes are shared: the sorted monotone list references the same probe entries as
    the disk points and Newton probes, so each candidate evaluates nu once per probe.
    """

    iv.dps = IV_DPS
    pad = iv.mpf(2) ** (3 - iv.prec)
    wobble = iv.mpf(1) + iv.mpf([-1, 1]) * pad

    probes: list[dict[str, Any]] = []

    def add_probe(y_interval: Any, sort_key: Any) -> int:
        u = 1 / iv.sqrt(y_interval)
        upowers = []
        power = iv.mpf(1)
        for _ in range(5):
            power = power * u
            upowers.append(power)
        probes.append({"y": y_interval, "upowers": tuple(upowers), "sort_key": sort_key})
        return len(probes) - 1

    with mp.workdps(GRID_MP_DPS):
        scale = mp.mpf(SCREEN_CONFIG["disk_scale_length"])
        disks: list[dict[str, Any]] = []
        for mass_text in SCREEN_CONFIG["disk_masses"]:
            mass_fraction = Fraction(mass_text)
            mass = mp.mpf(mass_fraction.numerator) / mp.mpf(mass_fraction.denominator)
            points = []
            for radius in [*SCREEN_CONFIG["inner_radii"], *SCREEN_CONFIG["outer_radii"]]:
                gbar_scalar = _disk_gbar(mass, mp.mpf(radius), scale)
                gbar_interval = iv.mpf(gbar_scalar) * wobble
                points.append(
                    {
                        "radius": radius,
                        "outer": radius in SCREEN_CONFIG["outer_radii"],
                        "probe": add_probe(gbar_interval, gbar_scalar),
                    }
                )
            disks.append({"mass_text": mass_text, "points": points})

    newton_indices = tuple(
        add_probe(iv.mpf(int(value)), int(value)) for value in SCREEN_CONFIG["newton_probe_y"]
    )

    monotone = sorted(range(len(probes)), key=lambda index: probes[index]["sort_key"])
    for previous, current in itertools.pairwise(monotone):
        if not probes[previous]["y"].b < probes[current]["y"].a:
            raise IntervalCertifierError("monotone probe intervals overlap; grid is ambiguous")

    masses = SCREEN_CONFIG["disk_masses"]
    span_fraction = Fraction(masses[-1]) / Fraction(masses[0])
    log_mass_span = iv.log(iv.mpf(span_fraction.numerator) / iv.mpf(span_fraction.denominator))

    return {
        "probes": probes,
        "disks": disks,
        "newton": newton_indices,
        "monotone": monotone,
        "log_mass_span": log_mass_span,
    }


# ---------------------------------------------------------------------------
# Interval evaluation of one candidate
# ---------------------------------------------------------------------------


def _evaluate_probes(
    candidate: Mapping[str, Any], grid: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate nu as an outward interval at every probe.

    Each result is ``{"verdict": .., "nu": interval-or-None, "margin": endpoint-or-None}``
    where ``verdict`` classifies definedness/positivity (denominator excludes zero and
    P/Q is strictly positive) and ``margin`` is the guaranteed positivity slack.
    """

    beta = candidate["beta"]
    zero = iv.mpf(0).a
    results = []
    for probe in grid["probes"]:
        numerator = iv.mpf(1)
        denominator = iv.mpf(1)
        for coefficient, power in zip(candidate["a"], probe["upowers"], strict=True):
            if coefficient:
                numerator = numerator + coefficient * power
        for coefficient, power in zip(candidate["b"], probe["upowers"], strict=True):
            if coefficient:
                denominator = denominator + coefficient * power
        if denominator.a <= zero <= denominator.b:
            if denominator.a == zero and denominator.b == zero:
                results.append({"verdict": CERTIFIED_FAIL, "nu": None, "margin": None})
            else:
                results.append({"verdict": UNRESOLVED, "nu": None, "margin": None})
            continue
        ratio = numerator / denominator
        margin = ratio.a
        if ratio.b <= zero:
            results.append({"verdict": CERTIFIED_FAIL, "nu": None, "margin": margin})
            continue
        if not ratio.a > zero:
            results.append({"verdict": UNRESOLVED, "nu": None, "margin": margin})
            continue
        if beta == "1":
            nu = ratio
        elif beta == "2":
            nu = ratio * ratio
        elif beta == "1/2":
            nu = iv.sqrt(ratio)
        elif beta == "1/3":
            nu = ratio ** (iv.mpf(1) / 3)
        else:  # pragma: no cover - the codec cannot produce other betas
            raise IntervalCertifierError(f"unknown beta {beta!r}")
        results.append({"verdict": CERTIFIED_PASS, "nu": nu, "margin": margin})
    return results


def certify_candidate(
    ordinal: int,
    grid: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify every screening criterion for one candidate with outward intervals.

    ``thresholds`` may override the fp64 thresholds (decimal strings) so control
    tests can construct deliberate straddles without touching any receipt.
    """

    iv.dps = IV_DPS
    if grid is None:
        grid = build_interval_grid()
    threshold_text = dict(SCREEN_CONFIG["fp64_thresholds"])
    if thresholds is not None:
        threshold_text.update(thresholds)
    bounds = {key: _iv_rational(value) for key, value in threshold_text.items()}

    candidate = decode_ordinal(int(ordinal))
    evaluated = _evaluate_probes(candidate, grid)
    one = iv.mpf(1)

    criteria: dict[str, str] = {}
    margins: list[Any] = []

    def record(name: str, verdicts: Sequence[str], local_margins: Sequence[Any]) -> None:
        criteria[name] = _combine(verdicts)
        margins.extend(margin for margin in local_margins if margin is not None)

    def bounded_check(probe_index: int, threshold_key: str) -> tuple[list[str], list[Any]]:
        """Definedness of one probe plus ``|nu - 1| <= threshold`` when nu exists."""

        result = evaluated[probe_index]
        verdicts = [result["verdict"]]
        local = [result["margin"]]
        if result["nu"] is not None:
            verdict, margin = _le_verdict(abs(result["nu"] - one), bounds[threshold_key])
            verdicts.append(verdict)
            local.append(margin)
        return verdicts, local

    near_index, far_index = grid["newton"]

    # Definedness/positivity mirrors verify_candidate_exact's "defined" key: the two
    # Newton probes.  Positivity at grid probes propagates into the criteria below.
    record(
        "defined",
        [evaluated[near_index]["verdict"], evaluated[far_index]["verdict"]],
        [evaluated[near_index]["margin"], evaluated[far_index]["margin"]],
    )
    record("newton_near", *bounded_check(near_index, "newton_near"))
    record("newton_far", *bounded_check(far_index, "newton_far"))

    # Monotone g_obs across the full sorted acceleration grid.
    verdicts = []
    local_margins = []
    previous = None
    for probe_index in grid["monotone"]:
        result = evaluated[probe_index]
        verdicts.append(result["verdict"])
        local_margins.append(result["margin"])
        if result["nu"] is None:
            previous = None
            continue
        gobs = grid["probes"][probe_index]["y"] * result["nu"]
        if previous is not None:
            verdict, margin = _gt_verdict(gobs, previous)
            verdicts.append(verdict)
            local_margins.append(margin)
        previous = gobs
    record("monotone", verdicts, local_margins)

    # Per-disk outer-curve flatness, then the cross-disk Tully-Fisher slope.
    vflat: list[Any] = []
    for disk_number, disk in enumerate(grid["disks"]):
        verdicts = []
        local_margins = []
        speeds = []
        for point in disk["points"]:
            if not point["outer"]:
                continue
            result = evaluated[point["probe"]]
            verdicts.append(result["verdict"])
            local_margins.append(result["margin"])
            if result["nu"] is not None:
                vsquared = grid["probes"][point["probe"]]["y"] * result["nu"] * point["radius"]
                speeds.append(iv.sqrt(vsquared))
        if len(speeds) == sum(1 for point in disk["points"] if point["outer"]):
            spread = _interval_max(speeds) - _interval_min(speeds)
            mean = sum(speeds[1:], speeds[0]) / len(speeds)
            verdict, margin = _le_verdict(spread, bounds["flatness"] * mean)
            verdicts.append(verdict)
            local_margins.append(margin)
            vflat.append(mean)
        else:
            vflat.append(None)
        record(f"flat_disk_{disk_number}", verdicts, local_margins)

    verdicts = [criteria[f"flat_disk_{number}"] for number in range(len(grid["disks"]))]
    local_margins = []
    if vflat[0] is not None and vflat[-1] is not None:
        slope = grid["log_mass_span"] / iv.log(vflat[-1] / vflat[0])
        verdict, margin = _le_verdict(abs(slope - 4), bounds["btfr_slope"])
        verdicts = [*verdicts, verdict]
        local_margins.append(margin)
    record("btfr_slope", verdicts, local_margins)

    certified = all(verdict == CERTIFIED_PASS for verdict in criteria.values())
    minimum = margins[0]
    for margin in margins[1:]:
        minimum = min(minimum, margin)
    return {
        "ordinal": int(ordinal),
        "formula": render_candidate(candidate),
        "criteria": criteria,
        "certified": certified,
        "min_margin": _decimal(minimum),
    }


def certify_coefficients(
    beta_index: int,
    a: Sequence[int],
    b: Sequence[int],
    grid: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Certify a crafted coefficient tuple directly; used by control tests."""

    return certify_candidate(encode_candidate(beta_index, a, b), grid=grid, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Receipt-level certification
# ---------------------------------------------------------------------------


def _check_source_receipt(receipt: Mapping[str, Any]) -> None:
    """Fail closed unless the screen receipt is sealed and on the frozen config."""

    if receipt.get("schema_version") != SCREEN_RESULT_SCHEMA:
        raise IntervalCertifierError("source receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise IntervalCertifierError("source receipt seal changed")
    if receipt.get("config_sha256") != canonical_sha256(receipt.get("config", {})):
        raise IntervalCertifierError("source receipt config binding changed")
    if receipt.get("config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise IntervalCertifierError("source receipt config differs from the frozen SCREEN_CONFIG")
    front = receipt.get("pareto_front")
    if not isinstance(front, list) or not front:
        raise IntervalCertifierError("source receipt has no pareto_front to certify")


def _certify_rows(front: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    grid = build_interval_grid()
    rows = []
    for entry in front:
        ordinal = entry.get("ordinal")
        if not isinstance(ordinal, int):
            raise IntervalCertifierError("pareto_front ordinal is not an integer")
        row = certify_candidate(ordinal, grid=grid)
        if row["formula"] != entry.get("formula"):
            raise IntervalCertifierError(f"formula binding drifted for ordinal {ordinal}")
        rows.append(row)
    # Reporting only: pick the smallest rendered margin at a fixed parse precision so
    # replay is deterministic.  Certification decisions never consume these strings.
    with mp.workdps(40):
        overall = min((row["min_margin"] for row in rows), key=mp.mpf)
    return rows, overall


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    certified = sum(1 for row in rows if row["certified"])
    failed = sum(
        1
        for row in rows
        if not row["certified"]
        and any(verdict == CERTIFIED_FAIL for verdict in row["criteria"].values())
    )
    return {
        "candidates": len(rows),
        "certified": certified,
        "uncertified": failed,
        "unresolved": len(rows) - certified - failed,
    }


def certify_receipt(receipt_path: str | Path) -> dict[str, Any]:
    """Certify every Pareto candidate of a sealed screen receipt; seal the result."""

    raw = Path(receipt_path).read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    _check_source_receipt(receipt)
    rows, min_margin = _certify_rows(receipt["pareto_front"])
    body: dict[str, Any] = {
        "candidates": rows,
        "claims": CLAIMS,
        "config_sha256": canonical_sha256(SCREEN_CONFIG),
        "counts": _counts(rows),
        "disk_grid_path": DISK_GRID_PATH,
        "grid_mp_dps": GRID_MP_DPS,
        "iv_dps": IV_DPS,
        "min_margin": min_margin,
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "source_content_sha256": receipt["content_sha256"],
        "source_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "source_schema_version": SCREEN_RESULT_SCHEMA,
        "thresholds": dict(SCREEN_CONFIG["fp64_thresholds"]),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(
    value: Mapping[str, Any], *, source_receipt_path: str | Path | None = None
) -> None:
    """Seal check plus exact replay of every certification row.  Fails closed."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise IntervalCertifierError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise IntervalCertifierError("result seal changed")
    if value.get("claims") != CLAIMS:
        raise IntervalCertifierError("claims changed")
    if value.get("config_sha256") != canonical_sha256(SCREEN_CONFIG):
        raise IntervalCertifierError("config binding changed")
    if value.get("disk_grid_path") != DISK_GRID_PATH:
        raise IntervalCertifierError("disk grid path changed")
    if value.get("iv_dps") != IV_DPS or value.get("grid_mp_dps") != GRID_MP_DPS:
        raise IntervalCertifierError("precision declaration changed")
    rows = value.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise IntervalCertifierError("result has no candidate rows")
    if value.get("counts") != _counts(rows):
        raise IntervalCertifierError("counts do not match candidate rows")
    replay_rows, replay_margin = _certify_rows(rows)
    if replay_rows != rows or replay_margin != value.get("min_margin"):
        raise IntervalCertifierError("exact replay diverged from the sealed rows")
    if source_receipt_path is not None:
        digest = hashlib.sha256(Path(source_receipt_path).read_bytes()).hexdigest()
        if digest != value.get("source_receipt_sha256"):
            raise IntervalCertifierError("source receipt bytes do not match the sealed binding")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Outward-rounded interval certification of screen threshold verdicts."
    )
    parser.add_argument("--receipt", help="sealed screen receipt (billion-v1.json)")
    parser.add_argument("--output", help="certification receipt path")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        if not args.output:
            raise IntervalCertifierError("--validate-checked requires --output")
        value = json.loads(Path(args.output).read_text(encoding="utf-8"))
        validate_result(value, source_receipt_path=args.receipt)
        print(json.dumps({"validated": True, "counts": value["counts"]}, indent=2))
        return 0
    if not args.receipt:
        raise IntervalCertifierError("--receipt is required")
    result = certify_receipt(args.receipt)
    if args.output:
        path = Path(args.output)
        encoded = canonical_json_bytes(result) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise IntervalCertifierError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "min_margin": result["min_margin"],
                "disk_grid_path": result["disk_grid_path"],
                "source_receipt_sha256": result["source_receipt_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
