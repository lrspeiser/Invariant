"""Source-only receipt for the bounded Galileo E14 development test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_galileo_e14_source_receipt_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_galileo_e14_source_receipt_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_galileo_e14_source_receipt_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-galileo-e14-source-receipt-v1/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "9b879e3add962a4ec4ff6c0dea26ef295b646810b1d8a31a4c16df847289a220"
_CONFIG_CONTENT_SHA256 = "8df029f9bd23418175f3d51a4d8e6ea943dd9adde42086a89f08ec14ac101ff3"
_MODULE_SEMANTIC_SHA256 = "c412c0438095857c1a4368f12f3373a15d72fa3dfef2ff1900ff663b117fe32c"
_TEST_RAW_SHA256 = "1a36f6285c50acac65b574c57f40633432ec198497f7e0bf3cead7f9842188f1"
_SCHEMA = "invariant-open-gravity-galileo-e14-source-receipt-1.0"


class GalileoE14SourceError(RuntimeError):
    """Raised when a frozen source-only invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GalileoE14SourceError(message)


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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GalileoE14SourceError(f"invalid JSON: {path}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "FROZEN_SOURCE_ONLY_NO_CLOCK_VALUES_OPENED", "status changed")
    auth = config["authorization"]
    _require(auth["development_prn"] == "E14", "development PRN changed")
    _require(auth["holdout_prn"] == "E18", "holdout PRN changed")
    _require(auth["holdout_response_access"] == "FORBIDDEN", "holdout access widened")
    _require(auth["other_prn_response_access"] == "FORBIDDEN", "control access widened")
    _require(auth["clock_value_rows_opened_by_this_package"] == 0, "CLK values authorized")
    names = [row["name"] for row in config["files"]]
    expected = [f"esoc2044{day}.{suffix}" for day in range(7) for suffix in ("clk", "sp3")]
    _require(names == expected, "file order or names changed")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "receipt path changed")


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH)
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _epoch(parts: Sequence[str]) -> datetime:
    second = float(parts[5])
    whole = math.floor(second)
    microsecond = round((second - whole) * 1_000_000)
    return datetime(
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
        int(parts[4]),
        whole,
        microsecond,
        tzinfo=UTC,
    )


def read_clk_header(path: Path) -> dict[str, Any]:
    """Read exactly the RINEX header and stop before the first response record."""
    raw = bytearray()
    text_lines: list[str] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            raw.extend(raw_line)
            line = raw_line.decode("ascii")
            text_lines.append(line.rstrip("\r\n"))
            if line[60:80].strip() == "END OF HEADER":
                break
        else:
            raise GalileoE14SourceError(f"CLK END OF HEADER absent: {path.name}")
    labels = [line[60:80].strip() for line in text_lines if len(line) >= 60]
    prns: list[str] = []
    for line in text_lines:
        if len(line) >= 60 and line[60:80].strip() == "PRN LIST":
            prns.extend(line[:60].split())
    return {
        "header_lines": len(text_lines),
        "header_bytes": len(raw),
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "rinex_version": text_lines[0][:20].strip(),
        "end_of_header_count": labels.count("END OF HEADER"),
        "advertised_prns": prns,
    }


def read_sp3_positions(path: Path) -> dict[str, dict[datetime, tuple[float, float, float]]]:
    """Read source coordinates while deliberately ignoring the SP3 clock column."""
    positions: dict[str, dict[datetime, tuple[float, float, float]]] = defaultdict(dict)
    current: datetime | None = None
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("*"):
                current = _epoch(line[1:].split())
            elif line.startswith("PE") and current is not None:
                prn = line[1:4]
                fields = line[4:].split()
                _require(len(fields) >= 3, f"short SP3 P record: {path.name}")
                positions[prn][current] = tuple(float(value) for value in fields[:3])
    _require("E14" in positions and "E18" in positions, f"eccentric PRNs absent: {path.name}")
    return dict(positions)


def _merge_sp3(
    daily: Sequence[dict[str, dict[datetime, tuple[float, float, float]]]],
) -> dict[str, dict[datetime, tuple[float, float, float]]]:
    merged: dict[str, dict[datetime, tuple[float, float, float]]] = defaultdict(dict)
    for day in daily:
        for prn, rows in day.items():
            for epoch, xyz in rows.items():
                if epoch not in merged[prn]:
                    merged[prn][epoch] = xyz
    return dict(merged)


def radial_fraction(rows: Mapping[datetime, Sequence[float]]) -> float:
    radii = np.asarray([np.linalg.norm(rows[epoch]) for epoch in sorted(rows)], dtype=float)
    return float(np.std(radii) / np.mean(radii))


def select_circular_controls(
    daily: Sequence[dict[str, dict[datetime, tuple[float, float, float]]]],
) -> list[dict[str, Any]]:
    complete = set(daily[0])
    for day in daily:
        epoch_count = len(day["E14"])
        complete &= {prn for prn, rows in day.items() if len(rows) == epoch_count}
    complete -= {"E14", "E18"}
    merged = _merge_sp3(daily)
    ranked = sorted((radial_fraction(merged[prn]), prn) for prn in complete)
    _require(len(ranked) >= 2, "fewer than two circular controls")
    return [
        {"prn": prn, "radial_std_over_mean": value, "sp3_epochs": len(merged[prn])}
        for value, prn in ranked[:2]
    ]


def source_period_and_shifts(
    rows: Mapping[datetime, Sequence[float]],
) -> tuple[int, float, list[int]]:
    radii = np.asarray([np.linalg.norm(rows[epoch]) for epoch in sorted(rows)], dtype=float)
    centered = radii - np.mean(radii)
    denominator = float(np.dot(centered, centered))
    _require(denominator > 0.0, "constant E14 radius")
    scored: list[tuple[float, int]] = []
    for lag in range(100, 221):
        left = centered[:-lag]
        right = centered[lag:]
        pair_norm = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        _require(pair_norm > 0.0, "zero pair norm in period search")
        correlation = float(np.dot(left, right) / pair_norm)
        scored.append((correlation, lag))
    correlation, period = max(scored, key=lambda row: (row[0], -row[1]))
    positive = [round(period * fraction) for fraction in (0.125, 0.25, 0.375, 0.5)]
    shifts = sorted(set(positive + [-value for value in positive]))
    return period, correlation, shifts


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = Path(config["predecessor"]["receipt_path"])
    _require(predecessor.is_file(), "predecessor receipt missing")
    _require(
        file_sha256(predecessor) == config["predecessor"]["receipt_sha256"],
        "predecessor receipt changed",
    )
    source = Path(config["source_directory"])
    headers: list[dict[str, Any]] = []
    daily_positions: list[dict[str, dict[datetime, tuple[float, float, float]]]] = []
    observed_files: list[dict[str, Any]] = []
    for frozen in config["files"]:
        path = source / frozen["name"]
        _require(path.is_file(), f"missing source file: {path}")
        _require(path.stat().st_size == frozen["bytes"], f"size changed: {path.name}")
        digest = file_sha256(path)
        _require(digest == frozen["sha256"], f"hash changed: {path.name}")
        observed_files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
        if path.suffix == ".clk":
            header = read_clk_header(path)
            _require(header["rinex_version"] == "2.00", f"CLK version changed: {path.name}")
            _require(header["end_of_header_count"] == 1, f"invalid CLK header: {path.name}")
            _require("E14" in header["advertised_prns"], f"E14 absent from header: {path.name}")
            headers.append({"name": path.name, **header})
        else:
            daily_positions.append(read_sp3_positions(path))
    _require(len(headers) == len(daily_positions) == 7, "daily source count changed")
    controls = select_circular_controls(daily_positions)
    merged = _merge_sp3(daily_positions)
    period, period_correlation, phase_shifts = source_period_and_shifts(merged["E14"])
    per_day = []
    for day, rows in enumerate(daily_positions):
        epochs = sorted(rows["E14"])
        per_day.append(
            {
                "gps_week_day": day,
                "sp3_epoch_count": len(epochs),
                "first_epoch": epochs[0].isoformat(),
                "last_epoch": epochs[-1].isoformat(),
                "candidate_after_edge_exclusion": len(epochs) - 2,
            }
        )
    artifact = {
        "schema": "invariant-open-gravity-galileo-e14-source-manifest-1.0",
        "clock_value_rows_opened": 0,
        "clk_headers": headers,
        "e14_sp3_unique_epochs": len(merged["E14"]),
        "daily_source_windows": per_day,
        "circular_controls_source_only": controls,
        "e14_source_period_samples": period,
        "e14_source_period_seconds": period * 300,
        "period_autocorrelation": period_correlation,
        "phase_scramble_shifts_samples": phase_shifts,
    }
    artifact_path = ARTIFACT_DIR / "source-only-manifest.json"
    _write_json(artifact_path, artifact)
    receipt = {
        "schema": "invariant-open-gravity-galileo-e14-source-receipt-1.0",
        "package_id": config["package_id"],
        "status": "PASS_SOURCE_ELIGIBLE_RESPONSE_PARSER_NOT_YET_RUN",
        "decision": "E14_DEVELOPMENT_MAY_OPEN_ONLY_AFTER_RESPONSE_SUCCESSOR_IS_HASH_FROZEN",
        "input_sha256": {"predecessor": config["predecessor"]["receipt_sha256"]},
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "source_bundle_sha256": content_sha256(observed_files),
        "observed_files": observed_files,
        "summary": artifact,
        "artifact": {
            "path": artifact_path.as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": file_sha256(artifact_path),
        },
        "access_accounting": {
            "payload_files_downloaded": 14,
            "clk_header_lines_opened": sum(row["header_lines"] for row in headers),
            "clk_value_rows_opened": 0,
            "sp3_position_rows_opened": sum(
                len(rows) for day in daily_positions for rows in day.values()
            ),
            "E18_clock_value_rows_opened": 0,
            "other_prn_clock_value_rows_opened": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "formula_retuning_events": 0,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def build(output: Path = OUTPUT_PATH) -> dict[str, Any]:
    config = load_config()
    receipt = build_receipt(config)
    _write_json(output, receipt)
    return receipt


def check(output: Path = OUTPUT_PATH) -> dict[str, Any]:
    _require(output.is_file(), "receipt missing")
    expected = build_receipt(load_config())
    observed = _read_json(output)
    _require(observed == expected, "receipt differs from deterministic rebuild")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    receipt = build(args.output) if args.command == "build" else check(args.output)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
