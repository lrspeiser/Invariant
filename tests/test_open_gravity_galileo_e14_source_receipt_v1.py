from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from sigma_theory_compiler import open_gravity_galileo_e14_source_receipt_v1 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_package_pins() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


def test_authorization_cannot_widen() -> None:
    config = _raw_config()
    config["authorization"]["holdout_response_access"] = "ALLOWED"  # type: ignore[index]
    with pytest.raises(subject.GalileoE14SourceError):
        subject.validate_config(config)


def test_clk_reader_stops_at_header_boundary(tmp_path) -> None:
    path = tmp_path / "tiny.clk"
    lines = [
        f"{'2.00':<60}RINEX VERSION / TYPE\n",
        f"{'E14 E18':<60}PRN LIST\n",
        f"{'':<60}END OF HEADER\n",
        "AS E14 2019 03 10 00 00 00.000000 1 9.999999999999E+99\n",
    ]
    path.write_text("".join(lines), encoding="ascii")
    result = subject.read_clk_header(path)
    assert result["header_lines"] == 3
    assert result["advertised_prns"] == ["E14", "E18"]
    assert result["header_bytes"] < path.stat().st_size


def _synthetic_day(day: int) -> dict[str, dict[datetime, tuple[float, float, float]]]:
    start = datetime(2019, 3, 10, tzinfo=UTC) + timedelta(days=day)
    epochs = [start + timedelta(seconds=300 * index) for index in range(8)]
    return {
        "E14": {epoch: (10.0 + index, 0.0, 0.0) for index, epoch in enumerate(epochs)},
        "E18": {epoch: (11.0 + index, 0.0, 0.0) for index, epoch in enumerate(epochs)},
        "E01": {epoch: (20.0 + 0.01 * index, 0.0, 0.0) for index, epoch in enumerate(epochs)},
        "E02": {epoch: (20.0 + 0.02 * index, 0.0, 0.0) for index, epoch in enumerate(epochs)},
        "E03": {epoch: (20.0 + 0.03 * index, 0.0, 0.0) for index, epoch in enumerate(epochs)},
    }


def test_control_selection_uses_only_complete_sp3_geometry() -> None:
    controls = subject.select_circular_controls([_synthetic_day(day) for day in range(7)])
    assert [row["prn"] for row in controls] == ["E01", "E02"]
    assert all(row["sp3_epochs"] == 56 for row in controls)


def test_period_and_phase_shifts_are_source_only_and_deterministic() -> None:
    start = datetime(2019, 3, 10, tzinfo=UTC)
    rows = {
        start + timedelta(seconds=300 * index): (20.0 + 2.0 * __import__("math").cos(2 * __import__("math").pi * index / 150), 0.0, 0.0)
        for index in range(1000)
    }
    period, correlation, shifts = subject.source_period_and_shifts(rows)
    assert period == 150
    assert correlation > 0.8
    assert shifts == [-75, -56, -38, -19, 19, 38, 56, 75]
