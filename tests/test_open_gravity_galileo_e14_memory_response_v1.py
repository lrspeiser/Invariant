from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_galileo_e14_memory_response_v1 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_package_pins() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authorization", "development_response_prns"), ["E14", "E18"]),
        (("authorization", "other_response_prns"), "ALLOWED"),
        (("source", "source_only_period_samples"), 154),
        (("physical_template", "tau_seconds"), [1.0]),
        (("development_decision", "E18_promotion_gate"), "OPEN"),
    ],
)
def test_frozen_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(subject.GalileoE14ResponseError):
        subject.validate_config(config)


def test_clk_parser_decodes_only_e14_numeric_payload(tmp_path) -> None:
    path = tmp_path / "tiny.clk"
    path.write_text(
        f"{'2.00':<60}RINEX VERSION / TYPE\n"
        f"{'':<60}END OF HEADER\n"
        "AS E18 THIS PAYLOAD MUST NEVER BE NUMERICALLY DECODED\n"
        "AS E14 2019 03 10 00 00 00.000000 1 1.250000000000E-03\n"
        "AS E25 THIS PAYLOAD MUST NEVER BE NUMERICALLY DECODED\n",
        encoding="ascii",
    )
    values, accounting = subject.read_e14_clk_values(path)
    assert list(values.values()) == [0.00125]
    assert accounting["e14_numeric_rows_decoded"] == 1
    assert accounting["non_e14_numeric_payload_rows_decoded"] == 0
    assert accounting["non_e14_payload_rows_prefix_skipped"] == 2


def test_periodic_memory_initial_state_closes_first_cycle() -> None:
    period = 12
    source = np.asarray([0.5 + 0.3 * math.sin(2 * math.pi * index / period) for index in range(40)])
    psi = subject.periodic_memory(source, 1800.0, period)
    a = math.exp(-300.0 / 1800.0)
    after_cycle = psi[0]
    for value in source[:period]:
        after_cycle = a * after_cycle + (1.0 - a) * value
    assert after_cycle == pytest.approx(psi[0], abs=1e-14)
    assert psi[1] == pytest.approx(a * psi[0] + (1 - a) * source[0])


def test_day_projection_removes_intercept_and_linear_trend() -> None:
    time = np.linspace(-1.0, 1.0, 101)
    residual = subject.project_day(4.0 + 2.0 * time + np.sin(3.0 * time))
    assert abs(float(np.mean(residual))) < 1e-14
    assert abs(float(np.dot(residual, time))) < 1e-12


def test_ar1_transform_has_no_cross_day_connection() -> None:
    first = subject.ar_transform(np.asarray([1.0, 2.0]), 0.5)
    second = subject.ar_transform(np.asarray([10.0, 11.0]), 0.5)
    assert first[1] == pytest.approx(1.5)
    assert second[0] == pytest.approx(math.sqrt(0.75) * 10.0)
    assert second[0] != pytest.approx(9.0)


def test_leave_one_day_out_recovers_frozen_synthetic_signal() -> None:
    y_days = []
    x_days = []
    for day in range(7):
        time = np.arange(160, dtype=float)
        x = np.sin(2 * math.pi * (time + 3 * day) / 47.0)
        noise = 0.03 * np.cos(2 * math.pi * (time + 5 * day) / 19.0)
        y = 2.5 * x + noise
        y_days.append(subject.project_day(y))
        x_days.append(subject.project_day(x)[:, None])
    result = subject.cross_validate(y_days, x_days)
    assert result["total_delta_log_likelihood"] > 100.0
    assert all(row["beta"][0] > 0.0 for row in result["folds"])
