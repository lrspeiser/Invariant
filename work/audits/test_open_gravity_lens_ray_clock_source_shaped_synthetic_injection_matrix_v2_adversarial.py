"""Independent adversarial audit for the frozen lens/ray/clock synthetic v2 package."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v2 as lens,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    ReplayEntry,
    SyntheticReplayLedger,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import array_sha256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / lens.OUTPUT_DIR


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (OUTPUT / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _ledger() -> SyntheticReplayLedger:
    payload = json.loads((OUTPUT / "ledger.json").read_text(encoding="utf-8"))
    entries = tuple(
        ReplayEntry(
            **{
                **entry,
                "status": DiscoveryStatus(entry["status"]),
                "reason_codes": tuple(entry["reason_codes"]),
                "observable_ids": tuple(entry["observable_ids"]),
            }
        )
        for entry in payload["entries"]
    )
    return SyntheticReplayLedger(payload["ledger_id"], entries, payload["schema_version"])


def _independent_prediction(
    features: dict[str, np.ndarray], formula_id: str, constants: dict[str, float]
) -> dict[str, np.ndarray]:
    phi = np.asarray(features["source.vector.phi-stars"], dtype=np.float64) + np.asarray(
        features["source.vector.phi-nfw"], dtype=np.float64
    )
    exposure = np.asarray(features["source.vector.path-exposure"], dtype=np.float64)
    endpoint = np.asarray(features["source.vector.endpoint-log-redshift"], dtype=np.float64)
    gr_delay = np.asarray(features["source.scalar.gr-time-delay-days"], dtype=np.float64)
    geometric_delay = np.asarray(
        features["source.scalar.geometric-time-delay-days"], dtype=np.float64
    )
    light = 2.0 * phi
    delay = gr_delay.copy()
    output_exposure = exposure.copy()
    delta = 0.0
    if formula_id == "ENDPOINT_LAPSE_CONTROL":
        endpoint = endpoint + np.asarray([float(np.mean(phi)), 0.0], dtype=np.float64)
    elif formula_id == "GEOMETRIC_TIME_DELAY_CONTROL":
        delay = geometric_delay.copy()
    elif formula_id == "PATH_AGED_WEYL_CLOCK":
        delta = constants["path_alpha"] * float(exposure[0] - exposure[1])
    elif formula_id == "PHI_PSI_SLIP_SAME_STATE":
        gamma = constants["same_state_slip_gamma"]
        scale = (1.0 + gamma) / 2.0
        light = (1.0 + gamma) * phi
        delay = gr_delay * scale
        output_exposure = exposure * math.sqrt(scale)
        delta = (
            constants["slip_path_fraction"]
            * (gamma - 1.0)
            * float(exposure[0] - exposure[1])
        )
    else:
        assert formula_id == "GR_STARS_NFW_CONTROL"
    return {
        "prediction.scalar.differential-log-redshift": np.asarray([delta], dtype=np.float64),
        "prediction.scalar.time-delay-days": np.asarray(delay, dtype=np.float64),
        "prediction.vector.endpoint-log-redshift": np.asarray(endpoint, dtype=np.float64),
        "prediction.vector.light-potential": np.asarray(light, dtype=np.float64),
        "prediction.vector.matter-potential": np.asarray(phi, dtype=np.float64),
        "prediction.vector.path-exposure": np.asarray(output_exposure, dtype=np.float64),
    }


def _independent_profile(
    candidate: dict[str, np.ndarray],
    response: dict[str, np.ndarray],
    variance: dict[str, np.ndarray],
) -> dict[str, float]:
    lens_sensitive = tuple(
        output for output in lens._OUTPUTS if "endpoint-log-redshift" not in output
    )
    x = np.concatenate([np.asarray(candidate[key]).reshape(-1) for key in lens_sensitive])
    y = np.concatenate([np.asarray(response[key]).reshape(-1) for key in lens_sensitive])
    var = np.concatenate([np.asarray(variance[key]).reshape(-1) for key in lens_sensitive])
    weight = 1.0 / var
    scale = float(np.dot(weight * x, y) / np.dot(weight * x, x))
    endpoint_key = "prediction.vector.endpoint-log-redshift"
    endpoint_weight = 1.0 / np.asarray(variance[endpoint_key])
    clock_offset = float(
        np.sum(
            endpoint_weight
            * (np.asarray(response[endpoint_key]) - np.asarray(candidate[endpoint_key]))
        )
        / np.sum(endpoint_weight)
    )
    fitted = {
        key: (
            np.asarray(candidate[key]) + clock_offset
            if key == endpoint_key
            else scale * np.asarray(candidate[key])
        )
        for key in lens._OUTPUTS
    }
    whitened: list[float] = []
    relative: list[float] = []
    for key in lens._OUTPUTS:
        residual = fitted[key] - np.asarray(response[key])
        whitened.extend((residual * residual / np.asarray(variance[key])).reshape(-1))
        relative.append(
            float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(np.asarray(response[key])), np.finfo(float).tiny)
            )
        )
    return {
        "profiled_whitened_rmse": math.sqrt(float(np.mean(whitened))),
        "mean_feature_relative_rmse": float(np.mean(relative)),
        "fitted_lens_amplitude_scale": scale,
        "fitted_endpoint_clock_offset": clock_offset,
    }


def test_exact_subject_receipt_artifact_and_source_bindings() -> None:
    expected = {
        lens.CONFIG_PATH: "09e5d8cb884bf207fd9587c7bab4e990cd00c5d49c39a34003d5b4b22df56bf9",
        Path(
            "src/sigma_theory_compiler/open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v2.py"
        ): "ce29c0cf58ee5b3c915a81b62c98af427944e53affeb1cd834889bb68158e30c",
        lens.TEST_PATH: "e6f19fda52f90c6666f145b8b0db7ad4a4488847ff0aea68801b869d22275bcc",
        lens.RECEIPT_PATH: "233ea33703d91cf8e0770b1f9b96807ae455598c213ebf5385956b5063106e10",
    }
    for path, sha256 in expected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha256
    receipt = json.loads((ROOT / lens.RECEIPT_PATH).read_text(encoding="utf-8"))
    content_sha256 = receipt.pop("content_sha256")
    assert _json_hash(receipt) == content_sha256 == (
        "6d8ac9bbe1c96ee07829e404b14b26c006991d150d07afd0813cc763f1ab1a6f"
    )
    config = lens.load_config()
    lens.validate_config(config)
    artifact_hashes = receipt["artifact_sha256"]
    assert set(artifact_hashes) == {
        "confusion-matrix.json",
        "invariance-and-identifiability.json",
        "ledger.json",
        "scenarios.jsonl",
        "values.npz",
    }
    for name, sha256 in artifact_hashes.items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == sha256
    assert {row["id"].split("_")[0] for row in config["source_anchors"]} == {
        "LANE1",
        "LANE7",
    }
    access = receipt["access_accounting"]
    assert all(
        access[key] == 0
        for key in (
            "eso_pixels_decoded",
            "eso_spectral_rows_decoded",
            "slacs_confirmation_files_opened",
            "lens_response_tables_opened",
            "lens_response_rows_opened",
            "theory_or_nuisance_tuning_events",
            "network_calls",
            "model_calls",
            "paid_calls",
        )
    )


def test_complete_population_direct_cache_and_zero_noise_truth_equality() -> None:
    config = lens.load_config()
    rows = _rows()
    expected_triples = set(
        itertools.product(config["lenses"], config["mechanisms"], config["noise_families"])
    )
    triples = {
        (row["lens"], row["truth_formula_id"], row["noise"]["family"]) for row in rows
    }
    assert len(rows) == len({row["scenario"]["scenario_id"] for row in rows}) == 120
    assert triples == expected_triples
    assert all(
        {candidate["candidate_formula_id"] for candidate in row["candidate_comparisons"]}
        == set(config["mechanisms"])
        for row in rows
    )
    direct_pairs: dict[tuple[str, str], tuple[str, tuple[tuple[str, str], ...]]] = {}
    with np.load(OUTPUT / "values.npz", allow_pickle=False) as arrays:
        for row in rows:
            for candidate in row["candidate_comparisons"]:
                pair = (row["lens"], candidate["candidate_formula_id"])
                signature = (
                    candidate["output_sha256"],
                    tuple(sorted(candidate["value_keys"].items())),
                )
                assert pair not in direct_pairs or direct_pairs[pair] == signature
                direct_pairs[pair] = signature
                for output_id, key in candidate["value_keys"].items():
                    assert array_sha256(arrays[key]) == candidate["value_sha256"][output_id]
            if row["noise"]["family"] == "zero-noise":
                truth = next(
                    candidate
                    for candidate in row["candidate_comparisons"]
                    if candidate["candidate_formula_id"] == row["truth_formula_id"]
                )
                for output_id in lens._OUTPUTS:
                    response_key = row["value_locators"]["responses"][output_id]["key"]
                    truth_key = truth["value_keys"][output_id]
                    np.testing.assert_array_equal(arrays[response_key], arrays[truth_key])
                    assert array_sha256(arrays[response_key]) == row["value_locators"][
                        "responses"
                    ][output_id]["sha256"]
    assert len(direct_pairs) == 8 * 5 == 40
    assert sum(row["noise"]["family"] == "zero-noise" for row in rows) == 8 * 5


def test_all_five_laws_units_and_path_time_redshift_limits() -> None:
    config = lens.load_config()
    mode_by_formula = {
        "ENDPOINT_LAPSE_CONTROL": "endpoint",
        "GEOMETRIC_TIME_DELAY_CONTROL": "geometric",
        "GR_STARS_NFW_CONTROL": "gr",
        "PATH_AGED_WEYL_CLOCK": "path",
        "PHI_PSI_SLIP_SAME_STATE": "slip",
    }
    rows = _rows()
    first_by_lens = {row["lens"]: row for row in rows}
    items = lens._source_items(config)
    with np.load(OUTPUT / "values.npz", allow_pickle=False) as arrays:
        for item in items:
            values = item["values"]
            row = first_by_lens[item["lens"]]
            comparison_by_formula = {
                candidate["candidate_formula_id"]: candidate
                for candidate in row["candidate_comparisons"]
            }
            expected_by_formula = {
                formula_id: _independent_prediction(
                    values, formula_id, dict(config["law_constants"])
                )
                for formula_id in config["mechanisms"]
            }
            for formula_id, expected in expected_by_formula.items():
                candidate = comparison_by_formula[formula_id]
                for output_id, expected_value in expected.items():
                    np.testing.assert_array_equal(
                        arrays[candidate["value_keys"][output_id]], expected_value
                    )
            gr = expected_by_formula["GR_STARS_NFW_CONTROL"]
            path = expected_by_formula["PATH_AGED_WEYL_CLOCK"]
            slip = expected_by_formula["PHI_PSI_SLIP_SAME_STATE"]
            endpoint = expected_by_formula["ENDPOINT_LAPSE_CONTROL"]
            geometric = expected_by_formula["GEOMETRIC_TIME_DELAY_CONTROL"]
            np.testing.assert_array_equal(
                gr["prediction.vector.light-potential"],
                2.0 * gr["prediction.vector.matter-potential"],
            )
            np.testing.assert_array_equal(
                path["prediction.scalar.time-delay-days"],
                gr["prediction.scalar.time-delay-days"],
            )
            np.testing.assert_array_equal(
                path["prediction.vector.endpoint-log-redshift"],
                gr["prediction.vector.endpoint-log-redshift"],
            )
            np.testing.assert_array_equal(
                geometric["prediction.scalar.time-delay-days"],
                values["source.scalar.geometric-time-delay-days"],
            )
            assert endpoint["prediction.vector.endpoint-log-redshift"][1] == gr[
                "prediction.vector.endpoint-log-redshift"
            ][1]
            np.testing.assert_array_equal(
                slip["prediction.vector.matter-potential"],
                gr["prediction.vector.matter-potential"],
            )
            alpha_zero = dict(config["law_constants"])
            alpha_zero["path_alpha"] = 0.0
            gamma_one = dict(config["law_constants"])
            gamma_one["same_state_slip_gamma"] = 1.0
            for output_id in lens._OUTPUTS:
                np.testing.assert_array_equal(
                    _independent_prediction(values, "PATH_AGED_WEYL_CLOCK", alpha_zero)[
                        output_id
                    ],
                    gr[output_id],
                )
                np.testing.assert_array_equal(
                    _independent_prediction(values, "PHI_PSI_SLIP_SAME_STATE", gamma_one)[
                        output_id
                    ],
                    gr[output_id],
                )
            equal = {key: np.array(value, copy=True) for key, value in values.items()}
            equal["source.vector.path-exposure"][:] = np.mean(
                equal["source.vector.path-exposure"]
            )
            assert _independent_prediction(
                equal, "PATH_AGED_WEYL_CLOCK", dict(config["law_constants"])
            )["prediction.scalar.differential-log-redshift"][0] == 0.0
            swapped = {key: np.array(value, copy=True) for key, value in values.items()}
            swapped["source.vector.path-exposure"] = swapped[
                "source.vector.path-exposure"
            ][::-1]
            swapped_delta = _independent_prediction(
                swapped, "PATH_AGED_WEYL_CLOCK", dict(config["law_constants"])
            )["prediction.scalar.differential-log-redshift"][0]
            assert swapped_delta == -path["prediction.scalar.differential-log-redshift"][0]
            delay_mutated = {key: np.array(value, copy=True) for key, value in values.items()}
            delay_mutated["source.scalar.image-time-delay-days"] += 1.0e9
            delay_mutated["source.scalar.image-time-delay-uncertainty-days"] += 1.0e9
            delay_mutated["source.scalar.image-time-delay-available"][:] = 1 - delay_mutated[
                "source.scalar.image-time-delay-available"
            ]
            for formula_id in config["mechanisms"]:
                for output_id in lens._OUTPUTS:
                    np.testing.assert_array_equal(
                        lens._prediction(delay_mutated, mode_by_formula[formula_id])[output_id],
                        lens._prediction(values, mode_by_formula[formula_id])[output_id],
                    )
    expected_units = {
        "prediction.scalar.differential-log-redshift": "1",
        "prediction.scalar.time-delay-days": "day",
        "prediction.vector.endpoint-log-redshift": "1",
        "prediction.vector.light-potential": "1",
        "prediction.vector.matter-potential": "1",
        "prediction.vector.path-exposure": "1",
    }
    for row in rows:
        assert {
            output["element_id"]: output["unit"]
            for output in row["scenario"]["expected_predictions"]
        } == expected_units
        assert {
            response["element_id"].replace("response.synthetic-", "prediction.", 1): response[
                "unit"
            ]
            for response in row["scenario"]["scoring_responses"]
        } == expected_units


def test_independent_metric_replay_recovers_exact_105_and_72_without_hand_ranking() -> None:
    config = lens.load_config()
    rows = _rows()
    confusion = json.loads((OUTPUT / "confusion-matrix.json").read_text(encoding="utf-8"))
    recovered_count = 0
    distinct_count = 0
    recovery_by_truth = {
        truth: {"scenarios": 0, "recovered": 0, "distinct": 0}
        for truth in config["mechanisms"]
    }
    winner_counts = {
        truth: {candidate: 0 for candidate in config["mechanisms"]}
        for truth in config["mechanisms"]
    }
    with np.load(OUTPUT / "values.npz", allow_pickle=False) as arrays:
        for row in rows:
            response = {
                output: arrays[row["value_locators"]["responses"][output]["key"]]
                for output in lens._OUTPUTS
            }
            variance = {
                output: arrays[row["value_locators"]["variances"][output]["key"]]
                for output in lens._OUTPUTS
            }
            scored = []
            for candidate in row["candidate_comparisons"]:
                prediction = {
                    output: arrays[candidate["value_keys"][output]] for output in lens._OUTPUTS
                }
                metric = _independent_profile(prediction, response, variance)
                for key, expected in candidate["metrics"].items():
                    assert math.isclose(metric[key], expected, rel_tol=0.0, abs_tol=1.0e-15)
                scored.append((metric["profiled_whitened_rmse"], candidate["candidate_formula_id"]))
            scored.sort()
            minimum = scored[0][0]
            winners = sorted(
                formula_id
                for value, formula_id in scored
                if math.isclose(
                    value,
                    minimum,
                    rel_tol=0.0,
                    abs_tol=config["scoring"]["winner_absolute_tolerance"],
                )
            )
            gap = scored[1][0] - minimum
            distinct = len(winners) == 1 and gap >= config["scoring"][
                "minimum_whitened_gap_for_distinct_signature"
            ]
            truth = row["truth_formula_id"]
            recovered = truth in winners
            assert row["injection_recovery"]["winner_formula_ids"] == winners
            assert math.isclose(
                row["injection_recovery"]["profiled_whitened_gap"],
                gap,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            assert row["injection_recovery"]["truth_recovered"] is recovered
            assert row["injection_recovery"]["truth_distinctly_recovered"] is (
                recovered and distinct
            )
            recovered_count += int(recovered)
            distinct_count += int(recovered and distinct)
            recovery_by_truth[truth]["scenarios"] += 1
            recovery_by_truth[truth]["recovered"] += int(recovered)
            recovery_by_truth[truth]["distinct"] += int(recovered and distinct)
            for winner in winners:
                winner_counts[truth][winner] += 1
    assert (recovered_count, distinct_count) == (105, 72)
    assert confusion["candidate_comparison_count"] == 600
    assert confusion["truth_recovered_count"] == recovered_count
    assert confusion["distinct_truth_recovered_count"] == distinct_count
    assert confusion["recovery_by_truth"] == recovery_by_truth
    assert confusion["winner_membership_counts"] == winner_counts
    assert confusion["numerical_failure_count"] == 0
    assert confusion["no_hand_ranking"] is True


def test_ledger_chain_has_2040_entries_600_completions_and_seven_blocks_per_scenario() -> None:
    config = lens.load_config()
    rows = _rows()
    ledger = _ledger()
    bindings = {binding.formula_id: binding for binding in lens._bindings(config)}
    expected_block_status = {
        block["formula_id"]: lens._BLOCK_STATUS[block["formula_id"]] for block in config["adapter_blocks"]
    }
    assert len(ledger.entries) == 2040
    assert len({entry.entry_sha256 for entry in ledger.entries}) == 2040
    assert Counter(entry.status.value for entry in ledger.entries) == {
        "AMBIGUOUS_WITH_COMPARATOR": 288,
        "ELIGIBLE_NOT_RUN": 600,
        "PROMISING_DISTINCT_SIGNATURE": 72,
        "SOURCE_BLOCKED": 720,
        "UNADAPTED": 120,
        "UNDERPOWERED": 240,
    }
    for scenario_index, row in enumerate(rows):
        chunk = ledger.entries[scenario_index * 17 : (scenario_index + 1) * 17]
        blocked = [entry for entry in chunk if entry.status.value in {"SOURCE_BLOCKED", "UNADAPTED"}]
        eligible = [entry for entry in chunk if entry.status is DiscoveryStatus.ELIGIBLE_NOT_RUN]
        completed = [
            entry
            for entry in chunk
            if entry.status
            in {
                DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
                DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE,
                DiscoveryStatus.UNDERPOWERED,
            }
        ]
        assert (len(blocked), len(eligible), len(completed)) == (7, 5, 5)
        assert {entry.formula_id for entry in blocked} == set(expected_block_status)
        assert all(entry.status.value == expected_block_status[entry.formula_id].value for entry in blocked)
        comparison_by_formula = {
            candidate["candidate_formula_id"]: candidate
            for candidate in row["candidate_comparisons"]
        }
        for completion in completed:
            candidate = comparison_by_formula[completion.formula_id]
            prior = ledger.entries[completion.sequence - 1]
            assert completion.scenario_id == row["scenario"]["scenario_id"]
            assert completion.entry_sha256 == candidate["completed_ledger_entry_sha256"]
            assert completion.sequence == candidate["completed_ledger_sequence"]
            assert completion.result_sha256 == candidate["output_sha256"]
            assert completion.metrics_sha256 == _json_hash(candidate["metrics"])
            assert completion.binding_sha256 == bindings[completion.formula_id].content_sha256
            assert prior.status is DiscoveryStatus.ELIGIBLE_NOT_RUN
            assert completion.prior_entry_sha256 == prior.entry_sha256
            assert prior.binding_sha256 == completion.binding_sha256
    assert sum(entry.result_sha256 is not None for entry in ledger.entries) == 600


def test_derive_release_executes_exactly_40_source_only_calls_and_no_response_paths(
    monkeypatch,
) -> None:
    config = lens.load_config()
    allowed_paths = {
        config["parameter_schema_path"],
        config["output_directory"],
        *(row["path"] for row in config["source_anchors"]),
        *(row["path"] for row in config["infrastructure_bindings"]),
        *(
            config["predecessor_binding"][f"{prefix}_path"]
            for prefix in ("config", "module", "test", "receipt")
        ),
    }
    opened: list[str] = []
    calls: list[tuple[str, str, frozenset[str]]] = []
    original_repo_path = lens._repo_path
    original_execute = lens.execute_binding_in_process

    def traced_repo_path(value):
        opened.append(str(value).replace("\\", "/"))
        return original_repo_path(value)

    def traced_execute(binding, catalogue, scenario, features, parameters):
        assert not parameters
        assert all(key.startswith("source.") for key in features)
        assert not any(key.startswith(("response.", "truth.")) for key in features)
        calls.append((scenario.object_id, binding.formula_id, frozenset(features)))
        return original_execute(binding, catalogue, scenario, features, parameters)

    monkeypatch.setattr(lens, "_repo_path", traced_repo_path)
    monkeypatch.setattr(lens, "execute_binding_in_process", traced_execute)
    receipt, *_payloads = lens.derive_release()
    assert receipt["common_abi_execution_count"] == 40
    assert len(calls) == len({(object_id, formula_id) for object_id, formula_id, _ in calls}) == 40
    assert all(feature_ids == frozenset(lens._FEATURES) for _, _, feature_ids in calls)
    assert set(opened) <= allowed_paths
    assert not any("eso" in path.lower() or "slacs" in path.lower() for path in opened)
    assert receipt["access_accounting"]["lens_response_tables_opened"] == 0
    assert receipt["access_accounting"]["lens_response_rows_opened"] == 0


def test_diagnostics_recompute_limits_and_geometry_invariants() -> None:
    diagnostics = json.loads(
        (OUTPUT / "invariance-and-identifiability.json").read_text(encoding="utf-8")
    )
    assert diagnostics["pass"] is True
    assert diagnostics["pair_count"] == 8 * math.comb(5, 2) == 80
    assert diagnostics["degenerate_pair_count"] == 0
    assert len(diagnostics["geometry_rows"]) == len(diagnostics["limit_rows"]) == 8
    for key, value in diagnostics.items():
        if key.startswith("maximum_"):
            assert math.isfinite(value)
            assert value <= 1.0e-12
    assert diagnostics["maximum_alpha_zero_error"] == 0.0
    assert diagnostics["maximum_gamma_one_error"] == 0.0
    assert diagnostics["maximum_equal_exposure_differential"] == 0.0
    assert diagnostics["maximum_image_swap_antisymmetry_error"] == 0.0
    assert diagnostics["maximum_zero_mass_lens_signal"] == 0.0
    assert diagnostics["maximum_gr_phi_psi_same_state_error"] == 0.0
