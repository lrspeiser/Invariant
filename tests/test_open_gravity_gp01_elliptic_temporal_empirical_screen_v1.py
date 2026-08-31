from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_gp01_elliptic_temporal_empirical_screen_v1 as screen


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config_path = screen._ROOT / screen.CONFIG_PATH
    config = screen._read_json(config_path, "test config")
    screen.validate_config(config)
    return config, screen.run_screen(config)


def test_exact_sealed_result_binding(packet: tuple[dict, dict]) -> None:
    config, _result = packet
    binding = config["sealed_result_binding"]
    assert len(binding["commit"]) == 40
    assert len(binding["sha256"]) == 64
    committed = screen._git_show(binding["commit"], binding["path"])
    assert screen.hashlib.sha256(committed).hexdigest() == binding["sha256"]


def test_exact_elliptic_population_and_best_cell(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    elliptic = result["elliptic"]
    assert elliptic["valid_frozen_cells"] == 1296
    assert elliptic["best_cell"] == "GP01E-n1-A8-rho10-T10-q2-p2-L0"
    assert elliptic["best_robust_loss"] == pytest.approx(356.59146688196233)
    assert elliptic["equilibrium_robust_loss"] == pytest.approx(209.84757921810345)


def test_elliptic_does_not_beat_equilibrium_on_any_cluster(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    assert result["elliptic"]["beats_equilibrium_on_objects"] == 0
    assert len(result["object_rows"]) == 8
    assert all(row["elliptic_to_equilibrium_loss_ratio"] > 1.0 for row in result["object_rows"])


def test_best_elliptic_cell_is_a_boundary_extension_signal(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    assert result["elliptic"]["boundary_extension_signal"] is True
    cell = result["elliptic"]["best_cell"]
    assert all(token in cell for token in ("-A8-", "-rho10-", "-T10-", "-q2-", "-p2-", "-L0"))


def test_log_entropy_retrospective_rank_signal(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    ratio = result["dynamical_proxy_correlations"]["LOG_K0_KEV_CM2"]["loss_ratio"]
    difference = result["dynamical_proxy_correlations"]["LOG_K0_KEV_CM2"]["loss_difference"]
    assert ratio["rho"] == pytest.approx(-0.8263621207201487)
    assert ratio["exact_two_sided_p"] == pytest.approx(1.0 / 60.0)
    assert difference["rho"] == pytest.approx(-0.8622909085775465)
    assert difference["exact_two_sided_p"] == pytest.approx(0.00873015873015873)


def test_other_morphology_proxies_are_not_overclaimed(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    correlations = result["dynamical_proxy_correlations"]
    assert correlations["CENTROID_SHIFT_X1E3"]["loss_ratio"]["exact_two_sided_p"] > 0.1
    assert correlations["C_Z"]["loss_ratio"]["exact_two_sided_p"] > 0.8


def test_cool_core_partition_signal_is_exact(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    split = result["cool_core_test"]
    assert split["partitions"] == 56
    assert split["difference"] == pytest.approx(0.28538920962697345)
    assert split["exact_two_sided_p"] == pytest.approx(1.0 / 28.0)


def test_temporal_branch_is_not_mislabeled_as_direct_fit(packet: tuple[dict, dict]) -> None:
    _config, result = packet
    temporal = result["temporal_interpretation"]
    assert temporal == {
        "direct_telegraph_fit": False,
        "static_equilibrium_data_can_identify_relaxation_time": False,
        "retrospective_history_proxy_signal": True,
        "independent_history_data_required": True,
    }


def test_every_frozen_check_passes(packet: tuple[dict, dict]) -> None:
    config, result = packet
    assert list(result["checks"]) == config["required_checks"]
    assert result["checks_passed"] == 8
    assert result["checks_failed"] == 0
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "sealed_result_binding",
        "primary_source_metadata",
        "analysis_contract",
        "required_checks",
        "next_experiments",
        "access_accounting",
        "claim_boundary",
    ),
)
def test_semantic_sections_are_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _result = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(screen.EmpiricalScreenError, match="config semantics changed"):
        screen.validate_config(changed)


def test_noncanonical_output_refuses_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(screen, "OUTPUT_PATH", tmp_path / "response.json")
    monkeypatch.setattr(screen, "_read_json", forbidden)
    with pytest.raises(screen.EmpiricalScreenError, match="output path changed"):
        screen.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _result = packet
    receipt = screen.build_receipt()
    screen.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["screen"]["temporal_interpretation"]["direct_telegraph_fit"] = True
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = screen.content_sha256(body)
    with pytest.raises(screen.EmpiricalScreenError, match="not reproducible"):
        screen.validate_receipt_payload(forged)


def test_access_and_claim_boundary(packet: tuple[dict, dict]) -> None:
    config, _result = packet
    access = config["access_accounting"]
    assert access["sealed_response_derived_ledgers_read"] == 1
    assert access["raw_scientific_response_files_read"] == 0
    assert access["new_scores_from_raw_rows"] == 0
    assert (
        "a direct temporal-memory or telegraph fit"
        in config["claim_boundary"]["does_not_establish"]
    )
    assert (
        "permission to tune each cluster independently"
        in config["claim_boundary"]["does_not_establish"]
    )
