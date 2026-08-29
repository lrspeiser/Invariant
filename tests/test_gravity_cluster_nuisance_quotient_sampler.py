from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / sampler.CONFIG_PATH
ARTIFACTS = ROOT / sampler.ARTIFACT_DIR
RECEIPT = ROOT / sampler.IMPLEMENTATION_RECEIPT


def config_hash() -> str:
    return sampler.file_sha256(CONFIG)


def smoke_summary() -> dict[str, object]:
    archive = np.load(ARTIFACTS / "bounded-smoke.npz", allow_pickle=False)
    return json.loads(str(archive["summary"].item()))


def test_canonical_config_and_receipt_fail_closed_validation() -> None:
    config = sampler.load_contract(CONFIG, config_hash())
    check = sampler.check_canonical_implementation(CONFIG, config_hash(), RECEIPT)
    assert config["exact_primitive_priors"] == sampler.PRIMITIVE_PRIORS
    assert len(config["exact_primitive_priors"]) == 17
    assert config["source_bindings"]["quotient_receipt"]["file_sha256"] == (
        "65dd1909e11a724e01b9614b1a0bb611d7a3c2f5baf8c54edba9bd27b229af08"
    )
    assert check == {
        "valid": True,
        "config_sha256": config_hash(),
        "implementation_receipt_sha256": sampler.file_sha256(RECEIPT),
        "production_authorized": False,
        "production_launches": 0,
        "completed_tasks": 59,
        "open_tasks": 63,
        "total_tasks": 122,
        "CP5_status": "PARTIAL",
    }
    with pytest.raises(RuntimeError, match="contract hash"):
        sampler.load_contract(CONFIG, "0" * 64)


def test_canonical_source_config_and_json_artifacts_have_no_work_binding() -> None:
    text_paths = [
        ROOT / "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sampler.py",
        CONFIG,
        RECEIPT,
        *(path for path in ARTIFACTS.glob("*.json")),
    ]
    for path in text_paths:
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text, path
        assert "v6r" not in text, path
        assert "source_smc" not in text, path


def test_train_packet_and_sobol_starts_preserve_frozen_boundaries() -> None:
    config = sampler.load_contract(CONFIG, config_hash())
    packets = sampler.load_train_packet(
        ROOT / config["train_packet"]["path"],
        config["train_packet"]["file_sha256"],
    )
    assert len(packets) == 8
    rows = [row for packet in packets for row in packet["rows"]]
    assert len(rows) == 80
    assert {row["split"] for row in rows} == {"development_train"}
    starts = np.load(
        ROOT / config["sobol_start_population"]["path"], allow_pickle=False
    )["particles"]
    assert starts.shape == (4, 512, 17)
    assert np.all((starts > 0.0) & (starts < 1.0))
    assert not np.array_equal(starts[0], starts[1])


def test_exact_prior_pushforward_retains_stellar_clipping_mixture() -> None:
    config = uncertainty.load_config(ROOT)
    assert config["continuous_priors"] == sampler.PRIMITIVE_PRIORS
    low = np.zeros(17)
    high = np.ones(17)
    low_composite = sampler.composite_values(low, config)
    high_composite = sampler.composite_values(high, config)
    stellar_index = sampler.COMPOSITES.index("published_stellar_acceleration_scale")
    assert low_composite[stellar_index] == pytest.approx(0.4 / (0.95 * 0.85) ** 2)
    assert high_composite[stellar_index] == pytest.approx(2.5 / (1.05 * 1.15) ** 2)


class ConstantEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _unit: np.ndarray) -> float:
        self.calls += 1
        return 0.0


def test_out_of_bounds_correlated_proposals_are_unevaluated_self_loops() -> None:
    particles = np.full((3, 17), 0.5)
    likelihood = np.zeros(3)
    before = particles.copy()
    evaluator = ConstantEvaluator()
    counts = sampler.active_transition(
        particles,
        likelihood,
        evaluator,  # type: ignore[arg-type]
        np.random.default_rng(813),
        np.eye(10),
        1e9,
    )
    assert counts == {
        "attempted": 3,
        "out_of_bounds_rejected": 3,
        "evaluated": 0,
        "accepted": 0,
    }
    assert evaluator.calls == 0
    np.testing.assert_array_equal(particles, before)


@pytest.mark.parametrize("move", sampler.ORBIT_NAMES)
def test_each_orbit_move_preserves_all_ten_composites(move: str) -> None:
    config = uncertainty.load_config(ROOT)
    physical = sampler.physical_values(np.full(17, 0.5), config)
    before = sampler.composite_values(sampler.unit_values(physical, config), config)
    rng = np.random.default_rng(5930)
    for _ in range(1000):
        proposed = physical.copy()
        if sampler.apply_orbit_move(proposed, move, rng, config, 0.08, 0.03, 0.02):
            after = sampler.composite_values(
                sampler.unit_values(proposed, config), config
            )
            np.testing.assert_allclose(after, before, atol=1e-12, rtol=0.0)
            return
    pytest.fail(f"no accepted {move} orbit move")


def test_diagnostics_controls_and_bounded_smoke_are_honestly_adjudicated() -> None:
    diagnostic = sampler.diagnostic_validation_control(sampler.DIAGNOSTIC_VALIDATION)
    constant = diagnostic["observed"]["constant_chain_negative_control"]
    assert diagnostic["passed"] is True
    assert constant == {
        "valid": False,
        "rhat": None,
        "bulk_ess": 0.0,
        "tail_ess": 0.0,
        "minimum_scaled_within_chain_variance": 0.0,
    }
    summary = smoke_summary()
    assert summary["mode"] == "smoke"
    assert summary["forward_call_accounting"]["total_forward_evaluations"] == 852
    assert summary["orbit_validation_passed"] is True
    assert summary["runtime_data_boundary"] == {
        "packet_sha256": (
            "a8efa90d569ea79f895a136b8177da1fb00681ce5bac1a169a1675a3515e32dc"
        ),
        "allowed_split": "development_train",
        "rows_loaded": 80,
        "holdout_rows_loaded": 0,
        "confirmation_rows_loaded": 0,
        "independent_rows_loaded": 0,
        "canonical_comparator_packet_builder_called_during_sampling": False,
    }
    assert summary["production_passed"] is False


def test_no_clobber_and_authorization_controls_pass_without_authorized_manifest() -> None:
    race = json.loads(
        (ARTIFACTS / "atomic-no-clobber-controls.json").read_text(encoding="utf-8")
    )
    transitions = json.loads(
        (ARTIFACTS / "authorization-transition-controls.json").read_text(
            encoding="utf-8"
        )
    )
    unauthorized_path = ARTIFACTS / "authorization-current-unauthorized.json"
    unauthorized = json.loads(unauthorized_path.read_text(encoding="utf-8"))
    assert race["passed"] is True
    assert race["npz_concurrent_creator"]["concurrent_creator_target_bytes_preserved"]
    assert race["json_concurrent_creator"]["concurrent_creator_target_bytes_preserved"]
    assert transitions["passed"] is True
    assert transitions["exact_replay_equality"] is True
    assert transitions["production_runs"] == 0
    assert unauthorized["production_authorization"]["authorized"] is False
    assert not (ARTIFACTS / "authorization-current-authorized.json").exists()
    with pytest.raises(RuntimeError, match="before contract or packet load"):
        sampler.validate_authorization(
            unauthorized_path,
            sampler.file_sha256(unauthorized_path),
            require_production=True,
        )
