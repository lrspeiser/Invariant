from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_kinetic_gate_novelty_benchmark as benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_symbolic_benchmarks_pass() -> None:
    checks = benchmark.symbolic_checks()
    assert len(checks) == 6
    assert all(checks.values())


def test_exact_shifted_power_thresholds() -> None:
    assert benchmark.shifted_power_threshold(0.5) == pytest.approx(1.0)
    assert benchmark.shifted_power_threshold(1.0) == pytest.approx(0.6)
    assert benchmark.shifted_power_threshold(2.0) == pytest.approx(1.0 / 3.0)
    assert benchmark.shifted_power_threshold(4.0) == pytest.approx(3.0 / 17.0)


def test_exponential_gate_thresholds() -> None:
    assert benchmark.exponential_threshold(1.0, 0.25) is None
    assert benchmark.exponential_threshold(1.0, 0.5) == pytest.approx(0.25)
    assert benchmark.exponential_threshold(1.0, 1.0) == pytest.approx(0.75)
    assert benchmark.exponential_threshold(1.0, 2.0) == pytest.approx((0.875) ** 0.5)


def test_dynamic_range_values() -> None:
    assert benchmark.maximum_ratio(1.0) == pytest.approx(2.44140625)
    assert benchmark.maximum_ratio(0.5) == pytest.approx(5.0625)
    assert benchmark.maximum_ratio(0.1) == pytest.approx(150.0625)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_family_inputs_fail_closed(value: float) -> None:
    with pytest.raises(benchmark.KineticGateNoveltyBenchmarkError):
        benchmark.shifted_power_threshold(value)
    with pytest.raises(benchmark.KineticGateNoveltyBenchmarkError):
        benchmark.maximum_ratio(value)
    with pytest.raises(benchmark.KineticGateNoveltyBenchmarkError):
        benchmark.exponential_threshold(value, 1.0)


def test_predecessor_commit_and_worktree_bytes_match() -> None:
    config = benchmark.load_config(ROOT)
    bound = benchmark._validate_predecessor(ROOT, config)
    assert bound["commit"] == "8d50d004287e546e37e6edef202a6f841acae9cb"
    assert bound["receipt_content_sha256"] == (
        "17975d63329997fed8fb758c6945da84e7bd7d9e756a0d7d7c79644011dea9f9"
    )


def test_primary_literature_inventory_has_exact_scope() -> None:
    config = benchmark.load_config(ROOT)
    literature = config["primary_literature"]
    assert len(literature) == 12
    assert len({item["arxiv"] for item in literature}) == 12
    current = next(item for item in literature if item["arxiv"] == "2603.13986v2")
    assert "W(chi)" in current["overlap"]
    assert "Z(X)" in current["overlap"]
    assert {"0806.0336", "1609.01272", "2304.12364", "2402.04460"}.issubset(
        {item["arxiv"] for item in literature}
    )
    assert not any(item["exact_finite_range_theorem_found"] for item in literature)


def test_source_or_paper_gate_is_machine_visible() -> None:
    config = benchmark.load_config(ROOT)
    gate = config["source_or_paper_gate"]
    assert gate["missing_source_action"] == "SOURCE_BLOCKED"
    assert gate["failed_benchmark_action"] == "THEORY_ONLY_RETAINED_NOT_PROMOTED"
    assert gate["observational_promotion_requires_real_data"] is True
    assert "0806.0336" in gate["formalism_anchor"]


def test_receipt_is_deterministic_and_restrictive() -> None:
    first = benchmark.build_receipt(ROOT)
    second = benchmark.build_receipt(ROOT)
    assert first == second
    assert first["content_sha256"] == benchmark._self_hash(first)
    assert first["checks_passed"] == first["checks_total"] == 12
    claims = first["claim_boundary"]
    assert claims["paper_anchored_formalism"] is True
    assert claims["candidate_explicit_corollary_not_found_in_reviewed_set"] is True
    assert claims["historical_novelty_established"] is False
    assert claims["causal_healthy_model"] is False
    assert claims["observational_support"] is False
    assert claims["publication_ready"] is False
    assert all(value == 0 for value in first["access_ledger"].values())


def test_config_semantic_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = benchmark.load_config(ROOT)
    config = copy.deepcopy(config)
    config["claim_boundary"]["publication_ready"] = True
    original = benchmark._read_json

    def forged(path: Path) -> dict[str, object]:
        if path.name == benchmark.CONFIG_PATH.name:
            return config
        return original(path)

    monkeypatch.setattr(benchmark, "_read_json", forged)
    with pytest.raises(benchmark.KineticGateNoveltyBenchmarkError, match="claim boundary"):
        benchmark.load_config(ROOT)


def test_module_and_test_integrity_are_pinned() -> None:
    bindings = benchmark._validate_local_integrity(ROOT)
    assert bindings["config_raw_sha256"] == benchmark.EXPECTED_CONFIG_RAW_SHA256
    assert bindings["module_semantic_sha256"] == (benchmark.EXPECTED_MODULE_SEMANTIC_SHA256)
    assert bindings["test_raw_sha256"] == benchmark.EXPECTED_TEST_RAW_SHA256


def test_atomic_no_clobber_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"a": 1, "content_sha256": "x"}
    assert benchmark._atomic_no_clobber(path, value) == "CREATED"
    assert benchmark._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(benchmark.KineticGateNoveltyBenchmarkError, match="nonidentical"):
        benchmark._atomic_no_clobber(path, value)


def test_receipt_rejects_claim_forgery() -> None:
    receipt = benchmark.build_receipt(ROOT)
    receipt["claim_boundary"]["historical_novelty_established"] = True
    receipt["content_sha256"] = benchmark._self_hash(receipt)
    assert receipt != benchmark.build_receipt(ROOT)
