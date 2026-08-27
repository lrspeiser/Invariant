"""Controls for the counterexample-driven G1 atlas repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g1_atlas_repair_v2 import (
    OUTPUT_PATH,
    GravityG1AtlasRepairError,
    build_receipt,
    load_config,
    predecessor_summary,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_repair_is_exactly_bound_to_the_single_v1_counterexample() -> None:
    config = _config()
    assert config["repair_galaxies"] == ["NGC2955"]
    assert config["predecessor_binding"]["required_covered_galaxies"] == 138
    assert config["predecessor_binding"]["required_uncovered_galaxies"] == ["NGC2955"]
    assert config["candidate_budget_per_repair_galaxy"] == 100_000_000
    assert sum(row["candidate_count"] for row in config["segments"]) == 100_000_000


def test_repair_lineage_does_not_claim_novelty_or_read_targets_for_proposal() -> None:
    shell = _config()["candidate_shell"]
    assert shell["known_base_family"] == "empirical_RAR_MOND_phenomenology"
    assert shell["origin_assessment"] == "new_combination_of_known_ideas"
    assert shell["historical_novelty_established"] is False
    assert shell["proposal_reads_vobs"] is False
    assert shell["maximum_local_constants"] == 2


def test_predecessor_replays_to_138_covered_and_ngc2955_uncovered() -> None:
    summary = predecessor_summary(ROOT, _config())
    assert summary["covered_galaxies"] == 138
    assert summary["uncovered_galaxies"] == ["NGC2955"]


def test_small_cpu_repair_cannot_pass_the_full_gate() -> None:
    receipt = build_receipt(ROOT, candidate_count_override=4, use_gpu=False)
    assert receipt["decision"] == "BLOCK_G1_REPAIR"
    assert receipt["counts"]["new_repair_candidate_galaxy_trials"] == 12
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["claims"]["historical_novelty_established"] is False
    assert [row["candidate_count"] for row in receipt["repair"]["trials"]] == [4, 4, 4]


def test_checked_repair_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 repair has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_checked_repair_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 repair has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1AtlasRepairError, match="confirmation access"):
        validate_receipt(tampered, root=ROOT)
