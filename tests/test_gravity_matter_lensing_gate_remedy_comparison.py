from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_gate_remedy_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / comparison.CONFIG_PATH).read_text(encoding="utf-8"))


def test_symbolic_suite_passes_exact_inventory() -> None:
    symbolic = comparison.run_symbolic_suite()
    assert tuple(item["check_id"] for item in symbolic["checks"]) == comparison.SYMBOLIC_CHECK_IDS
    assert symbolic["all_passed"] is True
    assert len(symbolic["checks"]) == 12


def test_receipt_is_deterministic_and_strictly_bounded() -> None:
    first = comparison.build_receipt(ROOT)
    second = comparison.build_receipt(ROOT)
    assert first == second
    assert first["counts"]["architectures_compared"] == 5
    assert first["counts"]["universal_metric_architectures"] == 5
    assert first["counts"]["independent_photon_multipliers"] == 0
    assert first["counts"]["symbolic_checks_passed"] == 12
    assert first["counts"]["healthy_architectures_established"] == 0
    assert first["counts"]["novel_architectures_established"] == 0
    assert all(value == 0 for value in first["zero_access_and_compute"].values())


def test_exact_five_origin_mapping_and_ranking() -> None:
    config = _config()
    architectures = config["architectures"]
    assert tuple(item["architecture_id"] for item in architectures) == comparison.ARCHITECTURE_IDS
    assert tuple(item["origin_remedy_id"] for item in architectures) == comparison.ORIGIN_IDS
    assert tuple(config["ranking_contract"]["order"]) == comparison.RANKING
    rank_by_id = {item["architecture_id"]: item["structural_rank"] for item in architectures}
    assert [rank_by_id[item] for item in comparison.RANKING] == [1, 2, 3, 4, 5]


def test_universal_metric_and_derivative_risk_are_explicit() -> None:
    config = _config()
    shared = config["shared_requirements"]
    assert "same single physical metric" in shared["universal_metric"]
    assert "higher derivatives" in shared["derivative_coupling_warning"]
    assert "second multiplier" in shared["forbidden_shortcut"]
    for item in config["architectures"]:
        assert item["universal_metric_preserved"] is True
        assert item["independent_photon_multiplier"] is False
        assert item["healthy_claim"] is False
        assert item["novelty_claim"] is False


def test_architecture_specific_claim_ceiling() -> None:
    architectures = {item["architecture_id"]: item for item in _config()["architectures"]}
    assert "Y=Y0>0" in architectures["B_SPLIT_KINETIC_MASS_GATES"]["minimal_action_term"]
    assert (
        "m_eff^2=m_chi^2*Z/Y" in architectures["B_SPLIT_KINETIC_MASS_GATES"]["range_and_amplitude"]
    )
    assert (
        "High and unresolved for s(X_phi)"
        in architectures["A_SOURCE_AMPLITUDE_GATE"]["higher_derivative_risk"]
    )
    assert (
        "does not itself create environmental screening"
        in architectures["C_POSITIVE_FIELD_SPACE_METRIC"]["range_and_amplitude"]
    )
    assert (
        "no Yukawa propagation length"
        in architectures["D_AUXILIARY_CHANNEL"]["range_and_amplitude"]
    )
    assert "UNRESOLVED" in architectures["E_DYNAMIC_XCHI_SUPPRESSION"]["degrees_of_freedom"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda c: c["shared_requirements"].update({"universal_metric": "two metrics"}),
            "universal metric",
        ),
        (
            lambda c: c["shared_requirements"].update({"derivative_coupling_warning": "safe"}),
            "derivative-coupling",
        ),
        (
            lambda c: c["architectures"][0].update({"independent_photon_multiplier": True}),
            "photon multiplier",
        ),
        (lambda c: c["architectures"][1].update({"healthy_claim": True}), "overclaimed"),
        (lambda c: c["architectures"][2].update({"structural_rank": 1}), "rank inconsistent"),
        (
            lambda c: c["ranking_contract"].update({"order": list(reversed(comparison.RANKING))}),
            "ranking changed",
        ),
        (
            lambda c: c["adjudication"].update({"novel_architecture_identified": True}),
            "health or novelty",
        ),
        (
            lambda c: c["claim_boundary"].update({"lensing_success_established": True}),
            "claim boundary",
        ),
        (lambda c: c["zero_access_and_compute"].update({"network_calls": 1}), "access state"),
    ],
)
def test_semantic_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    monkeypatch.setattr(comparison, "EXPECTED_CONFIG_CONTENT_SHA256", comparison._sha(config))
    with pytest.raises(comparison.RemedyComparisonError, match=message):
        comparison.validate_config(config)


def test_predecessor_receipt_mutation_fails_closed(tmp_path: Path) -> None:
    config = _config()
    needed = [comparison.CONFIG_PATH, comparison.SOURCE_PATH, comparison.TEST_PATH]
    needed.extend(Path(item["receipt_path"]) for item in config["predecessor_receipt_bindings"])
    for relative in needed:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    changed = tmp_path / config["predecessor_receipt_bindings"][0]["receipt_path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(comparison.RemedyComparisonError, match="predecessor changed"):
        comparison.build_receipt(tmp_path)


def test_receipt_architecture_mutation_fails_closed() -> None:
    config = comparison.load_config(ROOT)
    receipt = comparison.build_receipt(ROOT)
    receipt["architectures"][0]["healthy_claim"] = True
    body = dict(receipt)
    body.pop("content_sha256")
    receipt["content_sha256"] = comparison._sha(body)
    with pytest.raises(comparison.RemedyComparisonError, match="comparison changed"):
        comparison.validate_receipt(receipt, config)


def test_atomic_publication_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = b"sealed\n"
    assert comparison._atomic_no_replace(path, payload) == "CREATED"
    assert comparison._atomic_no_replace(path, payload) == "EXISTING_IDENTICAL"
    with pytest.raises(comparison.RemedyComparisonError, match="refusing to overwrite"):
        comparison._atomic_no_replace(path, b"different\n")
    assert path.read_bytes() == payload
