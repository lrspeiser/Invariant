from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_scalar_hessian_d2_integrability_gate import (
    FIRST_BLOCKER,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_scalar_hessian_d2_integrability_gate.json"
ARTIFACT = ROOT / "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _reseal(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    value["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_counts_and_blocker(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert artifact["first_blocker"] == FIRST_BLOCKER
    assert artifact["gate_counts"] == {
        "selected": 12,
        "ordered_D2_entries_materialized_per_candidate": 9801,
        "ordered_D2_entries_materialized_total": 117612,
        "nonzero_chunk_derivatives_per_candidate": 186,
        "nonzero_chunk_derivatives_total": 2232,
        "scalar_scalar_Schwarz_entries_checked_per_candidate": 891,
        "failed_ordered_family_pairs_per_candidate": 24,
        "nonzero_Schwarz_residuals_per_candidate": 30,
        "integrable_candidate_manifests": 0,
        "representative_four_entry_slices_replayed": 12,
        "full_ordered_D2_manifests_admitted": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }


def test_closed_world_submanifest_and_representative_replay(rebuilt: dict[str, object]) -> None:
    assert rebuilt["theorem"]["candidate_bound_submanifest_shape"] == [11, 9, 99]
    assert rebuilt["theorem"]["coverage_fraction"] == "11/289"
    for record in rebuilt["candidate_records"]:
        manifest = record["registered_chunk_extension"]
        assert manifest["closed_world_entries"] == 11 * 9 * 99
        assert manifest["nonzero_entries"] == 186
        assert len(manifest["blocks"]) == 81
        assert sum(block["entry_count"] for block in manifest["blocks"]) == 9801
        representative = next(
            block
            for block in manifest["blocks"]
            if block["low_direction"] == "s01[10]" and block["high_family"] == "s01"
        )
        assert representative["nonzero_count"] == 4
        assert record["representative_four_entry_slice_replayed"] is True


def test_exact_Schwarz_obstruction_is_candidate_bound(rebuilt: dict[str, object]) -> None:
    for record in rebuilt["candidate_records"]:
        schwarz = record["schwarz_integrability"]
        assert schwarz["ordered_atom_pairs_checked"] == 81
        assert schwarz["vector_entries_checked"] == 891
        assert schwarz["failed_ordered_family_pair_count"] == 24
        assert schwarz["nonzero_residual_entries"] == 30
        assert len(schwarz["residuals"]) == 24
        assert record["naive_chunk_extension_admitted_as_D2F"] is False
        assert record["candidate_decision"] == "blocked"
        assert record["candidate_rejection_authorized"] is False


def test_controls_and_seals_are_fail_closed(rebuilt: dict[str, object]) -> None:
    controls = rebuilt["exact_controls"]
    assert controls["representative_s01_s01_replay"]["passed"] is True
    assert all(
        control["rejected"]
        for key, control in controls.items()
        if key != "representative_s01_s01_replay"
    )
    assert rebuilt["claim_seals"]["naive_chunk_extension_obstructed"] is True
    assert not rebuilt["claim_seals"]["corrected_covariant_D2_tensor_ruled_out"]
    assert not rebuilt["claim_seals"]["candidate_theory_rejected"]
    assert not any(rebuilt["data_seals"].values())


@pytest.mark.parametrize(
    "mutation",
    ["candidate", "residual", "coverage", "promotion", "scope", "predecessor", "extra_key"],
)
def test_resealed_semantic_tampering_rejects(rebuilt: dict[str, object], mutation: str) -> None:
    value = copy.deepcopy(rebuilt)
    if mutation == "candidate":
        value["candidate_records"][0]["candidate_decision"] = "pass"
    elif mutation == "residual":
        value["candidate_records"][0]["schwarz_integrability"]["nonzero_residual_entries"] = 0
    elif mutation == "coverage":
        value["theorem"]["coverage_fraction"] = "1"
    elif mutation == "promotion":
        value["claim_seals"]["full_ordered_D2_tensor_registered"] = True
    elif mutation == "scope":
        value["scope"] = "global H7 proved"
    elif mutation == "predecessor":
        value["source_bindings"]["full_tensor_reconciliation"]["content_sha256"] = "0" * 64
    else:
        value["global_H7_pass"] = True
    _reseal(value)
    with pytest.raises(ValueError, match="scalar-Hessian"):
        _validate_result(value)


def test_config_and_local_hash_bindings_are_exact(rebuilt: dict[str, object]) -> None:
    config = _load(CONFIG)
    _validate_config(config)
    changed = copy.deepcopy(config)
    changed["policies"]["full_D2_promotion"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(changed)
    for label in ("config", "source", "test"):
        binding = rebuilt["source_bindings"][label]
        assert not Path(binding["path"]).is_absolute()
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == binding["file_sha256"]
        )


def test_source_has_no_runtime_data_or_gpu_surface() -> None:
    source = (
        (ROOT / "src/sigma_theory_compiler/quartic_scalar_hessian_d2_integrability_gate.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    for forbidden in (
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "cupy",
        "torch",
        "os.kill",
        "popen",
    ):
        assert forbidden not in source
