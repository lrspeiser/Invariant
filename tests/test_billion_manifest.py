import json
from pathlib import Path

from sigma_theory_compiler.high_throughput import crosscheck_manifest


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "generator-v2"
CONFIG = ROOT / "configs" / "generator_v2_billion.json"
COMPLETE = RUNS / "billion-authoritative.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_billion_manifest_accounts_for_declared_space() -> None:
    manifest = _load(COMPLETE)
    assert manifest["total_declared_actions"] == 1_088_651_720
    assert manifest["processed_actions"] == 1_088_651_720
    assert manifest["complete_declared_space"] is True
    assert sum(manifest["gate_counts"].values()) == manifest["processed_actions"]
    assert sum(block["processed"] for block in manifest["blocks"]) == manifest[
        "processed_actions"
    ]
    assert manifest["observational_data_opened"] is False
    assert len(manifest["rejection_witnesses"]) == 5
    sample_ordinals = [sample["ordinal"] for sample in manifest["survivor_samples"]]
    assert min(sample_ordinals) < 100_000_000
    assert max(sample_ordinals) > 900_000_000
    assert manifest["blocks_root_sha256"] == (
        "8c56d27cbf9c3ec28328a3244f745424148c6f8516ddd7c5922f6ad4c8b77a93"
    )


def test_complete_manifest_passes_independent_python_crosscheck() -> None:
    report = crosscheck_manifest(COMPLETE, CONFIG)
    assert report["all_accounting_checks_pass"] is True
    assert report["all_cross_language_samples_agree"] is True
    assert report["all_recorded_survivors_pass_python_sampled_convexity"] is True


def test_thread_count_does_not_change_commitment_or_gate_counts() -> None:
    one = _load(RUNS / "million-sound-asymptotic-t1.json")
    eight = _load(RUNS / "million-sound-asymptotic.json")
    assert one["blocks_root_sha256"] == eight["blocks_root_sha256"]
    assert one["gate_counts"] == eight["gate_counts"]


def test_windows_and_linux_builds_produce_identical_commitments() -> None:
    windows = _load(RUNS / "million-sound-asymptotic.json")
    linux = _load(RUNS / "million-linux-authoritative.json")
    assert windows["blocks_root_sha256"] == linux["blocks_root_sha256"]
    assert windows["basis_library_sha256"] == linux["basis_library_sha256"]
    assert windows["gate_counts"] == linux["gate_counts"]
    assert windows["survivor_samples"] == linux["survivor_samples"]


def test_checkpoint_replay_reuses_every_block_without_changing_result() -> None:
    initial = _load(RUNS / "checkpoint-sound-initial.json")
    resumed = _load(RUNS / "checkpoint-sound-resumed.json")
    assert initial["actions_computed_this_run"] == 1_000_000
    assert resumed["actions_computed_this_run"] == 0
    assert resumed["checkpoint_blocks_reused"] == resumed["block_count"] == 16
    assert initial["blocks_root_sha256"] == resumed["blocks_root_sha256"]
    assert initial["gate_counts"] == resumed["gate_counts"]
