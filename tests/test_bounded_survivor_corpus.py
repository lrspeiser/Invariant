import json
from pathlib import Path

import pytest

from sigma_theory_compiler.bounded_survivor_corpus import (
    BoundedSurvivorCorpusBuilder,
    benchmark_cached_cuda_manifest,
    verify_generated_manifest,
)
from sigma_theory_compiler.real_formula_execution import cuda_available

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bounded_survivor_corpus_1m.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
BINARY = ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(output: Path, formula_count: int = 20_000) -> dict:
    config = _load(CONFIG)
    config["generator_config_path"] = str(GENERATOR)
    config["generator_binary_path"] = str(BINARY)
    config["output_directory"] = str(output)
    config["formula_count"] = formula_count
    config["block_formula_count"] = formula_count
    config["threads"] = 2
    config["disk_budget_bytes"] = 4 * 1024 * 1024
    config["maximum_wall_seconds"] = 30
    config["equivalence_samples_per_block"] = 16
    return config


def test_bounded_builder_resumes_and_repairs_a_tampered_block(tmp_path: Path) -> None:
    if not BINARY.is_file():
        pytest.skip("release Rust generator is not built")
    config = _config(tmp_path / "corpus")
    first = BoundedSurvivorCorpusBuilder(config).build()
    assert first["generated_blocks"] == 1
    assert first["reused_blocks"] == 0
    assert first["formula_count"] == 20_000
    assert first["all_checks_passed"]
    assert first["data_eligibility"]["paid_llm_calls"] is False
    assert len(first["commands_executed"]) == 1
    assert "--survivor-dir" in first["commands_executed"][0]

    resumed = BoundedSurvivorCorpusBuilder(config).build()
    assert resumed["generated_blocks"] == 0
    assert resumed["reused_blocks"] == 1
    assert resumed["corpus_root_sha256"] == first["corpus_root_sha256"]

    block = Path(first["verified_manifests"][0]["blocks"][0]["file"])
    block = Path(config["output_directory"]) / block
    with block.open("ab") as handle:
        handle.write(b"tamper")
    repaired = BoundedSurvivorCorpusBuilder(config).build()
    assert repaired["generated_blocks"] == 1
    assert repaired["corpus_root_sha256"] == first["corpus_root_sha256"]


def test_manifest_verifier_rejects_wrong_interval(tmp_path: Path) -> None:
    if not BINARY.is_file():
        pytest.skip("release Rust generator is not built")
    config = _config(tmp_path / "verify", formula_count=4096)
    report = BoundedSurvivorCorpusBuilder(config).build()
    manifest = report["verified_manifests"][0]["manifest_path"]
    with pytest.raises(ValueError, match="interval mismatch"):
        verify_generated_manifest(
            manifest,
            config["output_directory"],
            GENERATOR,
            expected_start=1,
            expected_end=4097,
            equivalence_samples_per_block=4,
        )


def test_formula_and_disk_budgets_fail_before_subprocess(tmp_path: Path) -> None:
    if not BINARY.is_file():
        pytest.skip("release Rust generator is not built")
    config = _config(tmp_path / "limited")
    config["formula_count"] = 1_000_001
    config["block_formula_count"] = 1_000_001
    with pytest.raises(ValueError, match="one-million"):
        BoundedSurvivorCorpusBuilder(config)

    config = _config(tmp_path / "disk")
    config["disk_budget_bytes"] = 1024
    with pytest.raises(ValueError, match="disk budget"):
        BoundedSurvivorCorpusBuilder(config)


def test_bounded_cached_cuda_benchmark_is_root_equivalent(tmp_path: Path) -> None:
    available, reason = cuda_available()
    if not BINARY.is_file() or not available:
        pytest.skip(reason if not available else "release Rust generator is not built")
    config = _config(tmp_path / "benchmark", formula_count=20_000)
    corpus = BoundedSurvivorCorpusBuilder(config).build()
    report = benchmark_cached_cuda_manifest(
        corpus["verified_manifests"][0]["manifest_path"],
        config["output_directory"],
        GENERATOR,
        _load(EXECUTION),
        _load(PROFILE),
        cached_repeats=1,
        reference_records_per_second=0,
        output_path=tmp_path / "cuda-benchmark.json",
    )
    assert report["formula_count_in_manifest"] == 20_000
    assert report["cuda_assets_reused_for_every_measured_run"]
    assert report["cpu_gpu_status_root_equal"]
    assert report["cpu_gpu_counts_equal"]
    assert report["reference_reproduced_by_cached_median"]
    assert report["data_eligibility"]["paid_llm_calls"] is False
