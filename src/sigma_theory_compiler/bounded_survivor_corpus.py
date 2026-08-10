from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .high_throughput import build_basis, decode_ordinal, total_search_count
from .survivors import HEADER, MAGIC, RECORD

SCHEMA = "sigma-bounded-survivor-corpus-builder-1.0"
REPORT_SCHEMA = "sigma-bounded-survivor-corpus-report-1.0"
BENCHMARK_SCHEMA = "sigma-bounded-survivor-cuda-benchmark-1.0"
HARD_MAXIMUM_FORMULAS = 1_000_000
MANIFEST_ALLOWANCE_BYTES = 2 * 1024 * 1024
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sample_indices(count: int, requested: int, sha256: str) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    limit = min(count, requested)
    indices = {0, count - 1}
    state = bytes.fromhex(sha256)
    sequence = 0
    while len(indices) < limit:
        state = hashlib.sha256(state + sequence.to_bytes(8, "little")).digest()
        indices.add(int.from_bytes(state[:8], "little") % count)
        sequence += 1
    return sorted(indices)[:limit]


def verify_generated_manifest(
    manifest_path: str | Path,
    survivor_directory: str | Path,
    generator_config_path: str | Path,
    *,
    expected_start: int,
    expected_end: int,
    equivalence_samples_per_block: int,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    survivor_directory = Path(survivor_directory).resolve()
    generator_path = Path(generator_config_path).resolve()
    generator_raw = generator_path.read_bytes()
    generator = json.loads(generator_raw)
    config_sha = hashlib.sha256(generator_raw).hexdigest()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema_version") != "sigma-generator-v2-manifest-1.0":
        raise ValueError("unsupported Rust generator manifest")
    if manifest.get("observational_data_opened") is not False:
        raise ValueError("Rust generator manifest opened observational data")
    if (
        manifest.get("config_sha256") != config_sha
        or manifest.get("protocol_version") != generator.get("protocol_version")
        or int(manifest.get("basis_count", -1)) != int(generator["basis_count"])
        or int(manifest.get("max_action_terms", -1)) != int(generator["max_action_terms"])
    ):
        raise ValueError("Rust manifest/config identity mismatch")
    basis_payload = json.dumps(
        build_basis(int(generator["basis_count"])), separators=(",", ":"), ensure_ascii=False
    ).encode()
    if manifest.get("basis_library_sha256") != hashlib.sha256(basis_payload).hexdigest():
        raise ValueError("Rust manifest basis hash mismatch")
    declared_total = total_search_count(
        int(generator["basis_count"]), int(generator["max_action_terms"])
    )
    if (
        int(manifest.get("total_declared_actions", -1)) != declared_total
        or manifest.get("coefficient_alphabet") != generator.get("coefficient_alphabet")
    ):
        raise ValueError("Rust manifest declared generator space mismatch")
    if (
        int(manifest.get("start_ordinal", -1)) != expected_start
        or int(manifest.get("end_ordinal_exclusive", -1)) != expected_end
        or int(manifest.get("processed_actions", -1)) != expected_end - expected_start
    ):
        raise ValueError("Rust manifest formula interval mismatch")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Rust manifest contains no blocks")
    previous_end = expected_start
    record_count = file_bytes = samples_checked = 0
    block_commits: list[dict[str, Any]] = []
    blocks_root = hashlib.sha256()
    blocks_root.update(b"SIGMA-GENERATOR-V2-ROOT\0")
    for block in blocks:
        start = int(block["start_ordinal"])
        end = int(block["end_ordinal_exclusive"])
        if start != previous_end or not start < end <= expected_end:
            raise ValueError("Rust manifest block coverage is not contiguous")
        if int(block.get("processed", -1)) != end - start:
            raise ValueError("Rust manifest block formula accounting mismatch")
        previous_end = end
        digest_sha256 = str(block.get("digest_sha256", ""))
        if len(digest_sha256) != 64:
            raise ValueError("Rust manifest block digest is invalid")
        blocks_root.update(int(block["block_index"]).to_bytes(8, "little"))
        blocks_root.update(start.to_bytes(8, "little"))
        blocks_root.update(end.to_bytes(8, "little"))
        blocks_root.update(digest_sha256.encode("ascii"))
        export = block.get("survivor_export")
        if not isinstance(export, dict):
            raise TypeError("Rust manifest block lacks survivor export")
        path = survivor_directory / str(export["file"])
        expected_size = int(export["file_size_bytes"])
        expected_sha = str(export["file_sha256"])
        if int(block.get("survivors", -1)) != int(export["record_count"]):
            raise ValueError("Rust manifest block survivor accounting mismatch")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError("Rust survivor block size mismatch")
        actual_sha = _sha256_path(path)
        if actual_sha != expected_sha:
            raise ValueError("Rust survivor block SHA mismatch")
        with path.open("rb") as handle:
            raw_header = handle.read(HEADER.size)
            if len(raw_header) != HEADER.size:
                raise ValueError("truncated Rust survivor header")
            header = HEADER.unpack(raw_header)
            expected_header = (
                MAGIC,
                1,
                RECORD.size,
                int(block["block_index"]),
                start,
                end,
                int(export["record_count"]),
            )
            if header != expected_header:
                raise ValueError("Rust survivor header mismatch")
            records = []
            previous_ordinal = start - 1 if start else -1
            sample_indices = set(
                _sample_indices(
                    int(export["record_count"]), equivalence_samples_per_block, actual_sha
                )
            )
            for index in range(int(export["record_count"])):
                raw_record = handle.read(RECORD.size)
                if len(raw_record) != RECORD.size:
                    raise ValueError("truncated Rust survivor record")
                ordinal, term_count, sign_mask, reserved, *term_ids = RECORD.unpack(raw_record)
                if (
                    reserved != 0
                    or not 1 <= term_count <= 6
                    or not previous_ordinal < ordinal < end
                    or ordinal < start
                    or any(value >= int(generator["basis_count"]) for value in term_ids[:term_count])
                    or any(value != 0xFFFF for value in term_ids[term_count:])
                ):
                    raise ValueError("invalid Rust survivor record")
                previous_ordinal = ordinal
                if index in sample_indices:
                    records.append((ordinal, term_count, sign_mask, term_ids))
            if handle.read(1):
                raise ValueError("trailing Rust survivor bytes")
        for ordinal, term_count, sign_mask, term_ids in records:
            decoded = decode_ordinal(
                int(generator["basis_count"]), int(generator["max_action_terms"]), ordinal
            )
            decoded_mask = sum(
                1 << position
                for position, sign in enumerate(decoded["signs"])
                if sign > 0
            )
            if decoded["term_ids"] != list(term_ids[:term_count]) or decoded_mask != sign_mask:
                raise ValueError("Rust survivor differs from Python ordinal decoder")
        samples_checked += len(records)
        record_count += int(export["record_count"])
        file_bytes += expected_size
        block_commits.append(
            {
                "block_index": int(block["block_index"]),
                "start_ordinal": start,
                "end_ordinal_exclusive": end,
                "file": export["file"],
                "file_sha256": actual_sha,
                "record_count": int(export["record_count"]),
            }
        )
    gate_counts = manifest.get("gate_counts", {})
    if (
        previous_end != expected_end
        or record_count != int(manifest["survivor_count"])
        or sum(int(value) for value in gate_counts.values()) != expected_end - expected_start
        or int(gate_counts.get("survive_sampled_static", -1)) != record_count
        or manifest.get("blocks_root_sha256") != blocks_root.hexdigest()
    ):
        raise ValueError("Rust survivor corpus accounting mismatch")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "start_ordinal": expected_start,
        "end_ordinal_exclusive": expected_end,
        "processed_formulas": expected_end - expected_start,
        "record_count": record_count,
        "survivor_file_bytes": file_bytes,
        "equivalence_samples_checked": samples_checked,
        "all_checks_passed": True,
        "blocks": block_commits,
    }


class BoundedSurvivorCorpusBuilder:
    """Bounded, resumable orchestrator for trusted local generator-v2 blocks."""

    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("schema_version") != SCHEMA:
            raise ValueError("unsupported bounded corpus builder schema")
        if config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        if config.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("corpus data eligibility is not fail-closed")
        self.config = config
        self.generator_path = Path(config["generator_config_path"]).resolve()
        self.binary_path = Path(config["generator_binary_path"]).resolve()
        self.output_directory = Path(config["output_directory"]).resolve()
        if not self.generator_path.is_file():
            raise FileNotFoundError("generator config is unavailable")
        if not self.binary_path.is_file():
            raise FileNotFoundError("trusted local Rust generator binary is unavailable")
        generator_raw = self.generator_path.read_bytes()
        self.generator = json.loads(generator_raw)
        if self.generator.get("observational_data_opened") is not False:
            raise ValueError("generator config opens observational data")
        self.generator_sha = hashlib.sha256(generator_raw).hexdigest()
        self.binary_sha = _sha256_path(self.binary_path)
        self.start = int(config["start_ordinal"])
        self.formula_count = int(config["formula_count"])
        self.block_size = int(config["block_formula_count"])
        maximum = int(config["maximum_formula_count"])
        if not 1 <= self.formula_count <= maximum <= HARD_MAXIMUM_FORMULAS:
            raise ValueError("formula count exceeds the bounded one-million limit")
        if self.block_size <= 0 or self.block_size > self.formula_count:
            raise ValueError("invalid corpus block formula count")
        if self.start % self.block_size:
            raise ValueError("start ordinal must align to the corpus block size")
        total = total_search_count(
            int(self.generator["basis_count"]), int(self.generator["max_action_terms"])
        )
        self.end = self.start + self.formula_count
        if self.start < 0 or self.end > total:
            raise ValueError("bounded corpus interval exceeds the generator space")
        self.disk_budget = int(config["disk_budget_bytes"])
        block_count = (self.formula_count + self.block_size - 1) // self.block_size
        self.worst_case_bytes = self.formula_count * RECORD.size + block_count * (
            HEADER.size + MANIFEST_ALLOWANCE_BYTES
        )
        if self.worst_case_bytes > self.disk_budget:
            raise ValueError("disk budget cannot hold the worst-case bounded corpus")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_directory / "corpus-state.json"
        identity = {
            "generator_config_sha256": self.generator_sha,
            "generator_binary_sha256": self.binary_sha,
            "start_ordinal": self.start,
            "end_ordinal_exclusive": self.end,
            "block_formula_count": self.block_size,
            "threads": int(config["threads"]),
            "disk_budget_bytes": self.disk_budget,
        }
        self.identity_sha = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("identity_sha256") != self.identity_sha:
                raise ValueError("refusing to resume a changed corpus build")

    def command_for(self, start: int, end: int, manifest_path: Path) -> list[str]:
        return [
            str(self.binary_path),
            "run",
            "--config",
            str(self.generator_path),
            "--output",
            str(manifest_path),
            "--start",
            str(start),
            "--limit",
            str(end - start),
            "--threads",
            str(int(self.config["threads"])),
            "--block-size",
            str(self.block_size),
            "--survivor-dir",
            ".",
        ]

    def portable_command_for(self, start: int, end: int, manifest_path: Path) -> list[str]:
        """Return the provenance argv without committing workstation paths."""
        command = self.command_for(start, end, manifest_path)
        command[0] = "$SIGMA_GENERATOR_V2"
        command[3] = "$GENERATOR_CONFIG"
        command[5] = manifest_path.name
        return command

    def build(self) -> dict[str, Any]:
        started = time.perf_counter()
        maximum_seconds = float(self.config["maximum_wall_seconds"])
        generated = reused = 0
        verified: list[dict[str, Any]] = []
        commands: list[list[str]] = []
        cursor = self.start
        while cursor < self.end:
            end = min(cursor + self.block_size, self.end)
            sequence = (cursor - self.start) // self.block_size
            manifest_path = self.output_directory / f"manifest-{sequence:06}-{cursor}-{end}.json"
            audit = None
            if manifest_path.is_file():
                try:
                    audit = verify_generated_manifest(
                        manifest_path,
                        self.output_directory,
                        self.generator_path,
                        expected_start=cursor,
                        expected_end=end,
                        equivalence_samples_per_block=int(
                            self.config["equivalence_samples_per_block"]
                        ),
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    audit = None
            if audit is None:
                elapsed = time.perf_counter() - started
                remaining = maximum_seconds - elapsed
                if remaining <= 0:
                    raise TimeoutError("bounded corpus wall-time budget exhausted")
                current_bytes = _directory_bytes(self.output_directory)
                block_worst = (end - cursor) * RECORD.size + HEADER.size + MANIFEST_ALLOWANCE_BYTES
                if current_bytes + block_worst > self.disk_budget:
                    raise ValueError("disk budget exhausted before Rust block generation")
                command = self.command_for(cursor, end, manifest_path)
                commands.append(self.portable_command_for(cursor, end, manifest_path))
                completed = subprocess.run(
                    command,
                    cwd=self.output_directory,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Rust generator failed ({completed.returncode}): {completed.stderr[-2000:]}"
                    )
                audit = verify_generated_manifest(
                    manifest_path,
                    self.output_directory,
                    self.generator_path,
                    expected_start=cursor,
                    expected_end=end,
                    equivalence_samples_per_block=int(
                        self.config["equivalence_samples_per_block"]
                    ),
                )
                generated += 1
            else:
                reused += 1
            verified.append(audit)
            state = {
                "schema_version": SCHEMA,
                "identity_sha256": self.identity_sha,
                "next_ordinal": end,
                "end_ordinal_exclusive": self.end,
                "verified_manifest_sha256": [item["manifest_sha256"] for item in verified],
            }
            _write_json_atomic(self.state_path, state)
            cursor = end
        elapsed = time.perf_counter() - started
        disk_bytes = _directory_bytes(self.output_directory)
        if disk_bytes > self.disk_budget:
            raise ValueError("completed corpus exceeded its disk budget")
        commit = hashlib.sha256(
            _canonical(
                {
                    "identity_sha256": self.identity_sha,
                    "blocks": [block for item in verified for block in item["blocks"]],
                }
            ).encode()
        ).hexdigest()
        report = {
            "schema_version": REPORT_SCHEMA,
            "identity_sha256": self.identity_sha,
            "corpus_root_sha256": commit,
            "generator_config_sha256": self.generator_sha,
            "generator_binary_sha256": self.binary_sha,
            "start_ordinal": self.start,
            "end_ordinal_exclusive": self.end,
            "formula_count": self.formula_count,
            "generated_blocks": generated,
            "reused_blocks": reused,
            "survivor_records": sum(item["record_count"] for item in verified),
            "survivor_file_bytes": sum(item["survivor_file_bytes"] for item in verified),
            "disk_bytes": disk_bytes,
            "disk_budget_bytes": self.disk_budget,
            "worst_case_preflight_bytes": self.worst_case_bytes,
            "elapsed_seconds": elapsed,
            "formulas_per_second_this_invocation": (
                self.formula_count / elapsed if generated and elapsed > 0 else None
            ),
            "commands_executed": commands,
            "verified_manifests": verified,
            "all_checks_passed": True,
            "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
        }
        portable_report = json.loads(json.dumps(report))
        for item in portable_report["verified_manifests"]:
            item["manifest_path"] = Path(item["manifest_path"]).name
        _write_json_atomic(self.output_directory / "corpus-report.json", portable_report)
        return report


def benchmark_cached_cuda_manifest(
    manifest_path: str | Path,
    survivor_directory: str | Path,
    generator_config_path: str | Path,
    execution_config: dict[str, Any],
    resource_profile: dict[str, Any],
    *,
    cached_repeats: int = 3,
    reference_records_per_second: float = 7_570_000.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bounded benchmark of the largest exported block after one CUDA warmup."""
    if not 1 <= cached_repeats <= 5:
        raise ValueError("cached CUDA repeats must be between one and five")
    manifest_path = Path(manifest_path).resolve()
    survivor_directory = Path(survivor_directory).resolve()
    generator_path = Path(generator_config_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formulas = int(manifest["processed_actions"])
    if not 1 <= formulas <= HARD_MAXIMUM_FORMULAS:
        raise ValueError("benchmark manifest exceeds the one-million-formula limit")
    audit = verify_generated_manifest(
        manifest_path,
        survivor_directory,
        generator_path,
        expected_start=int(manifest["start_ordinal"]),
        expected_end=int(manifest["end_ordinal_exclusive"]),
        equivalence_samples_per_block=64,
    )
    exports = [
        (position, int(block["survivor_export"]["record_count"]))
        for position, block in enumerate(manifest["blocks"])
        if block.get("survivor_export")
    ]
    if not exports or max(count for _, count in exports) <= 0:
        raise ValueError("benchmark manifest contains no survivor records")
    position, record_count = max(exports, key=lambda item: item[1])

    from .binary_formula_execution import (  # local import keeps builders independent
        BinaryBlockQueueRefill,
        cpu_binary_block_evaluator,
        gpu_binary_block_evaluator,
    )
    from .persistent_parallel_search import PersistentParallelSearch

    configured = json.loads(json.dumps(execution_config))
    configured["queue"] = {
        **configured["queue"],
        "maximum_pending_work": 1,
    }
    configured["budget"] = {**configured["budget"], "maximum_tasks": 1}
    configured["cpu"] = {**configured["cpu"], "maximum_workers": 1}
    configured["supervisor"] = {
        **configured["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 1,
    }
    adapter = {
        "schema_version": "sigma-binary-block-refill-1.0",
        "external_paid_llm_calls": False,
        "manifest_path": str(manifest_path),
        "survivor_directory": str(survivor_directory),
        "generator_config_path": str(generator_path),
        "start_export_block": position,
        "stop_export_block_exclusive": position + 1,
        "target_pending_blocks": 1,
        "lane_cycle": ["gpu"],
        "equivalence_samples_per_block": 64,
        "ambiguity_guard": 1e-10,
        "data_eligibility": ELIGIBILITY,
    }
    with tempfile.TemporaryDirectory(prefix="sigma-cuda-benchmark-") as temporary:
        coordinator = PersistentParallelSearch(
            Path(temporary) / "benchmark.sqlite", configured, resource_profile
        )
        refill = BinaryBlockQueueRefill(coordinator, adapter)
        if refill.refill()["accepted_blocks"] != 1:
            raise RuntimeError("benchmark block was not admitted")
        lease = coordinator.claim("gpu", "bounded-benchmark", lease_seconds=60)
        if lease is None:
            raise RuntimeError("benchmark GPU lease was not available")
        warmup = gpu_binary_block_evaluator(lease)
        cached = [gpu_binary_block_evaluator(lease) for _ in range(cached_repeats)]
        cpu = cpu_binary_block_evaluator(lease)
    roots = {warmup["status_root_sha256"], cpu["status_root_sha256"]}
    roots.update(result["status_root_sha256"] for result in cached)
    throughputs = [float(result["records_per_second"]) for result in cached]
    median = statistics.median(throughputs)
    report = {
        "schema_version": BENCHMARK_SCHEMA,
        "manifest_sha256": audit["manifest_sha256"],
        "block_index": int(manifest["blocks"][position]["block_index"]),
        "formula_count_in_manifest": formulas,
        "survivor_records_benchmarked": record_count,
        "cuda_initialization_seconds": float(warmup["elapsed_seconds"]),
        "cached_repeats": cached_repeats,
        "cached_elapsed_seconds": [float(result["elapsed_seconds"]) for result in cached],
        "cached_records_per_second": throughputs,
        "cached_median_records_per_second": median,
        "cached_best_records_per_second": max(throughputs),
        "reference_records_per_second": reference_records_per_second,
        "reference_reproduced_by_cached_median": median >= reference_records_per_second,
        "cuda_assets_reused_for_every_measured_run": all(
            result["cuda_assets_reused"] is True for result in cached
        ),
        "cpu_gpu_status_root_equal": len(roots) == 1,
        "cpu_gpu_counts_equal": all(result["counts"] == cpu["counts"] for result in cached),
        "ordinal_equivalence_samples_checked": int(
            cached[0]["ordinal_equivalence"]["checked"]
        ),
        "all_checks_passed": len(roots) == 1,
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    if output_path is not None:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output, report)
    return report
