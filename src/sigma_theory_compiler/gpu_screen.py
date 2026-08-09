from __future__ import annotations

import hashlib
import json
import struct
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

STATUS_HEADER = struct.Struct("<8sHHQQ32s")
STATUS_MAGIC = b"SGDNS2\0\0"


@dataclass(frozen=True)
class Jet2:
    value: float
    gd: float = 0.0
    gp: float = 0.0
    hdd: float = 0.0
    hdp: float = 0.0
    hpp: float = 0.0

    def add(self, other: Jet2) -> Jet2:
        return Jet2(
            self.value + other.value,
            self.gd + other.gd,
            self.gp + other.gp,
            self.hdd + other.hdd,
            self.hdp + other.hdp,
            self.hpp + other.hpp,
        )

    def mul(self, other: Jet2) -> Jet2:
        return Jet2(
            self.value * other.value,
            self.gd * other.value + self.value * other.gd,
            self.gp * other.value + self.value * other.gp,
            self.hdd * other.value + self.value * other.hdd + 2 * self.gd * other.gd,
            self.hdp * other.value
            + self.value * other.hdp
            + self.gd * other.gp
            + self.gp * other.gd,
            self.hpp * other.value + self.value * other.hpp + 2 * self.gp * other.gp,
        )

    def unary(self, value: float, first: float, second: float) -> Jet2:
        return Jet2(
            value,
            first * self.gd,
            first * self.gp,
            first * self.hdd + second * self.gd * self.gd,
            first * self.hdp + second * self.gd * self.gp,
            first * self.hpp + second * self.gp * self.gp,
        )

    def powu(self, power: int) -> Jet2:
        result = Jet2(1.0)
        for _ in range(power):
            result = result.mul(self)
        return result


def _term_hessian(
    term: dict[str, Any], d: float, p: float, state: float
) -> tuple[float, float, float]:
    x = Jet2(d, gd=1.0).powu(2)
    q = Jet2(p, gp=1.0).powu(2)
    z = Jet2(state * state)
    monomial = x.powu(term["px"]).mul(q.powu(term["pq"])).mul(z.powu(term["pz"]))
    if term["transform"] == "Identity":
        jet = monomial
    elif term["transform"] == "Sqrt1pMinus1":
        value = monomial.value + 1.0
        root = value**0.5
        jet = monomial.unary(root - 1.0, 0.5 / root, -0.25 / (value * root))
    elif term["transform"] == "Saturate":
        denominator = monomial.value + 1.0
        jet = monomial.unary(
            monomial.value / denominator,
            1.0 / denominator**2,
            -2.0 / denominator**3,
        )
    else:
        raise ValueError(f"Unknown transform: {term['transform']}")
    return jet.hdd, jet.hdp, jet.hpp


def dense_grid() -> dict[str, list[float]]:
    return {
        "d": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "p": [0.0, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0],
        "state": [0.0, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0],
    }


def precompute_dense_hessians(
    basis: list[dict[str, Any]], grid: dict[str, list[float]]
) -> np.ndarray:
    rows = []
    for term in basis:
        term_rows = []
        for d in grid["d"]:
            for p in grid["p"]:
                for state in grid["state"]:
                    term_rows.append(_term_hessian(term, d, p, state))
        rows.append(term_rows)
    return np.ascontiguousarray(rows, dtype=np.float64)


KERNEL = r"""
extern "C" __global__
void dense_screen(
    const unsigned short* term_ids,
    const unsigned char* term_counts,
    const unsigned char* sign_masks,
    const double* hessians,
    const int sample_count,
    const double coupling,
    const double tolerance_low,
    const double tolerance_high,
    unsigned char* statuses,
    unsigned short* fail_samples,
    double* margins,
    const int candidate_count) {
  int candidate = blockDim.x * blockIdx.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  unsigned char term_count = term_counts[candidate];
  unsigned char sign_mask = sign_masks[candidate];
  bool ambiguous = false;
  unsigned short ambiguous_sample = 65535;
  double minimum_margin = 1.0e300;
  for (int sample = 0; sample < sample_count; ++sample) {
    double hdd = 1.0;
    double hdp = 0.0;
    double hpp = 0.0;
    for (int position = 0; position < term_count; ++position) {
      unsigned short term_id = term_ids[candidate * 6 + position];
      double scale = ((sign_mask >> position) & 1) ? coupling : -coupling;
      long long offset = ((long long)term_id * sample_count + sample) * 3;
      hdd += scale * hessians[offset];
      hdp += scale * hessians[offset + 1];
      hpp += scale * hessians[offset + 2];
    }
    double delta = hdd - hpp;
    double discriminant = delta * delta + 4.0 * hdp * hdp;
    double minimum = 0.5 * (hdd + hpp - sqrt(discriminant > 0.0 ? discriminant : 0.0));
    if (!isfinite(minimum) || minimum <= tolerance_low) {
      statuses[candidate] = 0;
      fail_samples[candidate] = (unsigned short)sample;
      margins[candidate] = minimum;
      return;
    }
    if (minimum < minimum_margin) minimum_margin = minimum;
    if (minimum <= tolerance_high && !ambiguous) {
      ambiguous = true;
      ambiguous_sample = (unsigned short)sample;
    }
  }
  statuses[candidate] = ambiguous ? 2 : 1;
  fail_samples[candidate] = ambiguous ? ambiguous_sample : 65535;
  margins[candidate] = minimum_margin;
}
"""


def _survivor_dtype() -> np.dtype:
    return np.dtype(
        [
            ("ordinal", "<u8"),
            ("term_count", "u1"),
            ("sign_mask", "u1"),
            ("reserved", "<u2"),
            ("term_ids", "<u2", (6,)),
        ]
    )


def run_dense_gpu_screen(
    manifest_path: str | Path,
    basis_path: str | Path,
    config_path: str | Path,
    survivor_directory: str | Path,
    status_directory: str | Path,
    output: str | Path,
    ambiguity_guard: float = 1e-10,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CuPy is required for the GPU dense-static tier") from error
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    basis = json.loads(Path(basis_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    survivor_directory = Path(survivor_directory)
    status_directory = Path(status_directory)
    status_directory.mkdir(parents=True, exist_ok=True)
    grid = dense_grid()
    grid_payload = json.dumps(grid, sort_keys=True, separators=(",", ":")).encode()
    grid_hash = hashlib.sha256(grid_payload).digest()
    host_hessians = precompute_dense_hessians(basis, grid)
    device_hessians = cp.asarray(host_hessians)
    kernel = cp.RawKernel(KERNEL, "dense_screen")
    tolerance = float(config["convexity_tolerance"])
    low = tolerance - ambiguity_guard
    high = tolerance + ambiguity_guard
    counts = Counter()
    block_reports = []
    pass_margins: list[float] = []
    rejection_witnesses = []
    device = cp.cuda.Device()
    for block in manifest["blocks"]:
        export = block.get("survivor_export")
        if not export:
            continue
        source = survivor_directory / export["file"]
        records = np.fromfile(
            source, dtype=_survivor_dtype(), offset=44, count=export["record_count"]
        )
        term_ids = cp.asarray(np.ascontiguousarray(records["term_ids"]))
        term_counts = cp.asarray(records["term_count"])
        sign_masks = cp.asarray(records["sign_mask"])
        statuses = cp.empty(len(records), dtype=cp.uint8)
        fail_samples = cp.empty(len(records), dtype=cp.uint16)
        margins = cp.empty(len(records), dtype=cp.float64)
        threads = 256
        kernel(
            ((len(records) + threads - 1) // threads,),
            (threads,),
            (
                term_ids,
                term_counts,
                sign_masks,
                device_hessians,
                np.int32(host_hessians.shape[1]),
                np.float64(config["coupling_magnitude"]),
                np.float64(low),
                np.float64(high),
                statuses,
                fail_samples,
                margins,
                np.int32(len(records)),
            ),
        )
        host_statuses = cp.asnumpy(statuses)
        host_fail_samples = cp.asnumpy(fail_samples)
        host_margins = cp.asnumpy(margins)
        block_counts = Counter(map(int, host_statuses))
        counts.update(block_counts)
        passed = host_margins[host_statuses == 1]
        if len(passed):
            pass_margins.extend([float(passed.min()), float(np.median(passed))])
        if len(rejection_witnesses) < 64:
            for index in np.flatnonzero(host_statuses != 1)[: 64 - len(rejection_witnesses)]:
                rejection_witnesses.append(
                    {
                        "ordinal": int(records["ordinal"][index]),
                        "status": "reject" if host_statuses[index] == 0 else "ambiguous",
                        "sample_index": int(host_fail_samples[index]),
                        "minimum_eigenvalue": float(host_margins[index]),
                    }
                )
        filename = f"dense-status-{block['block_index']:08}.bin"
        target = status_directory / filename
        payload = (
            STATUS_HEADER.pack(
                STATUS_MAGIC,
                1,
                1,
                block["block_index"],
                len(records),
                grid_hash,
            )
            + host_statuses.tobytes()
        )
        target.write_bytes(payload)
        block_reports.append(
            {
                "block_index": block["block_index"],
                "source_survivor_file": export["file"],
                "source_file_sha256": export["file_sha256"],
                "status_file": filename,
                "status_file_sha256": hashlib.sha256(payload).hexdigest(),
                "records": len(records),
                "reject": block_counts[0],
                "pass": block_counts[1],
                "ambiguous": block_counts[2],
            }
        )
    device.synchronize()
    device_name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    report = {
        "schema_version": "sigma-dense-static-gpu-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_manifest": str(manifest_path),
        "source_survivor_count": manifest["survivor_count"],
        "basis": str(basis_path),
        "config": str(config_path),
        "status_directory": str(status_directory),
        "gpu": {
            "name": device_name,
            "cupy_version": cp.__version__,
            "compute_capability": device.compute_capability,
        },
        "grid": grid,
        "grid_point_count": int(host_hessians.shape[1]),
        "grid_sha256": grid_hash.hex(),
        "convexity_tolerance": tolerance,
        "coupling_magnitude": float(config["coupling_magnitude"]),
        "ambiguity_guard": ambiguity_guard,
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {"reject": counts[0], "pass": counts[1], "ambiguous": counts[2]},
        "accounting_pass": sum(counts.values()) == manifest["survivor_count"],
        "rejection_witnesses": rejection_witnesses,
        "pass_margin_block_minima_and_medians": {
            "minimum": min(pass_margins) if pass_margins else None,
            "median": float(np.median(pass_margins)) if pass_margins else None,
        },
        "blocks": block_reports,
        "interpretation": (
            "Pass means only that the candidate survived this 343-point sampled-static lattice. "
            "Ambiguous values are quarantined and never counted as passes. This is not a proof of global convexity."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def crosscheck_dense_gpu_screen(
    dense_report_path: str | Path,
    basis_path: str | Path,
    survivor_directory: str | Path,
    status_directory: str | Path,
    output: str | Path,
    sample_limit: int = 1024,
) -> dict[str, Any]:
    dense_report = json.loads(Path(dense_report_path).read_text(encoding="utf-8"))
    basis = json.loads(Path(basis_path).read_text(encoding="utf-8"))
    survivor_directory = Path(survivor_directory)
    status_directory = Path(status_directory)
    hessians = precompute_dense_hessians(basis, dense_report["grid"])
    tolerance = dense_report["convexity_tolerance"]
    guard = dense_report["ambiguity_guard"]
    low, high = tolerance - guard, tolerance + guard
    samples: list[dict[str, Any]] = []
    file_errors = [
        f"sha256:{block['status_file']}"
        for block in dense_report["blocks"]
        if hashlib.sha256((status_directory / block["status_file"]).read_bytes()).hexdigest()
        != block["status_file_sha256"]
    ]
    for block in dense_report["blocks"]:
        source = survivor_directory / block["source_survivor_file"]
        status_path = status_directory / block["status_file"]
        if f"sha256:{block['status_file']}" in file_errors:
            continue
        records = np.fromfile(source, dtype=_survivor_dtype(), offset=44, count=block["records"])
        statuses = np.fromfile(
            status_path, dtype=np.uint8, offset=STATUS_HEADER.size, count=block["records"]
        )
        indices = []
        for code in (0, 1, 2):
            matches = np.flatnonzero(statuses == code)
            if len(matches):
                indices.append(int(matches[0]))
        for index in indices:
            term_count = int(records["term_count"][index])
            term_ids = records["term_ids"][index][:term_count]
            signs = np.array(
                [
                    1.0 if int(records["sign_mask"][index]) & (1 << position) else -1.0
                    for position in range(term_count)
                ]
            )
            candidate = np.array([1.0, 0.0, 0.0]) + dense_report["coupling_magnitude"] * np.sum(
                hessians[term_ids] * signs[:, None, None], axis=0
            )
            hdd, hdp, hpp = candidate.T
            eigenvalues = 0.5 * (
                hdd + hpp - np.sqrt(np.maximum((hdd - hpp) ** 2 + 4 * hdp**2, 0.0))
            )
            minimum = float(np.min(eigenvalues))
            expected = (
                0 if not np.isfinite(minimum) or minimum <= low else (2 if minimum <= high else 1)
            )
            actual = int(statuses[index])
            samples.append(
                {
                    "ordinal": int(records["ordinal"][index]),
                    "gpu_status": actual,
                    "cpu_status": expected,
                    "cpu_minimum_eigenvalue": minimum,
                    "agree": actual == expected,
                }
            )
            if len(samples) >= sample_limit:
                break
        if len(samples) >= sample_limit:
            break
    report = {
        "schema_version": "sigma-dense-static-crosscheck-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "dense_report": str(dense_report_path),
        "sample_rule": "first available reject/pass/ambiguous record per ordered block",
        "sample_count": len(samples),
        "all_cpu_gpu_samples_agree": bool(samples) and all(row["agree"] for row in samples),
        "all_status_file_hashes_pass": not file_errors,
        "file_errors": file_errors,
        "samples": samples,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
