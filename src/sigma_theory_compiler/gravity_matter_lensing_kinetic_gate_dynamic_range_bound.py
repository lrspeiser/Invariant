from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_dynamic_range_bound_v1.json")
CONFIG_CANONICAL_SHA256 = "02e0b50896490b001335990f2562ea8b18e30a046e07407897a6800af5ec599e"


class KineticGateDynamicRangeError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise KineticGateDynamicRangeError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["content_sha256"] = ""
    return _content_sha256(payload)


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = _read_json(base / CONFIG_PATH)
    if _content_sha256(config) != CONFIG_CANONICAL_SHA256:
        raise KineticGateDynamicRangeError("dynamic-range config semantics changed")
    if (
        config.get("schema_version")
        != "invariant-gravity-matter-lensing-kinetic-gate-dynamic-range-bound-1.0"
        or config.get("status") != "APPEND_ONLY_QUANTITATIVE_COROLLARY"
    ):
        raise KineticGateDynamicRangeError("unsupported dynamic-range config")
    for role in ("config", "module", "test", "receipt"):
        path = base / config["predecessor"][f"{role}_path"]
        expected = config["predecessor"][f"{role}_sha256"]
        if not path.is_file() or _sha256_file(path) != expected:
            raise KineticGateDynamicRangeError(f"predecessor {role} changed")
    predecessor = _read_json(base / config["predecessor"]["receipt_path"])
    if predecessor.get("content_sha256") != config["predecessor"]["receipt_content_sha256"]:
        raise KineticGateDynamicRangeError("predecessor receipt content changed")
    if config["claim_boundary"]["unconditional_action_no_go"] is not False:
        raise KineticGateDynamicRangeError("claim boundary changed")
    return config


def maximum_ratio(q0: float) -> float:
    if not math.isfinite(q0) or q0 <= 0.0:
        raise KineticGateDynamicRangeError("q0 must be finite and positive")
    return (1.0 + 1.0 / (4.0 * q0)) ** 4


def maximum_decades(q0: float) -> float:
    return math.log10(maximum_ratio(q0))


def maximum_initial_slope(required_ratio: float) -> float:
    if not math.isfinite(required_ratio) or required_ratio <= 1.0:
        raise KineticGateDynamicRangeError("required ratio must be finite and exceed one")
    return 1.0 / (4.0 * (required_ratio**0.25 - 1.0))


def comparison_q(q0: float, ratio: float) -> float:
    if ratio < 1.0 or ratio >= maximum_ratio(q0):
        raise KineticGateDynamicRangeError("comparison point lies outside finite branch")
    fourth_root = ratio**0.25
    return q0 * fourth_root / (1.0 + 4.0 * q0 * (1.0 - fourth_root))


def comparison_z_ratio(q0: float, ratio: float) -> float:
    if ratio < 1.0 or ratio >= maximum_ratio(q0):
        raise KineticGateDynamicRangeError("comparison point lies outside finite branch")
    return 1.0 / (1.0 + 4.0 * q0 * (1.0 - ratio**0.25))


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    slope_rows = []
    for q0 in config["frozen_design_points"]["initial_log_slopes"]:
        q_value = float(q0)
        slope_rows.append(
            {
                "q0": q_value,
                "maximum_ratio_strict_upper_boundary": maximum_ratio(q_value),
                "maximum_log10_decades_strict_upper_boundary": maximum_decades(q_value),
            }
        )
    range_rows = []
    for ratio in config["frozen_design_points"]["required_dynamic_ranges"]:
        ratio_value = float(ratio)
        range_rows.append(
            {
                "required_ratio": ratio_value,
                "maximum_allowed_initial_log_slope_strict": maximum_initial_slope(ratio_value),
            }
        )

    checks: dict[str, bool] = {}
    checks["D1_Q1_RANGE"] = math.isclose(maximum_ratio(1.0), 2.44140625)
    checks["D2_QHALF_RANGE"] = math.isclose(maximum_ratio(0.5), 5.0625)
    checks["D3_FOUR_DECADE_SLOPE"] = math.isclose(maximum_initial_slope(10000.0), 1.0 / 36.0)
    checks["D4_EIGHT_DECADE_SLOPE"] = math.isclose(maximum_initial_slope(100000000.0), 1.0 / 396.0)
    checks["D5_INVERSE_MAP"] = all(
        math.isclose(maximum_initial_slope(maximum_ratio(float(row))), float(row))
        for row in config["frozen_design_points"]["initial_log_slopes"]
    )
    checks["D6_COMPARISON_Q_POSITIVE"] = all(
        comparison_q(float(q0), math.sqrt(maximum_ratio(float(q0)))) > float(q0)
        for q0 in config["frozen_design_points"]["initial_log_slopes"]
    )
    checks["D7_COMPARISON_Z_GROWS"] = all(
        comparison_z_ratio(float(q0), math.sqrt(maximum_ratio(float(q0)))) > 1.0
        for q0 in config["frozen_design_points"]["initial_log_slopes"]
    )
    checks["D8_STRICT_MONOTONIC_TRADEOFF"] = all(
        slope_rows[index]["maximum_ratio_strict_upper_boundary"]
        < slope_rows[index + 1]["maximum_ratio_strict_upper_boundary"]
        for index in range(len(slope_rows) - 1)
    )
    checks["D9_CLAIM_CEILING"] = config["claim_boundary"][
        "quantitative_corollary_machine_verified"
    ] is True and all(
        config["claim_boundary"][key] is False
        for key in (
            "unconditional_action_no_go",
            "full_determinant_instability",
            "on_shell_background_excluded",
            "bounded_domain_gate_excluded",
            "alternative_architecture_excluded",
            "observational_support",
            "historical_novelty_established",
            "publication_ready",
        )
    )
    if not all(checks.values()):
        raise KineticGateDynamicRangeError("dynamic-range machine check failed")

    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-matter-lensing-kinetic-gate-dynamic-range-receipt-1.0",
        "analysis_id": config["analysis_id"],
        "status": "PASS_EXACT_QUANTITATIVE_DYNAMIC_RANGE_COROLLARY_SCOPE_RESTRICTED",
        "predecessor_receipt_content_sha256": config["predecessor"]["receipt_content_sha256"],
        "corollary": config["corollary"],
        "initial_slope_table": slope_rows,
        "required_range_table": range_rows,
        "machine_checks": checks,
        "machine_checks_passed": sum(checks.values()),
        "claim_boundary": config["claim_boundary"],
        "publication_value": config["publication_value"],
        "zero_access": config["zero_access"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() == encoded:
            return "EXISTING_IDENTICAL"
        raise KineticGateDynamicRangeError(f"refusing to replace existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    return _atomic_no_clobber(base / config["output_path"], build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    stored = _read_json(base / config["output_path"])
    expected = build_receipt(base)
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise KineticGateDynamicRangeError("stored dynamic-range receipt changed")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    else:
        receipt = check_receipt()
        if args.command == "check":
            print("VALID")
        else:
            print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
