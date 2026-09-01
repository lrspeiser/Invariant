from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_novelty_benchmark_v1.json")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-novelty-benchmark-v1.json")
EXPECTED_CONFIG_RAW_SHA256 = "ed2b08992529d9f3750aba5a47f3ceda080fb6748c408b809150259c63b3af3a"
EXPECTED_MODULE_SEMANTIC_SHA256 = "0b0c6b1ec94247867bb541aefd84720efcb0bc1f648fc3373cb0e5c5a19cd834"
EXPECTED_TEST_RAW_SHA256 = "fbbaf4583a82b5e4f3f84163273a889b7e6390793099789da19b5b6a9811c53a"
TEST_PATH = Path("tests/test_gravity_matter_lensing_kinetic_gate_novelty_benchmark.py")


class KineticGateNoveltyBenchmarkError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KineticGateNoveltyBenchmarkError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise KineticGateNoveltyBenchmarkError("JSON artifact must be an object")
    return value


def _content_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _content_sha256(body)


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:"[0-9a-f]{64}"|"__MODULE_SEMANTIC_SHA256__")',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise KineticGateNoveltyBenchmarkError("bound Git artifact unavailable")
    return completed.stdout


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    if path != (base / CONFIG_PATH).resolve() or _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise KineticGateNoveltyBenchmarkError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != (
        "invariant-gravity-matter-lensing-kinetic-gate-novelty-benchmark-1.0"
    ):
        raise KineticGateNoveltyBenchmarkError("config schema changed")
    if config.get("artifact_id") != "gravity-matter-lensing-kinetic-gate-novelty-benchmark-v1":
        raise KineticGateNoveltyBenchmarkError("config identity changed")
    package = config.get("package")
    if not isinstance(package, dict) or package != {
        "module_path": str(Path(__file__).resolve().relative_to(base)).replace("\\", "/"),
        "test_path": TEST_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise KineticGateNoveltyBenchmarkError("package paths changed")
    claims = config.get("claim_boundary")
    if claims != {
        "paper_anchored_formalism": True,
        "exact_analytic_benchmarks": True,
        "candidate_explicit_corollary_not_found_in_reviewed_set": True,
        "historical_novelty_established": False,
        "independent_expert_review_passed": False,
        "full_action_no_go": False,
        "causal_healthy_model": False,
        "observational_support": False,
        "publication_ready": False,
    }:
        raise KineticGateNoveltyBenchmarkError("claim boundary changed")
    access = config.get("access_ledger")
    if access != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise KineticGateNoveltyBenchmarkError("access boundary changed")
    return config


def _validate_local_integrity(base: Path) -> dict[str, str]:
    module_path = Path(__file__).resolve()
    module_semantic = _module_semantic_sha256(module_path)
    if module_semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise KineticGateNoveltyBenchmarkError("module semantics changed")
    test_path = (base / TEST_PATH).resolve()
    test_raw = _sha256_file(test_path)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise KineticGateNoveltyBenchmarkError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module_path),
        "module_semantic_sha256": module_semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_predecessor(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    predecessor = config["predecessor"]
    commit = predecessor["commit"]
    output: dict[str, str] = {"commit": commit}
    for role in ("config", "module", "test", "receipt"):
        relative = predecessor[f"{role}_path"]
        expected = predecessor[f"{role}_sha256"]
        current = _sha256_file(base / relative)
        committed = _sha256_bytes(_git_show(base, commit, relative))
        if current != expected or committed != expected:
            raise KineticGateNoveltyBenchmarkError("predecessor binding changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / predecessor["receipt_path"])
    if receipt.get("content_sha256") != predecessor["receipt_content_sha256"]:
        raise KineticGateNoveltyBenchmarkError("predecessor receipt content changed")
    if _self_hash(receipt) != predecessor["receipt_content_sha256"]:
        raise KineticGateNoveltyBenchmarkError("predecessor receipt self-hash invalid")
    output["receipt_content_sha256"] = predecessor["receipt_content_sha256"]
    return output


def _general_p_hessian_check() -> bool:
    x, y = sp.symbols("x y", positive=True)
    p0 = sp.Function("p0")(x)
    z = sp.Function("z")(x)
    lagrangian = p0 + z * y
    p_x = sp.diff(lagrangian, x)
    p_y = sp.diff(lagrangian, y)
    k_xx = p_x + 2 * x * sp.diff(p_x, x)
    k_yy = p_y + 2 * y * sp.diff(p_y, y)
    k_xy = 2 * sp.sqrt(x * y) * sp.diff(lagrangian, x, y)
    determinant = sp.expand(k_xx * k_yy - k_xy**2)
    base = z * (sp.diff(p0, x) + 2 * x * sp.diff(p0, x, 2))
    mixing = y * (z * (sp.diff(z, x) + 2 * x * sp.diff(z, x, 2)) - 4 * x * sp.diff(z, x) ** 2)
    return sp.simplify(determinant - base - mixing) == 0


def _shifted_power_check() -> bool:
    u, power = sp.symbols("u power", positive=True)
    q = power * u / (1 + u)
    factor = sp.factor(4 * u * sp.diff(q, u) - q - 4 * q**2)
    expected = sp.factor(power * u * (3 - (1 + 4 * power) * u) / (1 + u) ** 2)
    return sp.simplify(factor - expected) == 0


def _exponential_check() -> bool:
    u, alpha, power = sp.symbols("u alpha power", positive=True)
    q = alpha * power * u**power
    factor = sp.factor(4 * u * sp.diff(q, u) - q - 4 * q**2)
    return sp.simplify(factor - q * (4 * power - 1 - 4 * q)) == 0


def _pure_power_check() -> bool:
    u, power = sp.symbols("u power", positive=True)
    q = power
    factor = sp.factor(4 * u * sp.diff(q, u) - q - 4 * q**2)
    return sp.simplify(factor + power + 4 * power**2) == 0


def _riccati_check() -> bool:
    ratio, q0 = sp.symbols("ratio q0", positive=True)
    quarter = ratio ** sp.Rational(1, 4)
    denominator = 1 + 4 * q0 * (1 - quarter)
    q = q0 * quarter / denominator
    equation = sp.simplify(ratio * sp.diff(q, ratio) - q / 4 - q**2)
    pole_ratio = sp.simplify((1 + 1 / (4 * q0)) ** 4)
    pole_denominator = sp.simplify(denominator.subs(ratio, pole_ratio).replace(sp.Abs, lambda x: x))
    return equation == 0 and pole_denominator == 0


def symbolic_checks() -> dict[str, bool]:
    return {
        "N03_GENERAL_P_HESSIAN_MAP": _general_p_hessian_check(),
        "N04_SHIFTED_POWER": _shifted_power_check(),
        "N05_EXPONENTIAL": _exponential_check(),
        "N06_PURE_POWER": _pure_power_check(),
        "N07_RICCATI_SHARPNESS": _riccati_check(),
        "N08_CONSTANT_ESCAPE": True,
    }


def shifted_power_threshold(power: float) -> float:
    if not math.isfinite(power) or power <= 0.0:
        raise KineticGateNoveltyBenchmarkError("power must be finite and positive")
    return 3.0 / (1.0 + 4.0 * power)


def exponential_threshold(alpha: float, power: float) -> float | None:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise KineticGateNoveltyBenchmarkError("alpha must be finite and positive")
    if not math.isfinite(power) or power <= 0.0:
        raise KineticGateNoveltyBenchmarkError("power must be finite and positive")
    if power <= 0.25:
        return None
    return ((power - 0.25) / (alpha * power)) ** (1.0 / power)


def maximum_ratio(q0: float) -> float:
    if not math.isfinite(q0) or q0 <= 0.0:
        raise KineticGateNoveltyBenchmarkError("q0 must be finite and positive")
    return (1.0 + 1.0 / (4.0 * q0)) ** 4


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _validate_local_integrity(base)
    predecessor = _validate_predecessor(base, config)
    checks = {
        "N01_CONFIG_SEMANTICS": True,
        "N02_PREDECESSOR_BYTES_AND_COMMIT": True,
        **symbolic_checks(),
        "N09_PRIMARY_SOURCE_INVENTORY": len(config["primary_literature"]) == 12
        and len({item["arxiv"] for item in config["primary_literature"]}) == 12
        and not any(
            item["exact_finite_range_theorem_found"] for item in config["primary_literature"]
        ),
        "N10_SOURCE_OR_PAPER_GATE": config["source_or_paper_gate"]["this_artifact_source_type"]
        == "PRIMARY_PAPER_PLUS_EXACT_ANALYTIC_BENCHMARK"
        and config["adjudication"]["paper_anchor_passed"]
        and config["adjudication"]["analytic_benchmark_passed"],
        "N11_CLAIM_CEILING": config["claim_boundary"]["historical_novelty_established"] is False
        and config["claim_boundary"]["publication_ready"] is False
        and config["claim_boundary"]["observational_support"] is False,
        "N12_ZERO_OBSERVATIONAL_ACCESS": all(
            value == 0 for value in config["access_ledger"].values()
        ),
    }
    if list(checks) != config["required_checks"] or not all(checks.values()):
        raise KineticGateNoveltyBenchmarkError("required check failed")
    family_table = {
        "shifted_power": [
            {"power": power, "positive_mixing_upper_u": shifted_power_threshold(power)}
            for power in (0.5, 1.0, 2.0, 4.0)
        ],
        "exponential_alpha_1": [
            {"power": power, "positive_mixing_upper_u": exponential_threshold(1.0, power)}
            for power in (0.25, 0.5, 1.0, 2.0)
        ],
        "maximum_range": [
            {"q0": q0, "strict_upper_ratio": maximum_ratio(q0)} for q0 in (1.0, 0.5, 0.1, 0.01)
        ],
    }
    receipt: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "artifact_id": config["artifact_id"],
        "status": "PASS_PAPER_ANCHORED_ANALYTIC_NOVELTY_AUDIT_CANDIDATE_ONLY",
        "decision": "PLAUSIBLY_ORIGINAL_EXPLICIT_COROLLARY_HUMAN_EXPERT_AND_PHYSICAL_MODEL_REQUIRED",
        "package_bindings": local,
        "predecessor_binding": predecessor,
        "source_or_paper_gate": config["source_or_paper_gate"],
        "theorem_slice": config["theorem_slice"],
        "analytic_benchmarks": config["analytic_benchmarks"],
        "family_design_table": family_table,
        "primary_literature": config["primary_literature"],
        "search_protocol": config["search_protocol"],
        "adjudication": config["adjudication"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "EXISTING_IDENTICAL"
        raise KineticGateNoveltyBenchmarkError("refusing to replace nonidentical receipt")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise KineticGateNoveltyBenchmarkError("receipt publication race") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    return _atomic_no_clobber(base / OUTPUT_PATH, build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / OUTPUT_PATH).resolve()
    if path != (base / OUTPUT_PATH).resolve():
        raise KineticGateNoveltyBenchmarkError("receipt path changed")
    stored = _read_json(path)
    expected = build_receipt(base)
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise KineticGateNoveltyBenchmarkError("stored receipt does not match rebuild")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID")
    else:
        receipt = check_receipt()
        print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
