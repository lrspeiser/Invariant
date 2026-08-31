from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_publication_candidate_v1.json")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-publication-candidate-v1.json")
CONFIG_CANONICAL_SHA256 = "ceb78ed6071576b67337252eb1a994b97363c8af4e22772194e066458fc57819"


class KineticGatePublicationCandidateError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise KineticGatePublicationCandidateError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload["content_sha256"] = ""
    return _content_sha256(payload)


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise KineticGatePublicationCandidateError(
            f"missing committed binding: {commit}:{relative}"
        )
    return result.stdout


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = _read_json(base / CONFIG_PATH)
    if _content_sha256(config) != CONFIG_CANONICAL_SHA256:
        raise KineticGatePublicationCandidateError("publication-candidate config changed")
    if (
        config.get("schema_version")
        != "invariant-gravity-matter-lensing-kinetic-gate-publication-candidate-1.0"
        or config.get("status") != "CANDIDATE_SHORT_THEORY_NOTE_NOT_PREPRINT_READY"
        or config.get("output_path") != OUTPUT_PATH.as_posix()
    ):
        raise KineticGatePublicationCandidateError("publication-candidate identity changed")
    if [item["id"] for item in config["bindings"]] != [
        "CONDITIONAL_THEOREM",
        "DYNAMIC_RANGE_COROLLARY",
        "BOUNDED_ON_SHELL_WITNESS",
        "CONSTANT_KINETIC_ESCAPE",
    ]:
        raise KineticGatePublicationCandidateError("binding inventory changed")
    if config["claim_boundary"] != {
        "candidate_original_mathematical_result": True,
        "historical_novelty_established": False,
        "independent_expert_review_passed": False,
        "full_action_no_go": False,
        "causal_healthy_cosmology": False,
        "observational_support": False,
        "modified_gravity_success": False,
        "publication_ready": False,
    }:
        raise KineticGatePublicationCandidateError("claim boundary changed")
    return config


def _validate_bindings(base: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            path = base / relative
            if not path.is_file() or _sha256_file(path) != expected:
                raise KineticGatePublicationCandidateError(
                    f"working-tree binding changed: {binding['id']} {role}"
                )
            if _sha256_bytes(_git_show(base, binding["commit"], relative)) != expected:
                raise KineticGatePublicationCandidateError(
                    f"commit binding changed: {binding['id']} {role}"
                )
        receipt = _read_json(base / binding["receipt_path"])
        if receipt.get("content_sha256") != binding["receipt_content_sha256"]:
            raise KineticGatePublicationCandidateError(
                f"receipt content binding changed: {binding['id']}"
            )
        receipts[binding["id"]] = receipt
    return receipts


def maximum_ratio(q0: float) -> float:
    if not math.isfinite(q0) or q0 <= 0.0:
        raise KineticGatePublicationCandidateError("q0 must be finite and positive")
    return (1.0 + 1.0 / (4.0 * q0)) ** 4


def shifted_power_threshold(power: float) -> float:
    if not math.isfinite(power) or power <= 0.0:
        raise KineticGatePublicationCandidateError("power must be finite and positive")
    return 3.0 / (1.0 + 4.0 * power)


def symbolic_checks() -> dict[str, bool]:
    x, beta = sp.symbols("X beta", positive=True)
    u_symbol = sp.symbols("u", positive=True)
    w = sp.Function("w")
    u = beta * x**2
    z = sp.exp(w(u))
    zx = sp.diff(z, x)
    zxx = sp.diff(zx, x)
    q = u_symbol * sp.diff(w(u_symbol), u_symbol)
    q_dot = u_symbol * sp.diff(q, u_symbol)
    expected_m = (2 * z**2 / x) * (4 * q_dot - q - 4 * q**2).subs(u_symbol, u)
    actual_m = z * (zx + 2 * x * zxx) - 4 * x * zx**2

    t, t0, q0 = sp.symbols("t t0 q0", positive=True)
    growth = sp.exp((t - t0) / 4)
    denominator = 1 + 4 * q0 * (1 - growth)
    comparison = q0 * growth / denominator
    blowup = t0 + 4 * sp.log(1 + 1 / (4 * q0))

    power = sp.symbols("p", positive=True)
    u_power = sp.symbols("u_power", positive=True)
    z_power = (1 + u_power) ** power
    w_power = sp.log(z_power)
    power_bracket = sp.simplify(
        3 * sp.diff(w_power, u_power)
        + 4 * u_power * sp.diff(w_power, u_power, 2)
        - 4 * u_power * sp.diff(w_power, u_power) ** 2
    )
    expected_power = power * (3 - (1 + 4 * power) * u_power) / (1 + u_power) ** 2

    return {
        "P02_M_IDENTITY": sp.simplify(actual_m - expected_m) == 0,
        "P03_RICCATI_SOLUTION": sp.simplify(sp.diff(comparison, t) - comparison / 4 - comparison**2)
        == 0,
        "P04_FINITE_RANGE_BOUND": sp.simplify(denominator.subs(t, blowup)) == 0,
        "P05_SHIFTED_POWER_THRESHOLD": sp.simplify(power_bracket - expected_power) == 0,
    }


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    config = load_config(base)
    bound = _validate_bindings(base, config)
    witness = bound["BOUNDED_ON_SHELL_WITNESS"]
    dynamic = bound["DYNAMIC_RANGE_COROLLARY"]
    symbolic = symbolic_checks()
    witness_extrema = witness["witness"]["health_extrema"]

    checks = {
        "P01_EXACT_BINDINGS": True,
        **symbolic,
        "P06_ON_SHELL_WITNESS": witness["claim_boundary"][
            "coupled_metric_and_scalar_background_on_shell"
        ]
        is True
        and witness_extrema["u_max"] < 1.0 / 3.0
        and witness_extrema["kinetic_min_eigenvalue"] > 0.0
        and witness_extrema["sound_speed_squared_min"] > 0.0,
        "P07_EINSTEIN_CONSTRAINT_MAPPING": witness["checks"]["W21_GENERAL_MULTIFIELD_ADM_MAPPING"]
        is True
        and witness["claim_boundary"]["einstein_constrained_high_frequency_scalar_principal_block"]
        is True,
        "P08_SUPERLUMINAL_WARNING": witness_extrema["sound_speed_squared_max"] > 1.0
        and witness["claim_boundary"]["metric_cone_subluminality"] is False,
        "P09_ESCAPE_OUTSIDE_HYPOTHESES": config["counterexample_pair"][
            "structural_escape"
        ].startswith("A constant positive chi kinetic coefficient")
        and config["claim_boundary"]["full_action_no_go"] is False,
        "P10_LITERATURE_SCOPE": len(config["primary_literature_positioning"]) == 4
        and len({item["arxiv"] for item in config["primary_literature_positioning"]}) == 4
        and config["claim_boundary"]["historical_novelty_established"] is False,
        "P11_CLAIM_CEILING": config["claim_boundary"]["candidate_original_mathematical_result"]
        is True
        and all(
            config["claim_boundary"][key] is False
            for key in (
                "historical_novelty_established",
                "independent_expert_review_passed",
                "full_action_no_go",
                "causal_healthy_cosmology",
                "observational_support",
                "modified_gravity_success",
                "publication_ready",
            )
        ),
    }
    if set(checks) != set(config["required_checks"]):
        raise KineticGatePublicationCandidateError("required check inventory changed")
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise KineticGatePublicationCandidateError(f"publication checks failed: {failed}")

    design_table = [
        {
            "q0": q0,
            "strict_maximum_U_over_u0": maximum_ratio(q0),
            "strict_maximum_decades": math.log10(maximum_ratio(q0)),
        }
        for q0 in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    ]
    power_table = [
        {"power": power, "positive_M_upper_u": shifted_power_threshold(power)}
        for power in (0.5, 1.0, 2.0, 4.0)
    ]
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-matter-lensing-kinetic-gate-publication-candidate-receipt-1.0",
        "analysis_id": config["analysis_id"],
        "status": "PROMISING_ORIGINAL_THEOREM_CANDIDATE_NOT_PREPRINT_READY",
        "decision": config["publication_adjudication"]["decision"],
        "implementation_binding": {
            "module_path": config["implementation"]["module_path"],
            "module_sha256": _sha256_file(base / config["implementation"]["module_path"]),
            "test_path": config["implementation"]["test_path"],
            "test_sha256": _sha256_file(base / config["implementation"]["test_path"]),
        },
        "binding_receipt_content_sha256": {
            binding["id"]: binding["receipt_content_sha256"] for binding in config["bindings"]
        },
        "maximal_theorem": config["maximal_theorem"],
        "dynamic_range_design_table": design_table,
        "shifted_power_table": power_table,
        "witness_summary": {
            "status": witness["status"],
            "u_min": witness_extrema["u_min"],
            "u_max": witness_extrema["u_max"],
            "kinetic_min_eigenvalue": witness_extrema["kinetic_min_eigenvalue"],
            "sound_speed_squared_min": witness_extrema["sound_speed_squared_min"],
            "sound_speed_squared_max": witness_extrema["sound_speed_squared_max"],
            "trajectory_sha256": witness["witness"]["trajectory_sha256"],
        },
        "corollary_status": dynamic["status"],
        "counterexample_pair": config["counterexample_pair"],
        "primary_literature_positioning": config["primary_literature_positioning"],
        "publication_adjudication": config["publication_adjudication"],
        "draft_abstract": config["draft_abstract"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "claim_boundary": config["claim_boundary"],
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
        raise KineticGatePublicationCandidateError(f"refusing to replace artifact: {path}")
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
    return _atomic_no_clobber(base / OUTPUT_PATH, build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = _repo_root() if root is None else root.resolve()
    stored = _read_json(base / OUTPUT_PATH)
    expected = build_receipt(base)
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise KineticGatePublicationCandidateError("stored publication receipt changed")
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
