"""Publication adjudication for two exact derivative-gate obstruction theorems."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_publication_candidate_v2.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_kinetic_gate_publication_candidate_v2.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_kinetic_gate_publication_candidate_v2.py")
DRAFT_PATH = Path("docs/GRAVITY_KINETIC_GATE_TWO_OBSTRUCTIONS_THEORY_NOTE_V1.md")
POLICY_PATH = Path("docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-publication-candidate-v2.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-publication-candidate-2.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-publication-candidate-receipt-2.0"
DECISION = (
    "STRONG_NARROW_TWO_THEOREM_NOTE_CANDIDATE_PENDING_INDEPENDENT_EXPERT_AND_"
    "HISTORICAL_NOVELTY_REVIEW"
)

EXPECTED_CONFIG_RAW_SHA256 = "161253bdc1b60fdd2df5c9adda55c57cb16090390dffb153d19a7871bb41e978"
EXPECTED_MODULE_SEMANTIC_SHA256 = "e0e130d2762176ec029dc2473da911f9ac1de5c26d054b35db78af2bb5671110"
EXPECTED_TEST_RAW_SHA256 = "8472e9c46b16189153ced67f0f357e6909ca076d5bfb09499dd93687002832fe"


class KineticGatePublicationCandidateV2Error(RuntimeError):
    """Raised when a theorem, provenance, or publication gate fails."""


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


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical_bytes(body))


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:"[0-9a-f]{64}"|"__MODULE_SEMANTIC_SHA256__")',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KineticGatePublicationCandidateV2Error("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise KineticGatePublicationCandidateV2Error("JSON artifact must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KineticGatePublicationCandidateV2Error(message)


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise KineticGatePublicationCandidateV2Error("committed binding unavailable")
    return completed.stdout


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "GRAVITY-MATTER-LENSING-KINETIC-GATE-PUBLICATION-CANDIDATE-v2",
        "analysis identity changed",
    )
    _require(
        config.get("status") == "STRONG_NARROW_TWO_THEOREM_NOTE_CANDIDATE_PENDING_EXPERT_REVIEW",
        "status changed",
    )
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "draft_path": DRAFT_PATH.as_posix(),
            "draft_sha256": "abafb3bf03a6ed2746f20f43cc541454b2b3a1c938deed004825c59c3530fb9d",
            "output_path": OUTPUT_PATH.as_posix(),
        },
        "package paths or draft binding changed",
    )
    policy = config.get("admission_policy")
    _require(
        policy
        == {
            "path": POLICY_PATH.as_posix(),
            "sha256": "b3518291131b0ece9f05c966e55b40c40e549d5c27e87a2a128b70ad76a864fe",
            "theory_note_gate": (
                "PRIMARY_PAPERS_PLUS_INDEPENDENT_EXACT_ANALYTIC_AND_NUMERICAL_BENCHMARKS"
            ),
            "future_observational_gate": (
                "REAL_PUBLIC_SOURCE_DATASET_PLUS_PRIMARY_DATASET_PAPER_PLUS_INDEPENDENT_"
                "SOLVER_BENCHMARK_REQUIRED_BEFORE_RESPONSE_SCORING"
            ),
        },
        "admission policy changed",
    )
    bindings = config.get("bindings")
    _require(
        isinstance(bindings, list)
        and [item["id"] for item in bindings]
        == ["PUBLICATION_CANDIDATE_V1", "NOVELTY_BENCHMARK_V1", "CONE_STRADDLING_V1"],
        "binding inventory changed",
    )
    _require(
        bindings[0]["state"] == "COMMITTED"
        and bindings[0]["commit"] == "8d50d004"
        and all(
            item["state"] == "MUTATION_FROZEN_UNCOMMITTED" and item["commit"] is None
            for item in bindings[1:]
        ),
        "binding states changed",
    )
    core = config.get("two_theorem_core")
    _require(
        core["theorem_1"]["identity"] == "det(K-G)=-4 X Y Z_X^2<0"
        and core["theorem_1"]["conclusion"]
        == "the two generalized scalar speeds obey 0<c_-^2<1<c_+^2"
        and core["theorem_2"]["conclusion"] == "U/u0<[1+1/(4q0)]^4",
        "two-theorem core changed",
    )
    papers = config.get("primary_literature_boundary")
    _require(
        isinstance(papers, list)
        and len(papers) == 8
        and len({item["arxiv"] for item in papers}) == 8,
        "primary-literature boundary changed",
    )
    novelty = config.get("novelty_protocol")
    _require(
        novelty["exact_two_theorem_pair_found"] is False
        and novelty["historical_novelty_established"] is False,
        "novelty overclaimed",
    )
    adjudication = config.get("publication_adjudication")
    _require(
        adjudication["scientifically_interesting"] is True
        and adjudication["worth_preparing_as_narrow_theory_note"] is True
        and adjudication["worth_claiming_as_successful_gravity_model"] is False
        and adjudication["decision"] == DECISION,
        "publication adjudication changed",
    )
    _require(
        config.get("claim_boundary")
        == {
            "exact_two_theorem_pair_established": True,
            "scientifically_interesting_theory_note_candidate": True,
            "worth_independent_expert_review": True,
            "historical_novelty_established": False,
            "unconditional_causality_violation": False,
            "global_strong_hyperbolicity": False,
            "full_action_no_go": False,
            "healthy_modified_gravity_model": False,
            "observational_support": False,
            "publication_ready": False,
        },
        "claim boundary changed",
    )
    _require(
        config.get("access_ledger")
        == {
            "observational_files_opened": 0,
            "observational_rows_read": 0,
            "scores_computed": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access ledger changed",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    _require(_sha256_file(path) == EXPECTED_CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_local_integrity(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    module = (base / MODULE_PATH).resolve()
    test = (base / TEST_PATH).resolve()
    draft = (base / DRAFT_PATH).resolve()
    policy = (base / POLICY_PATH).resolve()
    _require(module == Path(__file__).resolve(), "module path changed")
    module_semantic = _module_semantic_sha256(module)
    _require(module_semantic == EXPECTED_MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(_sha256_file(test) == EXPECTED_TEST_RAW_SHA256, "test bytes changed")
    _require(_sha256_file(draft) == config["package"]["draft_sha256"], "draft bytes changed")
    _require(_sha256_file(policy) == config["admission_policy"]["sha256"], "policy bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": module_semantic,
        "test_raw_sha256": _sha256_file(test),
        "draft_raw_sha256": _sha256_file(draft),
        "admission_policy_raw_sha256": _sha256_file(policy),
    }


def _validate_bindings(base: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            _require(_sha256_file(base / relative) == expected, "bound worktree artifact changed")
            if binding["state"] == "COMMITTED":
                _require(
                    _sha256_bytes(_git_show(base, binding["commit"], relative)) == expected,
                    "bound committed artifact changed",
                )
        receipt = _read_json(base / binding["receipt_path"])
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            "bound receipt content changed",
        )
        _require(
            _self_hash(receipt) == binding["receipt_content_sha256"],
            "bound receipt self-hash invalid",
        )
        receipts[binding["id"]] = receipt
    return receipts


def symbolic_checks() -> dict[str, bool]:
    x, y = sp.symbols("X Y", positive=True)
    c, z, zx, zxx, p0xx, speed = sp.symbols("C Z Z_X Z_XX P0_XX s", real=True)
    k = sp.Matrix(
        [[c + 2 * x * (p0xx + y * zxx), 2 * zx * sp.sqrt(x * y)], [2 * zx * sp.sqrt(x * y), z]]
    )
    g = sp.diag(c, z)
    polynomial = sp.expand((g - speed * k).det())

    u = sp.symbols("u", positive=True)
    w = sp.Function("w")
    q = u * sp.diff(w(u), u)
    q_dot = u * sp.diff(q, u)
    bracket = sp.simplify(
        u * (3 * sp.diff(w(u), u) + 4 * u * sp.diff(w(u), u, 2) - 4 * u * sp.diff(w(u), u) ** 2)
    )
    t, t0, q0 = sp.symbols("t t0 q0", positive=True)
    growth = sp.exp((t - t0) / 4)
    denominator = 1 + 4 * q0 * (1 - growth)
    comparison = q0 * growth / denominator
    blowup = t0 + 4 * sp.log(1 + 1 / (4 * q0))
    checks = {
        "cone_determinant": sp.simplify((k - g).det() + 4 * x * y * zx**2) == 0,
        "cone_polynomial_at_zero": sp.simplify(polynomial.subs(speed, 0) - c * z) == 0,
        "cone_polynomial_at_one": sp.simplify(polynomial.subs(speed, 1) + 4 * x * y * zx**2) == 0,
        "cone_positive_leading_coefficient": sp.simplify(sp.Poly(polynomial, speed).LC() - k.det())
        == 0,
        "dynamic_range_log_slope_identity": sp.simplify(bracket - (4 * q_dot - q - 4 * q**2)) == 0,
        "riccati_equality_solution": sp.simplify(
            sp.diff(comparison, t) - comparison / 4 - comparison**2
        )
        == 0,
        "riccati_finite_pole": sp.simplify(denominator.subs(t, blowup)) == 0,
    }
    _require(all(checks.values()), "independent symbolic benchmark failed")
    return checks


def _validate_draft(base: Path) -> dict[str, bool]:
    text = (base / DRAFT_PATH).read_text(encoding="utf-8")
    checks = {
        "contains_cone_theorem": "Theorem 1: unavoidable metric-cone straddling" in text,
        "contains_cone_identity": "\\det(K-G)=-4XYZ_X^2<0" in text,
        "contains_dynamic_range_theorem": "Theorem 2: finite healthy-mixing range" in text,
        "contains_dynamic_range_bound": "\\frac{U}{u_0}<\\left(1+\\frac{1}{4q_0}\\right)^4" in text,
        "contains_causality_caveat": "does not, by itself, say" in text,
        "contains_source_policy": "real public source dataset" in text,
    }
    _require(all(checks.values()), "draft proof or scope section missing")
    return checks


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _validate_local_integrity(base, config)
    bound = _validate_bindings(base, config)
    symbolic = symbolic_checks()
    draft_checks = _validate_draft(base)

    v1 = bound["PUBLICATION_CANDIDATE_V1"]
    novelty = bound["NOVELTY_BENCHMARK_V1"]
    cone = bound["CONE_STRADDLING_V1"]
    witness_speed = float(v1["witness_summary"]["sound_speed_squared_max"])
    cone_speeds = [
        float(value) for value in cone["numeric_benchmarks"][0]["generalized_speed_squared"]
    ]
    witness_match = math.isclose(witness_speed, cone_speeds[1], rel_tol=2.0e-15, abs_tol=2.0e-15)
    escape_scope = (
        v1["counterexample_pair"]["structural_escape"].startswith(
            "A constant positive chi kinetic coefficient"
        )
        and config["claim_boundary"]["full_action_no_go"] is False
    )
    paper_arxiv = {item["arxiv"] for item in config["primary_literature_boundary"]}
    novelty_arxiv = {item["arxiv"] for item in novelty["primary_literature"]}
    checks = {
        "V201_PACKAGE_AND_POLICY_SEALS": True,
        "V202_BOUND_ARTIFACT_BYTES": True,
        "V203_V1_COMMIT_BINDING": config["bindings"][0]["state"] == "COMMITTED",
        "V204_CONE_DETERMINANT_IDENTITY": symbolic["cone_determinant"],
        "V205_CONE_ROOT_PLACEMENT": all(
            symbolic[key]
            for key in (
                "cone_polynomial_at_zero",
                "cone_polynomial_at_one",
                "cone_positive_leading_coefficient",
            )
        )
        and all(
            0.0
            < float(row["generalized_speed_squared"][0])
            < 1.0
            < float(row["generalized_speed_squared"][1])
            for row in cone["numeric_benchmarks"]
        ),
        "V206_DYNAMIC_RANGE_IDENTITY": symbolic["dynamic_range_log_slope_identity"],
        "V207_RICCATI_FINITE_BOUND": symbolic["riccati_equality_solution"]
        and symbolic["riccati_finite_pole"],
        "V208_BOUNDED_WITNESS_MATCH": witness_match,
        "V209_ESCAPE_ARCHITECTURE_SCOPE": escape_scope,
        "V210_PRIMARY_PAPER_BOUNDARY": len(paper_arxiv) == 8
        and paper_arxiv.issubset(novelty_arxiv | {"1510.01650", "0708.0561", "gr-qc/0607055"})
        and config["novelty_protocol"]["historical_novelty_established"] is False,
        "V211_DRAFT_CONTAINS_BOTH_PROOFS": all(draft_checks.values()),
        "V212_PUBLICATION_ADJUDICATION": config["publication_adjudication"][
            "scientifically_interesting"
        ]
        and config["publication_adjudication"]["worth_preparing_as_narrow_theory_note"]
        and not config["publication_adjudication"]["worth_claiming_as_successful_gravity_model"],
        "V213_CLAIM_CEILING": config["claim_boundary"]["exact_two_theorem_pair_established"]
        and config["claim_boundary"]["worth_independent_expert_review"]
        and all(
            config["claim_boundary"][key] is False
            for key in (
                "historical_novelty_established",
                "unconditional_causality_violation",
                "global_strong_hyperbolicity",
                "full_action_no_go",
                "healthy_modified_gravity_model",
                "observational_support",
                "publication_ready",
            )
        ),
        "V214_ZERO_OBSERVATIONAL_ACCESS": all(
            value == 0 for value in config["access_ledger"].values()
        ),
    }
    _require(list(checks) == config["required_checks"], "required check inventory changed")
    _require(all(checks.values()), "publication adjudication check failed")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "PASS_STRONG_NARROW_TWO_THEOREM_NOTE_CANDIDATE_NOT_PUBLICATION_READY",
        "decision": DECISION,
        "package_bindings": local,
        "artifact_binding_states": {item["id"]: item["state"] for item in config["bindings"]},
        "artifact_receipt_content_sha256": {
            item["id"]: item["receipt_content_sha256"] for item in config["bindings"]
        },
        "admission_policy": config["admission_policy"],
        "two_theorem_core": config["two_theorem_core"],
        "symbolic_checks": symbolic,
        "cone_numeric_speed_squared": [
            row["generalized_speed_squared"] for row in cone["numeric_benchmarks"]
        ],
        "bounded_witness_match": {
            "v1_max_speed_squared": format(witness_speed, ".17g"),
            "cone_theorem_case_max_speed_squared": format(cone_speeds[1], ".17g"),
            "matched": witness_match,
        },
        "primary_literature_boundary": config["primary_literature_boundary"],
        "novelty_protocol": config["novelty_protocol"],
        "publication_adjudication": config["publication_adjudication"],
        "draft_abstract": config["draft_abstract"],
        "draft_checks": draft_checks,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "remaining_before_preprint": [
            "independent expert verification of both proofs and principal matrices",
            "broader historical novelty review with citation chaining",
            "joint scalar-metric-matter causal-cone statement",
            "variable-background strong-hyperbolicity and lower-order analysis",
            "radiative-stability and cutoff analysis for any proposed realization",
        ],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("analysis_id") == config["analysis_id"], "receipt identity changed")
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(receipt.get("checks_passed") == receipt.get("checks_total") == 14, "checks incomplete")
    _require(receipt.get("two_theorem_core") == config["two_theorem_core"], "theorem core changed")
    _require(
        receipt.get("publication_adjudication") == config["publication_adjudication"],
        "publication adjudication changed",
    )
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    _require(receipt.get("access_ledger") == config["access_ledger"], "access ledger changed")
    _require(receipt.get("content_sha256") == _self_hash(receipt), "receipt self-hash invalid")


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "EXISTING_IDENTICAL"
        raise KineticGatePublicationCandidateV2Error("refusing to replace nonidentical receipt")
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
            raise KineticGatePublicationCandidateV2Error("receipt publication race") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    return _atomic_no_clobber(base / OUTPUT_PATH, build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    stored = _read_json((base / OUTPUT_PATH).resolve())
    expected = build_receipt(base)
    _require(stored == expected, "stored receipt does not match deterministic rebuild")
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
        print(check_receipt()["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
