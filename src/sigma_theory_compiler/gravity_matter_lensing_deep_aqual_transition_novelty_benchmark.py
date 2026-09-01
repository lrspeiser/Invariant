from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_deep_aqual_transition_novelty_benchmark_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_deep_aqual_transition_novelty_benchmark.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_deep_aqual_transition_novelty_benchmark.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-deep-aqual-transition-novelty-benchmark-v1.json"
)

EXPECTED_CONFIG_RAW_SHA256 = "c59df786150538a22108de8c313efcf9688450c24b0ad6fb9452d1e7c784a59d"
EXPECTED_MODULE_SEMANTIC_SHA256 = "a2df9da2d4055a78d4fc60f469593b4c1e6bee17b7a948f8d4514f7f75cab821"
EXPECTED_TEST_RAW_SHA256 = "1c17f9a7792811a084de2d1092c251da9021ec37edc1e45002ba102898040057"

SCHEMA = "invariant-gravity-matter-lensing-deep-aqual-transition-novelty-benchmark-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-matter-lensing-deep-aqual-transition-novelty-benchmark-receipt-1.0"
)
ARTIFACT_ID = "gravity-matter-lensing-deep-aqual-transition-novelty-benchmark-v1"
DECISION = (
    "KNOWN_RAQUAL_K_ESSENCE_DEGENERACY_SPECIALIZATION_USE_AS_DESIGN_CONSTRAINT_"
    "NOT_STANDALONE_PUBLICATION_CANDIDATE"
)


class DeepAqualNoveltyError(RuntimeError):
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
        raise DeepAqualNoveltyError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise DeepAqualNoveltyError("JSON artifact must be an object")
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
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:(?:"[0-9a-f]{64}")|(?:"__MODULE_SEMANTIC_SHA256__"))',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _git_show(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise DeepAqualNoveltyError("bound Git artifact unavailable")
    return completed.stdout


def _expected_claim_boundary() -> dict[str, bool]:
    return {
        "primary_paper_anchored": True,
        "independent_exact_benchmarks": True,
        "transition_obstruction_rederived": True,
        "static_aqual_invalidated": False,
        "mathematical_substance_preexisting": True,
        "exact_verbatim_duplicate_found": False,
        "historical_novelty_established": False,
        "standalone_publication_candidate": False,
        "useful_internal_design_constraint": True,
        "regulated_counterexample_established": True,
        "regulator_phenomenologically_derived": False,
        "full_action_health": False,
        "observational_support": False,
        "modified_gravity_success": False,
        "publication_ready": False,
    }


def load_config() -> dict[str, Any]:
    path = _repo_root() / CONFIG_PATH
    if _sha256_file(path) != EXPECTED_CONFIG_RAW_SHA256:
        raise DeepAqualNoveltyError("config semantics changed")
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA or config.get("artifact_id") != ARTIFACT_ID:
        raise DeepAqualNoveltyError("config identity changed")
    if config.get("package") != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "output_path": OUTPUT_PATH.as_posix(),
    }:
        raise DeepAqualNoveltyError("package paths changed")
    if config.get("claim_boundary") != _expected_claim_boundary():
        raise DeepAqualNoveltyError("claim boundary changed")
    if config.get("access_ledger") != {
        "observational_files_opened": 0,
        "observational_rows_read": 0,
        "scores_computed": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }:
        raise DeepAqualNoveltyError("access boundary changed")
    papers = config.get("primary_literature")
    expected_ids = {
        "DOI:10.1086/162570",
        "astro-ph/0403694",
        "astro-ph/0512425",
        "gr-qc/0607055",
        "0705.4043",
        "0708.0561",
        "hep-th/9904176",
    }
    if not isinstance(papers, list) or len(papers) != len(expected_ids):
        raise DeepAqualNoveltyError("primary-source inventory changed")
    if {item.get("id") for item in papers} != expected_ids:
        raise DeepAqualNoveltyError("primary-source identities changed")
    direct = [item for item in papers if item.get("overlap") == "DIRECT_MATHEMATICAL_SUBSTANCE"]
    if len(direct) != 1 or direct[0].get("id") != "0705.4043":
        raise DeepAqualNoveltyError("direct prior-art overlap changed")
    if any(item.get("exact_wording_found") is not False for item in papers):
        raise DeepAqualNoveltyError("verbatim novelty adjudication changed")
    if config["adjudication"]["mathematical_substance_preexisting"] is not True:
        raise DeepAqualNoveltyError("prior-art verdict changed")
    if config["adjudication"]["publication_value"] != (
        "DO_NOT_PROMOTE_AS_STANDALONE_NOTE_USE_AS_CITED_DESIGN_CONSTRAINT"
    ):
        raise DeepAqualNoveltyError("publication verdict changed")
    return config


def _validate_local_integrity() -> dict[str, str]:
    root = _repo_root()
    module_semantic = _module_semantic_sha256(root / MODULE_PATH)
    if module_semantic != EXPECTED_MODULE_SEMANTIC_SHA256:
        raise DeepAqualNoveltyError("module semantics changed")
    test_raw = _sha256_file(root / TEST_PATH)
    if test_raw != EXPECTED_TEST_RAW_SHA256:
        raise DeepAqualNoveltyError("test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(root / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(root / MODULE_PATH),
        "module_semantic_sha256": module_semantic,
        "test_raw_sha256": test_raw,
    }


def _validate_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    predecessor = config["predecessor"]
    root = _repo_root()
    commit = str(predecessor["commit"])
    output: dict[str, str] = {"commit": commit}
    for role in ("config", "module", "test", "receipt"):
        relative = str(predecessor[f"{role}_path"])
        expected = str(predecessor[f"{role}_sha256"])
        current = _sha256_file(root / relative)
        committed = _sha256_bytes(_git_show(commit, relative))
        if current != expected or committed != expected:
            raise DeepAqualNoveltyError("predecessor binding changed")
        output[f"{role}_sha256"] = expected
    receipt = _read_json(root / str(predecessor["receipt_path"]))
    expected_content = str(predecessor["receipt_content_sha256"])
    if receipt.get("content_sha256") != expected_content:
        raise DeepAqualNoveltyError("predecessor receipt content changed")
    output["receipt_content_sha256"] = expected_content
    return output


def _validate_policy(config: Mapping[str, Any]) -> dict[str, str]:
    policy = config["admission_policy"]
    raw = _sha256_file(_repo_root() / str(policy["path"]))
    if raw != policy["raw_sha256"]:
        raise DeepAqualNoveltyError("admission policy bytes changed")
    if policy["source_class"] != "PRIMARY_PAPERS_PLUS_EXACT_ANALYTIC_AND_NUMERIC_BENCHMARKS":
        raise DeepAqualNoveltyError("admission source class changed")
    return {"path": str(policy["path"]), "raw_sha256": raw}


def _symbolic_checks() -> dict[str, bool]:
    s, amplitude, exponent = sp.symbols("s A p", positive=True)
    mu0, slope = sp.symbols("mu0 B", positive=True)
    x_time = sp.symbols("X", nonnegative=True)
    mu = amplitude * s**exponent
    mu_s = sp.diff(mu, s)
    longitudinal = sp.simplify(mu + 2 * s * mu_s)
    determinant = sp.simplify(mu**3 * longitudinal)
    regulated = sp.sqrt(mu0**2 + slope**2 * s)
    regulated_s = sp.diff(regulated, s)
    regulated_longitudinal = sp.simplify(regulated + 2 * s * regulated_s)
    t = sp.symbols("t", positive=True)
    relative_error = sp.simplify(
        (regulated / (slope * sp.sqrt(s)) - 1).subs(s, t * mu0**2 / slope**2)
    )
    time_scaled = slope**2 * x_time / mu0**2
    time_c = mu0 / sp.sqrt(1 + time_scaled)
    time_k = mu0 / (1 + time_scaled) ** sp.Rational(3, 2)
    return {
        "N03_NOTATION_AND_PRINCIPAL_MAP": sp.simplify(-2 * (-s / 2) - s) == 0,
        "N04_GENERIC_DEEP_POWER": sp.simplify(
            longitudinal - (1 + 2 * exponent) * amplitude * s**exponent
        )
        == 0
        and sp.simplify(determinant - (1 + 2 * exponent) * amplitude**4 * s ** (4 * exponent)) == 0,
        "N05_ZERO_TRANSITION": sp.limit(mu, s, 0, dir="+") == 0
        and sp.limit(longitudinal, s, 0, dir="+") == 0
        and sp.limit(determinant, s, 0, dir="+") == 0,
        "N06_C2_OBSTRUCTION": sp.limit(1 / mu_s.subs(exponent, sp.Rational(1, 2)), s, 0, dir="+")
        == 0,
        "N07_DEEP_AQUAL_RATIO": sp.simplify(
            (longitudinal / mu).subs(exponent, sp.Rational(1, 2)) - 2
        )
        == 0,
        "N08_POSITIVE_FLOOR_ESCAPE": sp.limit(regulated, s, 0, dir="+") == mu0
        and sp.limit(regulated_longitudinal, s, 0, dir="+") == mu0
        and sp.limit(regulated / (slope * sp.sqrt(s)), s, sp.oo) == 1,
        "N09_REGULATOR_ACCURACY_COST": sp.simplify(relative_error - (sp.sqrt(1 + 1 / t) - 1)) == 0,
        "N10_TIMELIKE_CONE_COST": sp.simplify(time_c / time_k - (1 + time_scaled)) == 0,
    }


def _log_slope(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.log(y1 / y0) / math.log(x1 / x0)


def _numeric_power_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    s_values = [float(value) for value in config["numeric_probes"]["s_values"]]
    tolerance = float(config["numeric_probes"]["tolerance"])
    records: list[dict[str, Any]] = []
    all_passed = True
    for exponent in (float(value) for value in config["numeric_probes"]["p_values"]):
        transverse = [value**exponent for value in s_values]
        longitudinal = [(1.0 + 2.0 * exponent) * value for value in transverse]
        determinant = [transverse[index] ** 3 * longitudinal[index] for index in range(3)]
        derivatives = [exponent * value ** (exponent - 1.0) for value in s_values]
        transverse_slope = _log_slope(s_values[0], transverse[0], s_values[-1], transverse[-1])
        determinant_slope = _log_slope(s_values[0], determinant[0], s_values[-1], determinant[-1])
        if exponent < 1.0:
            derivative_behavior = "DIVERGES_TOWARD_ZERO"
            derivative_passed = derivatives[0] > derivatives[-1]
        elif exponent == 1.0:
            derivative_behavior = "FINITE_CONSTANT"
            derivative_passed = max(derivatives) == min(derivatives)
        else:
            derivative_behavior = "VANISHES_TOWARD_ZERO"
            derivative_passed = derivatives[0] < derivatives[-1]
        passed = (
            abs(transverse_slope - exponent) <= tolerance
            and abs(determinant_slope - 4.0 * exponent) <= tolerance
            and derivative_passed
        )
        all_passed = all_passed and passed
        records.append(
            {
                "p": exponent,
                "C_log_slope": transverse_slope,
                "determinant_log_slope": determinant_slope,
                "derivative_behavior": derivative_behavior,
                "derivative_values": derivatives,
                "passed": passed,
            }
        )
    return {"records": records, "all_passed": all_passed}


def _numeric_regulator_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    regulator = config["numeric_probes"]["regulator"]
    mu0 = float(regulator["mu0"])
    slope = float(regulator["A"])
    tolerance = float(config["numeric_probes"]["tolerance"])
    space_records: list[dict[str, Any]] = []
    for s_value in (float(value) for value in regulator["spacelike_s_values"]):
        c_value = math.sqrt(mu0**2 + slope**2 * s_value)
        h_value = (mu0**2 + 2.0 * slope**2 * s_value) / c_value
        relative_error = None
        if s_value > 0:
            relative_error = c_value / (slope * math.sqrt(s_value)) - 1.0
        space_records.append(
            {
                "s": s_value,
                "C": c_value,
                "H": h_value,
                "relative_aqual_error": relative_error,
            }
        )
    time_records: list[dict[str, Any]] = []
    for x_value in (float(value) for value in regulator["timelike_X_values"]):
        scaled = slope**2 * x_value / mu0**2
        c_value = mu0 / math.sqrt(1.0 + scaled)
        k_value = mu0 / (1.0 + scaled) ** 1.5
        time_records.append(
            {
                "X": x_value,
                "C": c_value,
                "K": k_value,
                "speed_squared": c_value / k_value,
            }
        )
    transition_passed = (
        abs(space_records[0]["C"] - mu0) <= tolerance
        and abs(space_records[0]["H"] - mu0) <= tolerance
        and abs(time_records[0]["speed_squared"] - 1.0) <= tolerance
    )
    errors = [
        float(item["relative_aqual_error"])
        for item in space_records
        if item["relative_aqual_error"] is not None
    ]
    accuracy_passed = (
        len(errors) == 2 and errors[1] < errors[0] and all(value > 0 for value in errors)
    )
    cone_passed = all(
        item["speed_squared"] > 1.0 + tolerance for item in time_records if item["X"] > 0
    )
    return {
        "spacelike_records": space_records,
        "timelike_records": time_records,
        "transition_passed": transition_passed,
        "accuracy_cost_passed": accuracy_passed,
        "timelike_cone_cost_passed": cone_passed,
        "all_passed": transition_passed and accuracy_passed and cone_passed,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    local = _validate_local_integrity()
    predecessor = _validate_predecessor(config)
    policy = _validate_policy(config)
    symbolic = _symbolic_checks()
    power = _numeric_power_evidence(config)
    regulator = _numeric_regulator_evidence(config)
    checks = {
        "N01_CONFIG_AND_POLICY_SEALS": local["config_raw_sha256"] == EXPECTED_CONFIG_RAW_SHA256,
        "N02_COMMITTED_PREDECESSOR_BYTES": predecessor["commit"] == config["predecessor"]["commit"],
        **symbolic,
        "N11_NUMERIC_POWER_PROBES": power["all_passed"],
        "N12_NUMERIC_REGULATOR_PROBES": regulator["all_passed"],
        "N13_PRIMARY_SOURCE_INVENTORY": len(config["primary_literature"]) == 7,
        "N14_PREEXISTING_SUBSTANCE_VERDICT": config["claim_boundary"][
            "mathematical_substance_preexisting"
        ]
        and not config["claim_boundary"]["standalone_publication_candidate"],
        "N15_STATIC_AQUAL_RETENTION": not config["claim_boundary"]["static_aqual_invalidated"]
        and config["adjudication"]["static_aqual_observational_testing_disposition"].startswith(
            "RETAIN_STATIC_AND_3D_AQUAL_TESTS"
        ),
        "N16_ZERO_OBSERVATIONAL_ACCESS": not any(config["access_ledger"].values()),
    }
    if list(checks) != config["required_checks"] or not all(checks.values()):
        raise DeepAqualNoveltyError("benchmark checks failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_id": ARTIFACT_ID,
        "status": config["status"],
        "decision": DECISION,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "predecessor_binding": predecessor,
        "policy_binding": policy,
        "implementation_binding": local,
        "notation_map": config["notation_map"],
        "primary_literature": config["primary_literature"],
        "search_protocol": config["search_protocol"],
        "symbolic_evidence": symbolic,
        "numeric_power_evidence": power,
        "numeric_regulator_evidence": regulator,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_receipt() -> dict[str, Any]:
    stored = _read_json(_repo_root() / OUTPUT_PATH)
    expected = build_receipt()
    if stored != expected or stored.get("content_sha256") != _self_hash(stored):
        raise DeepAqualNoveltyError("stored receipt differs from deterministic rebuild")
    return stored


def write_receipt() -> str:
    destination = _repo_root() / OUTPUT_PATH
    data = (json.dumps(build_receipt(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return "EXISTING_IDENTICAL"
        raise DeepAqualNoveltyError("refusing to replace nonidentical receipt")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise DeepAqualNoveltyError("receipt publication race") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        receipt = validate_receipt()
        print(f"VALID {receipt['content_sha256']}")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "checks_passed": receipt["checks_passed"],
                    "historical_novelty_established": receipt["claim_boundary"][
                        "historical_novelty_established"
                    ],
                    "static_aqual_invalidated": receipt["claim_boundary"][
                        "static_aqual_invalidated"
                    ],
                    "observational_rows_read": receipt["access_ledger"]["observational_rows_read"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
