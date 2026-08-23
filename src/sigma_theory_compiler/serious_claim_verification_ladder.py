"""Build a candidate-bound exact -> CAS -> SMT -> interval -> Lean release ladder.

The stored receipt calibrates the ladder on externally authored known controls and proves that
bounded-unknown candidates remain blocked.  It never releases a mathematical or novelty claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .external_creativity_multi_host import validate_receipt as validate_multi_host_receipt
from .external_creativity_validation import RECEIPT_SCHEMA as CAMPAIGN_SCHEMA
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/serious_claim_verification_ladder.json"
OUTPUT_PATH = "runs/math/serious-claim-verification-ladder/receipt.json"
CONFIG_SCHEMA = "invariant-serious-claim-verification-ladder-config-1.0"
SCHEMA_VERSION = "invariant-serious-claim-verification-ladder-1.0"
CHAIN_SCHEMA = "invariant-candidate-verification-chain-1.0"
STAGE_SCHEMA = "invariant-candidate-verification-stage-1.0"
REQUIRED_STAGES = ("exact_arithmetic", "cas", "smt", "interval", "lean")
MUTATIONS = (
    "missing_stage",
    "reordered_stages",
    "candidate_scope_substitution",
    "broken_previous_stage_link",
    "backend_unavailable",
)
_HEX = frozenset("0123456789abcdef")


class SeriousClaimVerificationError(ValueError):
    """The candidate verification ladder failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SeriousClaimVerificationError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SeriousClaimVerificationError(f"{label} is not a SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SeriousClaimVerificationError(f"could not read {label}") from error
    if not isinstance(value, dict):
        raise SeriousClaimVerificationError(f"{label} is not an object")
    return value


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise SeriousClaimVerificationError(f"{label} escapes the project root") from error
    if not path.is_file():
        raise SeriousClaimVerificationError(f"{label} is missing")
    return path


def _validate_seal(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SeriousClaimVerificationError(f"{label} content seal changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    value = _read_json(_under(root, CONFIG_PATH, "ladder config"), "ladder config")
    _strict(
        value,
        {
            "known_control_suite",
            "ladder_id",
            "mutation_controls",
            "negative_control_benchmark_ids",
            "release_policy",
            "required_stage_order",
            "schema_version",
            "sources",
        },
        "ladder config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise SeriousClaimVerificationError("ladder config schema changed")
    if tuple(value["required_stage_order"]) != REQUIRED_STAGES:
        raise SeriousClaimVerificationError("serious-claim backend order changed")
    if tuple(value["mutation_controls"]) != MUTATIONS:
        raise SeriousClaimVerificationError("ladder mutation controls changed")
    _strict(
        value["sources"],
        {"campaign_receipt", "lean_source", "multi_host_receipt"},
        "ladder sources",
    )
    _strict(
        value["known_control_suite"],
        {"benchmark_ids", "lean_target", "lean_theorems"},
        "known-control suite",
    )
    if len(value["known_control_suite"]["benchmark_ids"]) < 2:
        raise SeriousClaimVerificationError("too few known controls")
    policy = value["release_policy"]
    required_policy = {
        "backend_unavailable_is_block",
        "candidate_identity_bound_at_every_stage",
        "independent_reproduction_required",
        "known_control_calibration_can_release_serious_claim",
        "named_human_prior_art_review_required",
        "positive_result_required_at_every_stage",
        "previous_stage_hash_bound",
    }
    _strict(policy, required_policy, "ladder release policy")
    if any(
        policy[key] is not True
        for key in required_policy - {"known_control_calibration_can_release_serious_claim"}
    ) or policy["known_control_calibration_can_release_serious_claim"] is not False:
        raise SeriousClaimVerificationError("ladder release policy weakened")
    return value


def _campaign(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(
        _under(root, config["sources"]["campaign_receipt"], "campaign receipt"),
        "campaign receipt",
    )
    _validate_seal(value, "campaign receipt")
    if value.get("schema_version") != CAMPAIGN_SCHEMA:
        raise SeriousClaimVerificationError("campaign receipt schema changed")
    policy = value.get("serious_claim_policy", {})
    if tuple(policy.get("required_backends", ())) != REQUIRED_STAGES:
        raise SeriousClaimVerificationError("campaign backend policy changed")
    if policy.get("released_claims") != 0:
        raise SeriousClaimVerificationError("campaign unexpectedly released a serious claim")
    return value


def _multi_host(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = _read_json(
        _under(root, config["sources"]["multi_host_receipt"], "multi-host receipt"),
        "multi-host receipt",
    )
    validate_multi_host_receipt(value)
    return value


def _candidate_bindings(
    benchmarks: Mapping[str, Mapping[str, Any]], benchmark_ids: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for benchmark_id in benchmark_ids:
        benchmark = benchmarks.get(benchmark_id)
        if benchmark is None or benchmark.get("target_kind") != "known_formula":
            raise SeriousClaimVerificationError("known-control benchmark is missing or changed")
        candidates = benchmark.get("ranked_candidates", [])
        if not candidates:
            raise SeriousClaimVerificationError("known-control candidate is missing")
        candidate = candidates[0]
        if candidate.get("train_loss") != "0" or candidate.get("holdout_loss") != "0":
            raise SeriousClaimVerificationError("known-control best candidate is not exact")
        result.append(
            {
                "behavior_sha256": candidate["behavior_sha256"],
                "benchmark_id": benchmark_id,
                "candidate_id": candidate["candidate_id"],
                "expression_sha256": canonical_sha256({"expression": candidate["expression"]}),
                "target_commitment_sha256": benchmark["target_commitment_opened"],
            }
        )
    return result


def _stage(
    backend: str,
    order: int,
    candidate_scope_sha256: str,
    previous_stage_sha256: str | None,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": STAGE_SCHEMA,
        "backend": backend,
        "order": order,
        "candidate_scope_sha256": candidate_scope_sha256,
        "previous_stage_sha256": previous_stage_sha256,
        "backend_available": True,
        "positive_control_passed": True,
        "evidence": dict(evidence),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_candidate_chain(chain: Mapping[str, Any]) -> None:
    _strict(
        chain,
        {
            "candidate_bindings",
            "candidate_scope_sha256",
            "chain_id",
            "claims",
            "content_sha256",
            "purpose",
            "schema_version",
            "stages",
            "status",
        },
        "candidate verification chain",
    )
    _validate_seal(chain, "candidate verification chain")
    if chain["schema_version"] != CHAIN_SCHEMA:
        raise SeriousClaimVerificationError("candidate verification chain schema changed")
    bindings = chain["candidate_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise SeriousClaimVerificationError("candidate verification chain has no candidates")
    for binding in bindings:
        _strict(
            binding,
            {
                "behavior_sha256",
                "benchmark_id",
                "candidate_id",
                "expression_sha256",
                "target_commitment_sha256",
            },
            "candidate verification binding",
        )
        for key in (
            "behavior_sha256",
            "expression_sha256",
            "target_commitment_sha256",
        ):
            _sha(binding[key], f"candidate verification {key}")
    expected_scope = canonical_sha256({"candidate_bindings": bindings})
    if chain["candidate_scope_sha256"] != expected_scope:
        raise SeriousClaimVerificationError("candidate verification scope changed")
    stages = chain["stages"]
    if not isinstance(stages, list) or len(stages) != len(REQUIRED_STAGES):
        raise SeriousClaimVerificationError("candidate verification stage coverage changed")
    previous = None
    for order, (expected_backend, stage) in enumerate(zip(REQUIRED_STAGES, stages, strict=True)):
        _strict(
            stage,
            {
                "backend",
                "backend_available",
                "candidate_scope_sha256",
                "content_sha256",
                "evidence",
                "order",
                "positive_control_passed",
                "previous_stage_sha256",
                "schema_version",
            },
            "candidate verification stage",
        )
        _validate_seal(stage, "candidate verification stage")
        if (
            stage["schema_version"] != STAGE_SCHEMA
            or stage["backend"] != expected_backend
            or stage["order"] != order
            or stage["candidate_scope_sha256"] != expected_scope
            or stage["previous_stage_sha256"] != previous
            or stage["backend_available"] is not True
            or stage["positive_control_passed"] is not True
        ):
            raise SeriousClaimVerificationError("candidate verification stage policy failed")
        previous = stage["content_sha256"]
    if chain["status"] != "PASS_KNOWN_CONTROL_BACKEND_LADDER":
        raise SeriousClaimVerificationError("candidate verification chain status changed")
    claims = chain["claims"]
    _strict(
        claims,
        {
            "known_control_calibration_passed",
            "literature_novelty_established",
            "new_candidate_verified",
            "serious_claim_released",
        },
        "candidate verification claims",
    )
    if claims != {
        "known_control_calibration_passed": True,
        "literature_novelty_established": False,
        "new_candidate_verified": False,
        "serious_claim_released": False,
    }:
        raise SeriousClaimVerificationError("candidate verification claim boundary changed")


def _known_control_chain(
    root: Path,
    config: Mapping[str, Any],
    campaign: Mapping[str, Any],
    multi_host: Mapping[str, Any],
) -> dict[str, Any]:
    benchmarks = {item["benchmark_id"]: item for item in campaign["benchmarks"]}
    benchmark_ids = config["known_control_suite"]["benchmark_ids"]
    bindings = _candidate_bindings(benchmarks, benchmark_ids)
    scope = canonical_sha256({"candidate_bindings": bindings})
    stages = []
    previous = None
    for order, backend in enumerate(REQUIRED_STAGES):
        if backend == "lean":
            lean_source = _under(root, config["sources"]["lean_source"], "Lean source")
            source_text = lean_source.read_text(encoding="utf-8")
            theorem_names = config["known_control_suite"]["lean_theorems"]
            if any(name.rsplit(".", 1)[-1] not in source_text for name in theorem_names):
                raise SeriousClaimVerificationError("known-control Lean theorem is missing")
            if multi_host.get("lean", {}).get("kernel_checked") is not True:
                raise SeriousClaimVerificationError("known-control Lean artifact did not pass")
            evidence = {
                "artifact_content_sha256": multi_host["lean"]["content_sha256"],
                "artifact_id": multi_host["lean"]["artifact_id"],
                "kernel_checked": True,
                "source_sha256": _normalized_file_sha256(lean_source),
                "target": config["known_control_suite"]["lean_target"],
                "theorems": list(theorem_names),
            }
        else:
            rows = []
            for benchmark_id in benchmark_ids:
                benchmark = benchmarks[benchmark_id]
                formal = benchmark["formal_verification"]["backends"]
                if formal.get(backend) is not True:
                    raise SeriousClaimVerificationError(
                        f"known-control {backend} evidence did not pass"
                    )
                row = {
                    "backend_passed": True,
                    "benchmark_id": benchmark_id,
                    "candidate_id": benchmark["ranked_candidates"][0]["candidate_id"],
                }
                if backend == "exact_arithmetic":
                    row["independent_exact_match"] = benchmark[
                        "independent_exact_reproduction"
                    ]["match"]
                    if row["independent_exact_match"] is not True:
                        raise SeriousClaimVerificationError(
                            "known-control independent exact reproduction failed"
                        )
                rows.append(row)
            evidence = {
                "campaign_content_sha256": campaign["content_sha256"],
                "controls": rows,
            }
        current = _stage(backend, order, scope, previous, evidence)
        stages.append(current)
        previous = current["content_sha256"]
    body = {
        "schema_version": CHAIN_SCHEMA,
        "chain_id": config["ladder_id"] + ".known-controls",
        "purpose": "known_control_calibration_only",
        "candidate_bindings": bindings,
        "candidate_scope_sha256": scope,
        "stages": stages,
        "status": "PASS_KNOWN_CONTROL_BACKEND_LADDER",
        "claims": {
            "known_control_calibration_passed": True,
            "literature_novelty_established": False,
            "new_candidate_verified": False,
            "serious_claim_released": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_candidate_chain(body)
    return body


def _negative_controls(
    config: Mapping[str, Any], campaign: Mapping[str, Any]
) -> list[dict[str, Any]]:
    benchmarks = {item["benchmark_id"]: item for item in campaign["benchmarks"]}
    result = []
    for benchmark_id in config["negative_control_benchmark_ids"]:
        benchmark = benchmarks.get(benchmark_id)
        if benchmark is None or benchmark.get("target_kind") != "bounded_unknown":
            raise SeriousClaimVerificationError("bounded-unknown negative control changed")
        backends = benchmark["formal_verification"]["backends"]
        missing = [backend for backend in REQUIRED_STAGES if backends.get(backend) is not True]
        if not missing or benchmark["claims"]["serious_claim_released"] is not False:
            raise SeriousClaimVerificationError("bounded-unknown control did not fail closed")
        result.append(
            {
                "benchmark_id": benchmark_id,
                "missing_or_failed_backends": missing,
                "serious_claim_released": False,
                "status": "BLOCKED_INCOMPLETE_BACKEND_LADDER",
            }
        )
    return result


def _reseal_chain(chain: dict[str, Any]) -> None:
    previous = None
    for stage in chain["stages"]:
        stage["previous_stage_sha256"] = previous
        body = {key: item for key, item in stage.items() if key != "content_sha256"}
        stage["content_sha256"] = canonical_sha256(body)
        previous = stage["content_sha256"]
    body = {key: item for key, item in chain.items() if key != "content_sha256"}
    chain["content_sha256"] = canonical_sha256(body)


def _mutation_controls(chain: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = []
    for mutation in MUTATIONS:
        changed = deepcopy(chain)
        if mutation == "missing_stage":
            del changed["stages"][2]
        elif mutation == "reordered_stages":
            changed["stages"][1], changed["stages"][2] = (
                changed["stages"][2],
                changed["stages"][1],
            )
        elif mutation == "candidate_scope_substitution":
            changed["stages"][3]["candidate_scope_sha256"] = "0" * 64
        elif mutation == "broken_previous_stage_link":
            changed["stages"][4]["previous_stage_sha256"] = "0" * 64
        elif mutation == "backend_unavailable":
            changed["stages"][4]["backend_available"] = False
        if mutation in {"reordered_stages", "candidate_scope_substitution", "backend_unavailable"}:
            _reseal_chain(changed)
        else:
            body = {key: item for key, item in changed.items() if key != "content_sha256"}
            changed["content_sha256"] = canonical_sha256(body)
        try:
            validate_candidate_chain(changed)
        except SeriousClaimVerificationError as error:
            controls.append(
                {
                    "mutation_id": mutation,
                    "rejected": True,
                    "reason_class": type(error).__name__,
                }
            )
        else:
            raise SeriousClaimVerificationError(f"ladder mutation survived: {mutation}")
    return controls


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    campaign = _campaign(root, config)
    multi_host = _multi_host(root, config)
    chain = _known_control_chain(root, config, campaign, multi_host)
    negative = _negative_controls(config, campaign)
    mutations = _mutation_controls(chain)
    body = {
        "schema_version": SCHEMA_VERSION,
        "ladder_id": config["ladder_id"],
        "source_bindings": {
            "campaign_receipt": {
                "content_sha256": campaign["content_sha256"],
                "path": config["sources"]["campaign_receipt"],
            },
            "config": {
                "normalized_file_sha256": _normalized_file_sha256(root / CONFIG_PATH),
                "path": CONFIG_PATH,
            },
            "multi_host_receipt": {
                "content_sha256": multi_host["content_sha256"],
                "path": config["sources"]["multi_host_receipt"],
            },
        },
        "known_control_chain": chain,
        "negative_controls": negative,
        "mutation_controls": mutations,
        "summary": {
            "known_control_candidates": len(chain["candidate_bindings"]),
            "negative_controls_blocked": len(negative),
            "required_stage_order": list(REQUIRED_STAGES),
            "structural_mutations_rejected": len(mutations),
            "status": "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION",
        },
        "release_gate": {
            "candidate_specific_chain_required": True,
            "independent_reproduction_required": True,
            "named_human_prior_art_review_required": True,
            "serious_claims_released": 0,
            "status": "BLOCKED_NO_NEW_CANDIDATE_COMPLETE_LADDER",
        },
        "claims": {
            "backend_availability_is_proof": False,
            "known_control_calibration_is_novelty": False,
            "novel_formula_established": False,
            "serious_claim_released": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "known_control_chain",
            "ladder_id",
            "mutation_controls",
            "negative_controls",
            "release_gate",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "serious-claim ladder receipt",
    )
    _validate_seal(value, "serious-claim ladder receipt")
    if value["schema_version"] != SCHEMA_VERSION:
        raise SeriousClaimVerificationError("serious-claim ladder schema changed")
    validate_candidate_chain(value["known_control_chain"])
    summary = value["summary"]
    _strict(
        summary,
        {
            "known_control_candidates",
            "negative_controls_blocked",
            "required_stage_order",
            "status",
            "structural_mutations_rejected",
        },
        "serious-claim ladder summary",
    )
    if (
        summary.get("status") != "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION"
        or tuple(summary.get("required_stage_order", ())) != REQUIRED_STAGES
        or summary.get("known_control_candidates", 0) < 2
        or summary.get("negative_controls_blocked", 0) < 2
        or summary.get("structural_mutations_rejected") != len(MUTATIONS)
    ):
        raise SeriousClaimVerificationError("serious-claim ladder summary changed")
    negative = value["negative_controls"]
    for item in negative:
        _strict(
            item,
            {
                "benchmark_id",
                "missing_or_failed_backends",
                "serious_claim_released",
                "status",
            },
            "negative candidate control",
        )
    if any(
        item.get("status") != "BLOCKED_INCOMPLETE_BACKEND_LADDER"
        or item.get("serious_claim_released") is not False
        or not item.get("missing_or_failed_backends")
        for item in negative
    ):
        raise SeriousClaimVerificationError("negative candidate control changed")
    mutations = value["mutation_controls"]
    for item in mutations:
        _strict(
            item,
            {"mutation_id", "reason_class", "rejected"},
            "ladder mutation control",
        )
    if [item.get("mutation_id") for item in mutations] != list(MUTATIONS) or any(
        item.get("rejected") is not True for item in mutations
    ):
        raise SeriousClaimVerificationError("ladder mutation evidence changed")
    release = value["release_gate"]
    _strict(
        release,
        {
            "candidate_specific_chain_required",
            "independent_reproduction_required",
            "named_human_prior_art_review_required",
            "serious_claims_released",
            "status",
        },
        "serious-claim release gate",
    )
    if (
        release.get("status") != "BLOCKED_NO_NEW_CANDIDATE_COMPLETE_LADDER"
        or release.get("serious_claims_released") != 0
        or release.get("candidate_specific_chain_required") is not True
        or release.get("independent_reproduction_required") is not True
        or release.get("named_human_prior_art_review_required") is not True
    ):
        raise SeriousClaimVerificationError("serious-claim release boundary changed")
    _strict(
        value["claims"],
        {
            "backend_availability_is_proof",
            "known_control_calibration_is_novelty",
            "novel_formula_established",
            "serious_claim_released",
        },
        "serious-claim ladder claims",
    )
    if any(value["claims"].values()):
        raise SeriousClaimVerificationError("serious-claim ladder claim boundary changed")
    if root is not None:
        root = root.resolve()
        config = load_config(root)
        campaign = _campaign(root, config)
        multi_host = _multi_host(root, config)
        bindings = value["source_bindings"]
        _strict(
            bindings,
            {"campaign_receipt", "config", "multi_host_receipt"},
            "serious-claim source bindings",
        )
        _strict(
            bindings["campaign_receipt"],
            {"content_sha256", "path"},
            "campaign source binding",
        )
        _strict(
            bindings["config"],
            {"normalized_file_sha256", "path"},
            "ladder config source binding",
        )
        _strict(
            bindings["multi_host_receipt"],
            {"content_sha256", "path"},
            "multi-host source binding",
        )
        if (
            bindings["config"]["path"] != CONFIG_PATH
            or bindings["campaign_receipt"]["path"]
            != config["sources"]["campaign_receipt"]
            or bindings["multi_host_receipt"]["path"]
            != config["sources"]["multi_host_receipt"]
            or
            bindings.get("config", {}).get("normalized_file_sha256")
            != _normalized_file_sha256(root / CONFIG_PATH)
            or bindings.get("campaign_receipt", {}).get("content_sha256")
            != campaign["content_sha256"]
            or bindings.get("multi_host_receipt", {}).get("content_sha256")
            != multi_host["content_sha256"]
        ):
            raise SeriousClaimVerificationError("serious-claim ladder source binding changed")
        expected_chain = _known_control_chain(root, config, campaign, multi_host)
        if value["known_control_chain"] != expected_chain:
            raise SeriousClaimVerificationError("known-control ladder evidence changed")
        if value["negative_controls"] != _negative_controls(config, campaign):
            raise SeriousClaimVerificationError("negative-control ladder evidence changed")
        if value["mutation_controls"] != _mutation_controls(expected_chain):
            raise SeriousClaimVerificationError("ladder mutation audit changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    if args.command == "build":
        receipt = build_receipt(args.root)
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = _read_json(args.receipt.resolve(), "serious-claim ladder receipt")
        validate_receipt(receipt, args.root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
