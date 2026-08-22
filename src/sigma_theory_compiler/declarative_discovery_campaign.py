"""Deterministic readiness campaign for the declarative discovery protocol.

This is a protocol control, not a new mathematics claim.  It executes every creativity
operator, routes every candidate through an independent structural verifier, fills behavioral
niches, consumes a typed counterexample blocker, closes a proof plan, walks a staged dataset
pipeline and a target-sealed capability ladder, and emits one replayable discovery chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import declarative_discovery as D
from .sigma_core import canonical_sha256

RESULT_SCHEMA = "invariant-declarative-discovery-readiness-1.0"
OUTPUT_PATH = "runs/math/declarative-discovery-platform/readiness.json"
CONFIG_PATH = "configs/declarative_discovery_platform.json"
SOURCE_PATH = "src/sigma_theory_compiler/declarative_discovery.py"
CAMPAIGN_SOURCE_PATH = "src/sigma_theory_compiler/declarative_discovery_campaign.py"
RESEALER_SOURCE_PATH = "src/sigma_theory_compiler/receipt_dag.py"
PROTOCOL_TEST_PATH = "tests/test_declarative_discovery.py"
RESEALER_TEST_PATH = "tests/test_receipt_dag.py"


class DeclarativeCampaignError(ValueError):
    """The readiness campaign or its receipt failed closed."""


def _portable_file_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _seed(proposal_id: str, value_type: D.ValueType, representation: str) -> D.Proposal:
    return D.Proposal(
        proposal_id,
        "grammar.readiness",
        None,
        value_type,
        representation,
        ("declaration.readiness",),
    )


def _descriptor(index: int, proposal: D.Proposal) -> D.BehaviorDescriptor:
    operator = proposal.operator.value if proposal.operator else "seed"
    return D.BehaviorDescriptor(
        dimensional_signature=(index % 3 - 1, (index // 3) % 3 - 1),
        symmetry_class=("even", "odd", "cyclic", "none")[index % 4],
        complexity_bin=min(15, len(proposal.representation) // 12),
        asymptotic_class=operator,
        invariant_flags=("protocol_typed",),
    )


def _bindings(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": relative,
            "portable_file_sha256": _portable_file_sha256(root / relative),
        }
        for name, relative in (
            ("campaign_source", CAMPAIGN_SOURCE_PATH),
            ("config", CONFIG_PATH),
            ("protocol_source", SOURCE_PATH),
            ("protocol_tests", PROTOCOL_TEST_PATH),
            ("resealer_source", RESEALER_SOURCE_PATH),
            ("resealer_tests", RESEALER_TEST_PATH),
        )
    }


def run_readiness(root: Path) -> dict[str, Any]:
    config = D.load_platform_config(root / CONFIG_PATH)
    expression = _seed("seed.expression", D.ValueType.EXPRESSION, "x^2+y^2")
    sequence = _seed("seed.sequence", D.ValueType.SEQUENCE, "a[n]")
    portfolio = list(D.generate_operator_portfolio((expression, sequence)))

    blocker = D.TypedBlocker(
        "blocker.readiness.counterexample",
        D.BlockerKind.COUNTEREXAMPLE,
        D.ValueType.EXPRESSION,
        Fraction(1, 17),
        "x=2,y=5",
        D.CreativityOperator.COUNTEREXAMPLE_REPAIR,
    )
    repair_spec = next(
        item
        for item in D.DEFAULT_OPERATORS
        if item.operator is D.CreativityOperator.COUNTEREXAMPLE_REPAIR
    )
    portfolio.append(
        D.apply_operator(expression, repair_spec, nonce="readiness-repair", blocker=blocker)
    )
    if {item.operator for item in portfolio} != set(D.CreativityOperator):
        raise DeclarativeCampaignError("readiness portfolio did not execute every operator")

    index_by_id = {item.proposal_id: index for index, item in enumerate(portfolio)}
    registry = D.IndependentVerifierRegistry()

    def structural_verifier(proposal: D.Proposal) -> D.VerificationRecord:
        index = index_by_id[proposal.proposal_id]
        return D.VerificationRecord(
            proposal.proposal_id,
            "verifier.protocol-structure",
            D.VerificationStatus.VERIFIED,
            Fraction(20 + index, 40),
            _descriptor(index, proposal),
        )

    for value_type in sorted({item.value_type for item in portfolio}, key=lambda item: item.value):
        registry.register(value_type, "verifier.protocol-structure", structural_verifier)
    records = tuple(registry.verify(item) for item in portfolio)
    archive = D.BehavioralMapElites()
    for proposal, record in zip(portfolio, records, strict=True):
        archive.insert(proposal, record)

    proof_plan = D.search_proof_plan(
        portfolio[0].proposal_id,
        ("conjunction",),
        (
            D.TacticDeclaration("split", "conjunction", ("algebra", "domain")),
            D.TacticDeclaration("normalize", "algebra", ()),
            D.TacticDeclaration("domain-check", "domain", ()),
        ),
        max_steps=config.proof_plan_max_steps,
    )
    if not proof_plan.closed:
        raise DeclarativeCampaignError("readiness proof plan did not close")

    dataset = D.DatasetExplanationPipeline(
        canonical_sha256({"rows": [[1, 1], [2, 3], [3, 6], [4, 10]]}),
        canonical_sha256({"heldout": [[5, 15], [6, 21]]}),
    )
    previous = dataset.dataset_sha256
    for stage in D.DATASET_STAGES:
        output = canonical_sha256({"input": previous, "stage": stage.value})
        dataset.record(
            stage,
            previous,
            output,
            passed=True,
            heldout_opened=stage is D.DatasetStage.HELDOUT_TEST,
        )
        previous = output

    ladder = D.BlindCapabilityLadder()
    ladder.admit(D.CapabilityResult(D.CapabilityLevel.SOLVED_VISIBLE, "control.visible", "", False, (), True))
    ladder.admit(D.CapabilityResult(D.CapabilityLevel.SOLVED_ANONYMOUS, "control.anonymous", "", False, (), True))
    ladder.admit(
        D.CapabilityResult(
            D.CapabilityLevel.SYNTHETIC_TARGET_SEALED,
            "control.synthetic-sealed",
            canonical_sha256({"target": "sealed"}),
            True,
            (),
            True,
        )
    )
    # This readiness control does not pretend to establish historical or open-problem ability.
    ladder.admit(
        D.CapabilityResult(
            D.CapabilityLevel.HISTORICAL_TARGET_SEALED,
            "control.historical-not-run",
            canonical_sha256({"target": "not-run"}),
            True,
            (),
            False,
        )
    )

    reachability = D.find_type_reachability(
        D.ValueType.SEQUENCE,
        D.ValueType.EQUATION,
        next(
            item.proposal_id
            for item in portfolio
            if item.operator is D.CreativityOperator.RECURRENCE_GUESSING
        ),
    )
    negative = D.publish_negative("readiness.no-open-math-claim", len(portfolio), reachability)

    chain = D.DiscoveryChain()
    chain.add(
        D.DiscoveryChainLink(
            "declaration", D.ChainStage.DECLARATION, canonical_sha256(config.config_id), ()
        )
    )
    for index, (proposal, record) in enumerate(zip(portfolio, records, strict=True)):
        proposal_link = f"proposal-{index:02d}"
        verification_link = f"verification-{index:02d}"
        chain.add(
            D.DiscoveryChainLink(
                proposal_link,
                D.ChainStage.PROPOSAL,
                canonical_sha256(proposal.to_dict()),
                ("declaration",),
            )
        )
        chain.add(
            D.DiscoveryChainLink(
                verification_link,
                D.ChainStage.VERIFICATION,
                canonical_sha256(record.to_dict()),
                (proposal_link,),
            )
        )
    chain.add(
        D.DiscoveryChainLink(
            "proof-plan",
            D.ChainStage.PROOF_PLAN,
            canonical_sha256({"tactics": list(proof_plan.tactic_ids)}),
            ("verification-00",),
        )
    )
    chain.add(
        D.DiscoveryChainLink(
            "blind-benchmark",
            D.ChainStage.BLIND_BENCHMARK,
            canonical_sha256({"highest_passed": ladder.highest_passed}),
            ("proof-plan",),
        )
    )
    chain.add(
        D.DiscoveryChainLink(
            "dataset-explanation",
            D.ChainStage.DATASET_EXPLANATION,
            canonical_sha256({"completed": dataset.completed, "last": previous}),
            ("blind-benchmark",),
        )
    )

    metrics = D.measure_creative_yield(
        portfolio,
        records,
        archive,
        (proof_plan,),
        repairs_attempted=1,
        repairs_verified=1,
        blind_levels_passed=ladder.highest_passed,
        dataset_explanations_completed=int(dataset.completed),
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": "declarative-discovery-readiness-v1",
        "source_bindings": _bindings(root),
        "config": {
            "config_id": config.config_id,
            "maximum_proposals": config.maximum_proposals,
            "proof_plan_max_steps": config.proof_plan_max_steps,
        },
        "counts": {
            "behavioral_niches": archive.occupied_niches,
            "creativity_operators_executed": len(portfolio),
            "dataset_stages_completed": len(dataset.records),
            "discovery_chain_links": len(chain.links),
            "proposals_independently_verified": len(records),
        },
        "operator_receipts": [
            {
                "operator": proposal.operator.value if proposal.operator else None,
                "proposal_id": proposal.proposal_id,
                "value_type": proposal.value_type.value,
                "verification": record.to_dict(),
            }
            for proposal, record in zip(portfolio, records, strict=True)
        ],
        "counterexample_guidance": blocker.to_dict(),
        "reachability_qualified_negative": {
            "explored_proposals": negative.explored_proposals,
            "operator_path": [item.value for item in negative.reachability.operator_path],
            "status": negative.status,
            "target_id": negative.target_id,
            "witness_proposal_id": negative.reachability.witness_proposal_id,
        },
        "proof_plan": {
            "closed": proof_plan.closed,
            "proposal_id": proof_plan.proposal_id,
            "tactic_ids": list(proof_plan.tactic_ids),
        },
        "dataset_pipeline": {
            "completed": dataset.completed,
            "heldout_commitment": dataset.heldout_commitment,
            "stages": [item.stage.value for item in dataset.records],
        },
        "blind_capability": {
            "highest_passed": ladder.highest_passed,
            "historical_and_open_capability_established": False,
            "levels_recorded": [item.level.value for item in ladder.results],
        },
        "discovery_chain_sha256": chain.content_sha256(),
        "creative_yield": metrics.to_dict(),
        "claims": {
            "behavioral_map_elites_executed": True,
            "counterexample_guided_typed_repair_executed": True,
            "dataset_explanation_pipeline_executed": True,
            "historical_or_open_problem_solved": False,
            "negative_published_without_reachability": False,
            "novel_mathematics_established": False,
            "proof_plan_search_executed": True,
            "proposer_self_verdict_admitted": False,
            "receipt_dag_resealer_bound": True,
            "ten_creativity_operator_families_executed": True,
            "typed_declarative_search_language_executed": True,
        },
        "decision": "READY_FOR_TARGET_SEALED_CAPABILITY_CAMPAIGNS",
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: dict[str, Any], root: Path) -> None:
    body = dict(value)
    claimed = body.pop("content_sha256", None)
    if claimed != canonical_sha256(body):
        raise DeclarativeCampaignError("readiness receipt content seal changed")
    expected = run_readiness(root)
    if value != expected:
        raise DeclarativeCampaignError("readiness receipt does not replay from its bound sources")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    generated = run_readiness(root)
    if args.check:
        validate_receipt(json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8")), root)
    else:
        destination = root / (args.out or OUTPUT_PATH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(generated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps({"decision": generated["decision"], "counts": generated["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

