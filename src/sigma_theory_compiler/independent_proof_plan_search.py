"""Executable, candidate-independent search for diverse mathematical proof plans."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/independent_proof_plan_search.json"
OUTPUT_PATH = "runs/math/independent-proof-plan-search/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/independent_proof_plan_search.py"
TEST_PATH = "tests/test_independent_proof_plan_search.py"
CONFIG_SCHEMA = "invariant-independent-proof-plan-search-config-1.0"
RESULT_SCHEMA = "invariant-independent-proof-plan-search-result-1.0"
MECHANISMS = (
    "induction",
    "invariant_preservation",
    "bijection_or_involution",
    "minimal_counterexample_descent",
    "transform_and_extract",
    "contradiction",
)
APPLICABILITY_STATUSES = {
    "APPLICABLE_FEATURES_PRESENT",
    "REQUIRES_FEATURE_EVIDENCE_RETAINED",
}


class IndependentProofPlanSearchError(ValueError):
    """The proof-plan search, controls, or non-pruning contract changed."""


@dataclass(frozen=True, slots=True)
class SearchState:
    goals: tuple[str, ...]
    facts: tuple[str, ...] = ()
    induction_variables: tuple[str, ...] = ()
    normal_form: str = "raw"
    representation: str = "native"


@dataclass(frozen=True, slots=True)
class SearchResult:
    closed: bool
    explored_states: int
    tactic_ids: tuple[str, ...]
    remaining_goal_kinds: tuple[str, ...]
    facts: tuple[str, ...]
    induction_variables: tuple[str, ...]
    normal_form: str
    representation: str


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IndependentProofPlanSearchError(f"{label} keys changed")


def _normalized_sha256(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _string_tuple(value: Any, label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise IndependentProofPlanSearchError(f"{label} is not an admitted array")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result) or len(set(result)) != len(
        result
    ):
        raise IndependentProofPlanSearchError(f"{label} contains invalid or duplicate values")
    return result


def _validate_tactic(value: Mapping[str, Any]) -> None:
    _strict_keys(
        value,
        {
            "adds_facts",
            "consumes",
            "cost",
            "falsification_power",
            "instruction",
            "introduces_induction_variables",
            "normal_form",
            "premise_count",
            "produces",
            "proof_debt",
            "representation",
            "requires_facts",
            "tactic_id",
        },
        "proof tactic",
    )
    for key in (
        "adds_facts",
        "introduces_induction_variables",
        "produces",
        "requires_facts",
    ):
        _string_tuple(value[key], f"tactic {key}")
    for key in ("consumes", "instruction", "normal_form", "representation", "tactic_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise IndependentProofPlanSearchError(f"tactic {key} is invalid")
    for key in ("cost", "falsification_power", "premise_count", "proof_debt"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise IndependentProofPlanSearchError(f"tactic {key} is invalid")


def _load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict_keys(config, {"routes", "schema_version"}, "proof-plan config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise IndependentProofPlanSearchError("proof-plan config identity changed")
    routes = config["routes"]
    if not isinstance(routes, list) or len(routes) != len(MECHANISMS):
        raise IndependentProofPlanSearchError("proof-plan route count changed")
    seen_routes: set[str] = set()
    seen_tactics: set[str] = set()
    for route in routes:
        _strict_keys(
            route,
            {
                "initial_goal",
                "mechanism",
                "mutation_remove_tactic_id",
                "required_candidate_capabilities",
                "route_id",
                "tactics",
            },
            "proof-plan route",
        )
        if route["mechanism"] not in MECHANISMS or route["route_id"] in seen_routes:
            raise IndependentProofPlanSearchError("proof-plan route identity changed")
        seen_routes.add(route["route_id"])
        _string_tuple(
            route["required_candidate_capabilities"],
            "required candidate capabilities",
            allow_empty=False,
        )
        tactics = route["tactics"]
        if not isinstance(tactics, list) or len(tactics) < 2:
            raise IndependentProofPlanSearchError("proof-plan route lacks tactics")
        route_tactic_ids = set()
        for tactic in tactics:
            _validate_tactic(tactic)
            tactic_id = tactic["tactic_id"]
            if tactic_id in seen_tactics:
                raise IndependentProofPlanSearchError("proof tactic IDs are not globally unique")
            route_tactic_ids.add(tactic_id)
            seen_tactics.add(tactic_id)
        if route["mutation_remove_tactic_id"] not in route_tactic_ids:
            raise IndependentProofPlanSearchError("proof-plan mutation tactic is absent")
    if {route["mechanism"] for route in routes} != set(MECHANISMS):
        raise IndependentProofPlanSearchError("proof-plan mechanism coverage changed")
    return config


def search_tactic_graph(
    initial_goal: str, tactics: Sequence[Mapping[str, Any]], *, max_steps: int = 12
) -> SearchResult:
    """Use deterministic uniform-cost search over goal, fact, and representation states."""

    for tactic in tactics:
        _validate_tactic(tactic)
    start = SearchState((initial_goal,))
    queue: list[tuple[int, int, tuple[str, ...], int, SearchState]] = []
    counter = 0
    heapq.heappush(queue, (0, 0, (), counter, start))
    seen: dict[SearchState, int] = {start: 0}
    explored = 0
    best = start
    ordered = sorted(tactics, key=lambda item: item["tactic_id"])
    while queue:
        cost, steps, plan, _, state = heapq.heappop(queue)
        if cost != seen.get(state):
            continue
        explored += 1
        if not state.goals:
            return SearchResult(
                True,
                explored,
                plan,
                (),
                state.facts,
                state.induction_variables,
                state.normal_form,
                state.representation,
            )
        if len(state.goals) < len(best.goals):
            best = state
        if steps >= max_steps:
            continue
        goal = state.goals[0]
        facts = set(state.facts)
        for tactic in ordered:
            if tactic["consumes"] != goal or not set(tactic["requires_facts"]) <= facts:
                continue
            next_state = SearchState(
                (*tuple(tactic["produces"]), *state.goals[1:]),
                tuple(sorted(facts | set(tactic["adds_facts"]))),
                tuple(
                    sorted(
                        set(state.induction_variables)
                        | set(tactic["introduces_induction_variables"])
                    )
                ),
                tactic["normal_form"],
                tactic["representation"],
            )
            next_cost = cost + tactic["cost"]
            if next_cost >= seen.get(next_state, 10**12):
                continue
            seen[next_state] = next_cost
            counter += 1
            heapq.heappush(
                queue,
                (next_cost, steps + 1, (*plan, tactic["tactic_id"]), counter, next_state),
            )
    return SearchResult(
        False,
        explored,
        (),
        best.goals,
        best.facts,
        best.induction_variables,
        best.normal_form,
        best.representation,
    )


def _route_result(route: Mapping[str, Any]) -> dict[str, Any]:
    tactics = route["tactics"]
    positive = search_tactic_graph(route["initial_goal"], tactics)
    removed = route["mutation_remove_tactic_id"]
    mutation = search_tactic_graph(
        route["initial_goal"],
        [tactic for tactic in tactics if tactic["tactic_id"] != removed],
    )
    if not positive.closed or mutation.closed:
        raise IndependentProofPlanSearchError("proof-plan positive or mutation control failed")
    by_id = {tactic["tactic_id"]: tactic for tactic in tactics}
    used = [by_id[tactic_id] for tactic_id in positive.tactic_ids]
    metrics = {
        "cost": sum(tactic["cost"] for tactic in used),
        "falsification_power": sum(tactic["falsification_power"] for tactic in used),
        "premise_count": sum(tactic["premise_count"] for tactic in used),
        "proof_debt": sum(tactic["proof_debt"] for tactic in used),
    }
    mechanism_body = {
        "mechanism": route["mechanism"],
        "steps": [tactic["instruction"] for tactic in used],
        "tactic_ids": list(positive.tactic_ids),
    }
    template_body = {
        **mechanism_body,
        "required_candidate_capabilities": route["required_candidate_capabilities"],
        "route_id": route["route_id"],
    }
    return {
        "mechanism": route["mechanism"],
        "mutation_control": {
            "closed": mutation.closed,
            "explored_states": mutation.explored_states,
            "mutation_control_rejected": True,
            "remaining_goal_kinds": list(mutation.remaining_goal_kinds),
            "removed_tactic_id": removed,
        },
        "plan_template": {
            **template_body,
            "metrics": metrics,
            "proof_mechanism_sha256": canonical_sha256(mechanism_body),
            "template_id": "proof-template." + canonical_sha256(template_body)[:24],
        },
        "positive_control": {
            "closed": positive.closed,
            "explored_states": positive.explored_states,
            "facts": list(positive.facts),
            "induction_variables": list(positive.induction_variables),
            "normal_form": positive.normal_form,
            "representation": positive.representation,
            "tactic_ids": list(positive.tactic_ids),
        },
        "route_id": route["route_id"],
        "status": "PASS_EXECUTABLE_PROOF_PLAN_ROUTE",
    }


def _build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root)
    results = [_route_result(route) for route in config["routes"]]
    ranked = sorted(
        results,
        key=lambda row: (
            -row["plan_template"]["metrics"]["falsification_power"],
            row["plan_template"]["metrics"]["premise_count"],
            row["plan_template"]["metrics"]["proof_debt"],
            row["plan_template"]["metrics"]["cost"],
            row["route_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["plan_template"]["rank"] = rank
    results = sorted(ranked, key=lambda row: MECHANISMS.index(row["mechanism"]))
    paths = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": {
            "applicability_establishes_proof": False,
            "closed_abstract_route_establishes_candidate_theorem": False,
            "proof_mechanism_novelty_established": False,
        },
        "search_contract": {
            "candidate_content_used_during_library_search": False,
            "missing_features_delete_plan": False,
            "ranking_order": [
                "falsification_power_desc",
                "premise_count_asc",
                "proof_debt_asc",
                "cost_asc",
            ],
            "search_algorithm": "deterministic_uniform_cost_tactic_graph",
        },
        "source_bindings": {
            key: {"normalized_file_sha256": _normalized_sha256(root / path), "path": path}
            for key, path in sorted(paths.items())
        },
        "routes": results,
        "summary": {
            "mechanisms": list(MECHANISMS),
            "mutation_controls_rejected": len(results),
            "positive_routes_closed": len(results),
            "status": "PASS_INDEPENDENT_PROOF_PLAN_SEARCH",
            "total_routes": len(results),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def run_proof_plan_search(root: Path) -> dict[str, Any]:
    """Search, rank, and seal the candidate-independent proof-plan library."""

    receipt = _build_receipt(root)
    validate_proof_plan_search(receipt, root)
    return receipt


def validate_proof_plan_search(
    value: Mapping[str, Any], root: Path | None = None
) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise IndependentProofPlanSearchError("proof-plan search content seal changed")
    _strict_keys(
        value,
        {
            "claims",
            "content_sha256",
            "routes",
            "schema_version",
            "search_contract",
            "source_bindings",
            "summary",
        },
        "proof-plan receipt",
    )
    if value["schema_version"] != RESULT_SCHEMA:
        raise IndependentProofPlanSearchError("proof-plan search result schema changed")
    summary = value["summary"]
    contract = value["search_contract"]
    routes = value["routes"]
    if (
        summary
        != {
            "mechanisms": list(MECHANISMS),
            "mutation_controls_rejected": len(MECHANISMS),
            "positive_routes_closed": len(MECHANISMS),
            "status": "PASS_INDEPENDENT_PROOF_PLAN_SEARCH",
            "total_routes": len(MECHANISMS),
        }
        or contract.get("candidate_content_used_during_library_search") is not False
        or contract.get("missing_features_delete_plan") is not False
        or contract.get("ranking_order")
        != [
            "falsification_power_desc",
            "premise_count_asc",
            "proof_debt_asc",
            "cost_asc",
        ]
        or not isinstance(routes, list)
        or [row.get("mechanism") for row in routes] != list(MECHANISMS)
        or {row.get("plan_template", {}).get("rank") for row in routes}
        != set(range(1, len(MECHANISMS) + 1))
    ):
        raise IndependentProofPlanSearchError("proof-plan search contract changed")
    if set(value["claims"]) != {
        "applicability_establishes_proof",
        "closed_abstract_route_establishes_candidate_theorem",
        "proof_mechanism_novelty_established",
    } or any(value["claims"].values()):
        raise IndependentProofPlanSearchError("proof-plan claim boundary changed")
    for row in routes:
        positive = row.get("positive_control", {})
        mutation = row.get("mutation_control", {})
        template = row.get("plan_template", {})
        if (
            row.get("status") != "PASS_EXECUTABLE_PROOF_PLAN_ROUTE"
            or positive.get("closed") is not True
            or mutation.get("closed") is not False
            or mutation.get("mutation_control_rejected") is not True
            or not template.get("tactic_ids")
            or not template.get("steps")
            or template.get("proof_mechanism_sha256")
            != canonical_sha256(
                {
                    "mechanism": template.get("mechanism"),
                    "steps": template.get("steps"),
                    "tactic_ids": template.get("tactic_ids"),
                }
            )
        ):
            raise IndependentProofPlanSearchError("proof-plan executable evidence changed")
    expected_sources = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    if set(value["source_bindings"]) != set(expected_sources):
        raise IndependentProofPlanSearchError("proof-plan source bindings changed")
    for key, path in expected_sources.items():
        binding = value["source_bindings"][key]
        _strict_keys(binding, {"normalized_file_sha256", "path"}, "proof-plan binding")
        if binding["path"] != path:
            raise IndependentProofPlanSearchError("proof-plan source path changed")
    if root is not None:
        root = root.resolve()
        for key, path in expected_sources.items():
            bound_path = (root / path).resolve()
            try:
                bound_path.relative_to(root)
            except ValueError as error:
                raise IndependentProofPlanSearchError("proof-plan source escapes root") from error
            if value["source_bindings"][key]["normalized_file_sha256"] != _normalized_sha256(
                bound_path
            ):
                raise IndependentProofPlanSearchError("proof-plan source hash changed")
        if dict(value) != _build_receipt(root):
            raise IndependentProofPlanSearchError("proof-plan receipt does not exactly replay")


def infer_candidate_capabilities(idea: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer only structural applicability features; never infer truth or novelty."""

    capabilities: set[str] = set()
    representation = str(idea.get("representation", "")).lower()
    family = str(idea.get("family", "")).lower()
    domains = " ".join(str(item).lower() for item in idea.get("source_idea_domains", []))
    if idea.get("invariants"):
        capabilities.add("declared_invariant")
    if idea.get("falsifiers"):
        capabilities.add("falsifiable_boundary")
    if representation in {
        "finite_product",
        "finite_sum",
        "generating_function",
        "linear_recurrence",
        "modular_relation",
        "recurrence",
    } or any(token in family + " " + domains for token in ("discrete", "integer", "sequence")):
        capabilities.update({"discrete_domain", "well_founded_order"})
    if any(token in family + " " + domains for token in ("combinator", "graph", "partition")):
        capabilities.add("combinatorial_structure")
    if representation in {
        "finite_product",
        "finite_sum",
        "fourier_transform",
        "generating_function",
        "laplace_transform",
        "linear_recurrence",
        "recurrence",
        "z_transform",
    }:
        capabilities.add("transformable_representation")
    return tuple(sorted(capabilities))


def plan_templates(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return validated templates in evidence-defined rank order."""

    validate_proof_plan_search(value)
    return tuple(
        row["plan_template"]
        for row in sorted(value["routes"], key=lambda item: item["plan_template"]["rank"])
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run":
        receipt = run_proof_plan_search(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path = args.receipt if args.receipt.is_absolute() else root / args.receipt
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_proof_plan_search(receipt, root)
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
