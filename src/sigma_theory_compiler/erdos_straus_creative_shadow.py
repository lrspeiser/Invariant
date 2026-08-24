"""Bounded LLM-to-GPU shadow experiment for the Erdős--Straus conjecture.

The open-problem gate remains closed.  This module exercises the creative machinery on a
declared finite range, compiles well-formed suggestions into a tiny search DSL, preserves
malformed and unsuccessful ideas, and uses the existing exact sweeper as the authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isqrt
from pathlib import Path
from typing import Any

import numpy as np

from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeBudget,
    ClaudeCallResult,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeRole,
    ClaudeStructuredOutput,
    Transport,
    urllib_transport,
)
from .core_credential import activated_credential
from .durable_llm_attempt_journal import (
    AttemptJournalError,
    DurableAttemptJournal,
    JournaledScheduledTransport,
)
from .exponent_diophantine_sweeper import (
    _backend,
    _es_hard_members,
    _es_hard_rounds,
    _host_array,
    es_witness_is_exact,
    run_erdos_straus_sweep,
)
from .exponent_diophantine_sweeper import (
    validate_receipt as validate_sweep_receipt,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/erdos_straus_creative_shadow.json"
OUTPUT_PATH = "runs/math/erdos-straus-creative-shadow/live-runtime.json"
SCHEMA_VERSION = "invariant-erdos-straus-creative-shadow-runtime-1.2"
CONFIG_SCHEMA = "invariant-erdos-straus-creative-shadow-config-1.2"
ESDSL2_CONTRACT_SCHEMA = "invariant-esdsl2-basis-semantics-contract-1.0"
SOURCE_PATH = "src/sigma_theory_compiler/erdos_straus_creative_shadow.py"
SWEEPER_PATH = "src/sigma_theory_compiler/exponent_diophantine_sweeper.py"
JOURNAL_PATH = "work/erdos-straus-creative-shadow/llm-attempts.jsonl"

_DSL1 = re.compile(r"^ESDSL1\|basis=([a-z_]+)\|x=([0-9,]+)\|t=([0-9,]+)\|m=([0-9,]+)$")
_BASES = {
    "continued_fraction",
    "descent_graph",
    "divisor_pair",
    "greedy_offset",
    "lattice_transform",
    "modular_sieve",
    "polynomial_ansatz",
    "residue_cover",
}
_ESDSL2_FIELDS = {
    "continued_fraction": ("a", "scale", "m"),
    "descent_graph": ("start", "moves", "depth", "m"),
    "divisor_pair": ("n", "shift", "m"),
    "greedy_offset": ("x", "t", "budget", "m"),
    "lattice_transform": ("u", "v", "matrix", "shift", "m"),
    "modular_sieve": ("x", "t", "congruence", "m"),
    "polynomial_ansatz": ("k", "xcoef", "tcoef", "m"),
    "residue_cover": ("q", "residues", "lifts", "m"),
}
_ESDSL2_OPERATORS = {
    "continued_fraction": "ordered_continued_fraction_convergents",
    "descent_graph": "bounded_descent_graph_reachability",
    "divisor_pair": "exact_divisor_factor_pairs",
    "greedy_offset": "diagonal_cost_greedy_budget",
    "lattice_transform": "integer_affine_lattice_image",
    "modular_sieve": "linear_congruence_filtered_product",
    "polynomial_ansatz": "integer_polynomial_parameter_map",
    "residue_cover": "complementary_residue_lifts",
}
_ESDSL2_CONTROL_CASES = (
    {
        "basis": "continued_fraction",
        "expression": "ESDSL2|basis=continued_fraction|a=1,2,2,2|scale=1|m=24",
        "matched_control": "ESDSL2|basis=continued_fraction|a=1,2,2,2|scale=2|m=24",
    },
    {
        "basis": "descent_graph",
        "expression": (
            "ESDSL2|basis=descent_graph|start=4,4|moves=2,-1;-1,2|depth=2|m=24"
        ),
        "matched_control": (
            "ESDSL2|basis=descent_graph|start=4,4|moves=1,0;0,1|depth=2|m=24"
        ),
    },
    {
        "basis": "divisor_pair",
        "expression": "ESDSL2|basis=divisor_pair|n=12,18|shift=1,2|m=24",
        "matched_control": "ESDSL2|basis=divisor_pair|n=12,18|shift=2,1|m=24",
    },
    {
        "basis": "greedy_offset",
        "expression": "ESDSL2|basis=greedy_offset|x=0,4,9|t=1,5,10|budget=5|m=24",
        "matched_control": (
            "ESDSL2|basis=greedy_offset|x=0,4,9|t=2,6,11|budget=5|m=24"
        ),
    },
    {
        "basis": "lattice_transform",
        "expression": (
            "ESDSL2|basis=lattice_transform|u=0,1|v=0,2|"
            "matrix=2,1,1,3|shift=1,2|m=24"
        ),
        "matched_control": (
            "ESDSL2|basis=lattice_transform|u=0,1|v=0,2|"
            "matrix=3,1,1,2|shift=1,2|m=24"
        ),
    },
    {
        "basis": "modular_sieve",
        "expression": (
            "ESDSL2|basis=modular_sieve|x=0,1,2,3|t=0,1,2,3|"
            "congruence=1,1,0,2|m=24"
        ),
        "matched_control": (
            "ESDSL2|basis=modular_sieve|x=0,1,2,3|t=0,1,2,3|"
            "congruence=1,1,1,2|m=24"
        ),
    },
    {
        "basis": "polynomial_ansatz",
        "expression": (
            "ESDSL2|basis=polynomial_ansatz|k=0,1,2,3|xcoef=1,1|tcoef=0,0,1|m=24"
        ),
        "matched_control": (
            "ESDSL2|basis=polynomial_ansatz|k=0,1,2,3|xcoef=2,1|tcoef=0,0,1|m=24"
        ),
    },
    {
        "basis": "residue_cover",
        "expression": "ESDSL2|basis=residue_cover|q=5|residues=1,2,4|lifts=0,1|m=24",
        "matched_control": (
            "ESDSL2|basis=residue_cover|q=5|residues=0,2,3|lifts=0,1|m=24"
        ),
    },
)
_KNOWN_MECHANISM_TERMS = {
    "continued_fraction": ("continued fraction", "ceiling fraction"),
    "divisor_factorization": ("divisor", "factor", "b^2"),
    "greedy_egyptian_fraction": ("greedy", "egyptian"),
    "lattice_parameterization": ("lattice", "affine"),
    "modular_identity": ("modular", "residue", "congruence"),
    "polynomial_family": ("polynomial", "parametric"),
}


class ErdosStrausCreativeShadowError(ValueError):
    """Raised when the bounded experiment or its receipt fails closed."""


def _file_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load_config(root: Path, config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    path = root / config_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {
        "campaign_id",
        "claim_policy",
        "claude",
        "experiment",
        "literature",
        "problem",
        "recovery",
        "schema_version",
    }:
        raise ErdosStrausCreativeShadowError("config keys changed")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ErdosStrausCreativeShadowError("config schema changed")
    policy = value["claim_policy"]
    if (
        policy.get("famous_open_problem_attempt_authorized") is not False
        or policy.get("shadow_mechanism_experiment_only") is not True
        or policy.get("level5_process_passes", 1) >= policy.get("minimum_level5_process_passes", 0)
    ):
        raise ErdosStrausCreativeShadowError("open-problem claim gate is not closed")
    experiment = value["experiment"]
    if not (
        13
        <= experiment["calibration_n_max"]
        <= experiment["benchmark_n_max"]
        <= experiment["search_n_max"]
        <= 100_000_000
    ):
        raise ErdosStrausCreativeShadowError("experiment ranges are invalid")
    try:
        creative_roles = [ClaudeRole(item) for item in experiment["creative_roles"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ErdosStrausCreativeShadowError("creative role allocation is invalid") from error
    if not creative_roles or ClaudeRole.CRITIC in creative_roles:
        raise ErdosStrausCreativeShadowError("creative role allocation is invalid")
    critic_batch_size = experiment.get("llm_critic_batch_size")
    if (
        isinstance(critic_batch_size, bool)
        or not isinstance(critic_batch_size, int)
        or not 0 <= critic_batch_size <= 16
    ):
        raise ErdosStrausCreativeShadowError("LLM critic batch size is invalid")
    if experiment.get("proposal_dsl_version") not in {"ESDSL1", "ESDSL2"}:
        raise ErdosStrausCreativeShadowError("proposal DSL version is invalid")
    wall_clock_ceiling = experiment.get("matched_control_wall_clock_ceiling_seconds")
    if (
        isinstance(wall_clock_ceiling, bool)
        or not isinstance(wall_clock_ceiling, int)
        or not 1 <= wall_clock_ceiling <= 60
    ):
        raise ErdosStrausCreativeShadowError("matched control wall-clock ceiling is invalid")
    if value["claude"]["maximum_calls"] != 4 or value["claude"]["maximum_total_tokens"] > 32_000:
        raise ErdosStrausCreativeShadowError("open-problem shadow budget changed")
    return value


def _public_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    if experiment["proposal_dsl_version"] == "ESDSL1":
        machine_recipe = (
            "The expression field must be exactly ESDSL1|basis=B|x=X|t=T|m=M. "
            "B is one of continued_fraction, descent_graph, divisor_pair, greedy_offset, "
            "lattice_transform, modular_sieve, polynomial_ansatz, residue_cover. X and T "
            f"are comma-separated sets of 2 to {experiment['maximum_offsets_per_axis']} "
            f"integers from 0 to {experiment['maximum_offset']}; M is 1 to "
            f"{experiment['maximum_moduli_per_recipe']} moduli from 2 to 256."
        )
    else:
        machine_recipe = (
            "The expression field must use exactly one strict ESDSL2 basis form: "
            "continued_fraction|a=A|scale=S; "
            "descent_graph|start=X,T|moves=DX,DT;...|depth=D; "
            "divisor_pair|n=N|shift=X,T; greedy_offset|x=X|t=T|budget=K; "
            "lattice_transform|u=U|v=V|matrix=A,B,C,D|shift=X,T; "
            "modular_sieve|x=X|t=T|congruence=A,B,R,Q; "
            "polynomial_ansatz|k=K|xcoef=C|tcoef=C; or "
            "residue_cover|q=Q|residues=R|lifts=L. Prefix with ESDSL2|basis=B and "
            "suffix with |m=M. Preserve the shown field order and use only bounded integers."
        )
    return {
        "experiment_boundary": (
            "Finite mechanism shadow only. Do not claim a proof, a new verification bound, "
            "or mathematical novelty. Every idea is retained even when malformed or falsified."
        ),
        "known_computational_baseline": (
            "For n congruent to 1 mod 12, set x=floor(n/4)+1+dx, a=4x-n, b=nx, "
            "y=ceil(b/a)+t, and accept when d=ay-b divides by exactly testing d | by."
        ),
        "machine_recipe": machine_recipe,
        "problem_equation": config["problem"]["equation"],
        "problem_id": config["problem"]["id"],
        "requested_diversity": (
            "Prefer mechanisms from distant domains and different parameter scales. Label each "
            "as known_rewrite, cross_domain_synthesis, proposed_new_construction, or uncertain; "
            "the label will never be used to prune it."
        ),
    }


def _instruction(role: ClaudeRole, requested_ideas: int) -> str:
    return (
        f"Act as the {role.value} in a bounded Erdős--Straus mechanism experiment. Produce "
        f"exactly {requested_ideas} structurally different recipes spanning distant analogies, "
        "invented representations, recombinations, and independent proof strategies, not "
        "cosmetic variants. Follow machine_recipe exactly in every expression field. Put the "
        "mathematical explanation in rationale, invariants, proof_plan, known_analogues, "
        "source_idea_domains, and synthesis_note. Origin labels are fallible self-assessments "
        "and do not establish novelty. Failed ideas remain valuable."
    )


def _critic_instruction(batch_size: int) -> str:
    return (
        f"Critique exactly the {batch_size} supplied candidates in this bounded Erdős--Straus "
        "mechanism experiment. Return one typed steering action for every candidate. A reject "
        "or repair verdict must not delete an idea: identify a blocker and a concrete repair or "
        "recombination. Do not claim proof, novelty, or progress on the open conjecture."
    )


def _parse_numbers(raw: str, *, minimum: int, maximum: int, cap: int) -> tuple[int, ...] | None:
    if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", raw) is None:
        return None
    values = tuple(sorted({int(item) for item in raw.split(",")}))
    if not 1 <= len(values) <= cap or any(item < minimum or item > maximum for item in values):
        return None
    return values


def _parse_canonical_numbers(
    raw: str, *, minimum: int, maximum: int, cap: int
) -> tuple[int, ...] | None:
    values = _parse_numbers(raw, minimum=minimum, maximum=maximum, cap=cap)
    if values is None or raw != ",".join(str(item) for item in values):
        return None
    return values


def _parse_ordered_integers(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    minimum_count: int,
    maximum_count: int,
) -> tuple[int, ...] | None:
    if re.fullmatch(r"-?[0-9]+(?:,-?[0-9]+)*", raw) is None:
        return None
    values = tuple(int(item) for item in raw.split(","))
    if (
        not minimum_count <= len(values) <= maximum_count
        or any(item < minimum or item > maximum for item in values)
        or raw != ",".join(str(item) for item in values)
    ):
        return None
    return values


def _parse_scalar(raw: str, *, minimum: int, maximum: int) -> int | None:
    parsed = _parse_ordered_integers(
        raw,
        minimum=minimum,
        maximum=maximum,
        minimum_count=1,
        maximum_count=1,
    )
    return None if parsed is None else parsed[0]


def _parse_esdsl2_fields(expression: str) -> tuple[str, dict[str, str]] | None:
    parts = expression.strip().split("|")
    if len(parts) < 4 or parts[0] != "ESDSL2" or not parts[1].startswith("basis="):
        return None
    basis = parts[1][len("basis=") :]
    if basis not in _ESDSL2_FIELDS:
        return None
    fields: dict[str, str] = {}
    ordered_keys = []
    for part in parts[2:]:
        if part.count("=") != 1:
            return None
        key, raw = part.split("=", 1)
        if not key or not raw or key in fields:
            return None
        ordered_keys.append(key)
        fields[key] = raw
    if tuple(ordered_keys) != _ESDSL2_FIELDS[basis]:
        return None
    return basis, fields


def _ordered_unique_pairs(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(dict.fromkeys((int(dx), int(t)) for dx, t in pairs))


def _validate_esdsl2_schedule(
    pairs: Sequence[tuple[int, int]], experiment: Mapping[str, Any]
) -> tuple[tuple[int, int], ...]:
    result = _ordered_unique_pairs(pairs)
    maximum = int(experiment["maximum_offset"])
    cap = int(experiment["maximum_offsets_per_axis"]) ** 2
    if (
        not 1 <= len(result) <= cap
        or any(dx < 0 or t < 0 or dx > maximum or t > maximum for dx, t in result)
    ):
        raise ErdosStrausCreativeShadowError("ESDSL2 schedule is empty, oversized, or out of bounds")
    return result


def _schedule_esdsl2(
    recipe: Mapping[str, Any], experiment: Mapping[str, Any]
) -> tuple[tuple[int, int], ...]:
    basis = str(recipe["basis"])
    parameters = recipe["parameters"]
    pairs: list[tuple[int, int]] = []
    if basis == "continued_fraction":
        previous_previous_p, previous_p = 0, 1
        previous_previous_q, previous_q = 1, 0
        scale = int(parameters["scale"])
        for partial_quotient in parameters["partial_quotients"]:
            p = int(partial_quotient) * previous_p + previous_previous_p
            q = int(partial_quotient) * previous_q + previous_previous_q
            pairs.append((scale * p, scale * q))
            previous_previous_p, previous_p = previous_p, p
            previous_previous_q, previous_q = previous_q, q
    elif basis == "descent_graph":
        start = tuple(int(item) for item in parameters["start"])
        moves = [tuple(int(item) for item in move) for move in parameters["moves"]]
        maximum = int(experiment["maximum_offset"])
        seen = {start}
        frontier = [start]
        pairs.append(start)
        for _ in range(int(parameters["depth"])):
            next_frontier = sorted(
                {
                    (node[0] + move[0], node[1] + move[1])
                    for node in frontier
                    for move in moves
                    if 0 <= node[0] + move[0] <= maximum
                    and 0 <= node[1] + move[1] <= maximum
                }
                - seen
            )
            pairs.extend(next_frontier)
            seen.update(next_frontier)
            frontier = next_frontier
    elif basis == "divisor_pair":
        shift_x, shift_t = (int(item) for item in parameters["shift"])
        for composite in parameters["integers"]:
            value = int(composite)
            for divisor in range(1, isqrt(value) + 1):
                if value % divisor == 0:
                    pairs.append((divisor + shift_x, value // divisor + shift_t))
    elif basis == "greedy_offset":
        candidates = sorted(
            (
                (int(dx), int(t))
                for dx in parameters["x_offsets"]
                for t in parameters["t_offsets"]
            ),
            key=lambda item: (sum(item), abs(item[0] - item[1]), item),
        )
        pairs.extend(candidates[: int(parameters["budget"])])
    elif basis == "lattice_transform":
        a, b, c, d = (int(item) for item in parameters["matrix"])
        shift_x, shift_t = (int(item) for item in parameters["shift"])
        pairs.extend(
            (
                a * int(u) + b * int(v) + shift_x,
                c * int(u) + d * int(v) + shift_t,
            )
            for u in parameters["u_coordinates"]
            for v in parameters["v_coordinates"]
        )
    elif basis == "modular_sieve":
        coefficient_x, coefficient_t, residue, modulus = (
            int(item) for item in parameters["congruence"]
        )
        pairs.extend(
            (int(dx), int(t))
            for dx in parameters["x_offsets"]
            for t in parameters["t_offsets"]
            if (coefficient_x * int(dx) + coefficient_t * int(t) - residue) % modulus == 0
        )
    elif basis == "polynomial_ansatz":
        x_coefficients = tuple(int(item) for item in parameters["x_coefficients"])
        t_coefficients = tuple(int(item) for item in parameters["t_coefficients"])
        for raw_k in parameters["parameter_values"]:
            k = int(raw_k)
            pairs.append(
                (
                    sum(coefficient * k**power for power, coefficient in enumerate(x_coefficients)),
                    sum(coefficient * k**power for power, coefficient in enumerate(t_coefficients)),
                )
            )
    elif basis == "residue_cover":
        modulus = int(parameters["modulus"])
        pairs.extend(
            (
                int(residue) + modulus * int(lift),
                (-int(residue)) % modulus + modulus * int(lift),
            )
            for residue in parameters["residues"]
            for lift in parameters["lifts"]
        )
    else:
        raise ErdosStrausCreativeShadowError("ESDSL2 basis has no executable semantics")
    return _validate_esdsl2_schedule(pairs, experiment)


def _parse_esdsl2(expression: str, experiment: Mapping[str, Any]) -> dict[str, Any] | None:
    parsed_fields = _parse_esdsl2_fields(expression)
    if parsed_fields is None:
        return None
    basis, fields = parsed_fields
    cap = int(experiment["maximum_offsets_per_axis"])
    maximum = int(experiment["maximum_offset"])
    moduli = _parse_canonical_numbers(
        fields["m"],
        minimum=2,
        maximum=256,
        cap=int(experiment["maximum_moduli_per_recipe"]),
    )
    if moduli is None:
        return None
    parameters: dict[str, Any]
    if basis == "continued_fraction":
        partial_quotients = _parse_ordered_integers(
            fields["a"], minimum=1, maximum=16, minimum_count=2, maximum_count=cap
        )
        scale = _parse_scalar(fields["scale"], minimum=1, maximum=16)
        if partial_quotients is None or scale is None:
            return None
        parameters = {"partial_quotients": list(partial_quotients), "scale": scale}
    elif basis == "descent_graph":
        start = _parse_ordered_integers(
            fields["start"], minimum=0, maximum=maximum, minimum_count=2, maximum_count=2
        )
        raw_moves = fields["moves"].split(";")
        moves = [
            _parse_ordered_integers(
                raw, minimum=-maximum, maximum=maximum, minimum_count=2, maximum_count=2
            )
            for raw in raw_moves
        ]
        depth = _parse_scalar(fields["depth"], minimum=1, maximum=4)
        if (
            start is None
            or not 1 <= len(moves) <= 4
            or any(move is None or move == (0, 0) for move in moves)
            or depth is None
        ):
            return None
        parameters = {"depth": depth, "moves": [list(move) for move in moves], "start": list(start)}
    elif basis == "divisor_pair":
        integers = _parse_canonical_numbers(fields["n"], minimum=2, maximum=256, cap=4)
        shift = _parse_ordered_integers(
            fields["shift"], minimum=0, maximum=maximum, minimum_count=2, maximum_count=2
        )
        if integers is None or shift is None:
            return None
        parameters = {"integers": list(integers), "shift": list(shift)}
    elif basis == "greedy_offset":
        x_offsets = _parse_canonical_numbers(fields["x"], minimum=0, maximum=maximum, cap=cap)
        t_offsets = _parse_canonical_numbers(fields["t"], minimum=0, maximum=maximum, cap=cap)
        budget = _parse_scalar(fields["budget"], minimum=1, maximum=cap**2)
        if (
            x_offsets is None
            or t_offsets is None
            or budget is None
            or budget > len(x_offsets) * len(t_offsets)
        ):
            return None
        parameters = {
            "budget": budget,
            "t_offsets": list(t_offsets),
            "x_offsets": list(x_offsets),
        }
    elif basis == "lattice_transform":
        u_coordinates = _parse_canonical_numbers(fields["u"], minimum=0, maximum=16, cap=4)
        v_coordinates = _parse_canonical_numbers(fields["v"], minimum=0, maximum=16, cap=4)
        matrix = _parse_ordered_integers(
            fields["matrix"], minimum=-16, maximum=16, minimum_count=4, maximum_count=4
        )
        shift = _parse_ordered_integers(
            fields["shift"], minimum=0, maximum=maximum, minimum_count=2, maximum_count=2
        )
        if (
            u_coordinates is None
            or v_coordinates is None
            or matrix is None
            or matrix[0] * matrix[3] == matrix[1] * matrix[2]
            or shift is None
        ):
            return None
        parameters = {
            "matrix": list(matrix),
            "shift": list(shift),
            "u_coordinates": list(u_coordinates),
            "v_coordinates": list(v_coordinates),
        }
    elif basis == "modular_sieve":
        x_offsets = _parse_canonical_numbers(fields["x"], minimum=0, maximum=maximum, cap=cap)
        t_offsets = _parse_canonical_numbers(fields["t"], minimum=0, maximum=maximum, cap=cap)
        congruence = _parse_ordered_integers(
            fields["congruence"], minimum=-16, maximum=64, minimum_count=4, maximum_count=4
        )
        if x_offsets is None or t_offsets is None or congruence is None:
            return None
        coefficient_x, coefficient_t, residue, modulus = congruence
        if (
            coefficient_x == coefficient_t == 0
            or not 2 <= modulus <= 64
            or not 0 <= residue < modulus
        ):
            return None
        parameters = {
            "congruence": list(congruence),
            "t_offsets": list(t_offsets),
            "x_offsets": list(x_offsets),
        }
    elif basis == "polynomial_ansatz":
        parameter_values = _parse_canonical_numbers(
            fields["k"], minimum=0, maximum=16, cap=cap
        )
        x_coefficients = _parse_ordered_integers(
            fields["xcoef"], minimum=-16, maximum=16, minimum_count=1, maximum_count=3
        )
        t_coefficients = _parse_ordered_integers(
            fields["tcoef"], minimum=-16, maximum=16, minimum_count=1, maximum_count=3
        )
        if parameter_values is None or x_coefficients is None or t_coefficients is None:
            return None
        parameters = {
            "parameter_values": list(parameter_values),
            "t_coefficients": list(t_coefficients),
            "x_coefficients": list(x_coefficients),
        }
    else:
        modulus = _parse_scalar(fields["q"], minimum=2, maximum=64)
        if modulus is None:
            return None
        residues = _parse_canonical_numbers(
            fields["residues"], minimum=0, maximum=modulus - 1, cap=cap
        )
        lifts = _parse_canonical_numbers(fields["lifts"], minimum=0, maximum=cap, cap=cap)
        if residues is None or lifts is None:
            return None
        parameters = {"lifts": list(lifts), "modulus": modulus, "residues": list(residues)}
    recipe = {
        "basis": basis,
        "dsl_version": "ESDSL2",
        "moduli": list(moduli),
        "parameters": parameters,
        "semantic_operator": _ESDSL2_OPERATORS[basis],
    }
    try:
        _schedule_esdsl2(recipe, experiment)
    except ErdosStrausCreativeShadowError:
        return None
    return recipe


def parse_recipe(expression: str, experiment: Mapping[str, Any]) -> dict[str, Any] | None:
    match = _DSL1.fullmatch(expression.strip())
    if match is not None and match.group(1) in _BASES:
        cap = int(experiment["maximum_offsets_per_axis"])
        maximum = int(experiment["maximum_offset"])
        x_offsets = _parse_numbers(match.group(2), minimum=0, maximum=maximum, cap=cap)
        t_offsets = _parse_numbers(match.group(3), minimum=0, maximum=maximum, cap=cap)
        moduli = _parse_numbers(
            match.group(4),
            minimum=2,
            maximum=256,
            cap=int(experiment["maximum_moduli_per_recipe"]),
        )
        if x_offsets is None or t_offsets is None or moduli is None:
            return None
        return {
            "basis": match.group(1),
            "moduli": list(moduli),
            "t_offsets": list(t_offsets),
            "x_offsets": list(x_offsets),
        }
    return _parse_esdsl2(expression, experiment)


def _schedule_pairs(
    recipe: Mapping[str, Any], experiment: Mapping[str, Any] | None = None
) -> tuple[tuple[int, int], ...]:
    if recipe.get("dsl_version") == "ESDSL2":
        if experiment is None:
            raise ErdosStrausCreativeShadowError("ESDSL2 schedule requires experiment bounds")
        return _schedule_esdsl2(recipe, experiment)
    return tuple(
        sorted((int(dx), int(t)) for dx in recipe["x_offsets"] for t in recipe["t_offsets"])
    )


def _esdsl2_semantics_contract(experiment: Mapping[str, Any]) -> dict[str, Any]:
    legacy_expression = "ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120"
    legacy_recipe = parse_recipe(legacy_expression, experiment)
    legacy_expected_recipe = {
        "basis": "lattice_transform",
        "moduli": [24, 120],
        "t_offsets": [0, 7],
        "x_offsets": [0, 2, 65],
    }
    legacy_expected_schedule = ((0, 0), (0, 7), (2, 0), (2, 7), (65, 0), (65, 7))
    if (
        legacy_recipe != legacy_expected_recipe
        or _schedule_pairs(legacy_recipe or {}) != legacy_expected_schedule
    ):
        raise ErdosStrausCreativeShadowError("ESDSL1 compatibility contract changed")

    controls = []
    schedule_hashes = set()
    for case in _ESDSL2_CONTROL_CASES:
        recipe = parse_recipe(case["expression"], experiment)
        matched_recipe = parse_recipe(case["matched_control"], experiment)
        if (
            recipe is None
            or matched_recipe is None
            or recipe["basis"] != case["basis"]
            or matched_recipe["basis"] != case["basis"]
        ):
            raise ErdosStrausCreativeShadowError("ESDSL2 positive control did not compile")
        schedule = _schedule_pairs(recipe, experiment)
        matched_schedule = _schedule_pairs(matched_recipe, experiment)
        grammar_field_count_matched = len(case["expression"].split("|")) == len(
            case["matched_control"].split("|")
        )
        if (
            schedule == matched_schedule
            or len(schedule) != len(matched_schedule)
            or not grammar_field_count_matched
        ):
            raise ErdosStrausCreativeShadowError("ESDSL2 matched structural control changed")
        schedule_sha256 = canonical_sha256(schedule)
        schedule_hashes.add(schedule_sha256)
        controls.append(
            {
                "basis": case["basis"],
                "expression": case["expression"],
                "grammar_field_count_matched": grammar_field_count_matched,
                "matched_control_expression": case["matched_control"],
                "matched_control_pairs": [list(pair) for pair in matched_schedule],
                "matched_control_schedule_sha256": canonical_sha256(matched_schedule),
                "pair_count": len(schedule),
                "pair_count_matched": len(schedule) == len(matched_schedule),
                "schedule_pairs": [list(pair) for pair in schedule],
                "schedule_sha256": schedule_sha256,
                "semantic_operator": recipe["semantic_operator"],
                "verifier_lane_budget_matched": len(schedule) == len(matched_schedule),
            }
        )
    if len(controls) != len(_BASES) or len(schedule_hashes) != len(_BASES):
        raise ErdosStrausCreativeShadowError("ESDSL2 basis schedules are missing or collapsed")
    return {
        "all_basis_schedules_nonempty": all(item["pair_count"] > 0 for item in controls),
        "basis_controls": controls,
        "basis_schedule_sha256s_unique": True,
        "claim_boundary": {
            "basis_control_success_establishes_mathematical_novelty": False,
            "basis_label_establishes_causal_mechanism": False,
            "compiler_contract_decides_erdos_straus": False,
        },
        "esdsl1_compatibility": {
            "expression": legacy_expression,
            "parse_shape_unchanged": True,
            "recipe": legacy_expected_recipe,
            "schedule_pairs": [list(pair) for pair in legacy_expected_schedule],
            "schedule_sha256": canonical_sha256(legacy_expected_schedule),
        },
        "matched_control_contract": (
            "same basis, grammar field count, direct pair count, and exact verifier lane budget; "
            "one typed structural parameter changed"
        ),
        "schema_version": ESDSL2_CONTRACT_SCHEMA,
    }


def _run_pairs_with_attribution(
    xp: Any, members: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    started = time.perf_counter()
    n = xp.asarray(members)
    resolved = xp.zeros(n.shape, dtype=bool)
    wx = xp.zeros(n.shape, dtype=xp.int64)
    wy = xp.zeros(n.shape, dtype=xp.int64)
    winning_dx = xp.full(n.shape, -1, dtype=xp.int64)
    winning_t = xp.full(n.shape, -1, dtype=xp.int64)
    lane_tests = 0
    for dx, t in pairs:
        open_lanes = ~resolved
        open_count = int(open_lanes.sum())
        if open_count == 0:
            break
        lane_tests += open_count
        a = 3 + 4 * dx
        x = n // 4 + 1 + dx
        b = n * x
        y = (b + a - 1) // a + t
        d = a * y - b
        safe = xp.where(d > 0, d, xp.int64(1))
        ok = open_lanes & (d > 0) & (y >= x) & (((b % safe) * (y % safe)) % safe == 0)
        wx = xp.where(ok, x, wx)
        wy = xp.where(ok, y, wy)
        winning_dx = xp.where(ok, xp.int64(dx), winning_dx)
        winning_t = xp.where(ok, xp.int64(t), winning_t)
        resolved = resolved | ok
    elapsed = time.perf_counter() - started
    return (
        _host_array(wx),
        _host_array(wy),
        _host_array(resolved),
        _host_array(winning_dx),
        _host_array(winning_t),
        lane_tests,
        elapsed,
    )


def _run_pairs(
    xp: Any, members: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    wx, wy, resolved, _, _, lane_tests, elapsed = _run_pairs_with_attribution(
        xp, members, pairs
    )
    return wx, wy, resolved, lane_tests, elapsed


def _run_pairs_fixed_lane_budget(
    xp: Any, members: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Run every pair on every member while preserving first-success witness semantics."""

    started = time.perf_counter()
    n = xp.asarray(members)
    resolved = xp.zeros(n.shape, dtype=bool)
    wx = xp.zeros(n.shape, dtype=xp.int64)
    wy = xp.zeros(n.shape, dtype=xp.int64)
    lane_tests = 0
    for dx, t in pairs:
        lane_tests += int(n.size)
        a = 3 + 4 * dx
        x = n // 4 + 1 + dx
        b = n * x
        y = (b + a - 1) // a + t
        d = a * y - b
        safe = xp.where(d > 0, d, xp.int64(1))
        ok = (~resolved) & (d > 0) & (y >= x) & (((b % safe) * (y % safe)) % safe == 0)
        wx = xp.where(ok, x, wx)
        wy = xp.where(ok, y, wy)
        resolved = resolved | ok
    elapsed = time.perf_counter() - started
    return _host_array(wx), _host_array(wy), _host_array(resolved), lane_tests, elapsed


def _witness_sample(
    members: np.ndarray, wx: np.ndarray, wy: np.ndarray, resolved: np.ndarray, limit: int = 8
) -> list[dict[str, int]]:
    indices = np.flatnonzero(resolved)
    if indices.size > limit:
        indices = indices[np.linspace(0, indices.size - 1, limit, dtype=np.int64)]
    sample = []
    for index in indices:
        n, x, y = int(members[index]), int(wx[index]), int(wy[index])
        b = n * x
        d = (4 * x - n) * y - b
        z = (b * y) // d
        if not es_witness_is_exact(n, x, y, z):
            raise ErdosStrausCreativeShadowError("creative schedule produced a false witness")
        sample.append({"n": n, "x": x, "y": y, "z": z})
    return sample


def _residue_evidence(
    members: np.ndarray, resolved: np.ndarray, moduli: Sequence[int]
) -> list[dict[str, Any]]:
    evidence = []
    for modulus in moduli:
        all_counts = np.bincount(members % modulus, minlength=modulus)
        hit_counts = np.bincount(members[resolved] % modulus, minlength=modulus)
        complete = [
            residue
            for residue in range(modulus)
            if all_counts[residue] >= 32 and hit_counts[residue] == all_counts[residue]
        ]
        evidence.append(
            {
                "complete_residues_in_finite_calibration": complete,
                "modulus": modulus,
                "observed_residues": int(np.count_nonzero(hit_counts)),
            }
        )
    return evidence


def _known_overlap(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [
            str(hypothesis.get("expression", "")),
            str(hypothesis.get("rationale", "")),
            " ".join(hypothesis.get("known_analogues", [])),
            " ".join(hypothesis.get("source_idea_domains", [])),
        ]
    ).lower()
    matches = sorted(
        name
        for name, terms in _KNOWN_MECHANISM_TERMS.items()
        if any(term in text for term in terms)
    )
    return {
        "automated_mechanism_matches": matches,
        "human_prior_art_review": "NOT_PERFORMED",
        "novelty_established": False,
        "status": "POSSIBLE_KNOWN_OVERLAP" if matches else "NO_COARSE_MATCH_NOT_A_NOVELTY_RESULT",
    }


def _candidate_id(role: ClaudeRole, hypothesis_id: str, ordinal: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", hypothesis_id).strip("_.-")
    return f"{role.value}.{ordinal:02d}.{clean}"[:128]


def _journal_bindings(
    root: Path, config: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    return {
        "campaign_id": config["campaign_id"],
        "config_sha256": _file_sha256(root / config_path),
        "source_sha256": _file_sha256(root / SOURCE_PATH),
    }


def _open_attempt_journal(
    root: Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    journal_path: str | Path,
) -> DurableAttemptJournal:
    path = Path(journal_path)
    if not path.is_absolute():
        path = root / path
    expected = _journal_bindings(root, config, config_path)
    if path.exists():
        journal = DurableAttemptJournal.load(path)
        header = journal.header
        if (
            header.get("experiment_id") != config["campaign_id"]
            or header.get("source_bindings") != expected
        ):
            raise ErdosStrausCreativeShadowError(
                "attempt journal is bound to another campaign, config, or source"
            )
        return journal
    return DurableAttemptJournal.create(
        path,
        experiment_id=config["campaign_id"],
        source_bindings=expected,
        unblinding_key=secrets.token_bytes(32),
    )


def _call_id(config: Mapping[str, Any], phase: str, ordinal: int, role: ClaudeRole) -> str:
    return f"{config['campaign_id']}:{phase}.{ordinal:02d}:{role.value}"


def _result_from_dict(value: Mapping[str, Any] | None) -> ClaudeCallResult | None:
    if value is None:
        return None
    output_raw = value.get("output")
    output = None
    if output_raw is not None:
        parseable = {key: item for key, item in output_raw.items() if key != "quarantine"}
        output = ClaudeStructuredOutput.from_mapping(parseable)
    return ClaudeCallResult(
        ClaudeCallStatus(value["status"]),
        ClaudeRole(value["role"]),
        value["benchmark_id"],
        output,
        dict(value["evidence"]),
    )


def _outcome(journal: DurableAttemptJournal, call_id: str) -> dict[str, Any] | None:
    events = [
        item
        for item in journal.events_for(call_id)
        if item["event_kind"] == "scheduled_call_outcome"
    ]
    if len(events) > 1:
        raise ErdosStrausCreativeShadowError("scheduled LLM call has multiple outcomes")
    return None if not events else dict(events[0]["payload"])


def _append_outcome(
    journal: DurableAttemptJournal,
    call_id: str,
    *,
    phase: str,
    role: ClaudeRole,
    status: str,
    result: ClaudeCallResult | None,
    errors: Sequence[str],
) -> dict[str, Any]:
    event = journal.append(
        "scheduled_call_outcome",
        call_id,
        {
            "errors": [str(item)[:1024] for item in errors],
            "phase": phase,
            "result": None if result is None else result.to_dict(),
            "role": role.value,
            "status": status,
        },
    )
    return dict(event["payload"])


def _response_events(
    journal: DurableAttemptJournal, call_id: str | None = None
) -> list[dict[str, Any]]:
    events = journal.events if call_id is None else journal.events_for(call_id)
    return [dict(item) for item in events if item["event_kind"] == "message_response"]


def _restore_budget(
    client: ClaudeCreativityClient,
    journal: DurableAttemptJournal,
    *,
    exclude_call_id: str | None = None,
) -> None:
    calls = sum(
        1
        for event in journal.events
        if event["event_kind"] == "message_dispatch"
        and event["scheduled_call_id"] != exclude_call_id
    )
    input_tokens = 0
    output_tokens = 0
    for event in _response_events(journal):
        if event["scheduled_call_id"] == exclude_call_id:
            continue
        usage = event["payload"].get("response", {}).get("usage", {})
        raw_input = usage.get("input_tokens", 0)
        raw_output = usage.get("output_tokens", 0)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (raw_input, raw_output)
        ):
            raise ErdosStrausCreativeShadowError("journaled provider usage is invalid")
        input_tokens += raw_input
        output_tokens += raw_output
    client.budget = ClaudeBudget(calls, input_tokens, output_tokens)


def _restore_model_evidence(
    client: ClaudeCreativityClient, journal: DurableAttemptJournal
) -> None:
    probes = [
        item for item in journal.events if item["event_kind"] == "model_probe_response"
    ]
    if not probes:
        raise ErdosStrausCreativeShadowError(
            "journaled response cannot replay without its model-capability probe"
        )
    recovered = []
    for probe in probes:
        payload = probe["payload"]
        response = payload.get("response", {})
        capabilities = response.get("capabilities", {})
        structured = (
            capabilities.get("structured_outputs", {})
            if isinstance(capabilities, Mapping)
            else {}
        )
        if (
            payload.get("status") != 200
            or response.get("id") != client.config.model
            or not isinstance(structured, Mapping)
            or structured.get("supported") is not True
        ):
            raise ErdosStrausCreativeShadowError(
                "journaled model-capability evidence is invalid"
            )
        recovered.append(
            {
                "capabilities_sha256": canonical_sha256(capabilities),
                "model": client.config.model,
                "structured_outputs_supported": True,
            }
        )
    if any(item != recovered[0] for item in recovered[1:]):
        raise ErdosStrausCreativeShadowError("journaled model-capability probes disagree")
    client._model_evidence = recovered[0]


class _ReplayMessageTransport:
    """Replay one durable provider response against the exact original request bytes."""

    def __init__(self, dispatch: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        self.dispatch = dispatch
        self.response = response
        self.used = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        del headers, timeout
        if self.used or method != "POST" or body is None:
            raise ErdosStrausCreativeShadowError("journal replay attempted an unexpected request")
        expected = self.dispatch["payload"]
        if (
            expected.get("method") != method
            or expected.get("url") != url
            or expected.get("body_sha256") != hashlib.sha256(body).hexdigest()
        ):
            raise ErdosStrausCreativeShadowError(
                "journaled response request binding changed during replay"
            )
        self.used = True
        return int(self.response["payload"]["status"]), dict(
            self.response["payload"]["response"]
        )


def _run_scheduled_call(
    journal: DurableAttemptJournal,
    client: ClaudeCreativityClient,
    *,
    call_id: str,
    phase: str,
    role: ClaudeRole,
    public_payload: Mapping[str, Any],
    candidate_summaries: Sequence[Mapping[str, Any]],
    instruction: str,
    hypothesis_slots: int | None,
    base_transport: Transport,
) -> tuple[dict[str, Any], ClaudeCallResult | None]:
    existing = _outcome(journal, call_id)
    if existing is not None:
        return existing, _result_from_dict(existing.get("result"))
    previous = journal.events_for(call_id)
    dispatches = [item for item in previous if item["event_kind"] == "message_dispatch"]
    responses = [item for item in previous if item["event_kind"] == "message_response"]
    if len(dispatches) > 1 or len(responses) > 1:
        raise ErdosStrausCreativeShadowError("scheduled LLM call journal is not singular")
    if responses and not dispatches:
        raise ErdosStrausCreativeShadowError(
            "scheduled LLM response has no bound dispatch"
        )
    if dispatches and not responses:
        outcome = _append_outcome(
            journal,
            call_id,
            phase=phase,
            role=role,
            status="indeterminate_after_dispatch",
            result=None,
            errors=["process_stopped_after_dispatch_before_durable_response"],
        )
        return outcome, None
    _restore_budget(client, journal, exclude_call_id=call_id if responses else None)
    if responses:
        _restore_model_evidence(client, journal)
        client.transport = _ReplayMessageTransport(dispatches[0], responses[0])
    else:
        client.transport = JournaledScheduledTransport(
            journal,
            scheduled_call_id=call_id,
            arm="erdos_straus_shadow",
            task_id="erdos_straus_shadow",
            role=role.value,
            base_transport=base_transport,
        )
    try:
        result = client.run(
            role,
            "erdos_straus_shadow",
            public_payload,
            candidate_summaries=candidate_summaries,
            instruction_override=instruction,
            hypothesis_slots=hypothesis_slots,
        )
    except (
        ClaudeCreativityError,
        AttemptJournalError,
        ErdosStrausCreativeShadowError,
        OSError,
        TimeoutError,
    ) as error:
        outcome = _append_outcome(
            journal,
            call_id,
            phase=phase,
            role=role,
            status="client_or_contract_failure",
            result=None,
            errors=[f"{type(error).__name__}:{error!s}"],
        )
        return outcome, None
    if result.status is not ClaudeCallStatus.COMPLETED:
        return (
            {
                "errors": [f"pre_dispatch_status:{result.status.value}"],
                "phase": phase,
                "result": result.to_dict(),
                "role": role.value,
                "status": "pre_dispatch_blocked_retryable",
            },
            result,
        )
    outcome = _append_outcome(
        journal,
        call_id,
        phase=phase,
        role=role,
        status="completed",
        result=result,
        errors=[],
    )
    return outcome, result


def _candidate_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis = item["hypothesis"]
    return {
        "candidate_id": item["idea_id"],
        "expression": hypothesis["expression"],
        "known_analogues": hypothesis["known_analogues"],
        "llm_origin_assessment": hypothesis["llm_origin_assessment"],
        "representation": hypothesis["representation"],
        "source_idea_domains": hypothesis["source_idea_domains"],
    }


def _creative_calls(
    config: Mapping[str, Any],
    journal: DurableAttemptJournal,
    *,
    base_transport: Transport = urllib_transport,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claude = ClaudeCreativityClient(ClaudeAPIConfig.from_mapping(config["claude"]))
    base_payload = _public_payload(config)
    calls = []
    attempts = []
    ideas = []
    roles = [ClaudeRole(item) for item in config["experiment"]["creative_roles"]]
    requested = int(config["experiment"]["requested_ideas_per_creative_call"])
    for call_ordinal, role in enumerate(roles, 1):
        payload = dict(base_payload)
        if role is ClaudeRole.RECOMBINER:
            payload["parent_palette"] = [
                {
                    "expression": item["hypothesis"]["expression"],
                    "idea_id": item["idea_id"],
                    "origin": item["hypothesis"]["llm_origin_assessment"],
                    "source_domains": item["hypothesis"]["source_idea_domains"],
                }
                for item in ideas
            ]
        outcome, result = _run_scheduled_call(
            journal,
            claude,
            call_id=_call_id(config, "creative", call_ordinal, role),
            phase="creative",
            role=role,
            public_payload=payload,
            candidate_summaries=(),
            instruction=_instruction(role, requested),
            hypothesis_slots=None,
            base_transport=base_transport,
        )
        attempts.append(outcome)
        if (
            result is None
            or result.status is not ClaudeCallStatus.COMPLETED
            or result.output is None
        ):
            continue
        record = result.to_dict()
        calls.append(record)
        for ordinal, hypothesis in enumerate(record["output"]["hypotheses"], 1):
            ideas.append(
                {
                    "hypothesis": hypothesis,
                    "idea_id": _candidate_id(role, hypothesis["hypothesis_id"], ordinal),
                    "role": role.value,
                }
            )
    if not ideas:
        raise ErdosStrausCreativeShadowError("no journaled Claude idea call completed")
    critic_actions: dict[str, dict[str, Any]] = {}
    critic_batch_size = int(config["experiment"]["llm_critic_batch_size"])
    if critic_batch_size:
        batches = [
            ideas[index : index + critic_batch_size]
            for index in range(0, len(ideas), critic_batch_size)
        ]
        for batch_ordinal, batch in enumerate(batches, 1):
            summaries = [_candidate_summary(item) for item in batch]
            outcome, result = _run_scheduled_call(
                journal,
                claude,
                call_id=_call_id(config, "critic", batch_ordinal, ClaudeRole.CRITIC),
                phase="critic",
                role=ClaudeRole.CRITIC,
                public_payload=base_payload,
                candidate_summaries=summaries,
                instruction=_critic_instruction(len(batch)),
                hypothesis_slots=None,
                base_transport=base_transport,
            )
            attempts.append(outcome)
            if result is None or result.output is None:
                continue
            calls.append(result.to_dict())
            for action in result.output.steering_actions:
                critic_actions[action.candidate_id] = action.to_dict()
    for item in ideas:
        executable = (
            parse_recipe(item["hypothesis"]["expression"], config["experiment"]) is not None
        )
        deterministic = {
            "blocker_kind": "none" if executable else "machine_dsl_parse_failure",
            "candidate_id": item["idea_id"],
            "distance_denominator": 1,
            "distance_numerator": 0 if executable else 1,
            "repair": (
                "Proceed to exact finite execution; correctness and novelty remain unverified."
                if executable
                else (
                    "Retain the idea text and translate it into a valid "
                    f"{config['experiment']['proposal_dsl_version']} recipe."
                )
            ),
            "verdict": "retain" if executable else "repair",
        }
        item["critic"] = critic_actions.get(item["idea_id"], deterministic)
        item["critic_source"] = (
            "journaled_llm_critic_retained_without_pruning"
            if item["idea_id"] in critic_actions
            else "deterministic_machine_admission_not_llm_critique"
        )
    _restore_budget(claude, journal)
    return ideas, {
        "budget": claude.budget.to_dict(),
        "calls": calls,
        "attempt_journal": {
            "content_sha256": journal.content_sha256,
            "credential_values_persisted": False,
            "event_counts": journal.event_counts(),
            "private_path_persisted_in_receipt": False,
            "response_before_validation": True,
            "resume_redispatches_completed_slots": False,
        },
        "attempts": attempts,
        "critic_kind": (
            "journaled_llm_batches_with_deterministic_fallback_no_pruning"
            if critic_batch_size
            else "deterministic_machine_admission_not_llm_critique"
        ),
    }


def _evaluate_ideas(
    ideas: list[dict[str, Any]], config: Mapping[str, Any], xp: Any, device: str
) -> dict[str, Any]:
    experiment = config["experiment"]
    members = _es_hard_members(int(experiment["calibration_n_max"]))
    executable = 0
    all_pairs: set[tuple[int, int]] = set()
    total_lane_tests = 0
    for item in ideas:
        hypothesis = item["hypothesis"]
        recipe = parse_recipe(hypothesis["expression"], experiment)
        item["llm_self_assessed_origin"] = hypothesis["llm_origin_assessment"]
        item["prior_art"] = _known_overlap(hypothesis)
        if recipe is None:
            item["execution"] = {
                "admission": "RETAINED_NONEXECUTABLE",
                "reason": "expression_did_not_compile_to_esdsl1",
            }
            continue
        executable += 1
        pairs = _schedule_pairs(recipe, experiment)
        all_pairs.update(pairs)
        wx, wy, resolved, lane_tests, elapsed = _run_pairs(xp, members, pairs)
        total_lane_tests += lane_tests
        item["execution"] = {
            "admission": "EXECUTED_EXACT_MODULAR_SCREEN",
            "device": device,
            "direct_parameter_pairs": [list(pair) for pair in pairs],
            "elapsed_seconds": f"{elapsed:.6f}",
            "exact_lane_tests": lane_tests,
            "finite_calibration_hard_cases": int(members.size),
            "recipe": recipe,
            "residue_evidence": _residue_evidence(members, resolved, recipe["moduli"]),
            "resolved_hard_cases": int(resolved.sum()),
            "witness_sample": _witness_sample(members, wx, wy, resolved),
        }
    return {
        "direct_unique_parameter_pairs": len(all_pairs),
        "executable_ideas": executable,
        "ideas": ideas,
        "nonexecutable_ideas_retained": len(ideas) - executable,
        "parameter_pairs": sorted([list(pair) for pair in all_pairs]),
        "total_idea_lane_tests": total_lane_tests,
    }


def _mutated_pairs(
    direct_pairs: Sequence[Sequence[int]], experiment: Mapping[str, Any]
) -> tuple[tuple[int, int], ...]:
    radius = int(experiment["mutation_radius"])
    maximum = int(experiment["maximum_offset"])
    result = set()
    for raw_dx, raw_t in direct_pairs:
        for delta_x in range(-radius, radius + 1):
            for delta_t in range(-radius, radius + 1):
                dx, t = int(raw_dx) + delta_x, int(raw_t) + delta_t
                if 0 <= dx <= maximum and 0 <= t <= maximum:
                    result.add((dx, t))
    return tuple(sorted(result, key=lambda item: (item[0] + item[1], item)))


def _mutation_lineage(
    creative: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    radius = int(experiment["mutation_radius"])
    maximum = int(experiment["maximum_offset"])
    lineage: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for idea in creative["ideas"]:
        execution = idea["execution"]
        if execution["admission"] != "EXECUTED_EXACT_MODULAR_SCREEN":
            continue
        for raw_dx, raw_t in execution["direct_parameter_pairs"]:
            direct_dx, direct_t = int(raw_dx), int(raw_t)
            for delta_x in range(-radius, radius + 1):
                for delta_t in range(-radius, radius + 1):
                    dx, t = direct_dx + delta_x, direct_t + delta_t
                    if not (0 <= dx <= maximum and 0 <= t <= maximum):
                        continue
                    lineage.setdefault((dx, t), []).append(
                        {
                            "basis": execution["recipe"]["basis"],
                            "direct_pair": [direct_dx, direct_t],
                            "idea_id": idea["idea_id"],
                            "llm_self_assessed_origin": idea["llm_self_assessed_origin"],
                            "mutation_delta": [delta_x, delta_t],
                            "role": idea["role"],
                        }
                    )
    return {
        pair: sorted(records, key=canonical_sha256)
        for pair, records in sorted(
            lineage.items(), key=lambda item: (sum(item[0]), item[0])
        )
    }


def _summarize_lineage_attribution(
    records: Sequence[Mapping[str, Any]], creative: Mapping[str, Any]
) -> dict[str, Any]:
    idea_hits: dict[str, set[int]] = {}
    idea_pairs: dict[str, set[tuple[int, int]]] = {}
    basis_hits: dict[str, set[int]] = {}
    basis_pairs: dict[str, set[tuple[int, int]]] = {}
    winning_pair_hits: dict[tuple[int, int], int] = {}
    for record in records:
        n = int(record["n"])
        winning_pair = tuple(int(item) for item in record["winning_pair"])
        winning_pair_hits[winning_pair] = winning_pair_hits.get(winning_pair, 0) + 1
        for parent in record["parent_lineages"]:
            idea_id = parent["idea_id"]
            basis = parent["basis"]
            idea_hits.setdefault(idea_id, set()).add(n)
            idea_pairs.setdefault(idea_id, set()).add(winning_pair)
            basis_hits.setdefault(basis, set()).add(n)
            basis_pairs.setdefault(basis, set()).add(winning_pair)
    idea_metadata = {
        item["idea_id"]: {
            "basis": item["execution"].get("recipe", {}).get("basis"),
            "llm_self_assessed_origin": item["llm_self_assessed_origin"],
            "role": item["role"],
        }
        for item in creative["ideas"]
    }
    return {
        "all_resolved_hits_have_parent_lineage": all(
            bool(item["parent_lineages"]) for item in records
        ),
        "attribution_rule": "first_exact_success_in_canonical_mutated_pair_order",
        "basis_linked_hits": [
            {
                "basis": basis,
                "linked_resolved_hits": len(basis_hits[basis]),
                "winning_pairs": [list(pair) for pair in sorted(basis_pairs[basis])],
            }
            for basis in sorted(basis_hits)
        ],
        "causal_or_novelty_credit_claimed": False,
        "idea_linked_hits": [
            {
                "basis": idea_metadata[idea_id]["basis"],
                "idea_id": idea_id,
                "linked_resolved_hits": len(idea_hits[idea_id]),
                "llm_self_assessed_origin": idea_metadata[idea_id][
                    "llm_self_assessed_origin"
                ],
                "role": idea_metadata[idea_id]["role"],
                "winning_pairs": [list(pair) for pair in sorted(idea_pairs[idea_id])],
            }
            for idea_id in sorted(idea_hits)
        ],
        "multi_basis_lineage_hits": sum(
            len({parent["basis"] for parent in item["parent_lineages"]}) > 1
            for item in records
        ),
        "multi_idea_lineage_hits": sum(
            len({parent["idea_id"] for parent in item["parent_lineages"]}) > 1
            for item in records
        ),
        "multi_parent_lineage_hits": sum(
            len(item["parent_lineages"]) > 1 for item in records
        ),
        "resolved_hit_records": [dict(item) for item in records],
        "resolved_hit_records_sha256": canonical_sha256(records),
        "winning_pair_hit_counts": [
            {"resolved_hits": winning_pair_hits[pair], "winning_pair": list(pair)}
            for pair in sorted(winning_pair_hits)
        ],
    }


def _hard_tail_experiment(
    experiment: Mapping[str, Any], creative: Mapping[str, Any], xp: Any, device: str
) -> dict[str, Any]:
    n_max = int(experiment["search_n_max"])
    members = _es_hard_members(n_max)
    baseline_started = time.perf_counter()
    _, _, baseline_resolved, baseline_lane_tests = _es_hard_rounds(
        xp,
        members,
        int(experiment["baseline_x_rounds"]),
        int(experiment["baseline_t_rounds"]),
    )
    baseline_elapsed = time.perf_counter() - baseline_started
    tail = members[~baseline_resolved]
    direct = tuple(tuple(item) for item in creative["parameter_pairs"])
    lineage_by_pair = _mutation_lineage(creative, experiment)
    mutated = tuple(lineage_by_pair)
    if mutated != _mutated_pairs(creative["parameter_pairs"], experiment):
        raise ErdosStrausCreativeShadowError(
            "mutation lineage does not cover the exact union search schedule"
        )
    baseline_pairs = {
        (dx, t)
        for dx in range(int(experiment["baseline_x_rounds"]))
        for t in range(int(experiment["baseline_t_rounds"]))
    }
    novel_pairs = tuple(pair for pair in mutated if pair not in baseline_pairs)
    (
        wx,
        wy,
        resolved,
        winning_dx,
        winning_t,
        creative_lane_tests,
        creative_elapsed,
    ) = _run_pairs_with_attribution(xp, tail, novel_pairs)
    verified_witnesses = []
    attribution_records = []
    pair_order = {pair: ordinal for ordinal, pair in enumerate(novel_pairs)}
    for index in np.flatnonzero(resolved):
        n, x, y = int(tail[index]), int(wx[index]), int(wy[index])
        winning_pair = (int(winning_dx[index]), int(winning_t[index]))
        parent_lineages = lineage_by_pair.get(winning_pair, [])
        if not parent_lineages or winning_pair not in pair_order:
            raise ErdosStrausCreativeShadowError(
                "creative tail hit has no exact parent mutation lineage"
            )
        b = n * x
        d = (4 * x - n) * y - b
        z = (b * y) // d
        if not es_witness_is_exact(n, x, y, z):
            raise ErdosStrausCreativeShadowError(
                "creative tail GPU hit failed independent CPU exact verification"
            )
        witness = {"n": n, "x": x, "y": y, "z": z}
        verified_witnesses.append(witness)
        attribution_records.append(
            {
                "canonical_pair_order_ordinal": pair_order[winning_pair],
                "n": n,
                "parent_lineages": parent_lineages,
                "winning_pair": list(winning_pair),
                "witness": witness,
            }
        )
    attribution = _summarize_lineage_attribution(attribution_records, creative)
    attribution["all_resolved_hits_have_parent_lineage"] = (
        attribution["all_resolved_hits_have_parent_lineage"]
        and len(attribution_records) == int(resolved.sum())
    )
    uniform_universe = sorted(
        (dx, t)
        for dx in range(int(experiment["maximum_offset"]) + 1)
        for t in range(int(experiment["maximum_offset"]) + 1)
        if (dx, t) not in baseline_pairs
    )
    rng = np.random.default_rng(int(experiment["matched_control_seed"]))
    creative_count = int(resolved.sum())
    wall_clock_ceiling = int(experiment["matched_control_wall_clock_ceiling_seconds"])
    (
        fixed_wx,
        fixed_wy,
        fixed_resolved,
        fixed_creative_lane_tests,
        fixed_creative_elapsed,
    ) = _run_pairs_fixed_lane_budget(xp, tail, novel_pairs)
    expected_lane_tests_per_schedule = int(tail.size) * len(novel_pairs)
    if (
        fixed_creative_lane_tests != expected_lane_tests_per_schedule
        or not np.array_equal(fixed_resolved, resolved)
        or not np.array_equal(fixed_wx, wx)
        or not np.array_equal(fixed_wy, wy)
    ):
        raise ErdosStrausCreativeShadowError(
            "fixed-lane creative reference disagrees with first-success evaluation"
        )
    if fixed_creative_elapsed > wall_clock_ceiling:
        raise ErdosStrausCreativeShadowError(
            "fixed-lane creative reference exceeded matched wall-clock ceiling"
        )

    def summarize_controls(
        label: str,
        control_counts: Sequence[int],
        control_lane_tests: Sequence[int],
        control_elapsed: Sequence[float],
    ) -> dict[str, Any]:
        if (
            not len(control_counts) == len(control_lane_tests) == len(control_elapsed)
            or any(item != expected_lane_tests_per_schedule for item in control_lane_tests)
        ):
            raise ErdosStrausCreativeShadowError("fixed-lane control verifier budget changed")
        if any(item > wall_clock_ceiling for item in control_elapsed):
            raise ErdosStrausCreativeShadowError(
                "fixed-lane control exceeded matched wall-clock ceiling"
            )
        elapsed_strings = [f"{item:.6f}" for item in control_elapsed]
        rounded_elapsed = [float(item) for item in elapsed_strings]
        random_at_least_creative = sum(count >= creative_count for count in control_counts)
        return {
            "all_trials_within_wall_clock_ceiling": True,
            "control_kind": label,
            "control_resolved_counts": control_counts,
            "creative_outperformed_random_median": creative_count
            > float(np.median(control_counts)),
            "elapsed_seconds": elapsed_strings,
            "creative_percentile": (
                f"{sum(count <= creative_count for count in control_counts) / len(control_counts):.6f}"
            ),
            "empirical_one_sided_p": (
                f"{(random_at_least_creative + 1) / (len(control_counts) + 1):.6f}"
            ),
            "exact_lane_tests_per_trial": expected_lane_tests_per_schedule,
            "maximum_resolved": max(control_counts),
            "maximum_elapsed_seconds": f"{max(rounded_elapsed):.6f}",
            "mean_resolved": f"{float(np.mean(control_counts)):.6f}",
            "median_elapsed_seconds": f"{float(np.median(rounded_elapsed)):.6f}",
            "median_resolved": f"{float(np.median(control_counts)):.6f}",
            "minimum_elapsed_seconds": f"{min(rounded_elapsed):.6f}",
            "minimum_resolved": min(control_counts),
            "random_at_least_creative": random_at_least_creative,
            "total_exact_lane_tests": sum(control_lane_tests),
            "trials": len(control_counts),
            "wall_clock_ceiling_seconds": wall_clock_ceiling,
        }

    def run_controls(universe: Sequence[tuple[int, int]], label: str) -> dict[str, Any]:
        control_counts = []
        control_lane_tests = []
        control_elapsed = []
        for _ in range(int(experiment["matched_control_trials"])):
            indices = rng.choice(len(universe), size=len(novel_pairs), replace=False)
            control_pairs = tuple(universe[int(index)] for index in indices)
            _, _, control_resolved, lane_tests, elapsed = _run_pairs_fixed_lane_budget(
                xp, tail, control_pairs
            )
            control_counts.append(int(control_resolved.sum()))
            control_lane_tests.append(lane_tests)
            control_elapsed.append(elapsed)
        return summarize_controls(label, control_counts, control_lane_tests, control_elapsed)

    def rewire_pairs() -> tuple[tuple[tuple[int, int], ...], int]:
        edges = list(novel_pairs)
        edge_set = set(edges)
        successful = 0
        for _ in range(20 * len(edges)):
            first_index, second_index = rng.choice(len(edges), size=2, replace=False)
            first = edges[int(first_index)]
            second = edges[int(second_index)]
            if first[0] == second[0] or first[1] == second[1]:
                continue
            replacement_first = (first[0], second[1])
            replacement_second = (second[0], first[1])
            if (
                replacement_first in edge_set
                or replacement_second in edge_set
                or replacement_first in baseline_pairs
                or replacement_second in baseline_pairs
            ):
                continue
            edge_set.remove(first)
            edge_set.remove(second)
            edge_set.add(replacement_first)
            edge_set.add(replacement_second)
            edges[int(first_index)] = replacement_first
            edges[int(second_index)] = replacement_second
            successful += 1
        return tuple(edges), successful

    x_support = sorted({dx for dx, _ in novel_pairs})
    t_support = sorted({t for _, t in novel_pairs})
    support_universe = sorted(
        (dx, t) for dx in x_support for t in t_support if (dx, t) not in baseline_pairs
    )
    uniform_controls = run_controls(uniform_universe, "uniform_full_declared_domain")
    support_controls = run_controls(support_universe, "llm_offset_support_cross_product")
    rewired_counts = []
    rewired_lane_tests = []
    rewired_elapsed = []
    rewired_swaps = []
    for _ in range(int(experiment["matched_control_trials"])):
        rewired_pairs, successful_swaps = rewire_pairs()
        _, _, rewired_resolved, lane_tests, elapsed = _run_pairs_fixed_lane_budget(
            xp, tail, rewired_pairs
        )
        rewired_counts.append(int(rewired_resolved.sum()))
        rewired_lane_tests.append(lane_tests)
        rewired_elapsed.append(elapsed)
        rewired_swaps.append(successful_swaps)
    rewired_controls = summarize_controls(
        "degree_preserving_pair_rewire",
        rewired_counts,
        rewired_lane_tests,
        rewired_elapsed,
    )
    rewired_controls["exact_x_marginal_frequency_matched"] = True
    rewired_controls["exact_t_marginal_frequency_matched"] = True
    rewired_controls["minimum_successful_edge_swaps"] = min(rewired_swaps)
    matched_controls = {
        "candidate_pair_count_matched": True,
        "creative_fixed_lane_reference": {
            "all_within_wall_clock_ceiling": True,
            "device": device,
            "elapsed_seconds": f"{fixed_creative_elapsed:.6f}",
            "exact_lane_tests": fixed_creative_lane_tests,
            "first_success_result_agreement": True,
            "resolved": int(fixed_resolved.sum()),
            "wall_clock_ceiling_seconds": wall_clock_ceiling,
        },
        "early_stop_enabled": False,
        "early_stop_rule_matched": True,
        "exact_lane_budget_matched": True,
        "fixed_lane_evaluator": True,
        "llm_x_support": x_support,
        "llm_t_support": t_support,
        "parameter_domain_matched": True,
        "pairing_only_rewire": rewired_controls,
        "random_control_exact_lane_tests": (
            uniform_controls["total_exact_lane_tests"]
            + support_controls["total_exact_lane_tests"]
            + rewired_controls["total_exact_lane_tests"]
        ),
        "runtime_budget_contract": (
            "Every creative/control schedule executes the same pair-by-tail array shape on the "
            "same device and evaluator, with the same fail-closed wall-clock ceiling. Observed "
            "times are outcomes and are not asserted equal."
        ),
        "same_device_and_evaluator_kernel": True,
        "seed": int(experiment["matched_control_seed"]),
        "support_matched": support_controls,
        "total_exact_lane_tests": (
            fixed_creative_lane_tests
            + uniform_controls["total_exact_lane_tests"]
            + support_controls["total_exact_lane_tests"]
            + rewired_controls["total_exact_lane_tests"]
        ),
        "uniform_domain": uniform_controls,
        "wall_clock_budget_claimed_matched": True,
        "wall_clock_ceiling_seconds": wall_clock_ceiling,
    }
    return {
        "baseline": {
            "device": device,
            "elapsed_seconds": f"{baseline_elapsed:.6f}",
            "exact_lane_tests": baseline_lane_tests,
            "hard_cases": int(members.size),
            "resolved": int(baseline_resolved.sum()),
            "unresolved_tail": int(tail.size),
        },
        "creative_tail": {
            "direct_parameter_pairs": len(direct),
            "device": device,
            "elapsed_seconds": f"{creative_elapsed:.6f}",
            "exact_lane_tests": creative_lane_tests,
            "independent_cpu_exact_verified": len(verified_witnesses),
            "integer_gpu_screen_cpu_identity_agreement": len(verified_witnesses)
            == int(resolved.sum()),
            "mutated_parameter_pairs": len(mutated),
            "pairs_outside_fixed_baseline": len(novel_pairs),
            "parent_lineage_attribution": attribution,
            "resolved_from_baseline_tail": int(resolved.sum()),
            "still_requiring_complete_divisor_search": int(tail.size - resolved.sum()),
            "verified_witness_corpus_sha256": canonical_sha256(verified_witnesses),
            "witness_sample": _witness_sample(tail, wx, wy, resolved),
        },
        "matched_random_controls": matched_controls,
    }


def _benchmark(config: Mapping[str, Any]) -> dict[str, Any]:
    n_max = int(config["experiment"]["benchmark_n_max"])
    cpu = run_erdos_straus_sweep(n_max, use_gpu=False)
    gpu = run_erdos_straus_sweep(n_max, use_gpu=True)
    stable_match = (
        cpu["decision"] == gpu["decision"]
        and cpu["prefilter"]["gpu_lane_tests"] == gpu["prefilter"]["gpu_lane_tests"]
        and cpu["results"]["unsolvable_candidates"] == gpu["results"]["unsolvable_candidates"]
    )
    if not stable_match:
        raise ErdosStrausCreativeShadowError("CPU/GPU control disagreement")
    return {
        "cpu": {
            "device": cpu["device"],
            "elapsed_seconds": cpu["elapsed_seconds"],
            "throughput_per_second": cpu["throughput_per_second"],
        },
        "exact_result_match": True,
        "gpu": {
            "device": gpu["device"],
            "elapsed_seconds": gpu["elapsed_seconds"],
            "throughput_per_second": gpu["throughput_per_second"],
        },
        "n_max": n_max,
        "speedup": f"{float(cpu['elapsed_seconds']) / float(gpu['elapsed_seconds']):.6f}",
    }


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    ideas: list[dict[str, Any]],
    claude: Mapping[str, Any],
    credential_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    xp, device = _backend(True)
    if device == "cpu-numpy":
        raise ErdosStrausCreativeShadowError("creative shadow build requires a CUDA device")
    creative = _evaluate_ideas(ideas, config, xp, device)
    hard_tail = _hard_tail_experiment(config["experiment"], creative, xp, device)
    benchmark = _benchmark(config)
    sweep = run_erdos_straus_sweep(int(config["experiment"]["search_n_max"]), use_gpu=True)
    validate_sweep_receipt(sweep)
    origin_counts = Counter(item["llm_self_assessed_origin"] for item in creative["ideas"])
    critic_counts = Counter(item["critic"]["verdict"] for item in creative["ideas"])
    body: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "claim_boundary": {
            "finite_search_decides_conjecture": False,
            "famous_open_problem_attempt_authorized": False,
            "human_prior_art_review_complete": False,
            "llm_self_assessment_establishes_novelty": False,
            "new_verification_bound_claimed": False,
            "open_problem_solved": False,
        },
        "claude": {
            **claude,
            "completed_calls": len(claude["calls"]),
            "critic_verdict_counts": dict(sorted(critic_counts.items())),
            "credential_activation": credential_evidence,
            "origin_counts": dict(sorted(origin_counts.items())),
            "proposed_ideas": len(creative["ideas"]),
            "requested_ideas_per_creative_call": config["experiment"][
                "requested_ideas_per_creative_call"
            ],
        },
        "config": {
            "config_sha256": _file_sha256(root / config_path),
            "source_sha256": _file_sha256(root / SOURCE_PATH),
            "sweeper_sha256": _file_sha256(root / SWEEPER_PATH),
        },
        "creative_search": creative,
        "finite_sweep": sweep,
        "gpu_benchmark": benchmark,
        "hard_tail_funnel": hard_tail,
        "literature": config["literature"],
        "problem": config["problem"],
        "recovery": config["recovery"],
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM",
        "typed_schedule_compiler": _esdsl2_semantics_contract(config["experiment"]),
    }
    body["accounting"] = {
        "baseline_gpu_lane_tests": hard_tail["baseline"]["exact_lane_tests"],
        "creative_tail_lane_tests": hard_tail["creative_tail"]["exact_lane_tests"],
        "denominators_covered": sweep["results"]["coverage"]["class_total"],
        "executable_llm_ideas": creative["executable_ideas"],
        "llm_ideas_proposed": len(creative["ideas"]),
        "llm_provider_calls": (
            claude["budget"]["calls"]
            + sum(
                item["completed_provider_calls"]
                for item in config["recovery"]["discarded_attempts"]
            )
        ),
        "retained_llm_provider_calls": claude["budget"]["calls"],
        "mutated_parameter_pairs": hard_tail["creative_tail"]["mutated_parameter_pairs"],
        "matched_control_lane_tests": hard_tail["matched_random_controls"][
            "total_exact_lane_tests"
        ],
        "total_exact_modular_lane_tests": (
            creative["total_idea_lane_tests"]
            + hard_tail["baseline"]["exact_lane_tests"]
            + hard_tail["creative_tail"]["exact_lane_tests"]
            + hard_tail["matched_random_controls"]["total_exact_lane_tests"]
        ),
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root, config_path=config_path)
    return body


def run_live(
    root: Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    journal_path: str | Path = JOURNAL_PATH,
    base_transport: Transport = urllib_transport,
) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    journal = _open_attempt_journal(root, config, config_path, journal_path)
    with activated_credential(
        project_root=root,
        env_var=config["claude"]["credential_env_var"],
    ) as activation:
        ideas, claude = _creative_calls(config, journal, base_transport=base_transport)
        credential_evidence = activation.to_evidence()
    return _build_receipt(root, config, config_path, ideas, claude, credential_evidence)


def rebind_receipt(
    root: Path, previous: Mapping[str, Any], *, config_path: str | Path = CONFIG_PATH
) -> dict[str, Any]:
    """Rebuild deterministic evidence and source seals without another provider call."""

    root = root.resolve()
    config = _load_config(root, config_path)
    prior_claude = previous.get("claude", {})
    prior_ideas = previous.get("creative_search", {}).get("ideas", [])
    if not isinstance(prior_ideas, list) or not prior_ideas:
        raise ErdosStrausCreativeShadowError("previous receipt has no reusable ideas")
    ideas = [
        {
            "critic": dict(item["critic"]),
            "critic_source": item.get(
                "critic_source", "deterministic_machine_admission_not_llm_critique"
            ),
            "hypothesis": dict(item["hypothesis"]),
            "idea_id": item["idea_id"],
            "role": item["role"],
        }
        for item in prior_ideas
    ]
    claude = {
        "budget": dict(prior_claude["budget"]),
        "calls": list(prior_claude["calls"]),
        "critic_kind": prior_claude["critic_kind"],
    }
    for optional in ("attempt_journal", "attempts"):
        if optional in prior_claude:
            claude[optional] = prior_claude[optional]
    return _build_receipt(
        root,
        config,
        config_path,
        ideas,
        claude,
        dict(prior_claude["credential_activation"]),
    )


def validate_receipt(
    value: Mapping[str, Any], root: Path, *, config_path: str | Path = CONFIG_PATH
) -> None:
    root = root.resolve()
    config = _load_config(root, config_path)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != (
        "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM"
    ):
        raise ErdosStrausCreativeShadowError("receipt identity changed")
    if canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    ) != value.get("content_sha256"):
        raise ErdosStrausCreativeShadowError("receipt content seal failed")
    if value.get("config") != {
        "config_sha256": _file_sha256(root / config_path),
        "source_sha256": _file_sha256(root / SOURCE_PATH),
        "sweeper_sha256": _file_sha256(root / SWEEPER_PATH),
    }:
        raise ErdosStrausCreativeShadowError("receipt source bindings changed")
    if value.get("typed_schedule_compiler") != _esdsl2_semantics_contract(config["experiment"]):
        raise ErdosStrausCreativeShadowError("typed schedule compiler contract changed")
    boundary = value["claim_boundary"]
    if any(boundary.values()):
        raise ErdosStrausCreativeShadowError("a prohibited claim is true")
    claude = value["claude"]
    completed_calls = claude["completed_calls"]
    budget_calls = claude["budget"]["calls"]
    if (
        completed_calls != len(claude["calls"])
        or not 1 <= completed_calls <= budget_calls <= config["claude"]["maximum_calls"]
    ):
        raise ErdosStrausCreativeShadowError("Claude call accounting changed")
    proposed = claude["proposed_ideas"]
    if not 1 <= proposed <= 16 * len(config["experiment"]["creative_roles"]):
        raise ErdosStrausCreativeShadowError("idea count is outside the structured-output cap")
    if (
        claude["requested_ideas_per_creative_call"]
        != config["experiment"]["requested_ideas_per_creative_call"]
    ):
        raise ErdosStrausCreativeShadowError("requested idea allocation changed")
    if claude["credential_activation"].get("credential_persisted") is not False:
        raise ErdosStrausCreativeShadowError("credential persistence boundary changed")
    journal = claude.get("attempt_journal")
    if journal is not None and (
        journal.get("credential_values_persisted") is not False
        or journal.get("private_path_persisted_in_receipt") is not False
        or journal.get("response_before_validation") is not True
        or journal.get("resume_redispatches_completed_slots") is not False
        or journal.get("event_counts", {}).get("message_dispatch") != budget_calls
    ):
        raise ErdosStrausCreativeShadowError("durable attempt journal evidence changed")
    validate_sweep_receipt(value["finite_sweep"])
    direct_union: set[tuple[int, int]] = set()
    for idea in value["creative_search"]["ideas"]:
        if idea["execution"]["admission"] == "EXECUTED_EXACT_MODULAR_SCREEN":
            recipe = parse_recipe(idea["hypothesis"]["expression"], config["experiment"])
            expected_pairs = (
                _schedule_pairs(recipe, config["experiment"]) if recipe is not None else ()
            )
            if (
                recipe != idea["execution"]["recipe"]
                or [list(pair) for pair in expected_pairs]
                != idea["execution"]["direct_parameter_pairs"]
            ):
                raise ErdosStrausCreativeShadowError(
                    "idea direct schedule or typed recipe changed"
                )
            direct_union.update(expected_pairs)
            for witness in idea["execution"]["witness_sample"]:
                if not es_witness_is_exact(**witness):
                    raise ErdosStrausCreativeShadowError("idea witness failed exact replay")
        if idea["prior_art"]["novelty_established"] is not False:
            raise ErdosStrausCreativeShadowError("automated prior art claimed novelty")
    for witness in value["hard_tail_funnel"]["creative_tail"]["witness_sample"]:
        if not es_witness_is_exact(**witness):
            raise ErdosStrausCreativeShadowError("tail witness failed exact replay")
    tail = value["hard_tail_funnel"]["creative_tail"]
    if (
        tail["independent_cpu_exact_verified"] != tail["resolved_from_baseline_tail"]
        or tail["integer_gpu_screen_cpu_identity_agreement"] is not True
    ):
        raise ErdosStrausCreativeShadowError("creative tail independent verification changed")
    creative = value["creative_search"]
    if [list(pair) for pair in sorted(direct_union)] != creative["parameter_pairs"]:
        raise ErdosStrausCreativeShadowError("creative direct-pair union changed")
    lineage_by_pair = _mutation_lineage(creative, config["experiment"])
    baseline_pairs = {
        (dx, t)
        for dx in range(int(config["experiment"]["baseline_x_rounds"]))
        for t in range(int(config["experiment"]["baseline_t_rounds"]))
    }
    novel_pairs = tuple(pair for pair in lineage_by_pair if pair not in baseline_pairs)
    pair_order = {pair: ordinal for ordinal, pair in enumerate(novel_pairs)}
    attribution = tail.get("parent_lineage_attribution", {})
    records = attribution.get("resolved_hit_records", [])
    if (
        not isinstance(records, list)
        or len(records) != tail["resolved_from_baseline_tail"]
        or len({item.get("n") for item in records}) != len(records)
    ):
        raise ErdosStrausCreativeShadowError("creative lineage record coverage changed")
    for record in records:
        pair = tuple(record["winning_pair"])
        witness = record["witness"]
        if (
            pair not in pair_order
            or record["canonical_pair_order_ordinal"] != pair_order[pair]
            or record["parent_lineages"] != lineage_by_pair[pair]
            or record["n"] != witness["n"]
            or not es_witness_is_exact(**witness)
        ):
            raise ErdosStrausCreativeShadowError("creative lineage record failed exact replay")
        n = int(record["n"])
        dx, t = (int(item) for item in pair)
        x = n // 4 + 1 + dx
        a = 4 * x - n
        b = n * x
        y = (b + a - 1) // a + t
        d = a * y - b
        z = (b * y) // d if d > 0 and (b * y) % d == 0 else 0
        if witness != {"n": n, "x": x, "y": y, "z": z}:
            raise ErdosStrausCreativeShadowError(
                "creative lineage winning pair does not reproduce its witness"
            )
    expected_attribution = _summarize_lineage_attribution(records, creative)
    expected_attribution["all_resolved_hits_have_parent_lineage"] = (
        expected_attribution["all_resolved_hits_have_parent_lineage"]
        and len(records) == tail["resolved_from_baseline_tail"]
    )
    if attribution != expected_attribution:
        raise ErdosStrausCreativeShadowError("creative lineage aggregate changed")
    accounting = value["accounting"]
    if accounting["llm_ideas_proposed"] != value["claude"]["proposed_ideas"]:
        raise ErdosStrausCreativeShadowError("idea accounting mismatch")
    discarded_calls = sum(
        item["completed_provider_calls"] for item in config["recovery"]["discarded_attempts"]
    )
    if (
        accounting["llm_provider_calls"] != budget_calls + discarded_calls
        or accounting["retained_llm_provider_calls"] != budget_calls
    ):
        raise ErdosStrausCreativeShadowError("provider recovery accounting mismatch")
    if accounting["denominators_covered"] != config["experiment"]["search_n_max"] - 1:
        raise ErdosStrausCreativeShadowError("finite coverage accounting mismatch")
    controls = value["hard_tail_funnel"]["matched_random_controls"]
    trial_count = int(config["experiment"]["matched_control_trials"])
    wall_clock_ceiling = int(
        config["experiment"]["matched_control_wall_clock_ceiling_seconds"]
    )
    expected_lane_tests = (
        int(value["hard_tail_funnel"]["baseline"]["unresolved_tail"])
        * int(tail["pairs_outside_fixed_baseline"])
    )
    fixed_reference = controls.get("creative_fixed_lane_reference", {})
    if (
        controls.get("fixed_lane_evaluator") is not True
        or controls.get("early_stop_enabled") is not False
        or controls.get("exact_lane_budget_matched") is not True
        or controls.get("same_device_and_evaluator_kernel") is not True
        or controls.get("wall_clock_budget_claimed_matched") is not True
        or controls.get("wall_clock_ceiling_seconds") != wall_clock_ceiling
        or fixed_reference.get("device") != tail["device"]
        or fixed_reference.get("exact_lane_tests") != expected_lane_tests
        or fixed_reference.get("resolved") != tail["resolved_from_baseline_tail"]
        or fixed_reference.get("first_success_result_agreement") is not True
        or fixed_reference.get("all_within_wall_clock_ceiling") is not True
        or fixed_reference.get("wall_clock_ceiling_seconds") != wall_clock_ceiling
        or not 0 <= float(fixed_reference.get("elapsed_seconds", -1)) <= wall_clock_ceiling
        or controls["pairing_only_rewire"]["exact_x_marginal_frequency_matched"] is not True
        or controls["pairing_only_rewire"]["exact_t_marginal_frequency_matched"] is not True
        or controls["candidate_pair_count_matched"] is not True
        or controls["parameter_domain_matched"] is not True
        or controls["early_stop_rule_matched"] is not True
    ):
        raise ErdosStrausCreativeShadowError("matched random control contract changed")
    random_control_lane_tests = 0
    for control_name in ("uniform_domain", "support_matched", "pairing_only_rewire"):
        control = controls[control_name]
        elapsed = [float(item) for item in control.get("elapsed_seconds", [])]
        if (
            control.get("trials") != trial_count
            or len(control.get("control_resolved_counts", [])) != trial_count
            or len(elapsed) != trial_count
            or control.get("exact_lane_tests_per_trial") != expected_lane_tests
            or control.get("total_exact_lane_tests") != expected_lane_tests * trial_count
            or control.get("wall_clock_ceiling_seconds") != wall_clock_ceiling
            or control.get("all_trials_within_wall_clock_ceiling") is not True
            or any(item < 0 or item > wall_clock_ceiling for item in elapsed)
            or control.get("minimum_elapsed_seconds") != f"{min(elapsed):.6f}"
            or control.get("median_elapsed_seconds")
            != f"{float(np.median(elapsed)):.6f}"
            or control.get("maximum_elapsed_seconds") != f"{max(elapsed):.6f}"
        ):
            raise ErdosStrausCreativeShadowError(
                "fixed-lane random control budget or runtime evidence changed"
            )
        random_control_lane_tests += int(control["total_exact_lane_tests"])
    if (
        controls.get("random_control_exact_lane_tests") != random_control_lane_tests
        or controls.get("total_exact_lane_tests")
        != random_control_lane_tests + expected_lane_tests
        or accounting.get("matched_control_lane_tests") != controls["total_exact_lane_tests"]
        or accounting.get("total_exact_modular_lane_tests")
        != creative["total_idea_lane_tests"]
        + value["hard_tail_funnel"]["baseline"]["exact_lane_tests"]
        + tail["exact_lane_tests"]
        + controls["total_exact_lane_tests"]
    ):
        raise ErdosStrausCreativeShadowError("fixed-lane matched-control accounting changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-live", "rebind", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--journal", default=JOURNAL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / OUTPUT_PATH
    if args.command == "run-live":
        receipt = run_live(root, config_path=args.config, journal_path=args.journal)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "rebind":
        previous = json.loads(output.read_text(encoding="utf-8"))
        receipt = rebind_receipt(root, previous, config_path=args.config)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = json.loads(output.read_text(encoding="utf-8"))
        validate_receipt(receipt, root, config_path=args.config)
    print(
        json.dumps(
            {
                "accounting": receipt["accounting"],
                "content_sha256": receipt["content_sha256"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
