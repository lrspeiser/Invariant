from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import sympy as sp

TERM_NAMES = ("K1", "K2", "K3", "K4", "unit_constraint")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact arbitrary-jet 2D diffeomorphism identity for Einstein-Aether terms"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terms", nargs="+", choices=TERM_NAMES, default=list(TERM_NAMES))
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_jet_problem() -> dict[str, Any]:
    dimension = 2
    names = ("g00", "g01", "g11", "u0", "u1", "lambda")
    fields = sp.symbols(" ".join(names))
    first = [
        [sp.Symbol(f"{names[field]}_{mu}") for mu in range(dimension)]
        for field in range(len(fields))
    ]
    second = [
        [
            [
                sp.Symbol(f"{names[field]}_{min(mu, nu)}{max(mu, nu)}")
                for nu in range(dimension)
            ]
            for mu in range(dimension)
        ]
        for field in range(len(fields))
    ]
    third: dict[tuple[int, int, int, int], sp.Symbol] = {}

    def third_jet(field: int, i: int, j: int, k: int) -> sp.Symbol:
        ordered = tuple(sorted((i, j, k)))
        key = (field, *ordered)
        if key not in third:
            third[key] = sp.Symbol(f"{names[field]}_{ordered[0]}{ordered[1]}{ordered[2]}")
        return third[key]

    def total_derivative(expression: sp.Expr, direction: int, *, include_third: bool) -> sp.Expr:
        result = sum(
            sp.diff(expression, fields[field]) * first[field][direction]
            for field in range(len(fields))
        )
        result += sum(
            sp.diff(expression, first[field][mu]) * second[field][mu][direction]
            for field in range(len(fields))
            for mu in range(dimension)
        )
        if include_third:
            result += sum(
                sp.diff(expression, second[field][i][j])
                * third_jet(field, i, j, direction)
                for field in range(len(fields))
                for i in range(dimension)
                for j in range(i, dimension)
            )
        return result

    metric_field = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2}
    metric = sp.Matrix([[fields[0], fields[1]], [fields[1], fields[2]]])
    inverse_metric = sp.simplify(metric.inv())
    determinant = sp.factor(metric.det())
    connection = [
        [
            [
                sp.factor(
                    sum(
                        inverse_metric[upper, delta]
                        * (
                            first[metric_field[delta, beta]][alpha]
                            + first[metric_field[delta, alpha]][beta]
                            - first[metric_field[alpha, beta]][delta]
                        )
                        for delta in range(dimension)
                    )
                    / 2
                )
                for beta in range(dimension)
            ]
            for alpha in range(dimension)
        ]
        for upper in range(dimension)
    ]
    vector = sp.Matrix(fields[3:5])
    vector_up = sp.simplify(inverse_metric * vector)
    derivative = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda alpha, beta: first[3 + beta][alpha]
        - sum(
            connection[upper][alpha][beta] * vector[upper]
            for upper in range(dimension)
        ),
    )
    derivative_up = sp.simplify(inverse_metric * derivative * inverse_metric)
    k1 = sp.factor(
        sum(
            derivative[alpha, beta] * derivative_up[alpha, beta]
            for alpha in range(dimension)
            for beta in range(dimension)
        )
    )
    expansion = sp.factor(
        sum(
            inverse_metric[alpha, beta] * derivative[alpha, beta]
            for alpha in range(dimension)
            for beta in range(dimension)
        )
    )
    k2 = sp.factor(expansion**2)
    k3 = sp.factor(
        sum(
            derivative[alpha, beta] * derivative_up[beta, alpha]
            for alpha in range(dimension)
            for beta in range(dimension)
        )
    )
    acceleration = vector_up.T * derivative
    k4 = sp.factor((acceleration * inverse_metric * acceleration.T)[0])
    norm = sp.factor((vector.T * inverse_metric * vector)[0])
    density = sp.sqrt(-determinant)
    lagrangians = {
        "K1": density * k1,
        "K2": density * k2,
        "K3": density * k3,
        "K4": density * k4,
        "unit_constraint": density * fields[5] * (norm + 1),
    }

    representation = [
        [[sp.Integer(0) for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(len(fields))
    ]
    for alpha, beta, field in ((0, 0, 0), (0, 1, 1), (1, 1, 2)):
        for generator in range(dimension):
            for direction in range(dimension):
                representation[field][generator][direction] = (
                    metric[generator, beta] if direction == alpha else 0
                ) + (metric[alpha, generator] if direction == beta else 0)
    for alpha in range(dimension):
        for generator in range(dimension):
            for direction in range(dimension):
                representation[3 + alpha][generator][direction] = (
                    vector[generator] if direction == alpha else 0
                )
    return {
        "dimension": dimension,
        "names": names,
        "fields": fields,
        "first": first,
        "lagrangians": lagrangians,
        "representation": representation,
        "total_derivative": total_derivative,
    }


def verify_term(problem: dict[str, Any], term: str) -> dict[str, Any]:
    started = time.perf_counter()
    fields = problem["fields"]
    first = problem["first"]
    dimension = problem["dimension"]
    lagrangian = sp.factor(problem["lagrangians"][term])
    total_derivative = problem["total_derivative"]
    euler = [
        sp.factor(
            sp.diff(lagrangian, fields[field])
            - sum(
                total_derivative(
                    sp.diff(lagrangian, first[field][direction]),
                    direction,
                    include_third=False,
                )
                for direction in range(dimension)
            )
        )
        for field in range(len(fields))
    ]
    residuals: list[sp.Expr] = []
    raw_character_counts: list[int] = []
    for generator in range(dimension):
        residual = sum(
            euler[field] * first[field][generator] for field in range(len(fields))
        )
        residual -= sum(
            total_derivative(
                euler[field] * problem["representation"][field][generator][direction],
                direction,
                include_third=True,
            )
            for field in range(len(fields))
            for direction in range(dimension)
        )
        raw_character_counts.append(len(str(residual)))
        residuals.append(sp.factor(sp.together(residual)))
    return {
        "term": term,
        "euler_character_counts": [len(str(item)) for item in euler],
        "raw_residual_character_counts": raw_character_counts,
        "residuals": [str(item) for item in residuals],
        "passed": all(item == 0 for item in residuals),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = _parser().parse_args()
    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "sigma-aether-noether-jet-1.0",
        "verifier": str(script_path),
        "verifier_sha256": _sha256(script_path),
        "python": platform.python_version(),
        "sympy": sp.__version__,
        "dimension": 2,
        "field_jet_scope": "arbitrary g_ab, u_a, lambda and all symmetric derivatives through third order",
        "identity": "sum_A E_A q^A_,c - D_mu(sum_A E_A R^{A mu}_c) = 0",
        "terms_requested": args.terms,
        "terms": {},
        "complete": False,
        "passed": False,
    }
    _write_checkpoint(args.output, payload)
    problem = build_jet_problem()
    for term in args.terms:
        print(f"verifying {term}", flush=True)
        payload["terms"][term] = verify_term(problem, term)
        _write_checkpoint(args.output, payload)
    payload["complete"] = set(payload["terms"]) == set(args.terms)
    payload["passed"] = payload["complete"] and all(
        item["passed"] for item in payload["terms"].values()
    )
    _write_checkpoint(args.output, payload)
    print(json.dumps({"complete": payload["complete"], "passed": payload["passed"]}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
