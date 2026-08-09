from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

DIMENSION = 4
METRIC_PAIRS = tuple((a, b) for a in range(DIMENSION) for b in range(a, DIMENSION))
SECOND_PAIRS = METRIC_PAIRS
THIRD_TRIPLES = tuple(
    (a, b, c)
    for a in range(DIMENSION)
    for b in range(a, DIMENSION)
    for c in range(b, DIMENSION)
)
N_FIELDS = len(METRIC_PAIRS) + DIMENSION + 1
U_OFFSET = len(METRIC_PAIRS)
LAMBDA_INDEX = N_FIELDS - 1
TERM_NAMES = ("K1", "K2", "K3", "K4", "unit_constraint")
PAIR_INDEX = {pair: index for index, pair in enumerate(SECOND_PAIRS)}
TRIPLE_INDEX = {triple: index for index, triple in enumerate(THIRD_TRIPLES)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-coordinate 4D numerical arbitrary-jet Einstein-Aether Noether verifier"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terms", nargs="+", choices=TERM_NAMES, default=list(TERM_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[104729, 130363, 155921])
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-8)
    parser.add_argument("--negative-threshold", type=float, default=1.0e-5)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _symmetric_matrix_from_components(components: jax.Array) -> jax.Array:
    matrix = jnp.zeros((DIMENSION, DIMENSION), dtype=components.dtype)
    for index, (a, b) in enumerate(METRIC_PAIRS):
        matrix = matrix.at[a, b].set(components[index])
        matrix = matrix.at[b, a].set(components[index])
    return matrix


def _metric_derivatives(q1: jax.Array) -> jax.Array:
    derivatives = jnp.zeros((DIMENSION, DIMENSION, DIMENSION), dtype=q1.dtype)
    for field, (a, b) in enumerate(METRIC_PAIRS):
        for direction in range(DIMENSION):
            derivatives = derivatives.at[a, b, direction].set(q1[field, direction])
            derivatives = derivatives.at[b, a, direction].set(q1[field, direction])
    return derivatives


def _second(q2: jax.Array, field: int, first: int, second: int) -> jax.Array:
    return q2[field, PAIR_INDEX[tuple(sorted((first, second)))]]


def _third(q3: jax.Array, field: int, a: int, b: int, c: int) -> jax.Array:
    return q3[field, TRIPLE_INDEX[tuple(sorted((a, b, c)))]]


def lagrangian(term: str, q: jax.Array, q1: jax.Array) -> jax.Array:
    metric = _symmetric_matrix_from_components(q[:U_OFFSET])
    inverse = jnp.linalg.inv(metric)
    metric_first = _metric_derivatives(q1)
    connection = jnp.einsum(
        "cd,abd->cab",
        inverse,
        (
            jnp.transpose(metric_first, (2, 1, 0))
            + jnp.transpose(metric_first, (1, 2, 0))
            - metric_first
        ),
    ) / 2
    vector = q[U_OFFSET : U_OFFSET + DIMENSION]
    vector_up = inverse @ vector
    derivative = jnp.stack(
        [
            jnp.stack(
                [
                    q1[U_OFFSET + b, a] - jnp.dot(connection[:, a, b], vector)
                    for b in range(DIMENSION)
                ]
            )
            for a in range(DIMENSION)
        ]
    )
    derivative_up = inverse @ derivative @ inverse
    k1 = jnp.einsum("ab,ab->", derivative, derivative_up)
    expansion = jnp.einsum("ab,ab->", inverse, derivative)
    k2 = expansion**2
    k3 = jnp.einsum("ab,ba->", derivative, derivative_up)
    acceleration = vector_up @ derivative
    k4 = acceleration @ inverse @ acceleration
    norm = vector @ inverse @ vector
    density = jnp.sqrt(-jnp.linalg.det(metric))
    values = {
        "K1": density * k1,
        "K2": density * k2,
        "K3": density * k3,
        "K4": density * k4,
        "unit_constraint": density * q[LAMBDA_INDEX] * (norm + 1),
    }
    return values[term]


def euler_derivatives(
    term: str, q: jax.Array, q1: jax.Array, q2: jax.Array
) -> jax.Array:
    function = lambda q_value, q1_value: lagrangian(term, q_value, q1_value)
    derivative_q, _derivative_q1 = jax.grad(function, argnums=(0, 1))(q, q1)
    momentum = lambda q_value, q1_value: jax.grad(function, argnums=1)(q_value, q1_value)
    momentum_q, momentum_q1 = jax.jacrev(momentum, argnums=(0, 1))(q, q1)
    divergence = jnp.zeros(N_FIELDS, dtype=q.dtype)
    for field in range(N_FIELDS):
        for direction in range(DIMENSION):
            divergence = divergence.at[field].add(
                jnp.dot(momentum_q[field, direction], q1[:, direction])
            )
            for source_direction in range(DIMENSION):
                divergence = divergence.at[field].add(
                    jnp.dot(
                        momentum_q1[field, direction, :, source_direction],
                        jnp.asarray(
                            [
                                _second(q2, source, source_direction, direction)
                                for source in range(N_FIELDS)
                            ]
                        ),
                    )
                )
    return derivative_q - divergence


def representation(q: jax.Array, *, include_metric: bool) -> jax.Array:
    result = jnp.zeros((N_FIELDS, DIMENSION, DIMENSION), dtype=q.dtype)
    metric = _symmetric_matrix_from_components(q[:U_OFFSET])
    if include_metric:
        for field, (a, b) in enumerate(METRIC_PAIRS):
            for generator in range(DIMENSION):
                for direction in range(DIMENSION):
                    value = (metric[generator, b] if direction == a else 0.0) + (
                        metric[a, generator] if direction == b else 0.0
                    )
                    result = result.at[field, generator, direction].set(value)
    vector = q[U_OFFSET : U_OFFSET + DIMENSION]
    for a in range(DIMENSION):
        for generator in range(DIMENSION):
            result = result.at[U_OFFSET + a, generator, a].set(vector[generator])
    return result


def noether_residual(
    term: str,
    q: jax.Array,
    q1: jax.Array,
    q2: jax.Array,
    q3: jax.Array,
    *,
    include_metric_representation: bool,
) -> jax.Array:
    def current(q_value: jax.Array, q1_value: jax.Array, q2_value: jax.Array) -> jax.Array:
        euler = euler_derivatives(term, q_value, q1_value, q2_value)
        return jnp.einsum(
            "f,fcm->cm",
            euler,
            representation(q_value, include_metric=include_metric_representation),
        )

    euler = euler_derivatives(term, q, q1, q2)
    current_q, current_q1, current_q2 = jax.jacrev(current, argnums=(0, 1, 2))(q, q1, q2)
    divergence = jnp.zeros(DIMENSION, dtype=q.dtype)
    for generator in range(DIMENSION):
        for direction in range(DIMENSION):
            divergence = divergence.at[generator].add(
                jnp.dot(current_q[generator, direction], q1[:, direction])
            )
            for field in range(N_FIELDS):
                for source_direction in range(DIMENSION):
                    divergence = divergence.at[generator].add(
                        current_q1[generator, direction, field, source_direction]
                        * _second(q2, field, source_direction, direction)
                    )
                for first, second in SECOND_PAIRS:
                    divergence = divergence.at[generator].add(
                        current_q2[
                            generator,
                            direction,
                            field,
                            PAIR_INDEX[(first, second)],
                        ]
                        * _third(q3, field, first, second, direction)
                    )
    return jnp.einsum("f,fc->c", euler, q1) - divergence


def sample_jets(seed: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    random = np.random.default_rng(seed)
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    frame = np.eye(DIMENSION) + random.normal(scale=0.08, size=(DIMENSION, DIMENSION))
    metric = frame.T @ eta @ frame
    q = np.zeros(N_FIELDS, dtype=np.float64)
    for field, (a, b) in enumerate(METRIC_PAIRS):
        q[field] = metric[a, b]
    q[U_OFFSET : U_OFFSET + DIMENSION] = random.normal(scale=0.35, size=DIMENSION)
    q[LAMBDA_INDEX] = random.normal(scale=0.2)
    q1 = random.normal(scale=0.15, size=(N_FIELDS, DIMENSION))
    q2 = random.normal(scale=0.1, size=(N_FIELDS, len(SECOND_PAIRS)))
    q3 = random.normal(scale=0.06, size=(N_FIELDS, len(THIRD_TRIPLES)))
    return tuple(jnp.asarray(item) for item in (q, q1, q2, q3))


def verify_sample(term: str, seed: int, tolerance: float) -> dict[str, Any]:
    q, q1, q2, q3 = sample_jets(seed)
    started = time.perf_counter()
    residual = noether_residual(
        term, q, q1, q2, q3, include_metric_representation=True
    ).block_until_ready()
    values = np.asarray(residual, dtype=np.float64)
    elapsed = time.perf_counter() - started
    maximum = float(np.max(np.abs(values)))
    return {
        "seed": seed,
        "residuals": values.tolist(),
        "maximum_absolute_residual": maximum,
        "passed": bool(np.isfinite(values).all() and maximum <= tolerance),
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    args = _parser().parse_args()
    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "sigma-aether-noether-4d-numeric-1.0",
        "verifier": str(script_path),
        "verifier_sha256": _sha256(script_path),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "dimension": DIMENSION,
        "fields": N_FIELDS,
        "field_jet_scope": (
            "general Lorentzian g_ab, covector u_a, lambda, and arbitrary symmetric coordinate "
            "derivatives through third order"
        ),
        "identity": "sum_A E_A q^A_,c - D_mu(sum_A E_A R^{A mu}_c) = 0",
        "proof_scope": (
            "floating-point arbitrary-jet falsification control; this is not an exact symbolic proof"
        ),
        "absolute_tolerance": args.absolute_tolerance,
        "seeds": args.seeds,
        "terms_requested": args.terms,
        "terms": {},
        "negative_control": {},
        "complete": False,
        "passed": False,
    }
    _write_checkpoint(args.output, payload)
    for term in args.terms:
        print(f"verifying {term}", flush=True)
        samples = [verify_sample(term, seed, args.absolute_tolerance) for seed in args.seeds]
        payload["terms"][term] = {
            "samples": samples,
            "passed": all(sample["passed"] for sample in samples),
        }
        _write_checkpoint(args.output, payload)

    negative_term = args.terms[0]
    q, q1, q2, q3 = sample_jets(args.seeds[0])
    negative = noether_residual(
        negative_term,
        q,
        q1,
        q2,
        q3,
        include_metric_representation=False,
    ).block_until_ready()
    negative_values = np.asarray(negative, dtype=np.float64)
    negative_maximum = float(np.max(np.abs(negative_values)))
    payload["negative_control"] = {
        "term": negative_term,
        "mutation": "omit the metric Lie-derivative representation",
        "residuals": negative_values.tolist(),
        "maximum_absolute_residual": negative_maximum,
        "rejected": bool(
            np.isfinite(negative_values).all() and negative_maximum >= args.negative_threshold
        ),
        "threshold": args.negative_threshold,
    }
    payload["complete"] = set(payload["terms"]) == set(args.terms)
    payload["passed"] = (
        payload["complete"]
        and all(item["passed"] for item in payload["terms"].values())
        and payload["negative_control"]["rejected"]
    )
    _write_checkpoint(args.output, payload)
    print(json.dumps({"complete": payload["complete"], "passed": payload["passed"]}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
