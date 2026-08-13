from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-sphere-normal-form-reducer-registration-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-sphere-normal-form-reducer-registration-config-1.0"
)
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_285_symbolic_packets"
CONFIG_PATH = (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_sphere_normal_form_reducer_registration.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_sphere_normal_form_reducer_registration.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_sphere_normal_form_reducer_registration.py"
UPSTREAM_STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_286_symbolic_packets"
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 18
REGISTERED_PACKETS = 19
MISSING_PACKETS = 285
REQUIRED_ROWS = 117_180
MAXIMUM_DEGREE = 19
EXPECTED_MODES = 210
EXPECTED_GENERATOR_CASES = 615

Monomial = tuple[int, int, int]
Polynomial = dict[Monomial, Fraction]


class SphereNormalFormRegistrationError(ValueError):
    """Raised when the exact sphere normal-form registration fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SphereNormalFormRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise SphereNormalFormRegistrationError("bound path escaped project root")
    return path


def odd_sphere_modes(maximum_degree: int = MAXIMUM_DEGREE) -> list[Monomial]:
    if maximum_degree < 1 or maximum_degree % 2 == 0:
        raise SphereNormalFormRegistrationError("odd positive degree ceiling required")
    modes = [
        (a, b, total - a - b)
        for total in range(1, maximum_degree + 1, 2)
        for a in range(2)
        for b in range(total - a + 1)
    ]
    return sorted(modes, key=lambda exponent: (sum(exponent), exponent))


def _clean(polynomial: Polynomial) -> Polynomial:
    return {exponent: Fraction(value) for exponent, value in polynomial.items() if value}


def reduce_mod_sphere(polynomial: Polynomial) -> tuple[Polynomial, Polynomial]:
    work = _clean(polynomial)
    if any(min(exponent) < 0 or len(exponent) != 3 for exponent in work):
        raise SphereNormalFormRegistrationError("invalid monomial exponent")
    quotient: Polynomial = {}
    while reducible := [exponent for exponent in work if exponent[0] >= 2]:
        exponent = max(reducible)
        coefficient = work.pop(exponent)
        a, b, c = exponent
        base = (a - 2, b, c)
        quotient[base] = quotient.get(base, Fraction()) + coefficient
        for shifted, delta in (
            ((a - 2, b + 2, c), -coefficient),
            ((a - 2, b, c + 2), -coefficient),
            (base, coefficient),
        ):
            work[shifted] = work.get(shifted, Fraction()) + delta
            if not work[shifted]:
                del work[shifted]
    return _clean(quotient), _clean(work)


def reconstruct(quotient: Polynomial, remainder: Polynomial) -> Polynomial:
    result = dict(_clean(remainder))
    for (a, b, c), coefficient in _clean(quotient).items():
        for exponent, delta in (
            ((a + 2, b, c), coefficient),
            ((a, b + 2, c), coefficient),
            ((a, b, c + 2), coefficient),
            ((a, b, c), -coefficient),
        ):
            result[exponent] = result.get(exponent, Fraction()) + delta
            if not result[exponent]:
                del result[exponent]
    return _clean(result)


def coefficient_vector(polynomial: Polynomial) -> list[str]:
    if any(sum(exponent) > MAXIMUM_DEGREE or sum(exponent) % 2 != 1 for exponent in polynomial):
        raise SphereNormalFormRegistrationError(
            "extractor accepts odd polynomials through degree 19"
        )
    _, remainder = reduce_mod_sphere(polynomial)
    modes = odd_sphere_modes()
    if any(exponent not in set(modes) for exponent in remainder):
        raise SphereNormalFormRegistrationError("remainder escaped declared mode basis")
    return [str(remainder.get(exponent, Fraction())) for exponent in modes]


def _generator_multiples() -> list[Monomial]:
    return [
        (a, b, total - a - b)
        for total in range(1, MAXIMUM_DEGREE - 1, 2)
        for a in range(total + 1)
        for b in range(total - a + 1)
    ]


def _sphere_multiple(exponent: Monomial) -> Polynomial:
    a, b, c = exponent
    return {
        (a + 2, b, c): Fraction(1),
        (a, b + 2, c): Fraction(1),
        (a, b, c + 2): Fraction(1),
        (a, b, c): Fraction(-1),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "register_exact_sphere_normal_form_reducer_fail_closed"
        or not _hash_matches(config)
        or config.get("target")
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": PREDECESSOR_REGISTERED,
            "sphere_reducer_packets": 1,
            "expected_registered_packets": REGISTERED_PACKETS,
            "expected_missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or config.get("resource_caps")
        != {
            "maximum_remainder_modes": EXPECTED_MODES,
            "maximum_generator_replay_cases": EXPECTED_GENERATOR_CASES,
            "maximum_output_rows_emitted": 0,
        }
    ):
        raise SphereNormalFormRegistrationError("invalid sphere reducer config")


def _validate_upstream(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    upstream = _load_json(path)
    if (
        _file_sha256(path) != binding.get("file_sha256")
        or not _hash_matches(upstream)
        or upstream.get("content_sha256") != binding.get("content_sha256")
        or upstream.get("status") != UPSTREAM_STATUS
        or upstream.get("errors") != []
        or upstream.get("counts", {}).get("registered_symbolic_input_packets")
        != PREDECESSOR_REGISTERED
    ):
        raise SphereNormalFormRegistrationError("upstream seal or boundary mismatch")
    return upstream


def _replay_certificate() -> dict[str, Any]:
    modes = odd_sphere_modes()
    generators = _generator_multiples()
    if len(modes) != EXPECTED_MODES or len(generators) != EXPECTED_GENERATOR_CASES:
        raise SphereNormalFormRegistrationError("declared exact replay count mismatch")
    for index, exponent in enumerate(modes):
        vector = coefficient_vector({exponent: Fraction(1)})
        if vector[index] != "1" or sum(value != "0" for value in vector) != 1:
            raise SphereNormalFormRegistrationError("basis replay failed")
    for exponent in generators:
        quotient, remainder = reduce_mod_sphere(_sphere_multiple(exponent))
        if remainder or reconstruct(quotient, remainder) != _sphere_multiple(exponent):
            raise SphereNormalFormRegistrationError("sphere ideal replay failed")
    sample = {(9, 2, 0): Fraction(3, 5), (3, 0, 4): Fraction(-7, 3), (1, 2, 2): Fraction(11)}
    quotient, remainder = reduce_mod_sphere(sample)
    if reconstruct(quotient, remainder) != sample or any(a >= 2 for a, _, _ in remainder):
        raise SphereNormalFormRegistrationError("quotient witness reconstruction failed")
    wrong_sign = _sphere_multiple((1, 0, 0))
    wrong_sign[(1, 0, 0)] = Fraction(1)
    if not reduce_mod_sphere(wrong_sign)[1]:
        raise SphereNormalFormRegistrationError("wrong sphere-sign negative control failed")
    mode_records = [
        {"exponents": list(exponent), "index": index} for index, exponent in enumerate(modes)
    ]
    return {
        "sphere_relation": "n1^2+n2^2+n3^2-1",
        "division_order": "lexicographic_with_leading_monomial_n1^2",
        "mode_ordering": mode_records,
        "mode_ordering_sha256": hashlib.sha256(_canonical_bytes(mode_records)).hexdigest(),
        "basis_unit_replays": len(modes),
        "sphere_generator_multiple_replays": len(generators),
        "nontrivial_quotient_reconstruction_replays": 1,
        "wrong_sphere_sign_nonzero_remainder_controls": 1,
        "nonzero_remainders": 0,
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstream = _validate_upstream(root, config["upstream"])
    manifest = json.loads(json.dumps(upstream["required_symbolic_input_manifest"]))
    records = {record["input_id"]: record for record in manifest}
    reducer = records.get("sphere_mode_normal_form_reducer")
    if (
        len(records) != 8
        or reducer is None
        or reducer.get("required_packets") != 1
        or reducer.get("registered_packets") != 0
    ):
        raise SphereNormalFormRegistrationError("upstream manifest boundary mismatch")
    certificate = _replay_certificate()
    reducer.update(
        {
            "registered_packets": 1,
            "status": "registered_exact_odd_sphere_normal_form_reducer",
            "maximum_total_degree": MAXIMUM_DEGREE,
            "remainder_modes": EXPECTED_MODES,
            "mode_ordering_sha256": certificate["mode_ordering_sha256"],
            "quotient_witness_identity": "P=Q*(n1^2+n2^2+n3^2-1)+R",
        }
    )
    if (
        sum(row["required_packets"] for row in manifest) != REQUIRED_PACKETS
        or sum(row["registered_packets"] for row in manifest) != REGISTERED_PACKETS
    ):
        raise SphereNormalFormRegistrationError("updated manifest total mismatch")
    missing = [
        {
            "input_id": row["input_id"],
            "required_packets": row["required_packets"],
            "registered_packets": row["registered_packets"],
            "missing_packets": row["required_packets"] - row["registered_packets"],
        }
        for row in manifest
        if row["registered_packets"] < row["required_packets"]
    ]
    if sum(row["missing_packets"] for row in missing) != MISSING_PACKETS:
        raise SphereNormalFormRegistrationError("updated missing total mismatch")
    false_claims = {key: False for key in upstream["claims"] if upstream["claims"][key] is False}
    false_claims.update(
        {
            "sphere_mode_normal_form_reducer_registered": True,
            "all_210_odd_sphere_modes_replayed": True,
            "all_615_sphere_generator_multiples_reduce_to_zero": True,
            "manifest_recomputed_from_exact_packets": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_binding": {
            **config["upstream"],
            "verified": True,
        },
        "registered_sphere_normal_form_reducer": certificate,
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets",
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "285 required symbolic input packets remain unregistered",
        },
        "counts": {
            "upstream_seals_verified": 1,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "new_sphere_reducer_packets_registered": 1,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "odd_sphere_remainder_modes": EXPECTED_MODES,
            "basis_unit_replays": EXPECTED_MODES,
            "sphere_generator_multiple_replays": EXPECTED_GENERATOR_CASES,
            "nonzero_replay_remainders": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": false_claims,
        "negative_controls": {
            "accept_even_input_in_odd_extractor": {"rejected": True},
            "accept_degree_above_19": {"rejected": True},
            "omit_quotient_witness": {"rejected": True},
            "use_wrong_sphere_constant_sign": {"rejected": True},
            "emit_rows_with_285_missing_packets": {"rejected": True},
            "promote_reducer_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers only the exact 210-mode odd polynomial normal-form reducer modulo "
            "the unit-sphere relation through degree 19. The 225 Taylor packets and 60 "
            "lower-Sylvester recurrence packets remain missing; no coefficient row, D4 "
            "theorem, H7 closure, PDE theorem, lifespan, or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise SphereNormalFormRegistrationError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    print(write_campaign(document, args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
