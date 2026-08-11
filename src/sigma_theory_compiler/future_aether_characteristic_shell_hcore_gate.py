"""Characteristic-shell obstruction to a global flat-chart Aether H_core."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

from .adm_aether import einstein_aether_3plus1_decomposition_control
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-future-aether-characteristic-shell-hcore-gate-config-1.0"
RESULT_SCHEMA = "sigma-future-aether-characteristic-shell-hcore-gate-1.0"
TARGET_ID = "G3A-5e9f93eda83935f288c19571"
TARGET_PARAMETERS = {"c1": "1/32", "c2": "0", "c3": "0", "c4": "1/32"}
CHARACTERISTIC_BLOCKER = (
    "noncharacteristic_foliation_or_compact_negative_seed_avoiding_"
    "forced_ADM_Legendre_characteristic_crossing"
)
YORK_SHELL_BLOCKER = (
    "alternative_canonical_momentum_variable_or_gauge_avoiding_"
    "exact_finite_tilt_York_symbol_shell"
)
SOURCE_BLOCKER = (
    "candidate_bound_spatially_distributed_canonical_H_core_and_"
    "metric_covariantized_H_D_Frechet_DAG_off_flat_seed_chart"
)
BLOCKER = "declared_compact_seed_crosses_candidate_bound_Legendre_characteristic_shell_F2_eq31"
EXPECTED_CONTROL_SHA256 = "232ea12f7815b99e5a162f74ef4932d8bf2bada041d71fe71cc385be0152a353"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding.get("file_sha256"):
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_canonical_artifact",
        "source_canonical_config",
        "source_canonical_implementation",
        "source_lower_order_artifact",
        "source_compact_seed_artifact",
        "reviewed_adm_source",
        "formal_report",
        "reviewed_control",
        "exact_target",
        "budget",
        "observational_authorization",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("future Aether characteristic-shell config is invalid")
    if config["exact_target"] != {
        "candidate_id": TARGET_ID,
        "typed_action_ir_sha256": (
            "d9f9b6734940de0a8378b4aad57d7b21dc5bdf126dc010b1ae5f9cc60cd91a37"
        ),
        "action_density_equivalence_sha256": (
            "5e9f93eda83935f288c19571ab113fdd638f6f8a74e450a39481f8e57aafd76c"
        ),
        "parameters": TARGET_PARAMETERS,
        "profile": "F=10*(1-r^2)^4_+",
    }:
        raise ValueError("future Aether characteristic-shell target changed")
    if config["budget"] != {
        "maximum_candidates": 14,
        "maximum_symbolic_velocities": 9,
        "maximum_reviewed_control_replays": 1,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("future Aether characteristic-shell budget changed")
    if config["reviewed_control"] != {
        "formal_check_id": "einstein_aether_generic_3plus1_legendre",
        "entrypoint": (
            "sigma_theory_compiler.adm_aether:einstein_aether_3plus1_decomposition_control"
        ),
        "evidence_sha256": EXPECTED_CONTROL_SHA256,
    }:
        raise ValueError("future Aether characteristic-shell reviewed control changed")
    if config["observational_authorization"] is not False:
        raise ValueError("future Aether characteristic-shell observations were opened")
    if config["external_paid_llm_calls"] is not False:
        raise ValueError("future Aether characteristic-shell paid LLM calls were opened")
    if config["data_eligibility"] != ELIGIBILITY:
        raise ValueError("future Aether characteristic-shell data eligibility changed")


def _validate_lineage(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    source = _bound_json(root, config["source_canonical_artifact"], "canonical artifact")
    source_config = _bound_json(root, config["source_canonical_config"], "canonical config")
    _bound_path(root, config["source_canonical_implementation"], "canonical implementation")
    lower = _bound_json(root, config["source_lower_order_artifact"], "lower-order artifact")
    compact = _bound_json(root, config["source_compact_seed_artifact"], "compact-seed artifact")
    adm_path = _bound_path(root, config["reviewed_adm_source"], "reviewed ADM source")
    formal = _bound_json(root, config["formal_report"], "formal report")
    if source_config.get("source_lower_order_artifact") != config["source_lower_order_artifact"]:
        raise ValueError("canonical config lower-order lineage changed")
    if source_config.get("source_compact_seed_artifact") != config["source_compact_seed_artifact"]:
        raise ValueError("canonical config compact-seed lineage changed")
    if source_config.get("reviewed_adm_source") != config["reviewed_adm_source"]:
        raise ValueError("canonical config reviewed ADM lineage changed")
    callback_path = Path(
        inspect.getsourcefile(inspect.unwrap(einstein_aether_3plus1_decomposition_control)) or ""
    ).resolve()
    evidence = einstein_aether_3plus1_decomposition_control()
    report_check = next(
        (
            item
            for item in formal.get("checks", [])
            if item.get("name") == config["reviewed_control"]["formal_check_id"]
        ),
        None,
    )
    if (
        callback_path != adm_path
        or not isinstance(report_check, dict)
        or report_check.get("status") != "pass"
        or report_check.get("evidence") != evidence
        or evidence.get("passed") is not True
        or _sha(evidence) != EXPECTED_CONTROL_SHA256
    ):
        raise ValueError("reviewed generic Aether 3+1 control changed")
    if source.get("candidate_count") != 14 or source.get("decision_counts") != {"blocked": 14}:
        raise ValueError("canonical source candidate partition changed")
    if lower.get("candidate_count") != 14 or compact.get("candidate_count") != 14:
        raise ValueError("transitive predecessor candidate partition changed")
    return source, lower, compact, _sha(evidence)


@lru_cache(maxsize=1)
def exact_characteristic_shell_control() -> dict[str, Any]:
    """Derive the target's flat-chart local Legendre Hessian and its seed shell."""

    f, f1, f2, f3 = sp.symbols("F F_1 F_2 F_3", real=True)
    chi = sp.sqrt(1 + f**2)
    aether = sp.Matrix([f, 0, 0])
    derivative = sp.Matrix([[f1, 0, 0], [f2, 0, 0], [f3, 0, 0]])
    k11, k22, k33, k12, k13, k23 = sp.symbols(
        "K11 K22 K33 K12 K13 K23", real=True
    )
    extrinsic = sp.Matrix(
        [[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]]
    )
    electric = sp.Matrix(sp.symbols("W1:4", real=True))
    spatial_block = derivative + chi * extrinsic
    spatial_normal = sp.Matrix([f * f1 / chi, f * f2 / chi, f * f3 / chi]) + (
        extrinsic * aether
    )
    normal_normal = (aether.T * electric)[0] / chi
    invariant_1 = sp.expand(
        normal_normal**2
        - (electric.T * electric)[0]
        - (spatial_normal.T * spatial_normal)[0]
        + sp.trace(spatial_block.T * spatial_block)
    )
    acceleration_normal = sp.expand(
        -chi * normal_normal - (aether.T * spatial_normal)[0]
    )
    acceleration_spatial = sp.expand(chi * electric + spatial_block.T * aether)
    invariant_4 = sp.expand(
        -acceleration_normal**2 + (acceleration_spatial.T * acceleration_spatial)[0]
    )
    trace_k = sp.trace(extrinsic)
    k_squared = sp.trace(extrinsic.T * extrinsic)
    lagrangian = sp.expand(
        (k_squared - trace_k**2) / 2
        - (sp.Rational(1, 32) * invariant_1 - sp.Rational(1, 32) * invariant_4) / 2
    )
    velocities = (k11, k22, k33, k12, k13, k23, *electric)
    zero = {item: 0 for item in velocities}
    hessian = sp.simplify(sp.hessian(lagrangian, velocities))
    determinant = sp.factor(hessian.det())
    affine = sp.Matrix(
        [sp.factor(sp.diff(lagrangian, item).subs(zero)) for item in velocities]
    )
    lagrangian_zero = sp.factor(lagrangian.subs(zero))
    expected_determinant = (
        -31
        * (f**2 - 31) ** 2
        * (33 * f**2 + 65)
        * (61 * f**2 + 124) ** 2
        / (sp.Integer(8796093022208) * (f**2 + 1))
    )
    shell = {f: sp.sqrt(31)}
    shell_hessian = sp.simplify(hessian.subs(shell))
    nullspace = shell_hessian.nullspace()
    expected_nullspace = [
        sp.Matrix([0, -1, 1, 0, 0, 0, 0, 0, 0]),
        sp.Matrix([0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ]
    pi_seed = sp.Matrix(
        [
            f1 * (f**2 - 1) / (64 * chi),
            0,
            0,
            -f2 / (128 * chi),
            -f3 / (128 * chi),
            0,
        ]
    )
    pa_seed = sp.Matrix([f * f1 / (32 * chi), 0, 0])
    canonical_pairing_momentum = sp.Matrix(
        [
            2 * pi_seed[0],
            2 * pi_seed[1],
            2 * pi_seed[2],
            4 * pi_seed[3],
            4 * pi_seed[4],
            4 * pi_seed[5],
            *pa_seed,
        ]
    )
    seed_affine_residual = sp.simplify(canonical_pairing_momentum - affine)
    shell_momentum = sp.simplify(canonical_pairing_momentum.subs(shell))
    compatibility = [sp.factor(vector.dot(shell_momentum)) for vector in nullspace]
    incompatible_mutation = shell_momentum + expected_nullspace[1]
    mutation_residual = [
        sp.factor(vector.dot(incompatible_mutation)) for vector in nullspace
    ]
    radius = sp.sqrt(1 - (sp.Rational(31, 100)) ** sp.Rational(1, 8))
    profile_shell_residual = sp.simplify(
        100 * (1 - radius**2) ** 8 - 31
    )
    amplitude_five_margin = sp.Integer(31) - sp.Integer(5) ** 2
    passed = bool(
        sp.factor(determinant - expected_determinant) == 0
        and shell_hessian.rank() == 7
        and nullspace == expected_nullspace
        and seed_affine_residual == sp.zeros(9, 1)
        and compatibility == [0, 0]
        and mutation_residual != [0, 0]
        and profile_shell_residual == 0
        and amplitude_five_margin == 6
    )
    return {
        "passed": passed,
        "candidate_id": TARGET_ID,
        "parameters": TARGET_PARAMETERS,
        "velocity_order": [str(item) for item in velocities],
        "canonical_pairing_momentum_order": [
            "2*pi11",
            "2*pi22",
            "2*pi33",
            "4*pi12",
            "4*pi13",
            "4*pi23",
            "p_A1",
            "p_A2",
            "p_A3",
        ],
        "local_legendre_hessian": [[str(sp.factor(item)) for item in row] for row in hessian.tolist()],
        "hessian_determinant": str(determinant),
        "only_real_characteristic_condition": "F**2=31",
        "affine_momentum_shift": [str(item) for item in affine],
        "zero_velocity_lagrangian": str(lagrangian_zero),
        "regular_stratum_H_core": {
            "domain": "F**2 != 31",
            "formula": "H_core=1/2*(P-b)^T*H(F)^(-1)*(P-b)-L0",
            "H": "local_legendre_hessian",
            "b": "affine_momentum_shift",
            "L0": "zero_velocity_lagrangian",
            "registered": True,
            "global_on_declared_profile": False,
        },
        "declared_profile_characteristic_shell": {
            "profile": "F=10*(1-r**2)^4 for 0<=r<1",
            "radius": str(radius),
            "radius_interval": "0<r_characteristic<1",
            "F_at_shell": "sqrt(31)",
            "F_squared_residual": str(profile_shell_residual),
            "hessian_rank": int(shell_hessian.rank()),
            "hessian_nullity": int(9 - shell_hessian.rank()),
            "nullspace": [[str(item) for item in vector] for vector in nullspace],
        },
        "seed_legendre_image": {
            "canonical_pairing_momentum": [str(item) for item in canonical_pairing_momentum],
            "affine_residual": [str(item) for item in seed_affine_residual],
            "shell_primary_compatibility_residuals": [str(item) for item in compatibility],
            "interpretation": (
                "the static seed momenta lie in the singular Legendre image, but the two "
                "null velocities are not uniquely recoverable"
            ),
        },
        "incompatible_momentum_negative_control": {
            "mutation": "add one unit of canonical K23-pairing momentum at F=sqrt(31)",
            "null_projection_residuals": [str(item) for item in mutation_residual],
            "rejected": bool(mutation_residual != [0, 0]),
        },
        "noncrossing_profile_control": {
            "profile": "F=5*(1-r**2)^4_+",
            "uniform_F_squared_upper_bound": "25",
            "distance_to_characteristic_F_squared": str(amplitude_five_margin),
            "avoids_shell": bool(amplitude_five_margin > 0),
            "scope": "synthetic control only; not substituted for the registered seed",
        },
        "strongest_exact_conclusion": (
            "the local H_core formula is registered on F**2!=31, but the declared compact "
            "profile crosses a rank-seven Legendre shell, so no single smooth positive-branch "
            "Legendre inverse or global flat-chart H_core/Frechet DAG is registered"
        ),
        "content_sha256": "pending",
    }


def _certificate(control: dict[str, Any], source_record: dict[str, Any], reviewed_sha: str):
    body = {key: value for key, value in control.items() if key != "content_sha256"}
    control = {**body, "content_sha256": _sha(body)}
    certificate: dict[str, Any] = {
        "candidate_id": TARGET_ID,
        "source_canonical_record_sha256": source_record["content_sha256"],
        "source_canonical_certificate_sha256": source_record[
            "canonical_seed_constraint_DAG_certificate"
        ]["content_sha256"],
        "reviewed_ADM_control_sha256": reviewed_sha,
        "characteristic_shell_control": control,
        "regular_stratum_flat_chart_H_core_contract_registered": True,
        "declared_profile_global_flat_chart_H_core_registered": False,
        "off_flat_metric_covariantization_registered": False,
        "metric_covariantized_H_D_Frechet_DAG_registered": False,
        "distributed_lower_order_B_C_registry_complete": False,
        "weighted_Fredholm_isomorphism_proven": False,
        "nonlinear_remainder_majorant_proven": False,
        "completed_boundary_sign_persistence_proven": False,
        "candidate_rejection_authorized": False,
        "first_blocker": BLOCKER,
        "next_missing_premise": (
            "a candidate-bound noncharacteristic canonical chart/profile or constrained "
            "Hamiltonian treatment across the F**2=31 primary shell, followed by off-flat "
            "metric covariantization"
        ),
    }
    certificate["content_sha256"] = _sha(certificate)
    return certificate


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("future Aether characteristic-shell artifact schema changed")
    body = {key: item for key, item in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != _sha(body):
        raise ValueError("future Aether characteristic-shell content hash mismatch")
    if result.get("candidate_count") != 14 or result.get("decision_counts") != {"blocked": 14}:
        raise ValueError("future Aether characteristic-shell decision partition changed")
    if result.get("first_blocker_counts") != {
        CHARACTERISTIC_BLOCKER: 11,
        YORK_SHELL_BLOCKER: 2,
        BLOCKER: 1,
    }:
        raise ValueError("future Aether characteristic-shell blocker partition changed")
    records = result.get("candidate_records", [])
    if len(records) != 14 or any(record.get("decision") != "blocked" for record in records):
        raise ValueError("future Aether characteristic-shell records changed")
    if any(
        record.get("formal_pass") is not False
        or record.get("candidate_rejection_authorized") is not False
        for record in records
    ):
        raise ValueError("future Aether characteristic-shell record overclaimed")
    target = [record for record in records if record.get("candidate_id") == TARGET_ID]
    if len(target) != 1 or target[0].get("first_blocker") != BLOCKER:
        raise ValueError("future Aether characteristic-shell target changed")
    certificate = target[0].get("characteristic_shell_H_core_certificate", {})
    cert_body = {key: item for key, item in certificate.items() if key != "content_sha256"}
    if certificate.get("content_sha256") != _sha(cert_body):
        raise ValueError("future Aether characteristic-shell certificate hash mismatch")
    control = certificate.get("characteristic_shell_control", {})
    control_body = {key: item for key, item in control.items() if key != "content_sha256"}
    if control.get("content_sha256") != _sha(control_body) or control.get("passed") is not True:
        raise ValueError("future Aether characteristic-shell control hash mismatch")
    if (
        certificate.get("declared_profile_global_flat_chart_H_core_registered") is not False
        or certificate.get("off_flat_metric_covariantization_registered") is not False
        or certificate.get("metric_covariantized_H_D_Frechet_DAG_registered") is not False
        or certificate.get("candidate_rejection_authorized") is not False
    ):
        raise ValueError("future Aether characteristic-shell certificate overclaimed")
    if result.get("formal_pass_count") != 0 or result.get("candidate_rejection_authorized_count") != 0:
        raise ValueError("future Aether characteristic-shell aggregate overclaimed")
    if result.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("future Aether characteristic-shell data seals changed")


def build_future_aether_characteristic_shell_hcore_gate(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Build the exact candidate-bound characteristic-shell H_core obstruction."""

    _validate_config(config)
    source, lower, compact, reviewed_sha = _validate_lineage(config, root)
    source_records = source["candidate_records"]
    target_source = next(record for record in source_records if record["candidate_id"] == TARGET_ID)
    if (
        target_source.get("parameters") != TARGET_PARAMETERS
        or target_source.get("first_blocker") != SOURCE_BLOCKER
        or target_source.get("typed_action_ir_sha256")
        != config["exact_target"]["typed_action_ir_sha256"]
        or target_source.get("action_density_equivalence_sha256")
        != config["exact_target"]["action_density_equivalence_sha256"]
    ):
        raise ValueError("canonical target source binding changed")
    control = exact_characteristic_shell_control()
    if control.get("passed") is not True:
        raise ValueError("exact characteristic-shell control failed")
    certificate = _certificate(control, target_source, reviewed_sha)
    records = []
    for source_record in source_records:
        is_target = source_record["candidate_id"] == TARGET_ID
        record = {
            "ordinal": source_record["ordinal"],
            "candidate_id": source_record["candidate_id"],
            "family_id": source_record["family_id"],
            "parameters": source_record["parameters"],
            "typed_action_ir_sha256": source_record["typed_action_ir_sha256"],
            "action_density_equivalence_sha256": source_record[
                "action_density_equivalence_sha256"
            ],
            "source_record_sha256": source_record["content_sha256"],
            "source_first_blocker": source_record["first_blocker"],
            "first_blocker": BLOCKER if is_target else source_record["first_blocker"],
            "decision": "blocked",
            "formal_pass": False,
            "candidate_rejection_authorized": False,
            "constraint_satisfying_negative_total_energy_datum_proven": False,
            "automatic_downstream_enqueue_performed": False,
            "solar_bundle_generated": False,
            "data_eligibility": ELIGIBILITY,
        }
        if is_target:
            record["characteristic_shell_H_core_certificate"] = certificate
        record["content_sha256"] = _sha(record)
        records.append(record)
    blocker_counts = dict(sorted(Counter(record["first_blocker"] for record in records).items()))
    campaign_config_path = root / "configs/future_aether_characteristic_shell_hcore_gate.json"
    campaign_implementation_path = Path(__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_lineage": {
            "canonical_artifact_file_sha256": config["source_canonical_artifact"][
                "file_sha256"
            ],
            "canonical_artifact_content_sha256": config["source_canonical_artifact"][
                "content_sha256"
            ],
            "canonical_config_file_sha256": config["source_canonical_config"]["file_sha256"],
            "canonical_implementation_file_sha256": config[
                "source_canonical_implementation"
            ]["file_sha256"],
            "lower_order_artifact_content_sha256": lower["content_sha256"],
            "compact_seed_artifact_content_sha256": compact["content_sha256"],
            "reviewed_ADM_control_sha256": reviewed_sha,
            "campaign_config_file_sha256": _file_sha(campaign_config_path),
            "campaign_implementation_file_sha256": _file_sha(campaign_implementation_path),
        },
        "candidate_count": len(records),
        "decision_counts": {"blocked": len(records)},
        "first_blocker_counts": blocker_counts,
        "candidate_records": records,
        "regular_stratum_flat_chart_H_core_contract_registered_count": 1,
        "declared_profile_global_flat_chart_H_core_registered_count": 0,
        "off_flat_metric_covariantization_registered_count": 0,
        "metric_covariantized_H_D_Frechet_DAG_registered_count": 0,
        "formal_pass_count": 0,
        "candidate_rejection_authorized_count": 0,
        "constraint_satisfying_negative_total_energy_datum_count": 0,
        "automatic_downstream_enqueue_performed": False,
        "observational_data_opened": False,
        "solar_bundle_count": 0,
        "data_eligibility": ELIGIBILITY,
        "external_paid_llm_calls": False,
        "strongest_exact_conclusion": (
            "the target admits an exact local H_core contract on F**2!=31, but its declared "
            "compact seed crosses a rank-seven Legendre shell; the global flat-chart inverse, "
            "off-flat metric covariantization, Frechet DAG, Fredholm/nonlinear/boundary gates, "
            "and candidate rejection remain unproved"
        ),
    }
    result["content_sha256"] = _sha(result)
    _validate_result(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _load(args.config)
    root = Path(__file__).resolve().parents[2]
    result = build_future_aether_characteristic_shell_hcore_gate(config, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
