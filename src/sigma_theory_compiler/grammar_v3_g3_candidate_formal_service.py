from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from .cubic_bssn_domain import certify_cubic_bssn_domain
from .g3_asymptotically_flat_transition_audit import (
    _lapse_crossing_obstruction,
    _radial_profile_certificate,
    build_g3_asymptotically_flat_transition_audit,
)
from .g3_componentwise_interval_domain_audit import (
    build_g3_componentwise_interval_domain_audit,
)
from .g3_full_lapse_dirac_operator_audit import (
    _coercivity_certificate,
    _derive_full_delta,
    build_g3_full_lapse_dirac_operator_audit,
)
from .g3_seed_weak_cell_formal_audit import build_g3_seed_weak_cell_formal_audit
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .grammar_v3_promotion_admission_service import GrammarV3PromotionAdmissionService
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-g3-candidate-formal-service-config-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-g3-candidate-formal-work-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-g3-candidate-formal-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-g3-candidate-formal-status-1.0"

LOCAL_ADAPTERS = frozenset(
    {
        "generic_g2_g3_variation_noether",
        "generic_horndeski_adm_primary_degeneracy",
        "candidate_componentwise_principal_common_cone",
        "candidate_full_periodic_lapse_dirac",
        "candidate_af_profile_principal_common_cone",
        "candidate_af_lapse_obstruction",
    }
)

STATE_SQL = """
CREATE TABLE IF NOT EXISTS grammar_v3_g3_candidate_formal_service (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  immutable_config_sha256 TEXT NOT NULL,
  candidate_registry_root_sha256 TEXT NOT NULL,
  candidate_evidence_registry_root_sha256 TEXT NOT NULL,
  reviewed_adapter_registry_root_sha256 TEXT NOT NULL,
  promotion_status_content_sha256 TEXT NOT NULL
);
"""


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


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "promotion_status",
        "promotion_config",
        "seed_formal_audit",
        "seed_formal_config",
        "interval_audit",
        "interval_config",
        "lapse_audit",
        "lapse_config",
        "af_audit",
        "af_config",
        "reviewed_sources",
        "coordinator_config",
        "resource_profile",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 G3 candidate formal config is invalid")
    if not isinstance(config.get("execution_enabled"), bool):
        raise TypeError("G3 candidate formal execution_enabled must be boolean")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G3 candidate formal eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("G3 candidate formal enabled paid LLM calls")
    if not isinstance(config.get("reviewed_sources"), list) or not config["reviewed_sources"]:
        raise ValueError("G3 reviewed source allowlist is empty")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_tasks",
        "maximum_attempts",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_tasks"]) != 32
        or not 1 <= int(budget["maximum_attempts"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("G3 candidate formal budget is invalid or unbounded")


@lru_cache(maxsize=2)
def _rebuild_reviewed_campaigns(
    repo_root: str,
    seed_config_json: str,
    interval_config_json: str,
    lapse_config_json: str,
    af_config_json: str,
) -> tuple[dict[str, Any], ...]:
    root = Path(repo_root)
    seed = build_g3_seed_weak_cell_formal_audit(json.loads(seed_config_json), root)
    interval = build_g3_componentwise_interval_domain_audit(
        json.loads(interval_config_json), root
    )
    lapse = build_g3_full_lapse_dirac_operator_audit(json.loads(lapse_config_json), root)
    af = build_g3_asymptotically_flat_transition_audit(json.loads(af_config_json), root)
    return seed, interval, lapse, af


def _interval_inputs(
    *, action_sha256: str, beta: Fraction, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    domain = config["componentwise_domain"]
    beta_text = str(beta)
    ir_body = {
        "schema_version": "sigma-g3-candidate-bssn-adapter-ir-1.0",
        "source_action_sha256": action_sha256,
        "normalization": domain["normalization"],
        "formulation_classification": {
            "canonical_G2": "x",
            "canonical_G3": f"({beta_text})*x",
            "G4_X": "0",
        },
        "derivation": {
            "compiler_G2": "X_phi",
            "compiler_G3": f"({beta_text})*X_phi",
            "compiler_G4": "1/2",
            "source_normalized_G2": "0",
            "source_normalized_G3": f"-({beta_text})*x",
        },
    }
    ir = {**ir_body, "content_sha256": _sha(ir_body)}
    phi_lower, phi_upper = map(Fraction, domain["phi_interval"])
    x_lower, x_upper = map(Fraction, domain["anchor_X_interval"])
    anchor_body = {
        "schema_version": "sigma-g3-local-box-anchor-1.0",
        "status": "pass_interval_certified",
        "role": "candidate_local_box_anchor_not_on_shell_trajectory",
        "coefficients": {},
        "trajectory_hull": {
            "u": {"lower": float(phi_lower), "upper": float(phi_upper)},
            "x": {"lower": float(x_lower), "upper": float(x_upper)},
        },
        "source_action_sha256": action_sha256,
    }
    anchor = {**anchor_body, "content_sha256": _sha(anchor_body)}
    run = config["interval_run"]
    interval = {
        **run,
        "momentum_parameter_m": float(Fraction(domain["slicing"]["BSSN_m"])),
        "slicing_parameter_sigma": float(
            Fraction(domain["slicing"]["BSSN_sigma"])
        ),
        "domain_extension": {
            "phi_padding": 0.0,
            "normal_gradient_padding": float(Fraction(domain["normal_gradient_padding"])),
            "spatial_gradient_abs": float(
                Fraction(domain["spatial_gradient_component_abs"])
            ),
            "hessian_component_abs": float(
                Fraction(domain["symmetric_hessian_component_abs"])
            ),
            "riemann_component_abs": float(
                Fraction(domain["riemann_tetrad_component_abs"])
            ),
        },
        "frame_binding": domain["frame"],
        "lapse_interval": domain["lapse_interval"],
        "lapse_log_spatial_gradient_component_abs": domain[
            "lapse_log_spatial_gradient_component_abs"
        ],
        "extrinsic_curvature_component_abs": domain[
            "extrinsic_curvature_component_abs"
        ],
        "direction_sphere": domain["direction_sphere"],
    }
    return ir, anchor, interval


def _af_principal_certificate(beta: Fraction, length: Fraction) -> dict[str, Any]:
    spatial_lower = 1 - beta**2 / 4
    time_space_norm_squared = 2 * beta**2 / length**2
    cone_upper = -1 + 4 * beta / length
    if not (
        spatial_lower > 0
        and time_space_norm_squared >= 0
        and cone_upper < 0
        and beta > 0
        and beta <= Fraction(1, 100)
    ):
        raise ValueError("G3 AF principal/common-cone bound failed")
    body = {
        "effective_metric_on_profile": {
            "P00": "-(1+3*beta^2*X^2)",
            "P0i": "-2*beta*D_i(v)",
            "Pij": "(1-beta^2*X^2)*delta_ij",
        },
        "beta_specialization": str(beta),
        "length_specialization": str(length),
        "uniform_bounds_X_in_0_to_half": {
            "P00_upper": "-1",
            "spatial_eigenvalue_lower": str(spatial_lower),
            "time_space_norm_upper_squared": str(time_space_norm_squared),
            "characteristic_discriminant_lower": str(spatial_lower),
            "BSSN_sigma": "1",
            "slicing_cone_polynomial_upper": str(cone_upper),
        },
        "direction_sphere_method": (
            "isotropic_spatial_block_plus_exact_radial_time_space_norm_no_sampling"
        ),
        "status": "pass_on_complete_radial_reference_profile_including_X_limit_zero",
        "scope": "principal/common-cone profile theorem, not an Einstein-constraint solution",
    }
    return {**body, "content_sha256": _sha(body)}


class GrammarV3G3CandidateFormalService:
    """Exact bounded G3 local/periodic gates with AF and energy claims fail-closed."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        repo_root: str | Path,
        *,
        missing_adapter_ids: frozenset[str] = frozenset(),
    ) -> None:
        _validate(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.missing_adapter_ids = missing_adapter_ids
        if missing_adapter_ids - LOCAL_ADAPTERS:
            raise ValueError("unknown missing G3 reviewed adapter")
        self.promotion_status = self._bound_json("promotion_status", content=True)
        self.promotion_config = self._bound_json("promotion_config")
        self.seed_committed = self._bound_json("seed_formal_audit", content=True)
        self.seed_config = self._bound_json("seed_formal_config")
        self.interval_committed = self._bound_json("interval_audit", content=True)
        self.interval_config = self._bound_json("interval_config")
        self.lapse_committed = self._bound_json("lapse_audit", content=True)
        self.lapse_config = self._bound_json("lapse_config")
        self.af_committed = self._bound_json("af_audit", content=True)
        self.af_config = self._bound_json("af_config")
        self.base_coordinator = self._bound_json("coordinator_config")
        self.resource_profile = self._bound_json("resource_profile")
        for descriptor in config["reviewed_sources"]:
            self._path(descriptor, "reviewed_sources")
        self._validate_upstream_status()
        rebuilt = _rebuild_reviewed_campaigns(
            str(self.repo_root),
            _canonical(self.seed_config),
            _canonical(self.interval_config),
            _canonical(self.lapse_config),
            _canonical(self.af_config),
        )
        committed = (
            self.seed_committed,
            self.interval_committed,
            self.lapse_committed,
            self.af_committed,
        )
        if rebuilt != committed:
            raise ValueError("G3 reviewed campaign chain does not exactly rebuild")
        self.reviewed_adapter_registry_root_sha256 = _sha(
            {
                "seed_adapter_results": self.seed_committed["adapter_results"],
                "reviewed_campaign_content_roots": [item["content_sha256"] for item in committed],
                "reviewed_source_bindings": config["reviewed_sources"],
                "candidate_adapter_ids": sorted(LOCAL_ADAPTERS),
                "missing_adapter_ids": sorted(missing_adapter_ids),
            }
        )
        self.promotion = GrammarV3PromotionAdmissionService(
            self.directory / "promotion-attestation",
            self.promotion_config,
            self.repo_root,
        )
        if self.promotion_status["eligible_candidate_registry_root_sha256"] != (
            self.promotion.eligible_candidate_registry_root_sha256
        ):
            raise ValueError("G3 service promotion candidate registry changed")
        self.candidate_evidence: dict[str, dict[str, Any]] = {}
        self.work_items = self._work_items()
        self.candidate_registry_root_sha256 = _sha(
            [
                [
                    item["candidate_id"],
                    item["typed_action_ir_sha256"],
                    item["preflight_result_sha256"],
                    item["admission_result_sha256"],
                    item["candidate_evidence_sha256"],
                ]
                for item in self.work_items
            ]
        )
        self.candidate_evidence_registry_root_sha256 = _sha(
            [self.candidate_evidence[item["candidate_id"]] for item in self.work_items]
        )
        self.coordinator = PersistentParallelSearch(
            self.directory / "g3-candidate-formal.sqlite",
            self._coordinator_config(),
            self.resource_profile,
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _path(self, binding: dict[str, Any], label: str) -> Path:
        path = (self.repo_root / binding["path"]).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"G3 candidate formal {label} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"G3 candidate formal {label} file hash mismatch")
        return path

    def _bound_json(self, key: str, *, content: bool = False) -> dict[str, Any]:
        binding = self.config[key]
        value = _load(self._path(binding, key))
        if content:
            body = {name: item for name, item in value.items() if name != "content_sha256"}
            if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
                "content_sha256"
            ]:
                raise ValueError(f"G3 candidate formal {key} content hash mismatch")
        return value

    def _validate_upstream_status(self) -> None:
        status = self.promotion_status
        queue = "grammar_v3_g3_candidate_adm_formal"
        if (
            status.get("decision_counts") != {"pass": 162}
            or status.get("target_queue_counts", {}).get(queue) != 32
            or status.get("target_queue_registry_roots", {}).get(queue) is None
            or status.get("downstream_expensive_execution_started") is not False
            or status.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
        ):
            raise ValueError("G3 promotion-admission status is ineligible")

    def _candidate_parameters(self) -> dict[str, dict[str, str]]:
        cells = iter_parameter_cells(
            self.promotion.preflight.cell_manifest,
            self.promotion.preflight.source_manifest,
        )
        parameters: dict[str, dict[str, str]] = {}
        for cell in cells:
            if cell["family_id"] != "CUBIC_HORNDESKI_G3_WEAK_CELL":
                continue
            candidate_id = "G3A-" + _sha(_action_density_key(cell))[:24]
            parameters[candidate_id] = {
                "beta": cell["rational_coordinates"]["beta"],
                "G3": cell["parameters"]["G3"],
                "cell_lineage_sha256": cell["parameter_cell_lineage_sha256"],
            }
        if len(parameters) != 32:
            raise ValueError("G3 canonical action count changed")
        return parameters

    def _evidence(self, action_sha256: str, beta: Fraction) -> dict[str, Any]:
        ir, anchor, interval_run = _interval_inputs(
            action_sha256=action_sha256, beta=beta, config=self.interval_config
        )
        principal = certify_cubic_bssn_domain(ir, anchor, interval_run)
        if principal.get("status") != "pass_uniform_local_jet_box":
            raise ValueError("G3 candidate componentwise principal certificate failed")
        derivation = _derive_full_delta(beta)
        coercivity = _coercivity_certificate(
            {"componentwise_domain": self.interval_config["componentwise_domain"]},
            beta,
            self.lapse_config["operator_domain"],
        )
        length = Fraction(self.af_config["transition_domain"]["transition_length_L"])
        profile = _radial_profile_certificate(length)
        af_principal = _af_principal_certificate(beta, length)
        af_lapse = _lapse_crossing_obstruction(beta, length)
        body = {
            "beta": str(beta),
            "candidate_adapter_ir_sha256": ir["content_sha256"],
            "local_box_anchor_sha256": anchor["content_sha256"],
            "principal_certificate_sha256": principal["content_sha256"],
            "principal_uniform_margins": principal["uniform_proof"],
            "lapse_derivation_sha256": derivation["content_sha256"],
            "periodic_lapse_coercivity_sha256": coercivity["content_sha256"],
            "periodic_lapse_lower_bound": coercivity["Delta_N_lower_bound"]["exact"],
            "af_profile_sha256": profile["content_sha256"],
            "af_principal_certificate_sha256": af_principal["content_sha256"],
            "af_lapse_obstruction_sha256": af_lapse["content_sha256"],
            "af_lapse_status": af_lapse["Dirac_operator_status"],
        }
        return {**body, "content_sha256": _sha(body)}

    def _work_items(self) -> list[dict[str, Any]]:
        queue = "grammar_v3_g3_candidate_adm_formal"
        promotion_by_id = {
            item["candidate_id"]: item
            for item in self.promotion.work_items
            if item["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL"
        }
        parameters = self._candidate_parameters()
        items = []
        for candidate_id in sorted(promotion_by_id):
            source = promotion_by_id[candidate_id]
            fake = WorkLease(
                work_id="attestation",
                ordinal=int(source["ordinal"]),
                lane="cpu",
                seed=0,
                attempt=1,
                max_attempts=1,
                payload=source,
            )
            admission = self.promotion.execute_lease(fake)
            if admission["decision"] != "pass":
                raise ValueError("G3 promotion admission does not replay as pass")
            parameter = parameters.get(candidate_id)
            if parameter is None:
                raise ValueError("admitted G3 candidate lacks exact parameter cell")
            beta = Fraction(parameter["beta"])
            if not Fraction(0) < beta <= Fraction(1, 100):
                raise ValueError("G3 beta left the reviewed weak-cell envelope")
            evidence = self._evidence(source["typed_action_ir_sha256"], beta)
            self.candidate_evidence[candidate_id] = evidence
            body = {
                "schema_version": PAYLOAD_SCHEMA,
                "ordinal": len(items),
                "candidate_id": candidate_id,
                "typed_action_ir_sha256": source["typed_action_ir_sha256"],
                "preflight_result_sha256": source["preflight_result_sha256"],
                "admission_result_sha256": admission["content_sha256"],
                "promotion_g3_queue_root_sha256": self.promotion_status[
                    "target_queue_registry_roots"
                ][queue],
                "parameter_cell_lineage_sha256": parameter["cell_lineage_sha256"],
                "beta": str(beta),
                "G3": parameter["G3"],
                "candidate_evidence_sha256": evidence["content_sha256"],
                "reviewed_seed_campaign_content_sha256": self.seed_committed[
                    "content_sha256"
                ],
                "reviewed_interval_campaign_content_sha256": self.interval_committed[
                    "content_sha256"
                ],
                "reviewed_lapse_campaign_content_sha256": self.lapse_committed[
                    "content_sha256"
                ],
                "reviewed_af_campaign_content_sha256": self.af_committed["content_sha256"],
                "reviewed_adapter_registry_root_sha256": (
                    self.reviewed_adapter_registry_root_sha256
                ),
                "data_eligibility": dict(ELIGIBILITY),
            }
            items.append({**body, "input_lineage_sha256": _sha(body)})
        if len(items) != 32 or {item["beta"] for item in items} != {
            str(Fraction(index, 3200)) for index in range(1, 33)
        }:
            raise ValueError("G3 formal service requires the exact 32-cell beta grid")
        return items

    def _coordinator_config(self) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        config["queue"].update(
            maximum_pending_work=32,
            maximum_attempts=int(self.config["budget"]["maximum_attempts"]),
            lease_seconds=int(self.config["budget"]["maximum_wall_seconds"]),
            checkpoint_every_completions=4,
        )
        config["budget"] = {
            "maximum_tasks": 32,
            "maximum_wall_seconds": float(self.config["budget"]["maximum_wall_seconds"]),
        }
        config["cpu"]["maximum_workers"] = min(8, int(config["cpu"]["maximum_workers"]))
        config["external_paid_llm_calls"] = False
        return config

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "immutable_config_sha256": _sha(self.config),
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "candidate_evidence_registry_root_sha256": (
                self.candidate_evidence_registry_root_sha256
            ),
            "reviewed_adapter_registry_root_sha256": (
                self.reviewed_adapter_registry_root_sha256
            ),
            "promotion_status_content_sha256": self.promotion_status["content_sha256"],
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT * FROM grammar_v3_g3_candidate_formal_service WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_g3_candidate_formal_service VALUES (1,?,?,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume changed G3 candidate formal service")
            for row in connection.execute("SELECT payload_json FROM work"):
                if json.loads(row[0]).get("schema_version") != PAYLOAD_SCHEMA:
                    raise ValueError("G3 candidate formal service requires a dedicated DB")

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def enqueue(self) -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 G3 candidate formal service is disabled")
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("G3 candidate formal disk budget exhausted")
        admitted = self.coordinator.enqueue(
            self.work_items,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts"]),
        )
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": 32,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        if lease.ordinal >= 32 or lease.payload != self.work_items[lease.ordinal]:
            raise ValueError("G3 candidate formal leased payload changed")
        payload = lease.payload
        evidence = self.candidate_evidence[payload["candidate_id"]]
        missing = sorted(self.missing_adapter_ids)
        variation = (
            "blocked" if "generic_g2_g3_variation_noether" in missing else "pass"
        )
        adm = (
            "blocked" if "generic_horndeski_adm_primary_degeneracy" in missing else "pass"
        )
        local_principal = (
            "blocked"
            if "candidate_componentwise_principal_common_cone" in missing
            else "pass"
        )
        periodic_lapse = (
            "blocked" if "candidate_full_periodic_lapse_dirac" in missing else "pass"
        )
        af_principal = (
            "blocked"
            if "candidate_af_profile_principal_common_cone" in missing
            else "pass"
        )
        af_obstruction = "candidate_af_lapse_obstruction" not in missing
        blocker = (
            "reviewed_g3_adapter_missing:" + ",".join(missing)
            if missing
            else "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain"
        )
        gates = {
            "candidate_action_preflight_admission_binding": "pass",
            "exact_parameter_cell_and_weak_envelope": "pass",
            "covariant_G2_G3_variation_noether": variation,
            "adm_primary_degeneracy": adm,
            "uniform_local_principal_symbol": local_principal,
            "uniform_local_common_time_and_BSSN_cone": local_principal,
            "all_spatial_covector_directions": local_principal,
            "full_candidate_lapse_operator_derivation": periodic_lapse,
            "periodic_lapse_coercivity_and_zero_mode_exclusion": periodic_lapse,
            "distributed_Dirac_on_periodic_cell": periodic_lapse,
            "af_finite_scalar_energy_tail": "pass",
            "af_reference_principal_common_cone": af_principal,
            "af_uniform_lapse_Dirac_invertibility": "blocked",
            "af_Einstein_constraint_solution": "blocked",
            "global_hamiltonian_energy": "blocked",
            "full_formal_completion": "blocked",
        }
        body = {
            "schema_version": RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "preflight_result_sha256": payload["preflight_result_sha256"],
            "admission_result_sha256": payload["admission_result_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "beta": payload["beta"],
            "candidate_evidence_sha256": evidence["content_sha256"],
            "principal_certificate_sha256": evidence["principal_certificate_sha256"],
            "periodic_lapse_coercivity_sha256": evidence[
                "periodic_lapse_coercivity_sha256"
            ],
            "af_lapse_obstruction_sha256": evidence["af_lapse_obstruction_sha256"],
            "reviewed_adapter_registry_root_sha256": (
                self.reviewed_adapter_registry_root_sha256
            ),
            "gate_ledger": gates,
            "decision": "blocked",
            "first_missing_premise": blocker,
            "af_lapse_approximate_zero_mode_obstruction_verified": af_obstruction,
            "af_global_constraint_solution_proved": False,
            "global_positive_energy_proved": False,
            "full_formal_pass": False,
            "necessary_condition_rejection_found": False,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-g3-formal") -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 G3 candidate formal service is disabled")
        started = time.monotonic()
        current = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(current[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(32):
            if time.monotonic() - started > float(self.config["budget"]["maximum_wall_seconds"]):
                raise TimeoutError("G3 candidate formal wall budget exhausted")
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("G3 candidate formal lease was lost")
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
                raise
            executed += 1
        checkpoint = self.coordinator.checkpoint()
        return {
            "executed": executed,
            "recovered": recovered,
            "checkpoint_sha256": checkpoint["content_sha256"],
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        work_counts: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        gate_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        blocker_counts: Counter[str] = Counter()
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT ordinal,payload_json,state,attempt,result_json,error_text FROM work "
                "ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            ordinal = int(row["ordinal"])
            payload = json.loads(row["payload_json"])
            if ordinal >= 32 or payload != self.work_items[ordinal]:
                raise ValueError("stored G3 candidate formal payload was tampered")
            work_counts[str(row["state"])] += 1
            result_sha = blocker = None
            if row["result_json"]:
                result = json.loads(row["result_json"])
                expected = self.execute_lease(
                    WorkLease(
                        work_id="result-attestation",
                        ordinal=ordinal,
                        lane="cpu",
                        seed=0,
                        attempt=int(row["attempt"]),
                        max_attempts=int(self.config["budget"]["maximum_attempts"]),
                        payload=payload,
                    )
                )
                if result != expected:
                    raise ValueError("stored G3 candidate formal result binding changed")
                decisions[result["decision"]] += 1
                for gate, state in result["gate_ledger"].items():
                    gate_counts[gate][state] += 1
                blocker = result["first_missing_premise"]
                blocker_counts[blocker] += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
                    "preflight_result_sha256": payload["preflight_result_sha256"],
                    "admission_result_sha256": payload["admission_result_sha256"],
                    "beta": payload["beta"],
                    "candidate_evidence_sha256": payload["candidate_evidence_sha256"],
                    "state": row["state"],
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "first_missing_premise": blocker,
                    "error_text": row["error_text"],
                }
            )
        body = {
            "schema_version": STATUS_SCHEMA,
            "execution_enabled": self.config["execution_enabled"],
            "immutable_config_sha256": _sha(self.config),
            "candidate_count": 32,
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "candidate_evidence_registry_root_sha256": (
                self.candidate_evidence_registry_root_sha256
            ),
            "reviewed_adapter_registry_root_sha256": (
                self.reviewed_adapter_registry_root_sha256
            ),
            "promotion_status_binding": self.config["promotion_status"],
            "promotion_config_binding": self.config["promotion_config"],
            "reviewed_campaign_bindings": {
                "seed_formal_audit": self.config["seed_formal_audit"],
                "interval_audit": self.config["interval_audit"],
                "lapse_audit": self.config["lapse_audit"],
                "af_audit": self.config["af_audit"],
            },
            "reviewed_config_bindings": {
                "seed_formal_config": self.config["seed_formal_config"],
                "interval_config": self.config["interval_config"],
                "lapse_config": self.config["lapse_config"],
                "af_config": self.config["af_config"],
            },
            "reviewed_source_registry_root_sha256": _sha(self.config["reviewed_sources"]),
            "work_state_counts": dict(sorted(work_counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "gate_counts": {
                gate: dict(sorted(counts.items())) for gate, counts in sorted(gate_counts.items())
            },
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "record_registry_root_sha256": _sha(records),
            "checkpoint_sequence": self.coordinator.telemetry()["checkpoint_sequence"],
            "disk_bytes": self._disk_bytes(),
            "af_global_constraint_solution_proved": False,
            "global_positive_energy_proved": False,
            "full_formal_pass_count": 0,
            "necessary_condition_rejection_count": 0,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}


def portable_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in status.items()
        if key not in {"content_sha256", "checkpoint_sequence", "disk_bytes"}
    }
    return {**body, "content_sha256": _sha(body)}
