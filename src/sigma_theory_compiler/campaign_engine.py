from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sympy as sp

from .action_health import analyze_action_health
from .campaign import CampaignStore, ClaimedTask, canonical_json, stable_id, utc_now
from .equation_universe import EquationUniverse
from .formal_backend import run_formal_control_suite, write_formal_report
from .gpu_screen import run_dense_gpu_screen
from .knowledge import GateOntology, KnowledgeBuilder
from .observation_eligibility import (
    audit_theory_observation_eligibility,
    write_observation_eligibility,
)
from .relativity import run_relativity_reference_suite
from .static_dictionary import classify_generator_expression


@dataclass
class WorkerOutcome:
    status: str = "succeeded"
    result: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate(
    gate_id: str,
    outcome: str,
    stage: int,
    *,
    hard: bool = True,
    payload: dict[str, Any] | None = None,
    margin: float | None = None,
    units: str | None = None,
    version: str = "1.0",
    evidence_class: str = "theory",
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "gate_version": version,
        "stage": stage,
        "is_hard": hard,
        "outcome": outcome,
        "margin": margin,
        "units": units,
        "evidence_class": evidence_class,
        "payload": payload or {},
    }


def initialize_campaign(
    store: CampaignStore,
    config: dict[str, Any],
    source_queue: str | Path,
) -> str:
    store.initialize()
    source_queue = Path(source_queue).resolve()
    config = json.loads(json.dumps(config))
    config["project_root"] = str(Path(config.get("project_root", ".")).resolve())
    config["source_queue"] = str(source_queue)
    output_root = Path(config["output_root"]).resolve()
    config["output_root"] = str(output_root)
    campaign_id = store.create_campaign(config)
    store.register_artifact(campaign_id, source_queue, kind="dense_pareto_source")
    queue = _read_json(source_queue)
    seed_limit = int(config.get("seed", {}).get("candidate_limit", 6))
    seed_rows = sorted(
        queue["work_queue"], key=lambda row: (row["pareto_front"], row["family_id"])
    )[:seed_limit]
    for row in seed_rows:
        canonical = {
            "source": "generated-priority-dense",
            "family_id": row["family_id"],
            "ordinal": row["ordinal"],
            "term_ids": row["term_ids"],
            "sign_mask": row["sign_mask"],
            "correction_expression": row["correction_expression"],
        }
        candidate_id = store.add_candidate(
            campaign_id,
            kind="generated_static",
            expression=row["correction_expression"],
            canonical=canonical,
            family_id=row["family_id"],
            generation=0,
            pareto_front=row["pareto_front"],
            mechanism_tags=row["mechanism_tags"],
        )
        for evidence in (
            _gate("finite_grammar_membership", "pass", 0, payload={"ordinal": row["ordinal"]}),
            _gate("generator_v2_structural_static", "pass", 1),
            _gate("dense_sampled_static_343", "pass", 1, version="1.0-343-point"),
        ):
            store.record_evidence(campaign_id, candidate_id, None, evidence)
        policy = store.add_task(
            campaign_id,
            "policy_validate",
            stage=0,
            payload={"candidate_id": candidate_id},
            candidate_id=candidate_id,
            priority=100.0,
            diversity_bucket=row["family_id"],
        )
        store.add_task(
            campaign_id,
            "covariant_lift",
            stage=2,
            payload={"candidate_id": candidate_id},
            candidate_id=candidate_id,
            priority=90.0,
            diversity_bucket=row["family_id"],
            depends_on=[policy],
        )

    if config.get("seed", {}).get("include_gr_control", True):
        control = {
            "action": "S_EH=(M_Pl^2/2) integral sqrt(-g) R + S_m[g,psi]",
            "fields": ["g_mu_nu"],
            "matter_metric": "g_mu_nu",
            "role": "golden_control_not_discovery",
        }
        control_id = store.add_candidate(
            campaign_id,
            kind="gr_control",
            expression=control["action"],
            canonical=control,
            family_id="CONTROL-EINSTEIN-HILBERT",
            mechanism_tags=["tensor_polarization"],
        )
        policy = store.add_task(
            campaign_id,
            "policy_validate",
            stage=0,
            payload={"candidate_id": control_id},
            candidate_id=control_id,
            priority=120.0,
            diversity_bucket="control",
        )
        formal = store.add_task(
            campaign_id,
            "formal_reference_controls",
            stage=2,
            payload={"candidate_id": control_id},
            candidate_id=control_id,
            priority=115.0,
            diversity_bucket="control",
            depends_on=[policy],
        )
        store.add_task(
            campaign_id,
            "reference_control",
            stage=5,
            payload={"candidate_id": control_id},
            candidate_id=control_id,
            priority=110.0,
            diversity_bucket="control",
            depends_on=[formal],
        )

    audit_task = store.add_task(
        campaign_id,
        "pipeline_artifact_audit",
        stage=0,
        payload=config.get("existing_artifacts", {}),
        priority=130.0,
        diversity_bucket="infrastructure",
    )
    cluster_task = store.add_task(
        campaign_id,
        "failure_cluster",
        stage=90,
        payload={"campaign_id": campaign_id},
        priority=-100.0,
        diversity_bucket="research-cycle-0",
        depends_on=[audit_task],
    )
    store.add_task(
        campaign_id,
        "llm_research",
        stage=91,
        payload={"campaign_id": campaign_id, "cycle": 0},
        priority=-110.0,
        diversity_bucket="research-cycle-0",
        depends_on=[cluster_task],
    )
    return campaign_id


class CampaignEngine:
    def __init__(
        self,
        store: CampaignStore,
        campaign_id: str,
        worker_id: str,
        allowed_task_types: set[str] | None = None,
    ):
        self.store = store
        self.campaign_id = campaign_id
        self.worker_id = worker_id
        self.allowed_task_types = allowed_task_types
        campaign = store.campaign(campaign_id)
        self.config = json.loads(campaign["config_json"])
        self.project_root = Path(self.config["project_root"])
        self.output_root = Path(self.config["output_root"]) / campaign_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        prior_art_config = self.config.get("prior_art", {})
        prior_art_database = Path(
            prior_art_config.get(
                "equation_universe_database",
                "runs/equation-universe/equations-v1.sqlite",
            )
        )
        if not prior_art_database.is_absolute():
            prior_art_database = self.project_root / prior_art_database
        self.prior_art_database = prior_art_database.resolve()
        self.prior_art_nearest_limit = int(prior_art_config.get("nearest_limit", 5))
        self.lease_seconds = int(self.config.get("runtime", {}).get("lease_seconds", 300))
        self.handlers: dict[str, Callable[[ClaimedTask], WorkerOutcome]] = {
            "policy_validate": self._policy_validate,
            "measurement_policy": self._policy_validate,
            "pipeline_artifact_audit": self._pipeline_artifact_audit,
            "covariant_lift": self._covariant_lift,
            "symbolic_proxy": self._symbolic_proxy,
            "constraint_analysis": self._constraint_analysis,
            "formal_reference_controls": self._formal_reference_controls,
            "reference_control": self._reference_control,
            "failure_cluster": self._failure_cluster,
            "llm_research": self._llm_research,
            "candidate_dossier": self._candidate_dossier,
            "proposal_compile": self._proposal_compile,
            "generator_export": self._generator_export,
            "gpu_dense_screen": self._gpu_dense_screen,
            "knowledge_build": self._knowledge_build,
        }

    def run(
        self,
        *,
        max_tasks: int | None = None,
        duration_seconds: float | None = None,
        wait_for_work: bool = False,
        poll_seconds: float = 5.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        processed = 0
        retries = 0
        self.store.recover_expired_leases(self.campaign_id)
        self.store.requeue_cancelled_dossiers(self.campaign_id)
        self.store.reconcile_candidate_states(self.campaign_id)
        while max_tasks is None or processed < max_tasks:
            if duration_seconds is not None and time.monotonic() - started >= duration_seconds:
                break
            task = self.store.claim_task(
                self.campaign_id,
                self.worker_id,
                self.lease_seconds,
                self.allowed_task_types,
            )
            if task is None:
                campaign_state = self.store.campaign(self.campaign_id)["state"]
                if campaign_state == "active":
                    self._schedule_next_cycle()
                if not wait_for_work or campaign_state != "active":
                    break
                time.sleep(min(poll_seconds, max(0.05, duration_seconds or poll_seconds)))
                continue
            stop_heartbeat = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop, args=(task.task_id, stop_heartbeat), daemon=True
            )
            heartbeat.start()
            try:
                handler = self.handlers.get(task.task_type)
                if handler is None:
                    raise ValueError(f"No worker registered for task type {task.task_type}")
                outcome = handler(task)
                for evidence in outcome.evidence:
                    if task.candidate_id:
                        self.store.record_evidence(
                            self.campaign_id, task.candidate_id, task.task_id, evidence
                        )
                for artifact in outcome.artifacts:
                    self.store.register_artifact(
                        self.campaign_id,
                        artifact["path"],
                        kind=artifact["kind"],
                        task_id=task.task_id,
                        candidate_id=task.candidate_id,
                        metadata=artifact.get("metadata"),
                        known_sha256=artifact.get("sha256"),
                    )
                self.store.finish_task(
                    task,
                    self.worker_id,
                    outcome.status,
                    outcome.result,
                    outcome.error,
                )
            except Exception as error:  # noqa: BLE001 - worker errors become durable evidence
                state = self.store.retry_or_fail_task(
                    task,
                    self.worker_id,
                    f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
                )
                retries += int(state == "queued")
            finally:
                stop_heartbeat.set()
                heartbeat.join(timeout=1.0)
            processed += 1
        return {
            "campaign_id": self.campaign_id,
            "worker_id": self.worker_id,
            "allowed_task_types": (
                None if self.allowed_task_types is None else sorted(self.allowed_task_types)
            ),
            "processed_tasks": processed,
            "retries_scheduled": retries,
            "elapsed_seconds": time.monotonic() - started,
            "status": self.store.status(self.campaign_id),
        }

    def _schedule_next_cycle(self) -> bool:
        campaign = self.store.campaign(self.campaign_id)
        cycle = int(campaign["cycles_completed"]) + 1
        if cycle > int(campaign["max_cycles"]):
            return False
        with self.store.connect() as connection:
            existing = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE campaign_id=? AND status IN ('queued','running')",
                (self.campaign_id,),
            ).fetchone()[0]
            if existing:
                return False
            connection.execute(
                "UPDATE campaigns SET cycles_completed=?,updated_utc=? WHERE campaign_id=?",
                (cycle, utc_now(), self.campaign_id),
            )
        interval = int(self.config.get("runtime", {}).get("research_cycle_seconds", 3600))
        not_before = datetime.now(UTC).timestamp() + interval
        not_before_utc = datetime.fromtimestamp(not_before, UTC).isoformat()
        cluster = self.store.add_task(
            self.campaign_id,
            "failure_cluster",
            stage=90,
            payload={"campaign_id": self.campaign_id, "cycle": cycle},
            priority=-100.0,
            diversity_bucket=f"research-cycle-{cycle}",
            not_before_utc=not_before_utc,
            idempotency_key=f"{self.campaign_id}:failure-cluster:{cycle}",
        )
        self.store.add_task(
            self.campaign_id,
            "llm_research",
            stage=91,
            payload={"campaign_id": self.campaign_id, "cycle": cycle},
            priority=-110.0,
            diversity_bucket=f"research-cycle-{cycle}",
            depends_on=[cluster],
            not_before_utc=not_before_utc,
            idempotency_key=f"{self.campaign_id}:llm-research:{cycle}",
        )
        return True

    def _heartbeat_loop(self, task_id: str, stop: threading.Event) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not stop.wait(interval):
            if not self.store.heartbeat(task_id, self.worker_id, self.lease_seconds):
                return

    def _policy_validate(self, task: ClaimedTask) -> WorkerOutcome:
        candidate = self.store.candidate(task.candidate_id or task.payload["candidate_id"])
        text = f"{candidate['expression']} {candidate['canonical_json']}".casefold()
        forbidden = [
            pattern.casefold()
            for pattern in self.config["scientific_contract"]["prohibited_evidence_patterns"]
        ]
        matches = sorted(pattern for pattern in forbidden if pattern in text)
        outcome = "reject" if matches else "pass"
        return WorkerOutcome(
            result={"matches": matches},
            evidence=[
                _gate(
                    "observational_evidence_policy",
                    outcome,
                    0,
                    payload={"prohibited_matches": matches},
                    evidence_class="policy",
                )
            ],
        )

    def _pipeline_artifact_audit(self, task: ClaimedTask) -> WorkerOutcome:
        checks = []
        artifacts = []
        required = {
            "survivor_audit": ("all_checks_pass", True),
            "dense_gpu_report": ("accounting_pass", True),
            "dense_crosscheck": ("all_cpu_gpu_samples_agree", True),
            "priority_queue": ("schema_version", "sigma-generated-priority-1.0"),
        }
        for name, (check_field, expected) in required.items():
            value = task.payload.get(name)
            if not value:
                checks.append({"artifact": name, "pass": False, "reason": "missing path"})
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.project_root / path
            payload = _read_json(path)
            passed = payload.get(check_field) == expected
            checks.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "field": check_field,
                    "expected": expected,
                    "pass": passed,
                }
            )
            artifacts.append({"path": path, "kind": name})
        passed = all(check["pass"] for check in checks)
        return WorkerOutcome(
            status="succeeded" if passed else "failed",
            result={"checks": checks, "all_pass": passed},
            artifacts=artifacts,
            error=None if passed else "one or more required pipeline artifacts failed audit",
        )

    def _covariant_lift(self, task: ClaimedTask) -> WorkerOutcome:
        parent = self.store.candidate(task.candidate_id or "")
        parent_canonical = json.loads(parent["canonical_json"])
        expression = parent["expression"]
        # The legacy templates below are retained only for reproducibility of pre-contract runs.
        # New campaigns fail closed: naming a tensor expression is not a derivation of the static
        # dictionary, and the old baryonic z is not an admissible gravitational-action atom.
        if self.config.get("formal", {}).get("fail_closed", True):
            classification = classify_generator_expression(
                expression, aether_x_available=True
            )
            if classification["decision"] == "reject_forbidden_baryonic_action_atom":
                return WorkerOutcome(
                    result={
                        "decision": "reject_legacy_covariant_lift",
                        "reason": "z_b is diagnostic-only under universal minimal matter coupling",
                        "field_contract": "sigma-covariant-field-contract-1.0",
                        "static_lift_classification": classification,
                        "child_candidates": [],
                    },
                    evidence=[
                        _gate(
                            "universal_minimal_matter_coupling",
                            "reject",
                            2,
                            payload={
                                "invariant": "z_b",
                                "definition": "-g_mu_nu J_b^mu J_b^nu/n_0^2",
                                "reason": "baryon number selects a matter species and is forbidden as an S_grav atom",
                            },
                        )
                    ],
                )
            return WorkerOutcome(
                status="deferred",
                result={
                    "decision": "awaiting_derived_static_covariant_dictionary",
                    "reason": classification["reason"],
                    "static_lift_classification": classification,
                    "child_candidates": [],
                },
                evidence=[
                    _gate(
                        "static_covariant_dictionary",
                        "unresolved",
                        2,
                        hard=True,
                        payload={
                            "required": (
                                "implement the missing covariant adapter named by the static-lift "
                                "classification and derive it from candidate field equations"
                            ),
                            "classification": classification,
                            "legacy_templates_executed": False,
                        },
                    )
                ],
                error="covariant lift is fail-closed until the static dictionary is derived",
            )
        templates = [
            {
                "template_id": "scalar_second_derivative_v1",
                "fields": ["g_mu_nu", "phi"],
                "symmetries": ["diffeomorphism"],
                "matter_metric": "g_mu_nu",
                "invariants": {
                    "x": "-nabla_mu(phi)nabla^mu(phi)/a_sigma^2",
                    "q": "L_sigma^2 nabla_mu_nabla_nu(phi)nabla^mu_nabla^nu(phi)/a_sigma^2",
                    "z": "declared baryonic scalar Z_b^2/Z_0^2",
                },
                "derivative_order_in_action": 2,
                "degeneracy_conditions": [],
                "expected_dof": "unknown until constraint analysis",
                "risk_flags": ["higher_derivative_scalar", "matter_invariant_definition_required"],
            },
            {
                "template_id": "constrained_vector_flux_v1",
                "fields": ["g_mu_nu", "A_mu", "lambda"],
                "symmetries": ["diffeomorphism"],
                "matter_metric": "g_mu_nu",
                "invariants": {
                    "x": "-A_mu A^mu/a_sigma^2",
                    "q": "L_sigma^2 nabla_mu(A_nu)nabla^mu(A^nu)/a_sigma^2",
                    "z": "declared baryonic scalar Z_b^2/Z_0^2",
                },
                "derivative_order_in_action": 1,
                "constraints": ["A_mu A^mu + a_sigma^2 = 0 enforced by lambda"],
                "expected_dof": "unknown until ADM/Dirac analysis",
                "risk_flags": [
                    "constraint_algebra_required",
                    "matter_invariant_definition_required",
                ],
            },
        ]
        children = []
        for template in templates:
            canonical = {
                "proposal_class": "covariant_lift_hypothesis",
                "parent": parent_canonical,
                "correction_function": expression,
                "template": template,
                "universal_constants": ["M_Pl", "a_sigma", "L_sigma", "Z_0", "epsilon"],
                "static_limit_status": "claimed_not_yet_derived",
            }
            child_id = self.store.add_candidate(
                self.campaign_id,
                kind="covariant_lift",
                expression=f"{template['template_id']}: F={expression}",
                canonical=canonical,
                family_id=parent["family_id"],
                parent_candidate_id=parent["candidate_id"],
                generation=parent["generation"] + 1,
                pareto_front=parent["pareto_front"],
                mechanism_tags=json.loads(parent["mechanism_tags_json"]),
            )
            policy = self.store.add_task(
                self.campaign_id,
                "policy_validate",
                stage=0,
                payload={"candidate_id": child_id},
                candidate_id=child_id,
                priority=85.0,
                diversity_bucket=parent["family_id"] or "unclassified",
            )
            symbolic = self.store.add_task(
                self.campaign_id,
                "symbolic_proxy",
                stage=2,
                payload={"candidate_id": child_id},
                candidate_id=child_id,
                priority=80.0,
                diversity_bucket=parent["family_id"] or "unclassified",
                depends_on=[policy],
            )
            self.store.add_task(
                self.campaign_id,
                "constraint_analysis",
                stage=3,
                payload={"candidate_id": child_id},
                candidate_id=child_id,
                priority=70.0,
                diversity_bucket=parent["family_id"] or "unclassified",
                depends_on=[symbolic],
            )
            self.store.add_task(
                self.campaign_id,
                "candidate_dossier",
                stage=80,
                payload={"candidate_id": child_id},
                candidate_id=child_id,
                priority=5.0,
                diversity_bucket=parent["family_id"] or "unclassified",
                depends_on=[symbolic],
            )
            children.append(child_id)
        return WorkerOutcome(
            result={"parent_candidate_id": parent["candidate_id"], "child_candidates": children},
            evidence=[
                _gate(
                    "covariant_lift_hypotheses_generated",
                    "pass",
                    2,
                    hard=False,
                    payload={"template_count": len(children)},
                )
            ],
        )

    def _symbolic_proxy(self, task: ClaimedTask) -> WorkerOutcome:
        candidate = self.store.candidate(task.candidate_id or "")
        canonical = json.loads(candidate["canonical_json"])
        expression = canonical["correction_function"].replace("^", "**")
        x, q, z = sp.symbols("x q z", nonnegative=True, finite=True)
        parsed = sp.sympify(expression, locals={"x": x, "q": q, "z": z, "sqrt": sp.sqrt})
        derivatives = {
            "dF_dx": str(sp.diff(parsed, x)),
            "dF_dq": str(sp.diff(parsed, q)),
            "dF_dz": str(sp.diff(parsed, z)),
            "d2F_dq2": str(sp.diff(parsed, q, 2)),
            "mixed_xq": str(sp.diff(parsed, x, q)),
            "mixed_xz": str(sp.diff(parsed, x, z)),
            "mixed_qz": str(sp.diff(parsed, q, z)),
        }
        prior_art = self._equation_prior_art(candidate["candidate_id"], expression)
        report_path = _write_json(
            self.output_root / "symbolic" / f"{candidate['candidate_id']}.json",
            {
                "scope": "reduced invariant proxy; not covariant tensor variation",
                "candidate_id": candidate["candidate_id"],
                "expression": str(parsed),
                "derivatives": derivatives,
                "equation_prior_art": prior_art,
            },
        )
        return WorkerOutcome(
            result={"parsed_expression": str(parsed), "derivatives": derivatives},
            evidence=[
                _gate("reduced_symbolic_compilation", "pass", 2, payload=derivatives),
                _gate(
                    "covariant_variation",
                    "unresolved",
                    2,
                    hard=False,
                    payload={"reason": "tensor variation backend not yet completed"},
                ),
            ],
            artifacts=[{"path": report_path, "kind": "symbolic_proxy_report"}],
        )

    def _equation_prior_art(
        self,
        candidate_id: str,
        expression: str,
        *,
        representation: str = "scalar_sympy",
    ) -> dict[str, Any]:
        """Screen a reduced correction or exact action record without claiming completeness."""
        scalar = representation == "scalar_sympy"
        base = {
            "candidate_id": candidate_id,
            "database": str(self.prior_art_database),
            "query_relation": f"F = {expression}" if scalar else expression,
            "representation": representation,
            "scope": (
                "reduced dimensionless correction function; not a covariant-action proof"
                if scalar
                else "exact normalized tensor/action text; structural tensor equivalence pending"
            ),
            "novelty_claim_allowed": False,
        }
        if not self.prior_art_database.is_file():
            return {
                **base,
                "status": "unavailable",
                "classification": "not_screened",
                "novelty_warning": "the equation-universe database is unavailable",
            }
        record = {
            "name": f"campaign correction function {candidate_id}",
            "domain": "sigma_reduced_candidate" if scalar else "covariant_action_candidate",
            "representation": representation,
            "expression": f"F = {expression}" if scalar else expression,
            "variables": [] if not scalar else [
                {
                    "symbol": symbol,
                    "canonical_name": symbol,
                    "meaning": meaning,
                    "dimension": {},
                    "field_kind": "scalar",
                    "tensor_rank": 0,
                }
                for symbol, meaning in (
                    ("F", "dimensionless correction function"),
                    ("x", "dimensionless weak-field invariant"),
                    ("q", "dimensionless gradient-state invariant"),
                    ("z", "dimensionless baryonic diagnostic invariant"),
                )
            ],
        }
        if scalar:
            record["formula_space_expression"] = expression
        try:
            result = EquationUniverse(self.prior_art_database).classify(
                record, nearest_limit=self.prior_art_nearest_limit
            )
        except (OSError, TypeError, ValueError, sqlite3.DatabaseError, sp.SympifyError) as error:
            return {
                **base,
                "status": "error",
                "classification": "not_screened",
                "error": f"{type(error).__name__}: {error}",
                "novelty_warning": "a failed screen cannot support a novelty claim",
            }
        return {**base, "status": "screened", **result}

    @staticmethod
    def _prior_art_gate(prior_art: dict[str, Any]) -> dict[str, Any]:
        screened = prior_art.get("status") == "screened"
        payload = {
            "status": prior_art.get("status"),
            "classification": prior_art.get("classification"),
            "database": prior_art.get("database"),
            "semantic_matches": prior_art.get("semantic_matches", []),
            "structural_matches": prior_art.get("structural_matches", []),
            "generator_history_matches": prior_art.get("generator_history_matches", []),
            "novelty_claim_allowed": False,
            "warning": prior_art.get("novelty_warning"),
        }
        if prior_art.get("error"):
            payload["error"] = prior_art["error"]
        return _gate(
            "equation_prior_art_screen",
            "pass" if screened else "unresolved",
            1,
            hard=False,
            payload=payload,
            version="1.0",
            evidence_class="provenance",
        )

    def _constraint_analysis(self, task: ClaimedTask) -> WorkerOutcome:
        candidate = self.store.candidate(task.candidate_id or "")
        canonical = json.loads(candidate["canonical_json"])
        template = canonical.get("template", canonical)
        template_id = template.get("template_id", "bounded_research_proposal")
        derivative_order = template.get(
            "derivative_order_in_action", template.get("derivative_order", 99)
        )
        if derivative_order > 1 and not template.get("degeneracy_conditions"):
            evidence = _gate(
                "higher_derivative_degeneracy_declaration",
                "reject",
                3,
                payload={
                    "reason": "higher-derivative action has no declared degeneracy relation",
                    "template_id": template_id,
                    "scope": "grammar-level rejection; not a full Hamiltonian proof",
                },
            )
            return WorkerOutcome(result={"decision": "reject_lift"}, evidence=[evidence])
        evidence = [
            _gate(
                "derivative_order_bound",
                "pass",
                3,
                payload={"derivative_order_in_action": derivative_order},
            ),
            _gate(
                "constraint_algebra",
                "unresolved",
                3,
                hard=True,
                payload={"required_backend": "ADM/Dirac symbolic constraint worker"},
            ),
        ]
        return WorkerOutcome(
            status="deferred",
            result={"decision": "awaiting_adm_dirac_backend"},
            evidence=evidence,
            error="constraint algebra remains unresolved; candidate is not promoted",
        )

    def _proposal_compile(self, task: ClaimedTask) -> WorkerOutcome:
        candidate = self.store.candidate(task.candidate_id or "")
        proposal = json.loads(candidate["canonical_json"])
        validation = validate_proposal(proposal, self.config["scientific_contract"])
        if not validation["valid"]:
            return WorkerOutcome(
                result=validation,
                evidence=[_gate("bounded_proposal_schema", "reject", 0, payload=validation)],
            )
        report = {
            "candidate_id": candidate["candidate_id"],
            "field_count": len(proposal["fields"]),
            "derivative_order": proposal["derivative_order"],
            "matter_metric": proposal["matter_metric"],
            "status": "schema_compiled_tensor_variation_pending",
        }
        return WorkerOutcome(
            result=report,
            evidence=[
                _gate("bounded_proposal_schema", "pass", 0, payload=report),
                _gate(
                    "covariant_variation",
                    "unresolved",
                    2,
                    hard=False,
                    payload={"reason": "formal tensor variation backend required"},
                ),
            ],
        )

    def _reference_control(self, task: ClaimedTask) -> WorkerOutcome:
        eligibility = audit_theory_observation_eligibility(
            self.output_root
            / "controls"
            / "action-health"
            / "einstein_hilbert_control"
            / "action-health.json",
            self.project_root / "configs" / "observational_evidence_policy.json",
            mode="known_answer_reference",
        )
        eligibility_path = write_observation_eligibility(
            eligibility, self.output_root / "controls" / "gr-reference-eligibility.json"
        )
        report = run_relativity_reference_suite(eligibility)
        output = _write_json(self.output_root / "controls" / "gr-reference.json", report)
        passed = (
            eligibility["status"] == "eligible"
            and report["counts"]["failed"] == 0
            and report["counts"]["blocked"] == 0
        )
        return WorkerOutcome(
            status="succeeded" if passed else "failed",
            result={"counts": report["counts"], "report": str(output)},
            evidence=[
                _gate("gr_limit", "pass" if passed else "reject", 5, payload=report["counts"]),
                _gate(
                    "solar_system_controls",
                    "pass" if passed else "reject",
                    5,
                    payload={"golden_checks": report["counts"]["golden_total"]},
                ),
            ],
            artifacts=[
                {"path": eligibility_path, "kind": "gr_solar_formal_eligibility"},
                {"path": output, "kind": "gr_solar_reference_report"},
            ],
            error=None if passed else "GR/Solar golden control failed",
        )

    def _formal_reference_controls(self, task: ClaimedTask) -> WorkerOutcome:
        formal_config = self.config.get("formal", {})
        contract_path = Path(
            formal_config.get("field_contract", "configs/covariant_field_contract.json")
        )
        if not contract_path.is_absolute():
            contract_path = self.project_root / contract_path
        report = run_formal_control_suite(contract_path, self.project_root)
        json_path, markdown_path = write_formal_report(
            report, self.output_root / "controls" / "formal"
        )
        grammar_path = self.project_root / "configs" / "covariant_action_grammar.json"
        health_expectations = {
            "einstein_hilbert_control.json": "pass",
            "canonical_scalar_control.json": "pass",
            "proca_control.json": "pass",
            "einstein_aether_control.json": "unresolved",
        }
        health_results: dict[str, dict[str, Any]] = {}
        health_artifacts: list[dict[str, Any]] = []
        for filename, expected in health_expectations.items():
            key = filename.removesuffix(".json")
            result = analyze_action_health(
                self.project_root / "configs" / "actions" / filename,
                grammar_path,
                contract_path,
                self.output_root / "controls" / "action-health" / key,
                project_root=self.project_root,
            )
            health_results[key] = {
                "status": result["status"],
                "expected_status": expected,
                "matches_expected": result["status"] == expected,
                "family": result.get("family"),
                "physical_dof": result.get("physical_dof"),
                "promotion_allowed": result["promotion_allowed"],
                "discovery_blockers": result.get("discovery_blockers", []),
            }
            health_artifacts.append(
                {"path": Path(result["report_path"]), "kind": "action_health_control_report"}
            )
        formal_passed = report["counts"]["failed"] == 0
        health_passed = all(item["matches_expected"] for item in health_results.values())
        passed = formal_passed and health_passed
        return WorkerOutcome(
            status="succeeded" if passed else "failed",
            result={
                "counts": report["counts"],
                "cadabra": report["backends"]["cadabra2"],
                "candidate_readiness": report["candidate_readiness"],
                "action_health_controls": health_results,
            },
            evidence=[
                _gate(
                    "formal_known_answer_controls",
                    "pass" if passed else "reject",
                    2,
                    hard=True,
                    payload={
                        "counts": report["counts"],
                        "scope": "reference controls only; not a generated-candidate health claim",
                    },
                ),
                _gate(
                    "linearized_covariance_identity",
                    "pass" if passed else "reject",
                    3,
                    hard=False,
                    payload={
                        "identity": "k^mu G^(1)_mu_nu = 0",
                        "background": "Minkowski",
                        "nonlinear_identity_status": "pass",
                    },
                ),
                _gate(
                    "nonlinear_covariance_identity",
                    "pass" if formal_passed else "reject",
                    3,
                    hard=True,
                    payload={
                        "identity": "nabla^mu G_mu_nu = 0",
                        "scope": "exact nonlinear contracted Bianchi known-answer control",
                    },
                ),
                _gate(
                    "action_health_known_answer_controls",
                    "pass" if health_passed else "reject",
                    4,
                    hard=True,
                    payload=health_results,
                ),
            ],
            artifacts=[
                {"path": json_path, "kind": "formal_control_report"},
                {"path": markdown_path, "kind": "formal_control_summary"},
                *health_artifacts,
            ],
            error=None if passed else "one or more formal known-answer controls failed",
        )

    def _failure_cluster(self, task: ClaimedTask) -> WorkerOutcome:
        count = self.store.build_failure_clusters(self.campaign_id)
        return WorkerOutcome(result={"failure_cluster_count": count})

    def _llm_research(self, task: ClaimedTask) -> WorkerOutcome:
        with self.store.connect() as connection:
            clusters = [
                dict(row)
                for row in connection.execute(
                    "SELECT gate_id,mechanism_tag,rejection_count,summary FROM failure_clusters "
                    "WHERE campaign_id=? ORDER BY rejection_count DESC,gate_id",
                    (self.campaign_id,),
                ).fetchall()
            ]
        prompt = {
            "schema_version": "sigma-llm-research-packet-1.0",
            "campaign_id": self.campaign_id,
            "role": "propose bounded falsifiable theory grammar additions; do not judge truth",
            "scientific_contract": self.config["scientific_contract"],
            "failure_clusters": clusters,
            "required_output_schema": proposal_schema(),
        }
        packet = _write_json(self.output_root / "llm-outbox" / f"{task.task_id}.json", prompt)
        command = self.config.get("llm", {}).get("command") or []
        if not command:
            return WorkerOutcome(
                status="deferred",
                result={"mode": "offline", "proposal_packet": str(packet)},
                artifacts=[{"path": packet, "kind": "llm_research_packet"}],
                error="No trusted LLM command adapter configured; packet retained for submission",
            )
        llm_config = self.config.get("llm", {})
        per_call = float(llm_config.get("per_call_budget_usd", 0))
        if per_call <= 0:
            raise RuntimeError("LLM command configured without a positive per_call_budget_usd")
        self._validate_llm_command_cap(command, per_call)
        if self.store.llm_budget_status(self.campaign_id) is None:
            self.store.configure_llm_budget(
                self.campaign_id,
                total_budget_usd=float(llm_config.get("total_budget_usd", 0)),
                max_calls=int(llm_config.get("max_calls", 0)),
            )
        call_id = self.store.reserve_llm_call(
            self.campaign_id,
            task_id=task.task_id,
            provider=str(llm_config.get("provider", "command-adapter")),
            model=str(llm_config.get("model", "unspecified")),
            max_cost_usd=per_call,
        )
        completed = None
        try:
            completed = subprocess.run(
                command,
                input=canonical_json(prompt),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(llm_config.get("timeout_seconds", 900)),
                check=False,
                shell=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"LLM adapter failed: {completed.stderr[-2000:]}")
            adapter_result = json.loads(completed.stdout)
            if adapter_result.get("schema_version") == "sigma-llm-adapter-result-1.0":
                proposal = adapter_result["proposal"]
                cost = adapter_result.get("total_cost_usd")
                metadata = {key: value for key, value in adapter_result.items() if key != "proposal"}
            else:
                proposal = adapter_result
                cost = None
                metadata = {"legacy_unmetered_adapter": True}
            self.store.settle_llm_call(
                call_id, actual_cost_usd=cost, status="succeeded", metadata=metadata
            )
        except Exception:
            self.store.settle_llm_call(
                call_id,
                actual_cost_usd=None,
                status="failed_unmetered",
                metadata={
                    "returncode": completed.returncode if completed is not None else None,
                    "reason": "full reservation charged because reliable provider cost was unavailable",
                },
            )
            raise
        proposal_id, validation = self.submit_proposal(proposal, source_task_id=task.task_id)
        return WorkerOutcome(
            result={
                "mode": "command",
                "proposal_id": proposal_id,
                "validation": validation,
                "llm_call_id": call_id,
                "llm_budget": self.store.llm_budget_status(self.campaign_id),
            },
            artifacts=[{"path": packet, "kind": "llm_research_packet"}],
        )

    @staticmethod
    def _validate_llm_command_cap(command: list[str], configured_cap: float) -> None:
        try:
            index = command.index("--max-budget-usd")
            command_cap = float(command[index + 1])
        except (ValueError, IndexError, TypeError) as error:
            raise RuntimeError("LLM command must include --max-budget-usd VALUE") from error
        if command_cap <= 0 or command_cap > configured_cap + 1e-12:
            raise RuntimeError("LLM command dollar cap exceeds per_call_budget_usd")

    def _candidate_dossier(self, task: ClaimedTask) -> WorkerOutcome:
        candidate = self.store.candidate(task.candidate_id or "")
        gates = self.store.gate_summary(candidate["candidate_id"])
        tasks = self.store.candidate_tasks(candidate["candidate_id"])
        unresolved = self.store.unresolved_claims(candidate["candidate_id"])
        parent = (
            dict(self.store.candidate(candidate["parent_candidate_id"]))
            if candidate["parent_candidate_id"]
            else None
        )
        with self.store.connect() as connection:
            siblings = [
                {
                    "candidate_id": row["candidate_id"],
                    "kind": row["kind"],
                    "expression": row["expression"],
                    "status": row["status"],
                    "hard_gate_status": row["hard_gate_status"],
                }
                for row in connection.execute(
                    "SELECT candidate_id,kind,expression,status,hard_gate_status FROM candidates "
                    "WHERE campaign_id=? AND family_id=? AND candidate_id!=? ORDER BY generation,candidate_id",
                    (self.campaign_id, candidate["family_id"], candidate["candidate_id"]),
                ).fetchall()
            ]
            provenance = [
                dict(row)
                for row in connection.execute(
                    "SELECT artifact_id,task_id,kind,path,sha256,size_bytes FROM artifacts "
                    "WHERE campaign_id=? AND (candidate_id=? OR candidate_id IS NULL) ORDER BY created_utc",
                    (self.campaign_id, candidate["candidate_id"]),
                ).fetchall()
            ]
        canonical = json.loads(candidate["canonical_json"])
        correction = canonical.get("correction_function", candidate["expression"])
        prior_art = self._equation_prior_art(
            candidate["candidate_id"],
            correction,
            representation=(
                "scalar_sympy"
                if "correction_function" in canonical
                or candidate["kind"] in {"generated_static", "covariant_lift"}
                else "tensor_dsl"
            ),
        )
        ablations = structural_ablations(correction)
        failures = [gate for gate in gates if gate["outcome"] in {"reject", "unresolved"}]
        dossier = {
            "schema_version": "sigma-candidate-dossier-1.0",
            "candidate": dict(candidate),
            "canonical": canonical,
            "parent": parent,
            "family_comparisons": siblings,
            "gate_evidence": gates,
            "gate_failures_or_unresolved": failures,
            "structural_ablations": ablations,
            "equation_prior_art": prior_art,
            "provenance_artifacts": provenance,
            "tasks": tasks,
            "remaining_claims": unresolved,
            "why_prioritized": {
                "inherited_pareto_front": candidate["pareto_front"],
                "completed_hard_gate_passes": sum(
                    gate["is_hard"] and gate["outcome"] == "pass" for gate in gates
                ),
                "terminal_hard_gate_rejections": sum(
                    gate["is_hard"] and gate["outcome"] == "reject" for gate in gates
                ),
                "mechanism_tags": json.loads(candidate["mechanism_tags_json"]),
                "semantics": "work priority, not truth probability",
            },
            "interpretation": (
                "Work-priority evidence only. Completed passes do not imply truth, and any hard "
                "rejection is terminal for this exact candidate."
            ),
        }
        json_path = _write_json(
            self.output_root / "dossiers" / f"{candidate['candidate_id']}.json", dossier
        )
        markdown_path = json_path.with_suffix(".md")
        lines = [
            f"# Candidate {candidate['candidate_id']}",
            "",
            f"Expression: `{candidate['expression']}`",
            "",
            f"Status: **{candidate['status']}**; hard-gate state: **{candidate['hard_gate_status']}**.",
            "",
            "## Completed evidence",
            "",
        ]
        lines.extend(
            f"- `{gate['gate_id']}`: **{gate['outcome']}** (stage {gate['stage']})"
            for gate in gates
        )
        lines.extend(["", "## Remaining claims", ""])
        lines.extend(f"- `{claim}`" for claim in unresolved)
        lines.extend(["", "## Structural ablations", ""])
        lines.extend(
            f"- Remove `{item['removed_term']}` -> `{item['ablated_expression']}`"
            for item in ablations
        )
        lines.extend(
            [
                "",
                "## Equation-universe prior-art screen",
                "",
                f"- Status: `{prior_art['status']}`",
                f"- Classification: `{prior_art['classification']}`",
                "- Novelty claim allowed: `false`",
                "- An unmatched result means only that this finite corpus has no match.",
            ]
        )
        lines.extend(["", "## Family comparisons", ""])
        lines.extend(
            f"- `{item['candidate_id']}`: {item['status']} / {item['hard_gate_status']}"
            for item in siblings
        )
        lines.extend(
            [
                "",
                "This dossier explains work priority only. It is not a probability of truth.",
                "",
            ]
        )
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return WorkerOutcome(
            result={"json": str(json_path), "markdown": str(markdown_path)},
            evidence=[self._prior_art_gate(prior_art)],
            artifacts=[
                {"path": json_path, "kind": "candidate_dossier_json"},
                {"path": markdown_path, "kind": "candidate_dossier_markdown"},
            ],
        )

    def queue_dossier_refresh(self, version: str = "2") -> int:
        with self.store.connect() as connection:
            candidates = connection.execute(
                "SELECT candidate_id,family_id FROM candidates WHERE campaign_id=?",
                (self.campaign_id,),
            ).fetchall()
        for candidate in candidates:
            self.store.add_task(
                self.campaign_id,
                "candidate_dossier",
                stage=80,
                payload={"candidate_id": candidate["candidate_id"], "dossier_version": version},
                candidate_id=candidate["candidate_id"],
                priority=5.0,
                diversity_bucket=candidate["family_id"] or "unclassified",
                idempotency_key=f"{candidate['candidate_id']}:dossier:{version}",
            )
        return len(candidates)

    def queue_formal_reference_controls(self, version: str = "1") -> int:
        with self.store.connect() as connection:
            controls = connection.execute(
                "SELECT candidate_id FROM candidates WHERE campaign_id=? AND kind='gr_control'",
                (self.campaign_id,),
            ).fetchall()
        for control in controls:
            self.store.add_task(
                self.campaign_id,
                "formal_reference_controls",
                stage=2,
                payload={"candidate_id": control["candidate_id"], "control_version": version},
                candidate_id=control["candidate_id"],
                priority=115.0,
                diversity_bucket="control",
                idempotency_key=f"{control['candidate_id']}:formal-reference-controls:{version}",
            )
        return len(controls)

    def enforce_covariant_field_contract(self, version: str = "1.0") -> dict[str, int]:
        """Reclassify legacy exact candidates; never rewrite their expressions in place."""

        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT candidate_id,kind,expression,canonical_json FROM candidates "
                "WHERE campaign_id=? AND kind IN ('generated_static','covariant_lift')",
                (self.campaign_id,),
            ).fetchall()
        audited = 0
        rejected = 0
        retained = 0
        z_symbol = sp.Symbol("z")
        for row in rows:
            canonical = json.loads(row["canonical_json"])
            expression = (
                canonical.get("correction_function", row["expression"])
                if row["kind"] == "covariant_lift"
                else row["expression"]
            )
            try:
                parsed = sp.sympify(
                    expression.replace("^", "**"),
                    locals={"x": sp.Symbol("x"), "q": sp.Symbol("q"), "z": z_symbol},
                )
                uses_z = z_symbol in parsed.free_symbols
            except (sp.SympifyError, TypeError):
                uses_z = "z" in expression.casefold()
            audited += 1
            if uses_z:
                self.store.record_evidence(
                    self.campaign_id,
                    row["candidate_id"],
                    None,
                    _gate(
                        "universal_minimal_matter_coupling",
                        "reject",
                        2,
                        version=f"field-contract-{version}",
                        payload={
                            "invariant": "z_b",
                            "definition": "-g_mu_nu J_b^mu J_b^nu/n_0^2",
                            "migration": "legacy exact candidate reclassified; expression preserved",
                        },
                    ),
                )
                rejected += 1
            else:
                retained += 1
        return {"audited": audited, "rejected": rejected, "retained": retained}

    def _generator_export(self, task: ClaimedTask) -> WorkerOutcome:
        executable = Path(task.payload["executable"])
        args = [str(executable), "run"] + [str(value) for value in task.payload["arguments"]]
        completed = subprocess.run(args, capture_output=True, text=True, check=False, shell=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-4000:])
        manifest = Path(task.payload["manifest"])
        return WorkerOutcome(
            result={"stdout": completed.stdout, "manifest": str(manifest)},
            artifacts=[{"path": manifest, "kind": "generator_manifest"}],
        )

    def _gpu_dense_screen(self, task: ClaimedTask) -> WorkerOutcome:
        report = run_dense_gpu_screen(**task.payload)
        return WorkerOutcome(
            result={"counts": report["counts"], "accounting": report["accounting_pass"]}
        )

    def _knowledge_build(self, task: ClaimedTask) -> WorkerOutcome:
        ontology = GateOntology.from_path(task.payload["ontology"])
        summary = KnowledgeBuilder(task.payload["repo"], ontology).build(
            task.payload["database"], task.payload["summary"]
        )
        return WorkerOutcome(
            status="succeeded" if summary["integrity_check"] == "ok" else "failed",
            result=summary,
            artifacts=[
                {"path": task.payload["database"], "kind": "knowledge_database"},
                {"path": task.payload["summary"], "kind": "knowledge_summary"},
            ],
        )

    def submit_proposal(
        self,
        proposal: dict[str, Any],
        *,
        source_task_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        validation = validate_proposal(proposal, self.config["scientific_contract"])
        proposal_id = stable_id("PROP", self.campaign_id, canonical_json(proposal))
        parent_id = proposal.get("parent_candidate_id")
        with self.store.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO proposals VALUES (?,?,?,?,?,?,?,?)",
                (
                    proposal_id,
                    self.campaign_id,
                    source_task_id,
                    parent_id,
                    "accepted" if validation["valid"] else "rejected",
                    canonical_json(proposal),
                    canonical_json(validation),
                    utc_now(),
                ),
            )
        if validation["valid"]:
            generation = 1
            family_id = proposal.get("family_id", "LLM-BOUNDED-PROPOSAL")
            tags = proposal.get("mechanism_tags", [])
            candidate_id = self.store.add_candidate(
                self.campaign_id,
                kind="bounded_research_proposal",
                expression=proposal["action"],
                canonical=proposal,
                family_id=family_id,
                parent_candidate_id=parent_id,
                generation=generation,
                mechanism_tags=tags,
            )
            policy = self.store.add_task(
                self.campaign_id,
                "policy_validate",
                stage=0,
                payload={"candidate_id": candidate_id},
                candidate_id=candidate_id,
                priority=75.0,
                diversity_bucket=family_id,
            )
            compiled = self.store.add_task(
                self.campaign_id,
                "proposal_compile",
                stage=2,
                payload={"candidate_id": candidate_id},
                candidate_id=candidate_id,
                priority=70.0,
                diversity_bucket=family_id,
                depends_on=[policy],
            )
            self.store.add_task(
                self.campaign_id,
                "constraint_analysis",
                stage=3,
                payload={"candidate_id": candidate_id},
                candidate_id=candidate_id,
                priority=65.0,
                diversity_bucket=family_id,
                depends_on=[compiled],
            )
        return proposal_id, validation


def proposal_schema() -> dict[str, Any]:
    return {
        "required": [
            "proposal_type",
            "action",
            "fields",
            "symmetries",
            "universal_constants",
            "derivative_order",
            "degeneracy_conditions",
            "matter_metric",
            "claimed_static_limit",
            "expected_dof",
            "evasion_rationale",
            "falsification_tests",
            "literature_overlap",
            "bounded_grammar",
        ],
        "bounded_grammar_required": ["basis", "max_terms", "coefficient_alphabet"],
    }


def structural_ablations(expression: str) -> list[dict[str, str]]:
    x, q, z = sp.symbols("x q z", nonnegative=True, finite=True)
    try:
        parsed = sp.sympify(expression.replace("^", "**"), locals={"x": x, "q": q, "z": z})
    except (sp.SympifyError, TypeError, SyntaxError):
        return []
    terms = sp.Add.make_args(sp.expand(parsed))
    if len(terms) <= 1:
        return []
    return [
        {
            "removed_term": str(term),
            "ablated_expression": str(sp.simplify(parsed - term)),
            "status": "not promoted; must re-enter all hard gates as a distinct candidate",
        }
        for term in terms
    ]


def validate_proposal(proposal: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    schema = proposal_schema()
    errors = [
        f"missing required field: {field}" for field in schema["required"] if field not in proposal
    ]
    bounded = proposal.get("bounded_grammar")
    if not isinstance(bounded, dict):
        errors.append("bounded_grammar must be an object")
    else:
        errors.extend(
            f"missing bounded_grammar field: {field}"
            for field in schema["bounded_grammar_required"]
            if field not in bounded
        )
        if isinstance(bounded.get("max_terms"), int) and bounded["max_terms"] > 12:
            errors.append("max_terms exceeds campaign safety cap of 12")
    text = canonical_json(proposal).casefold()
    prohibited = sorted(
        pattern
        for pattern in contract["prohibited_evidence_patterns"]
        if pattern.casefold() in text
    )
    if prohibited:
        errors.append(f"prohibited evidence patterns: {', '.join(prohibited)}")
    if proposal.get("matter_metric") in (None, "multiple", "object_specific"):
        errors.append("one universal matter metric is required")
    degeneracy_conditions = proposal.get("degeneracy_conditions")
    if not isinstance(degeneracy_conditions, list):
        errors.append("degeneracy_conditions must be an array")
    else:
        required_condition_fields = {"id", "expression", "equals", "variables", "status"}
        for index, condition in enumerate(degeneracy_conditions):
            if not isinstance(condition, dict):
                errors.append(f"degeneracy condition {index} must be an object")
                continue
            missing = sorted(required_condition_fields - set(condition))
            if missing:
                errors.append(
                    f"degeneracy condition {index} missing fields: {', '.join(missing)}"
                )
            if condition.get("status") != "declared_unverified":
                errors.append(
                    f"degeneracy condition {index} status must be declared_unverified"
                )
            if not isinstance(condition.get("variables"), list) or not condition.get(
                "variables"
            ):
                errors.append(f"degeneracy condition {index} variables must be nonempty")
    if proposal.get("derivative_order", 0) > 1 and not degeneracy_conditions:
        errors.append(
            "derivative_order > 1 requires a nonempty machine-readable degeneracy_conditions array"
        )
    return {"valid": not errors, "errors": errors, "prohibited_matches": prohibited}
