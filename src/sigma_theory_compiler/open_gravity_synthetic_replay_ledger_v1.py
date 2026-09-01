"""Append-only lineage for exploratory synthetic discovery replays."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    EligibilityDecision,
    EligibilityStatus,
    FormulaExecutionBinding,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

SCHEMA_VERSION = "open-gravity-synthetic-replay-ledger-1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DiscoveryStatus(str, Enum):
    PROMISING_DISTINCT_SIGNATURE = "PROMISING_DISTINCT_SIGNATURE"
    AMBIGUOUS_WITH_COMPARATOR = "AMBIGUOUS_WITH_COMPARATOR"
    UNDERPOWERED = "UNDERPOWERED"
    SOURCE_ELEMENT_MISSING = "SOURCE_ELEMENT_MISSING"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"
    OUT_OF_REGISTERED_DOMAIN = "OUT_OF_REGISTERED_DOMAIN"
    ELIGIBLE_NOT_RUN = "ELIGIBLE_NOT_RUN"
    UNADAPTED = "UNADAPTED"
    THEORY_ONLY = "THEORY_ONLY"
    SOURCE_BLOCKED = "SOURCE_BLOCKED"
    QUARANTINED = "QUARANTINED"
    FORBIDDEN_RESPONSE_ACCESS = "FORBIDDEN_RESPONSE_ACCESS"
    INCOMPATIBLE_FEATURE_SET = "INCOMPATIBLE_FEATURE_SET"


_ELIGIBILITY_DISCOVERY = {
    EligibilityStatus.ELIGIBLE: DiscoveryStatus.ELIGIBLE_NOT_RUN,
    EligibilityStatus.UNADAPTED: DiscoveryStatus.UNADAPTED,
    EligibilityStatus.THEORY_ONLY: DiscoveryStatus.THEORY_ONLY,
    EligibilityStatus.SOURCE_BLOCKED: DiscoveryStatus.SOURCE_BLOCKED,
    EligibilityStatus.QUARANTINED: DiscoveryStatus.QUARANTINED,
    EligibilityStatus.INCOMPATIBLE_FEATURE_SET: DiscoveryStatus.INCOMPATIBLE_FEATURE_SET,
    EligibilityStatus.FORBIDDEN_RESPONSE_ACCESS: DiscoveryStatus.FORBIDDEN_RESPONSE_ACCESS,
    EligibilityStatus.OUT_OF_REGISTERED_DOMAIN: DiscoveryStatus.OUT_OF_REGISTERED_DOMAIN,
}


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SchemaViolation(f"{label} is not a canonical identifier")
    return value


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaViolation(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class SyntheticSuiteRelease:
    suite_id: str
    version: str
    release_sha256: str
    ontology_sha256: str
    generator_sha256: str
    observation_operator_sha256: str
    changed_feature_ids: tuple[str, ...]
    change_level: str
    response_calibrated: bool
    prediction_semantics_changed: bool = False

    def __post_init__(self) -> None:
        _identifier(self.suite_id, "suite_id")
        _identifier(self.version, "suite version")
        for label, value in (
            ("release_sha256", self.release_sha256),
            ("ontology_sha256", self.ontology_sha256),
            ("generator_sha256", self.generator_sha256),
            ("observation_operator_sha256", self.observation_operator_sha256),
        ):
            _hash(value, label)
        if self.changed_feature_ids != tuple(sorted(set(self.changed_feature_ids))):
            raise SchemaViolation("changed feature IDs must be unique and sorted")
        for feature in self.changed_feature_ids:
            _identifier(feature, "changed feature ID")
        if self.change_level not in {"MAJOR", "MINOR", "PATCH"}:
            raise SchemaViolation("change level must be MAJOR, MINOR, or PATCH")
        if self.change_level == "PATCH" and self.changed_feature_ids:
            raise SchemaViolation("PATCH release cannot change feature semantics")
        if type(self.prediction_semantics_changed) is not bool:
            raise SchemaViolation("prediction_semantics_changed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "version": self.version,
            "release_sha256": self.release_sha256,
            "ontology_sha256": self.ontology_sha256,
            "generator_sha256": self.generator_sha256,
            "observation_operator_sha256": self.observation_operator_sha256,
            "changed_feature_ids": list(self.changed_feature_ids),
            "change_level": self.change_level,
            "response_calibrated": self.response_calibrated,
            "prediction_semantics_changed": self.prediction_semantics_changed,
        }


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    sequence: int
    suite_id: str
    suite_version: str
    suite_sha256: str
    formula_id: str
    formula_version: str
    formula_sha256: str
    binding_sha256: str
    adapter_sha256: str | None
    domain: str
    experiment_id: str
    status: DiscoveryStatus
    reason_codes: tuple[str, ...]
    result_sha256: str | None
    scenario_id: str | None
    object_id: str | None
    truth_world_id: str | None
    seed_lineage_sha256: str | None
    nuisance_draw: int | None
    parameter_cell_id: str | None
    observable_ids: tuple[str, ...]
    metrics_sha256: str | None
    diagnostics_sha256: str | None
    claim_class: str
    prior_entry_sha256: str | None
    entry_sha256: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise SchemaViolation("replay sequence must be a nonnegative integer")
        for label, value in (
            ("suite_id", self.suite_id),
            ("suite_version", self.suite_version),
            ("formula_id", self.formula_id),
            ("formula_version", self.formula_version),
            ("domain", self.domain),
            ("experiment_id", self.experiment_id),
        ):
            _identifier(value, label)
        for label, value in (
            ("suite_sha256", self.suite_sha256),
            ("formula_sha256", self.formula_sha256),
            ("binding_sha256", self.binding_sha256),
            ("entry_sha256", self.entry_sha256),
        ):
            _hash(value, label)
        for label, value in (
            ("adapter_sha256", self.adapter_sha256),
            ("result_sha256", self.result_sha256),
            ("prior_entry_sha256", self.prior_entry_sha256),
            ("seed_lineage_sha256", self.seed_lineage_sha256),
            ("metrics_sha256", self.metrics_sha256),
            ("diagnostics_sha256", self.diagnostics_sha256),
        ):
            if value is not None:
                _hash(value, label)
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise SchemaViolation("reason codes must be unique and sorted")
        for reason in self.reason_codes:
            _identifier(reason, "reason code")
        for label, value in (
            ("scenario_id", self.scenario_id),
            ("object_id", self.object_id),
            ("truth_world_id", self.truth_world_id),
            ("parameter_cell_id", self.parameter_cell_id),
        ):
            if value is not None:
                _identifier(value, label)
        if self.nuisance_draw is not None and (
            type(self.nuisance_draw) is not int or self.nuisance_draw < 0
        ):
            raise SchemaViolation("nuisance_draw must be a nonnegative integer")
        if self.observable_ids != tuple(sorted(set(self.observable_ids))):
            raise SchemaViolation("observable IDs must be unique and sorted")
        for observable in self.observable_ids:
            _identifier(observable, "observable ID")
        if self.claim_class != "SYNTHETIC_DIRECTIONAL_SIGNAL":
            raise SchemaViolation("synthetic replay cannot claim empirical support or rejection")
        completed = self.status in {
            DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE,
            DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
            DiscoveryStatus.UNDERPOWERED,
            DiscoveryStatus.NUMERICAL_INVALID,
        }
        completion_fields = (
            self.result_sha256,
            self.scenario_id,
            self.object_id,
            self.truth_world_id,
            self.seed_lineage_sha256,
            self.nuisance_draw,
            self.parameter_cell_id,
            self.metrics_sha256,
            self.diagnostics_sha256,
        )
        if completed and (
            any(value is None for value in completion_fields) or not self.observable_ids
        ):
            raise SchemaViolation("completed replay entry lacks exact matrix-cell evidence")
        if not completed and (
            any(value is not None for value in completion_fields) or self.observable_ids
        ):
            raise SchemaViolation("noncompleted replay entry cannot bind result evidence")
        if self.entry_sha256 != canonical_sha256(self._body()):
            raise SchemaViolation("replay entry hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "suite_sha256": self.suite_sha256,
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "formula_sha256": self.formula_sha256,
            "binding_sha256": self.binding_sha256,
            "adapter_sha256": self.adapter_sha256,
            "domain": self.domain,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "result_sha256": self.result_sha256,
            "scenario_id": self.scenario_id,
            "object_id": self.object_id,
            "truth_world_id": self.truth_world_id,
            "seed_lineage_sha256": self.seed_lineage_sha256,
            "nuisance_draw": self.nuisance_draw,
            "parameter_cell_id": self.parameter_cell_id,
            "observable_ids": list(self.observable_ids),
            "metrics_sha256": self.metrics_sha256,
            "diagnostics_sha256": self.diagnostics_sha256,
            "claim_class": self.claim_class,
            "prior_entry_sha256": self.prior_entry_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "entry_sha256": self.entry_sha256}

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        release: SyntheticSuiteRelease,
        binding: FormulaExecutionBinding,
        adapter_sha256: str | None,
        domain: str,
        experiment_id: str,
        status: DiscoveryStatus,
        reason_codes: Sequence[str] = (),
        result_sha256: str | None = None,
        scenario_id: str | None = None,
        object_id: str | None = None,
        truth_world_id: str | None = None,
        seed_lineage_sha256: str | None = None,
        nuisance_draw: int | None = None,
        parameter_cell_id: str | None = None,
        observable_ids: Sequence[str] = (),
        metrics_sha256: str | None = None,
        diagnostics_sha256: str | None = None,
        prior_entry_sha256: str | None = None,
    ) -> ReplayEntry:
        body = {
            "sequence": sequence,
            "suite_id": release.suite_id,
            "suite_version": release.version,
            "suite_sha256": release.release_sha256,
            "formula_id": binding.formula_id,
            "formula_version": binding.formula_version,
            "formula_sha256": binding.formula_sha256,
            "binding_sha256": binding.content_sha256,
            "adapter_sha256": adapter_sha256,
            "domain": domain,
            "experiment_id": experiment_id,
            "status": status,
            "reason_codes": tuple(sorted(set(reason_codes))),
            "result_sha256": result_sha256,
            "scenario_id": scenario_id,
            "object_id": object_id,
            "truth_world_id": truth_world_id,
            "seed_lineage_sha256": seed_lineage_sha256,
            "nuisance_draw": nuisance_draw,
            "parameter_cell_id": parameter_cell_id,
            "observable_ids": tuple(sorted(set(observable_ids))),
            "metrics_sha256": metrics_sha256,
            "diagnostics_sha256": diagnostics_sha256,
            "claim_class": "SYNTHETIC_DIRECTIONAL_SIGNAL",
            "prior_entry_sha256": prior_entry_sha256,
        }
        canonical_body = {
            **body,
            "status": status.value,
            "reason_codes": list(body["reason_codes"]),
            "observable_ids": list(body["observable_ids"]),
        }
        return cls(**body, entry_sha256=canonical_sha256(canonical_body))


@dataclass(frozen=True, slots=True)
class SyntheticReplayLedger:
    ledger_id: str
    entries: tuple[ReplayEntry, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.ledger_id, "ledger_id")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaViolation("synthetic replay ledger schema changed")
        prior: str | None = None
        for sequence, entry in enumerate(self.entries):
            if entry.sequence != sequence or entry.prior_entry_sha256 != prior:
                raise SchemaViolation("synthetic replay chain changed")
            prior = entry.entry_sha256

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def append(
        self,
        *,
        release: SyntheticSuiteRelease,
        binding: FormulaExecutionBinding,
        eligibility: EligibilityDecision,
        adapter_sha256: str | None,
        domain: str,
        experiment_id: str,
    ) -> SyntheticReplayLedger:
        status = _ELIGIBILITY_DISCOVERY[eligibility.status]
        reasons = tuple(
            sorted(
                [
                    *(f"missing.{item}" for item in eligibility.missing_features),
                    *(f"forbidden.{item}" for item in eligibility.forbidden_features),
                ]
            )
        )
        entry = ReplayEntry.create(
            sequence=len(self.entries),
            release=release,
            binding=binding,
            adapter_sha256=adapter_sha256,
            domain=domain,
            experiment_id=experiment_id,
            status=status,
            reason_codes=reasons,
            prior_entry_sha256=self.entries[-1].entry_sha256 if self.entries else None,
        )
        return SyntheticReplayLedger(self.ledger_id, (*self.entries, entry), self.schema_version)

    def complete_last_eligible(
        self,
        *,
        release: SyntheticSuiteRelease,
        binding: FormulaExecutionBinding,
        adapter_sha256: str,
        domain: str,
        experiment_id: str,
        status: DiscoveryStatus,
        scenario_id: str,
        object_id: str,
        truth_world_id: str,
        seed_lineage_sha256: str,
        nuisance_draw: int,
        parameter_cell_id: str,
        observable_ids: Sequence[str],
        result_sha256: str,
        metrics_sha256: str,
        diagnostics_sha256: str,
        reason_codes: Sequence[str] = (),
    ) -> SyntheticReplayLedger:
        if not self.entries or self.entries[-1].status is not DiscoveryStatus.ELIGIBLE_NOT_RUN:
            raise SchemaViolation("replay completion requires the immediately prior eligible entry")
        prior = self.entries[-1]
        if (
            prior.suite_sha256 != release.release_sha256
            or prior.binding_sha256 != binding.content_sha256
            or prior.domain != domain
            or prior.experiment_id != experiment_id
        ):
            raise SchemaViolation("replay completion identity differs from eligible entry")
        if status not in {
            DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE,
            DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR,
            DiscoveryStatus.UNDERPOWERED,
            DiscoveryStatus.NUMERICAL_INVALID,
        }:
            raise SchemaViolation("replay completion status is not a calculated discovery result")
        entry = ReplayEntry.create(
            sequence=len(self.entries),
            release=release,
            binding=binding,
            adapter_sha256=adapter_sha256,
            domain=domain,
            experiment_id=experiment_id,
            status=status,
            reason_codes=reason_codes,
            result_sha256=result_sha256,
            scenario_id=scenario_id,
            object_id=object_id,
            truth_world_id=truth_world_id,
            seed_lineage_sha256=seed_lineage_sha256,
            nuisance_draw=nuisance_draw,
            parameter_cell_id=parameter_cell_id,
            observable_ids=observable_ids,
            metrics_sha256=metrics_sha256,
            diagnostics_sha256=diagnostics_sha256,
            prior_entry_sha256=prior.entry_sha256,
        )
        return SyntheticReplayLedger(self.ledger_id, (*self.entries, entry), self.schema_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ledger_id": self.ledger_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def affected_formula_ids(
    release: SyntheticSuiteRelease,
    bindings: Sequence[FormulaExecutionBinding],
    *,
    affected_domains: Sequence[str] = (),
) -> tuple[str, ...]:
    domains = set(affected_domains)
    changed = set(release.changed_feature_ids)
    if release.change_level == "PATCH" and not release.prediction_semantics_changed:
        return ()
    affected: set[str] = set()
    for binding in bindings:
        features = set(
            binding.required_features + binding.optional_features + binding.emitted_features
        )
        if features & changed or domains & set(binding.domains):
            affected.add(binding.formula_id)
    if (
        not changed
        and not domains
        and (release.change_level in {"MAJOR", "MINOR"} or release.prediction_semantics_changed)
    ):
        affected.update(binding.formula_id for binding in bindings)
    return tuple(sorted(affected))


def status_from_result(
    *,
    distinct_from_comparators: bool,
    self_injection_recovered: bool,
    numerical_valid: bool,
    powered: bool,
) -> DiscoveryStatus:
    if not numerical_valid:
        return DiscoveryStatus.NUMERICAL_INVALID
    if not powered:
        return DiscoveryStatus.UNDERPOWERED
    if self_injection_recovered and distinct_from_comparators:
        return DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE
    return DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR


__all__ = [
    "SCHEMA_VERSION",
    "DiscoveryStatus",
    "ReplayEntry",
    "SyntheticReplayLedger",
    "SyntheticSuiteRelease",
    "affected_formula_ids",
    "status_from_result",
]
