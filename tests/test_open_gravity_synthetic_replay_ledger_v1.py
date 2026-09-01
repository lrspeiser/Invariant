from __future__ import annotations

import pytest

from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import (
    BindingStatus,
    EligibilityDecision,
    EligibilityStatus,
    FormulaExecutionBinding,
    ResourceBounds,
)
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    SyntheticReplayLedger,
    SyntheticSuiteRelease,
    affected_formula_ids,
    status_from_result,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

HASH = "3" * 64


def binding(formula_id: str, feature: str, domain: str = "galaxy") -> FormulaExecutionBinding:
    return FormulaExecutionBinding(
        binding_id=f"binding.{formula_id}.v1",
        formula_id=formula_id,
        formula_version="v1.0.0",
        formula_sha256=HASH,
        status=BindingStatus.EXECUTABLE,
        entrypoint=(
            "sigma_theory_compiler.open_gravity_formula_execution_protocol_v1:decide_eligibility"
        ),
        required_features=(feature,),
        optional_features=(),
        emitted_features=("prediction.vector.acceleration",),
        domains=(domain,),
        geometry_support=("nonspherical3d",),
        time_support=("static",),
        parameter_schema_path="configs/open_gravity_vector_scale_control_parameters_v1.schema.json",
        parameter_schema_sha256=HASH,
        approximation_ceiling="synthetic direction finding only",
        health_gates=("dimension",),
        resource_bounds=ResourceBounds(60, 1_000_000, 100_000),
    )


def release(level: str = "MINOR") -> SyntheticSuiteRelease:
    return SyntheticSuiteRelease(
        suite_id="gravity.real-shaped.synthetic",
        version="v2.0.0",
        release_sha256=HASH,
        ontology_sha256=HASH,
        generator_sha256=HASH,
        observation_operator_sha256=HASH,
        changed_feature_ids=("source.tensor.shape",) if level != "PATCH" else (),
        change_level=level,
        response_calibrated=False,
    )


def test_affected_formulas_are_dependency_selected() -> None:
    shape = binding("GQNS01", "source.tensor.shape")
    density = binding("NEWTON", "source.scalar.density", "cluster")
    assert affected_formula_ids(release(), (shape, density)) == ("GQNS01",)
    assert affected_formula_ids(release(), (shape, density), affected_domains=("cluster",)) == (
        "GQNS01",
        "NEWTON",
    )
    assert affected_formula_ids(release("PATCH"), (shape, density)) == ()


def test_append_retains_nonexecuted_reason_and_hash_chain() -> None:
    candidate = binding("DPEL01", "source.tensor.shape")
    decision = EligibilityDecision(
        EligibilityStatus.INCOMPATIBLE_FEATURE_SET,
        ("source.tensor.shape",),
        (),
    )
    ledger = SyntheticReplayLedger("gravity.synthetic.replays", ())
    updated = ledger.append(
        release=release(),
        binding=candidate,
        eligibility=decision,
        adapter_sha256=None,
        domain="galaxy",
        experiment_id="galaxy.synthetic.v1",
    )
    assert updated.entries[0].status is DiscoveryStatus.INCOMPATIBLE_FEATURE_SET
    assert updated.entries[0].reason_codes == ("missing.source.tensor.shape",)
    assert updated.content_sha256

    broken = updated.entries[0].to_dict()
    broken["reason_codes"] = ()
    broken["observable_ids"] = ()
    broken["status"] = DiscoveryStatus(broken["status"])
    with pytest.raises(SchemaViolation, match="hash"):
        type(updated.entries[0])(**broken)


def test_discovery_status_never_implies_empirical_adjudication() -> None:
    assert (
        status_from_result(
            distinct_from_comparators=True,
            self_injection_recovered=True,
            numerical_valid=True,
            powered=True,
        )
        is DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE
    )
    assert (
        status_from_result(
            distinct_from_comparators=False,
            self_injection_recovered=True,
            numerical_valid=True,
            powered=True,
        )
        is DiscoveryStatus.AMBIGUOUS_WITH_COMPARATOR
    )


def test_completed_replay_requires_exact_matrix_cell_and_result_evidence() -> None:
    candidate = binding("DPEL01", "source.tensor.shape")
    ledger = SyntheticReplayLedger("gravity.synthetic.replays", ()).append(
        release=release(),
        binding=candidate,
        eligibility=EligibilityDecision(EligibilityStatus.ELIGIBLE, (), ()),
        adapter_sha256=HASH,
        domain="galaxy",
        experiment_id="galaxy.synthetic.v1",
    )
    completed = ledger.complete_last_eligible(
        release=release(),
        binding=candidate,
        adapter_sha256=HASH,
        domain="galaxy",
        experiment_id="galaxy.synthetic.v1",
        status=DiscoveryStatus.PROMISING_DISTINCT_SIGNATURE,
        scenario_id="galaxy.disk.v1",
        object_id="ngc2903",
        truth_world_id="dpel01",
        seed_lineage_sha256=HASH,
        nuisance_draw=0,
        parameter_cell_id="cell.001",
        observable_ids=("response.vector.velocity-field",),
        result_sha256=HASH,
        metrics_sha256=HASH,
        diagnostics_sha256=HASH,
    )
    assert completed.entries[-1].result_sha256 == HASH
    assert completed.entries[-1].claim_class == "SYNTHETIC_DIRECTIONAL_SIGNAL"

    with pytest.raises(SchemaViolation, match="prior eligible"):
        completed.complete_last_eligible(
            release=release(),
            binding=candidate,
            adapter_sha256=HASH,
            domain="galaxy",
            experiment_id="galaxy.synthetic.v1",
            status=DiscoveryStatus.UNDERPOWERED,
            scenario_id="galaxy.disk.v1",
            object_id="ngc2903",
            truth_world_id="dpel01",
            seed_lineage_sha256=HASH,
            nuisance_draw=0,
            parameter_cell_id="cell.001",
            observable_ids=("response.vector.velocity-field",),
            result_sha256=HASH,
            metrics_sha256=HASH,
            diagnostics_sha256=HASH,
        )
