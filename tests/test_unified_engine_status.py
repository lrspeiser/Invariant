from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sigma_theory_compiler.unified_engine_status import (
    DEFAULT_MAXIMUM_OUTPUT_BYTES,
    _read_deferred_gpu_handoff_status,
    _read_unified_live_service_status,
    build_unified_snapshot,
    load_config,
    main,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "runs/engine/rust-streaming-billion-status.json",
    "runs/benchmarks/cpu-real-formula-overlap-15-16.json",
    "runs/formal-controls-v1/formal-controls-portable.json",
    "runs/engine/continuous-scientific-pipeline-admission-readiness.json",
    "runs/engine/continuous-scientific-pipeline-service-readiness.json",
    "runs/engine/continuous-scientific-pipeline-service-result.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-genesis.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-result.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/result.json",
    "runs/engine/composite-promotion-overlay-production-status.json",
    "runs/engine/grammar-v3-parameter-cell-execution-status.json",
    "runs/engine/grammar-v3-parameter-cell-expansion-service-status.json",
    "runs/engine/covariant-grammar-v3-seed-manifest.json",
    "runs/engine/grammar-v3-parameter-cell-manifest.json",
    "runs/engine/grammar-v3-parameter-cell-compilation-campaign.json",
    "runs/engine/generated-candidate-formal-export.json",
    "runs/engine/generated-candidate-metric-variation-execution.json",
    "runs/engine/generated-candidate-formula-gpu-stress-campaign.json",
    "runs/engine/kastner-schlatter-transactional-gravity-intake.json",
    "runs/engine/kastner-schlatter-equation-graph-admission.json",
    "runs/engine/kastner-schlatter-cuda-consequence-campaign.json",
    "runs/engine/kastner-schlatter-observational-readiness-contract.json",
    "runs/engine/kastner-schlatter-cuda-falsification-design.json",
    "runs/engine/kastner-schlatter-candidate-action-completion.json",
    "runs/engine/kastner-schlatter-action-equivalence-audit.json",
    "runs/engine/kastner-schlatter-candidate-action-formal-admission.json",
    "runs/engine/kastner-schlatter-scalar-intensity-cuda-falsification.json",
    "runs/engine/kastner-schlatter-extended-geometry-cuda-stress.json",
    "runs/engine/kastner-schlatter-de-sitter-energy-prerequisite.json",
    "runs/engine/kastner-schlatter-poisson-action-compatibility.json",
    "runs/engine/kastner-schlatter-positive-intensity-preservation-gate.json",
    "runs/engine/kastner-schlatter-positive-reparameterization-gate.json",
    "runs/engine/kastner-schlatter-covariant-point-process-measure-gate.json",
    "runs/engine/kastner-schlatter-poisson-selector-contract-gate.json",
    "runs/engine/kastner-schlatter-conditional-poisson-kernel-completion-gate.json",
    "runs/engine/kastner-schlatter-actualization-history-map-audit.json",
    "runs/engine/kastner-schlatter-qed-actualization-poisson-derivation-audit.json",
    "runs/engine/kastner-schlatter-deterministic-compensator-admission-gate.json",
    "runs/engine/kastner-schlatter-canonical-probability-space-gate.json",
    "runs/engine/kastner-schlatter-deterministic-feature-selector-no-go.json",
    "runs/engine/kastner-schlatter-second-order-selector-no-go.json",
    "runs/engine/kastner-schlatter-finite-factorial-hierarchy-no-go.json",
    "runs/engine/kastner-schlatter-countable-full-law-selector-admission.json",
    "runs/engine/kastner-schlatter-source-selector-type-inhabitation-audit.json",
    "runs/engine/kastner-schlatter-actualization-probability-bridge-contract.json",
    "runs/engine/kastner-schlatter-history-kernel-projective-admission.json",
    "runs/engine/kastner-schlatter-transaction-event-observable-exposure-gate.json",
    "runs/engine/kastner-schlatter-poisson-cox-cuda-power-campaign.json",
    "runs/engine/kastner-schlatter-set-indexed-cuda-falsification-campaign.json",
    "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness.json",
    "runs/engine/kastner-schlatter-deferred-gpu-ownership-readiness.json",
    "runs/engine/kastner-schlatter-deferred-gpu-handoff-readiness.json",
    "runs/engine/generic-g4-b4-termwise-normalization-campaign.json",
    "runs/engine/einstein-aether-coupling-boundary-kkt-gate.json",
    "runs/engine/grammar-v3-formal-preflight-status.json",
    "runs/engine/grammar-v3-promotion-admission-status.json",
    "runs/engine/grammar-v3-g2-candidate-formal-status.json",
    "runs/engine/g2-scalable-nonmaximal-positive-mass-audit.json",
    "runs/engine/g2-scalable-solar-prediction-readiness.json",
    "runs/engine/g2-solar-heldout-transfer-registration.json",
    "runs/engine/scalable-campaign-staged-epoch-status.json",
    "runs/engine/scalable-future-parameter-chunk-001-status.json",
    "runs/engine/reviewed-future-parameter-formal-preflight-001.json",
    "runs/engine/future-aether-candidate-formal-followup.json",
    "runs/engine/future-aether-constraint-boundary-embedding-audit.json",
    "runs/engine/future-aether-pure-twist-ae-no-go-audit.json",
    "runs/engine/future-aether-weak-field-ae-constraint-gate.json",
    "runs/engine/future-aether-finite-amplitude-negative-seed-gate.json",
    "runs/engine/future-aether-nonlinear-lift-characteristic-gate.json",
    "runs/engine/future-aether-regular-adm-inverse-margin-gate.json",
    "runs/engine/future-aether-weighted-ift-contract-gate.json",
    "runs/engine/future-aether-weighted-reference-operator-gate.json",
    "runs/engine/future-aether-fixed-free-data-principal-gate.json",
    "runs/engine/future-aether-finite-tilt-york-symbol-gate.json",
    "runs/engine/future-aether-principal-inverse-fredholm-gate.json",
    "runs/engine/future-aether-lower-order-coefficient-contract-gate.json",
    "runs/engine/future-aether-canonical-seed-constraint-dag-gate.json",
    "runs/engine/future-aether-characteristic-shell-hcore-gate.json",
    "runs/engine/future-g3-componentwise-domain-contract-campaign.json",
    "runs/engine/future-g3-action-bound-jet-box-campaign.json",
    "runs/engine/future-g3-af-transition-obstruction-campaign.json",
    "runs/engine/future-g3-nonunitary-af-constraint-gate-campaign.json",
    "runs/engine/future-g3-radial-conformal-constraint-reduction-campaign.json",
    "runs/engine/future-g3-radial-lichnerowicz-bvp-no-go-campaign.json",
    "runs/engine/future-g3-nonradial-york-bounded-mean-curvature-no-go-campaign.json",
    "runs/engine/future-g3-york-mean-curvature-frontier-campaign.json",
    "runs/engine/future-g3-york-analytic-mean-curvature-threshold-campaign.json",
    "runs/engine/future-g3-york-tracefree-compensation-no-go-campaign.json",
    "runs/engine/future-g3-general-geometry-curvature-shortfall-no-go-campaign.json",
    "runs/engine/future-g3-general-geometry-surplus-mismatch-no-go-campaign.json",
    "runs/engine/future-g3-flat-radial-matched-constraints-asymptotic-no-go-campaign.json",
    "runs/engine/future-candidate-action-dossier.json",
    "runs/engine/grammar-v3-g3-candidate-formal-status.json",
    "runs/engine/g4-scalable-action-formal-followup.json",
    "runs/engine/aether-parameter-cell-formal-gate-status.json",
    "runs/engine/scalable-candidate-structural-metrics.json",
    "runs/engine/scalable-candidate-explanation-dossier-bridge.json",
    "runs/engine/grammar-v3-evidence-pareto-report.json",
    "runs/engine/grammar-v3-followup-service-g4-final-status.json",
    "runs/engine/grammar-v3-followup-queue-g4-final-status.json",
    "configs/resource_profile_5090.json",
    "runs/engine/llm-formula-proposal-adapter-readiness.json",
    "runs/engine/campaign-llm-proposal-bridge-readiness.json",
    "runs/engine/reviewed-g4-candidate-solar-evaluator-readiness.json",
    "runs/engine/grammar-v3-g4-solar-reviewed-execution-status.json",
    "runs/engine/reviewed-g4-candidate-galaxy-evaluator-readiness.json",
    "runs/engine/grammar-v3-g4-galaxy-reviewed-execution-status.json",
    "runs/engine/typed-dsl-campaign-admission-readiness.json",
    "runs/engine/compiler-receipt-registry-bridge-readiness.json",
    "runs/engine/reviewed-local-formula-epoch-status.json",
    "runs/engine/reviewed-local-formula-service-readiness.json",
    "runs/engine/g4-scalar-free-galaxy-forward-model.json",
    "runs/engine/g4-galaxy-branch-distance-registration.json",
    "runs/engine/g4-galaxy-calibration-evaluation-registration.json",
    "runs/engine/g4-galaxy-prediction-contract-transform-registration.json",
    "runs/engine/g4-galaxy-manifest-bundle-tooling-readiness.json",
    "runs/engine/g4-galaxy-source-registry-admission-readiness.json",
    "runs/physics-language/quartic-anti-wick-composition-campaign/campaign.json",
    "runs/physics-language/quartic-annular-k55-c6-campaign/campaign.json",
    "runs/physics-language/quartic-bounded-frequency-defect-campaign/campaign.json",
    "runs/physics-language/quartic-dyadic-localization-campaign/campaign.json",
    "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-campaign/campaign.json",
    "runs/physics-language/quartic-full-tensor-good-unknown-reconciliation-gate/campaign.json",
    "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json",
    "runs/physics-language/quartic-scalar-hessian-curl-invariance-gate/campaign.json",
    "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json",
    "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json",
    "runs/physics-language/quartic-principal-high-atom-connection-extension-gate/campaign.json",
    "runs/physics-language/quartic-tc2-ck1-p55-tube-envelope-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-quadratic-deltak-extension-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-diagonal-third-jet-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-basis-reduction-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-reduction-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-fourth-jet-range-obligation-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000000.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000032.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000064.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000096.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000128.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000160.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000192.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/chunks/obligation-offset-000224.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/checkpoint.json",
    "runs/physics-language/quartic-tc2-fourth-jet-obligation-service/service-status.json",
    "runs/physics-language/quartic-tc2-d4-obstruction-cokernel-certificate/campaign.json",
    "runs/physics-language/quartic-tc2-d4-homogeneous-freedom-reduction/campaign.json",
    "runs/physics-language/quartic-tc2-d4-minimal-tc2-escape-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-registered-operator-origin-no-go-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-topology-changing-origin-classification-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-curl-constraint-admission-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-curl-companion-range-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-axis2-base-rhs-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-spatial-gradient-annihilator-no-go-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-full-linear-gradient-annihilator-no-go-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-parity-cubic-angular-escape-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-parity-cubic-generic-direction-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-matrix-curl-rank-one-completion-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-degree-three-matrix-curl-sphere-extension-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-degree-three-c23-great-circle-escape-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-degree-three-rank-two-xyz-completion-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-degree-three-sixth-frame-completion-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-rational-chart-determining-gate/campaign.json",
    "runs/physics-language/quartic-tc2-d4-degree-five-counterexample-escape-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-revised-symbol-rational-counterexample-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-revised-eight-frame-rational-counterexample-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-d4-revised-nine-frame-rational-counterexample-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000000.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000064.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000128.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000192.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000256.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000320.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/chunks/obligation-offset-000384.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/checkpoint.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service/service-status.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-chunk-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/chunks/offset-000064.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/chunks/offset-000128.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/checkpoint.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/service-status.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000192.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000256.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000320.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000384.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000448.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000512.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000576.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000640.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000704.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000768.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000832.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000896.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-000960.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001024.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001088.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001152.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001216.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001280.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001344.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001408.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001472.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/chunks/offset-001536.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/checkpoint.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service/service-status.json",
    "runs/engine/quartic-tc2-mixed-third-jet-parallel-supervisor-readiness.json",
    "runs/engine/unified-engine-live-service-readiness.json",
    "runs/engine/unified-engine-live-service-safety-readiness.json",
]

RECOVERY_CONFIG_PATHS = (
    "configs/backgrounds/quartic_anti_wick_composition_campaign.json",
    "configs/backgrounds/quartic_annular_k55_c6_campaign.json",
    "configs/backgrounds/quartic_bounded_frequency_defect_campaign.json",
    "configs/backgrounds/quartic_dyadic_localization_campaign.json",
    "configs/backgrounds/quartic_finite_sobolev_hierarchy_no_go_campaign.json",
    "configs/backgrounds/quartic_full_tensor_good_unknown_reconciliation_gate.json",
    "configs/backgrounds/quartic_scalar_hessian_d2_integrability_gate.json",
    "configs/backgrounds/quartic_scalar_hessian_curl_invariance_gate.json",
    "configs/backgrounds/quartic_scalar_hessian_output_bundle_repair_gate.json",
    "configs/backgrounds/quartic_full_d2f_high_atom_coverage_gate.json",
    "configs/backgrounds/quartic_principal_high_atom_connection_extension_gate.json",
)
FINITE_SOBOLEV_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_finite_sobolev_hierarchy_no_go_campaign.py",
    "tests/test_quartic_finite_sobolev_hierarchy_no_go_campaign.py",
)
FULL_TENSOR_RECONCILIATION_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_full_tensor_good_unknown_reconciliation_gate.py",
    "tests/test_quartic_full_tensor_good_unknown_reconciliation_gate.py",
)
SCALAR_HESSIAN_D2_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_scalar_hessian_d2_integrability_gate.py",
    "tests/test_quartic_scalar_hessian_d2_integrability_gate.py",
)
SCALAR_HESSIAN_CURL_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_scalar_hessian_curl_invariance_gate.py",
    "tests/test_quartic_scalar_hessian_curl_invariance_gate.py",
)
SCALAR_HESSIAN_OUTPUT_BUNDLE_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_scalar_hessian_output_bundle_repair_gate.py",
    "tests/test_quartic_scalar_hessian_output_bundle_repair_gate.py",
)
FULL_D2F_HIGH_ATOM_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_full_d2f_high_atom_coverage_gate.py",
    "tests/test_quartic_full_d2f_high_atom_coverage_gate.py",
)
PRINCIPAL_HIGH_ATOM_CONNECTION_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_principal_high_atom_connection_extension_gate.py",
    "tests/test_quartic_principal_high_atom_connection_extension_gate.py",
)
SIXTH_FRAME_CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_degree_three_sixth_frame_completion_campaign.json"
)
RATIONAL_CHART_CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_rational_chart_determining_gate.json"
)
REVISED_SYMBOL_CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_revised_symbol_rational_counterexample_campaign.json"
)
REVISED_SYMBOL_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_tc2_d4_revised_symbol_rational_counterexample_campaign.py",
    "tests/test_quartic_tc2_d4_revised_symbol_rational_counterexample_campaign.py",
)
REVISED_EIGHT_FRAME_CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_revised_eight_frame_rational_counterexample_campaign.json"
)
REVISED_EIGHT_FRAME_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_tc2_d4_revised_eight_frame_rational_counterexample_campaign.py",
    "tests/test_quartic_tc2_d4_revised_eight_frame_rational_counterexample_campaign.py",
)
REVISED_NINE_FRAME_CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_revised_nine_frame_rational_counterexample_campaign.json"
)
REVISED_NINE_FRAME_DEPENDENCIES = (
    "src/sigma_theory_compiler/quartic_tc2_d4_revised_nine_frame_rational_counterexample_campaign.py",
    "tests/test_quartic_tc2_d4_revised_nine_frame_rational_counterexample_campaign.py",
)
PORTABLE_FORMAL_DEPENDENCIES = (
    "configs/formal_controls_portable_report.json",
    "src/sigma_theory_compiler/formal_controls_portable_report.py",
    "tests/test_formal_controls_portable_report.py",
    "runs/formal-controls-v1/formal-controls.json",
)
CONTINUOUS_PIPELINE_DEPENDENCIES = (
    "configs/continuous_scientific_pipeline_admission.json",
    "src/sigma_theory_compiler/continuous_scientific_pipeline_admission.py",
    "tests/test_continuous_scientific_pipeline_admission.py",
    "configs/continuous_scientific_pipeline_service.json",
    "src/sigma_theory_compiler/continuous_scientific_pipeline_service.py",
    "tests/test_continuous_scientific_pipeline_service.py",
    "configs/continuous_formula_formal_backend.json",
    "src/sigma_theory_compiler/continuous_formula_formal_backend.py",
    "tests/test_continuous_formula_formal_backend.py",
    "configs/continuous_scientific_pipeline_epoch_003.json",
    "src/sigma_theory_compiler/continuous_scientific_pipeline_epoch.py",
    "tests/test_continuous_scientific_pipeline_epoch.py",
    "src/sigma_theory_compiler/continuous_scientific_pipeline_epoch_result.py",
    "tests/test_continuous_scientific_pipeline_epoch_result.py",
    "runs/engine/continuous-scientific-pipeline-epoch-003-preflight.json",
    "configs/generator_v2_billion.json",
    "configs/covariant_action_grammar.json",
    "configs/covariant_field_contract.json",
    "src/sigma_theory_compiler/production_covariant_provenance.py",
    "src/sigma_theory_compiler/action_health.py",
    "src/sigma_theory_compiler/persistent_parallel_search.py",
    "src/sigma_theory_compiler/persistent_parallel_supervisor.py",
    "src/sigma_theory_compiler/real_formula_execution.py",
    "src/sigma_theory_compiler/high_throughput.py",
    "src/sigma_theory_compiler/gpu_screen.py",
    "src/sigma_theory_compiler/scientific_leaderboards.py",
    "configs/scientific_leaderboards.json",
    "configs/continuous_scientific_pipeline_epoch_003_candidate_followup.json",
    "src/sigma_theory_compiler/continuous_scientific_pipeline_candidate_followup.py",
    "tests/test_continuous_scientific_pipeline_candidate_followup.py",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-01.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-02.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-03.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-05.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-06.json",
    "runs/engine/continuous-scientific-pipeline-epoch-003-candidate-followup/batch-07.json",
)
LABELS = [
    "billion_streaming",
    "cpu_real_formula_overlap_benchmark",
    "formal_controls_portable_report",
    "continuous_scientific_pipeline_admission",
    "continuous_scientific_pipeline_service_readiness",
    "continuous_scientific_pipeline_service_result",
    "continuous_scientific_pipeline_epoch_003_genesis",
    "continuous_scientific_pipeline_epoch_003_result",
    "continuous_scientific_pipeline_epoch_003_candidate_followup",
    "promotion_overlay",
    "grammar_parameter_cells",
    "grammar_parameter_cell_expansion_service",
    "grammar_v3_seed_manifest",
    "grammar_parameter_cell_manifest",
    "grammar_parameter_cell_compilation",
    "generated_candidate_formal_export",
    "generated_candidate_metric_variation_specialization",
    "generated_candidate_formula_gpu_stress",
    "kastner_schlatter_transactional_gravity_intake",
    "kastner_schlatter_equation_graph_admission",
    "kastner_schlatter_cuda_consequence_campaign",
    "kastner_schlatter_observational_readiness_contract",
    "kastner_schlatter_cuda_falsification_design",
    "kastner_schlatter_candidate_action_completion",
    "kastner_schlatter_action_equivalence_audit",
    "kastner_schlatter_candidate_action_formal_admission",
    "kastner_schlatter_scalar_intensity_cuda_falsification",
    "kastner_schlatter_extended_geometry_cuda_stress",
    "kastner_schlatter_de_sitter_energy_prerequisite",
    "kastner_schlatter_poisson_action_compatibility",
    "kastner_schlatter_positive_intensity_preservation",
    "kastner_schlatter_positive_reparameterization",
    "kastner_schlatter_covariant_point_process_measure",
    "kastner_schlatter_poisson_selector_contract",
    "kastner_schlatter_conditional_poisson_kernel_completion",
    "kastner_schlatter_actualization_history_map_audit",
    "kastner_schlatter_qed_actualization_poisson_derivation",
    "kastner_schlatter_deterministic_compensator_admission",
    "kastner_schlatter_canonical_probability_space",
    "kastner_schlatter_deterministic_feature_selector_no_go",
    "kastner_schlatter_second_order_selector_no_go",
    "kastner_schlatter_finite_factorial_hierarchy_no_go",
    "kastner_schlatter_countable_full_law_selector_admission",
    "kastner_schlatter_source_selector_type_inhabitation_audit",
    "kastner_schlatter_actualization_probability_bridge_contract",
    "kastner_schlatter_history_kernel_projective_admission",
    "kastner_schlatter_transaction_event_observable_exposure",
    "kastner_schlatter_poisson_cox_cuda_power",
    "kastner_schlatter_set_indexed_cuda_falsification",
    "kastner_schlatter_set_indexed_gpu_scheduler_adapter",
    "kastner_schlatter_deferred_gpu_ownership",
    "kastner_schlatter_deferred_gpu_handoff_service",
    "generic_g4_b4_termwise_normalization",
    "einstein_aether_coupling_boundary_kkt",
    "grammar_v3_formal_preflight",
    "grammar_v3_promotion_admission",
    "grammar_v3_g2_candidate_formal",
    "grammar_v3_g2_nonmaximal_positive_mass_followup",
    "grammar_v3_g2_solar_readiness",
    "grammar_v3_g2_solar_heldout_transfer",
    "scalable_campaign_epoch",
    "scalable_future_parameter_chunk",
    "scalable_future_formal_preflight",
    "future_aether_formal_followup",
    "future_aether_constraint_followup",
    "future_aether_pure_twist_ae_no_go",
    "future_aether_weak_field_ae_constraint_gate",
    "future_aether_finite_amplitude_negative_seed_gate",
    "future_aether_nonlinear_lift_characteristic_gate",
    "future_aether_regular_adm_inverse_margin_gate",
    "future_aether_weighted_ift_contract_gate",
    "future_aether_weighted_reference_operator_gate",
    "future_aether_fixed_free_data_principal_gate",
    "future_aether_finite_tilt_york_symbol_gate",
    "future_aether_principal_inverse_fredholm_gate",
    "future_aether_lower_order_coefficient_contract_gate",
    "future_aether_canonical_seed_constraint_dag_gate",
    "future_aether_characteristic_shell_hcore_gate",
    "future_g3_domain_followup",
    "future_g3_action_bound_followup",
    "future_g3_af_transition_obstruction",
    "future_g3_nonunitary_af_constraint_gate",
    "future_g3_radial_conformal_constraint_reduction",
    "future_g3_radial_lichnerowicz_bvp_no_go",
    "future_g3_nonradial_york_bounded_mean_curvature_no_go",
    "future_g3_york_mean_curvature_frontier",
    "future_g3_york_analytic_threshold",
    "future_g3_york_tracefree_compensation",
    "future_g3_general_geometry_curvature_shortfall",
    "future_g3_general_geometry_surplus_mismatch",
    "future_g3_flat_radial_matched_constraints_asymptotic_no_go",
    "future_candidate_action_dossier",
    "grammar_v3_g3_candidate_formal",
    "grammar_v3_g4_scalable_formal_followup",
    "grammar_v3_aether_candidate_formal",
    "scalable_structural_metrics",
    "scalable_explanation_dossiers",
    "evidence_pareto",
    "followup_service",
    "followup_queue",
    "resource_profile",
    "llm_proposal_adapter",
    "llm_campaign_bridge",
    "g4_solar_evaluator",
    "g4_solar_execution",
    "g4_galaxy_evaluator",
    "g4_galaxy_execution",
    "typed_dsl_admission",
    "compiler_registry_bridge",
    "reviewed_local_formula_epoch",
    "reviewed_local_formula_service",
    "g4_galaxy_forward_model",
    "g4_galaxy_branch_distance",
    "g4_galaxy_calibration_evaluation",
    "g4_galaxy_prediction_contract_transform",
    "g4_galaxy_manifest_bundle_tooling",
    "g4_galaxy_source_registry_admission",
    "quartic_anti_wick_composition_campaign",
    "quartic_annular_k55_c6_campaign",
    "quartic_bounded_frequency_defect_campaign",
    "quartic_dyadic_localization_campaign",
    "quartic_finite_sobolev_hierarchy_no_go_campaign",
    "quartic_full_tensor_good_unknown_reconciliation_gate",
    "quartic_scalar_hessian_d2_integrability_gate",
    "quartic_scalar_hessian_curl_invariance_gate",
    "quartic_scalar_hessian_output_bundle_repair_gate",
    "quartic_full_d2f_high_atom_coverage_gate",
    "quartic_principal_high_atom_connection_extension_gate",
    "quartic_ck1_p55_tube_envelope",
    "quartic_tc2_quadratic_deltak_extension",
    "quartic_tc2_diagonal_third_jet",
    "quartic_tc2_mixed_third_jet_basis_reduction",
    "quartic_tc2_mixed_third_jet_reranked_reduction",
    "quartic_tc2_fourth_jet_range_obligations",
    "quartic_tc2_fourth_jet_chunk_0",
    "quartic_tc2_fourth_jet_chunk_32",
    "quartic_tc2_fourth_jet_chunk_64",
    "quartic_tc2_fourth_jet_chunk_96",
    "quartic_tc2_fourth_jet_chunk_128",
    "quartic_tc2_fourth_jet_chunk_160",
    "quartic_tc2_fourth_jet_chunk_192",
    "quartic_tc2_fourth_jet_chunk_224",
    "quartic_tc2_fourth_jet_checkpoint",
    "quartic_tc2_fourth_jet_status",
    "quartic_tc2_d4_obstruction_cokernel_certificate",
    "quartic_tc2_d4_homogeneous_freedom_reduction",
    "quartic_tc2_d4_minimal_tc2_escape",
    "quartic_tc2_d4_registered_operator_origin_no_go",
    "quartic_tc2_d4_topology_changing_origin_classification",
    "quartic_tc2_d4_curl_constraint_admission",
    "quartic_tc2_d4_curl_companion_range",
    "quartic_tc2_d4_axis2_base_rhs",
    "quartic_tc2_d4_spatial_gradient_annihilator_no_go",
    "quartic_tc2_d4_full_linear_gradient_annihilator_no_go",
    "quartic_tc2_d4_parity_cubic_angular_escape",
    "quartic_tc2_d4_parity_cubic_generic_direction",
    "quartic_tc2_d4_matrix_curl_rank_one_completion",
    "quartic_tc2_d4_degree_three_matrix_curl_sphere_extension",
    "quartic_tc2_d4_degree_three_c23_great_circle_escape",
    "quartic_tc2_d4_degree_three_rank_two_xyz_completion",
    "quartic_tc2_d4_degree_three_sixth_frame_completion",
    "quartic_tc2_d4_rational_chart_determining_gate",
    "quartic_tc2_d4_degree_five_counterexample_escape",
    "quartic_tc2_d4_revised_symbol_rational_counterexample",
    "quartic_tc2_d4_revised_eight_frame_rational_counterexample",
    "quartic_tc2_d4_revised_nine_frame_rational_counterexample",
    "quartic_tc2_reranked_obligation_chunk_0",
    "quartic_tc2_reranked_obligation_chunk_64",
    "quartic_tc2_reranked_obligation_chunk_128",
    "quartic_tc2_reranked_obligation_chunk_192",
    "quartic_tc2_reranked_obligation_chunk_256",
    "quartic_tc2_reranked_obligation_chunk_320",
    "quartic_tc2_reranked_obligation_chunk_384",
    "quartic_tc2_reranked_obligation_checkpoint",
    "quartic_tc2_reranked_obligation_status",
    "quartic_tc2_mixed_third_jet_chunk",
    "quartic_tc2_mixed_third_jet_chunk_64",
    "quartic_tc2_mixed_third_jet_chunk_128",
    "quartic_tc2_mixed_third_jet_checkpoint",
    "quartic_tc2_mixed_third_jet_continuation_status",
    "quartic_tc2_mixed_third_jet_parallel_chunk_192",
    "quartic_tc2_mixed_third_jet_parallel_chunk_256",
    "quartic_tc2_mixed_third_jet_parallel_chunk_320",
    "quartic_tc2_mixed_third_jet_parallel_chunk_384",
    "quartic_tc2_mixed_third_jet_parallel_chunk_448",
    "quartic_tc2_mixed_third_jet_parallel_chunk_512",
    "quartic_tc2_mixed_third_jet_parallel_chunk_576",
    "quartic_tc2_mixed_third_jet_parallel_chunk_640",
    "quartic_tc2_mixed_third_jet_parallel_chunk_704",
    "quartic_tc2_mixed_third_jet_parallel_chunk_768",
    "quartic_tc2_mixed_third_jet_parallel_chunk_832",
    "quartic_tc2_mixed_third_jet_parallel_chunk_896",
    "quartic_tc2_mixed_third_jet_parallel_chunk_960",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1024",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1088",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1152",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1216",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1280",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1344",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1408",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1472",
    "quartic_tc2_mixed_third_jet_parallel_chunk_1536",
    "quartic_tc2_mixed_third_jet_parallel_checkpoint",
    "quartic_tc2_mixed_third_jet_parallel_status",
    "quartic_tc2_mixed_third_jet_parallel_supervisor_readiness",
    "unified_live_dashboard_service_readiness",
    "unified_live_dashboard_service_safety_readiness",
]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    specs = []
    for label, rel in zip(LABELS, SOURCE_PATHS, strict=True):
        source = REPO / rel
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        raw = target.read_bytes()
        value = json.loads(raw)
        claimed = value.get("content_sha256")
        if claimed is None:
            claimed = hashlib.sha256(_canonical(value)).hexdigest()
        specs.append(
            {
                "label": label,
                "path": rel,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": claimed,
            }
        )
    for config_rel in RECOVERY_CONFIG_PATHS:
        source_config = REPO / config_rel
        target_config = tmp_path / config_rel
        target_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_config, target_config)
        recovery_config = json.loads(source_config.read_text(encoding="utf-8"))
        for binding in recovery_config["predecessors"].values():
            predecessor_rel = binding["path"]
            predecessor_source = REPO / predecessor_rel
            predecessor_target = tmp_path / predecessor_rel
            predecessor_target.parent.mkdir(parents=True, exist_ok=True)
            if not predecessor_target.exists():
                shutil.copyfile(predecessor_source, predecessor_target)
    for relative in PORTABLE_FORMAL_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in CONTINUOUS_PIPELINE_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    leaderboard_config = json.loads(
        (REPO / "configs/scientific_leaderboards.json").read_text(encoding="utf-8")
    )
    for binding in leaderboard_config["sources"].values():
        relative = binding["path"]
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    for relative in FINITE_SOBOLEV_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in FULL_TENSOR_RECONCILIATION_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in SCALAR_HESSIAN_D2_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in SCALAR_HESSIAN_CURL_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in SCALAR_HESSIAN_OUTPUT_BUNDLE_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in FULL_D2F_HIGH_ATOM_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in PRINCIPAL_HIGH_ATOM_CONNECTION_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in REVISED_SYMBOL_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in REVISED_EIGHT_FRAME_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    revised_eight_config_source = REPO / REVISED_EIGHT_FRAME_CONFIG_PATH
    revised_eight_config_target = tmp_path / REVISED_EIGHT_FRAME_CONFIG_PATH
    revised_eight_config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(revised_eight_config_source, revised_eight_config_target)
    revised_eight_config = json.loads(
        revised_eight_config_source.read_text(encoding="utf-8")
    )
    for key in (
        "campaign_source",
        "campaign_test",
        "revised_predecessor",
        "degree_five_predecessor",
        "rational_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    ):
        binding = revised_eight_config[key]
        relative = binding["path"]
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    for relative in REVISED_NINE_FRAME_DEPENDENCIES:
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    revised_nine_config_source = REPO / REVISED_NINE_FRAME_CONFIG_PATH
    revised_nine_config_target = tmp_path / REVISED_NINE_FRAME_CONFIG_PATH
    revised_nine_config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(revised_nine_config_source, revised_nine_config_target)
    revised_nine_config = json.loads(
        revised_nine_config_source.read_text(encoding="utf-8")
    )
    for key in (
        "campaign_source",
        "campaign_test",
        "nine_frame_predecessor",
        "revised_predecessor",
        "degree_five_predecessor",
        "rational_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    ):
        binding = revised_nine_config[key]
        relative = binding["path"]
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    sixth_config_source = REPO / SIXTH_FRAME_CONFIG_PATH
    sixth_config_target = tmp_path / SIXTH_FRAME_CONFIG_PATH
    sixth_config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sixth_config_source, sixth_config_target)
    sixth_config = json.loads(sixth_config_source.read_text(encoding="utf-8"))
    for key in (
        "campaign_source",
        "campaign_test",
        "c23_predecessor",
        "fourth_campaign",
        "minimal_escape",
        "xyz_predecessor",
    ):
        relative = sixth_config[key]["path"]
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    rational_config_source = REPO / RATIONAL_CHART_CONFIG_PATH
    rational_config_target = tmp_path / RATIONAL_CHART_CONFIG_PATH
    rational_config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rational_config_source, rational_config_target)
    rational_config = json.loads(rational_config_source.read_text(encoding="utf-8"))
    for key in (
        "campaign_source",
        "campaign_test",
        "c23_predecessor",
        "fourth_campaign",
        "minimal_escape",
        "sixth_predecessor",
        "xyz_predecessor",
    ):
        relative = rational_config[key]["path"]
        source = REPO / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    database = tmp_path / "runs/campaigns/watchdog.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE campaigns (
          campaign_id TEXT, state TEXT, deadline_utc TEXT, max_tasks INTEGER,
          tasks_started INTEGER, tasks_succeeded INTEGER, tasks_failed INTEGER,
          max_cycles INTEGER, cycles_completed INTEGER, stop_reason TEXT
        );
        CREATE TABLE tasks (task_type TEXT, status TEXT);
        CREATE TABLE candidates (status TEXT);
        CREATE TABLE evidence (outcome TEXT);
        CREATE TABLE llm_budgets (
          limit_microusd INTEGER, reserved_microusd INTEGER, spent_microusd INTEGER,
          max_calls INTEGER, calls_started INTEGER, calls_completed INTEGER
        );
        CREATE TABLE events (created_utc TEXT);
        INSERT INTO campaigns VALUES
          ('fixture','active','2026-08-21T00:00:00+00:00',100,4,1,0,8,1,NULL);
        INSERT INTO tasks VALUES ('covariant_lift','queued'),('llm_research','running'),
          ('candidate_dossier','deferred');
        INSERT INTO candidates VALUES ('active'),('rejected'),('deferred');
        INSERT INTO evidence VALUES ('pass'),('reject'),('unresolved');
        INSERT INTO llm_budgets VALUES (500000000,0,1250000,250,2,2);
        INSERT INTO events VALUES ('2026-08-10T20:00:00+00:00');
    """)
    connection.commit()
    connection.close()
    config = {
        "watchdog_database": "runs/campaigns/watchdog.sqlite",
        "watchdog_stale_after_seconds": 1800,
        "sources": specs,
    }
    return tmp_path, config, database


def test_hardened_live_service_checkpoint_is_hash_and_route_bound(tmp_path: Path) -> None:
    config_path = tmp_path / "configs/unified_engine_live_service_safety.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    expected_tail = [
        "-m",
        "sigma_theory_compiler.unified_engine_live_service_safety",
        "worker",
        "--project-root",
        str(tmp_path.resolve()),
        "--config",
        "configs/unified_engine_live_service_safety.json",
    ]
    body = {
        "schema_version": "sigma-unified-engine-live-service-safety-checkpoint-1.0",
        "runtime_epoch": "unified-live-dashboard-safety-epoch-test",
        "runtime_directory": "runs/engine/unified-live-dashboard-safety-service",
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "worker_argv_sha256": hashlib.sha256(_canonical(expected_tail)).hexdigest(),
        "state": "stopped",
        "pid": None,
        "refresh_count": 3,
        "consecutive_failures": 0,
        "reload_count": 0,
        "last_error": None,
        "last_refresh": None,
        "stop_reason": "external_stop_requested",
    }
    checkpoint = {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    checkpoint_path = tmp_path / "runs/engine/unified-live-dashboard-safety-service/checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    config = {
        "unified_live_service_checkpoint": checkpoint_path.relative_to(tmp_path).as_posix(),
        "unified_live_service_config": config_path.relative_to(tmp_path).as_posix(),
    }

    status = _read_unified_live_service_status(tmp_path, config)
    assert status["implementation"] == "hardened_safety"
    assert status["config_current"] is True
    assert status["alive"] is False
    assert status["pid_identity_verified"] is False

    checkpoint["runtime_directory"] = "runs/engine/other-runtime"
    tampered = dict(checkpoint)
    tampered.pop("content_sha256")
    checkpoint["content_sha256"] = hashlib.sha256(_canonical(tampered)).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="hardened unified live-service checkpoint"):
        _read_unified_live_service_status(tmp_path, config)


def test_deferred_gpu_handoff_checkpoint_is_hash_and_seal_bound(tmp_path: Path) -> None:
    service_config = {
        "service_id": "deferred-test",
        "service_epoch": "epoch-test",
        "runtime_directory": "runs/engine/deferred-test-runtime",
        "service_lease_name": "service.lease.json",
        "maximum_state_bytes": 131072,
        "maximum_service_cycles": 24,
        "maximum_wait_polls_per_cycle": 721,
        "required_consecutive_safe_samples": 3,
        "maximum_gpu_utilization_percent": 20,
        "minimum_free_gpu_memory_mib": 8192,
    }
    service_config_path = tmp_path / "configs/deferred.json"
    service_config_path.parent.mkdir(parents=True)
    service_config_path.write_text(json.dumps(service_config), encoding="utf-8")
    body = {
        "schema_version": "sigma-kastner-schlatter-deferred-gpu-handoff-checkpoint-1.0",
        "service_id": "deferred-test",
        "service_epoch": "epoch-test",
        "config_file_sha256": hashlib.sha256(service_config_path.read_bytes()).hexdigest(),
        "state": "stopped",
        "pid": None,
        "process_argv_sha256": None,
        "attempted_cycles": 0,
        "executed_cycles": 0,
        "polls_in_cycle": 0,
        "consecutive_safe_samples": 0,
        "last_nvml_sample": None,
        "last_scheduler_receipt": None,
        "error": None,
        "waiting_cuda_context_created": False,
        "handoff_service_direct_sqlite_accessed": False,
        "live_campaign_sqlite_accessed": False,
        "existing_process_signaled": False,
    }
    checkpoint = {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    checkpoint_path = tmp_path / "runs/engine/deferred-test-runtime/checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    config = {
        "deferred_gpu_handoff_checkpoint": checkpoint_path.relative_to(tmp_path).as_posix(),
        "deferred_gpu_handoff_config": service_config_path.relative_to(tmp_path).as_posix(),
    }

    status = _read_deferred_gpu_handoff_status(tmp_path, config)
    assert status["availability"] == "available"
    assert status["state"] == "stopped"
    assert status["alive"] is False
    assert status["safe_now"] is False

    checkpoint["live_campaign_sqlite_accessed"] = True
    tampered = dict(checkpoint)
    tampered.pop("content_sha256")
    checkpoint["content_sha256"] = hashlib.sha256(_canonical(tampered)).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="deferred GPU handoff checkpoint"):
        _read_deferred_gpu_handoff_status(tmp_path, config)


def test_read_only_snapshot_is_deterministic_and_does_not_mutate_database(tmp_path: Path) -> None:
    root, config, database = _fixture(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    sampled_at = datetime(2026, 8, 10, 20, 10, tzinfo=UTC)
    first = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_cpu={"availability": "available", "utilization_percent": 16.0},
        physical_gpu={"availability": "available", "utilization_percent": 4.0},
    )
    second = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_cpu={"availability": "available", "utilization_percent": 81.0},
        physical_gpu={"availability": "available", "utilization_percent": 99.0},
    )
    after = hashlib.sha256(database.read_bytes()).hexdigest()

    assert before == after
    assert first["core"] == second["core"]
    assert first["core_content_sha256"] == second["core_content_sha256"]
    assert first["volatile"] != second["volatile"]
    assert first["volatile"]["physical_cpu"]["utilization_percent"] == 16.0
    watchdog = first["core"]["campaign_watchdog"]
    assert watchdog["read_contract"] == "sqlite_uri_mode_ro_plus_query_only_transaction"
    assert watchdog["candidate_counts"] == {"active": 1, "deferred": 1, "rejected": 1}
    assert watchdog["evidence_outcome_counts"] == {"pass": 1, "reject": 1, "unresolved": 1}
    assert first["core"]["scheduler_lanes"]["llm_research"] == {
        "capacity": 4,
        "running": 1,
        "queued": 0,
        "scheduler_occupancy_fraction": 0.25,
    }
    assert first["volatile"]["campaign_watchdog_freshness"]["stale"] is False
    assert first["core"]["llm"]["spent_usd"] == 1.25
    assert first["core"]["llm"]["proposal_adapter"] == {
        "default_paid_calls_enabled": False,
        "maximum_call_usd": "5.000000",
        "maximum_total_usd": "500.000000",
        "network_calls_made": 0,
        "output_status": "quarantine_until_downstream_validation",
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_no_network_no_spend",
    }
    assert first["core"]["llm"]["campaign_bridge"] == {
        "admission_callback_configured": False,
        "campaign_task_type": "reviewed_llm_formula_proposal",
        "compiler_tasks_enqueued": 0,
        "default_execution_enabled": False,
        "network_calls_made": 0,
        "paid_spend_usd": "0.000000",
        "raw_body_persistence": False,
        "status": "ready_disabled_quarantine_only",
    }
    assert first["core"]["llm"]["typed_dsl_admission"] == {
        "compiler_queue_task_type": "reviewed_covariant_compiler_admission",
        "default_execution_enabled": False,
        "fixture_expected_counts": {
            "block": 1,
            "enqueue": 1,
            "pass": 1,
            "reject": 9,
        },
        "formula_body_persistence": False,
        "status": "ready_disabled_hash_only",
    }
    assert first["core"]["llm"]["compiler_registry_bridge"] == {
        "candidate_body_persistence": False,
        "default_execution_enabled": False,
        "fixture_expected_counts": {
            "block": 1,
            "dedup": 1,
            "enqueue": 1,
            "pass": 1,
            "reject": 7,
        },
        "next_stage_adapter_registered": False,
        "novelty_claim_allowed": False,
        "status": "ready_disabled_hash_only",
    }
    assert first["core"]["llm"]["reviewed_local_epoch"] == {
        "default_execution_enabled": False,
        "expected_bounded_status": {
            "candidate_count": 1,
            "compiler_receipt_pass_count": 2,
            "decision_counts": {
                "block": 1,
                "dedup": 1,
                "pass": 1,
                "reject": 1,
            },
            "network_calls": 0,
            "next_stage_enqueue_count": 1,
            "paid_spend_usd": "0.000000",
            "policy_pass_count": 1,
            "proposal_quarantine_count": 4,
        },
        "formula_body_persistence": False,
        "network_calls": 0,
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_bounded_mock_only",
    }
    assert first["core"]["llm"]["reviewed_local_service"] == {
        "budgets": {
            "maximum_attempts": 3,
            "maximum_disk_bytes": 100_000_000,
            "maximum_tasks": 1,
            "maximum_wall_seconds": 120,
        },
        "default_execution_enabled": False,
        "deterministic_export": True,
        "network_allowed": False,
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_bounded_local_only",
    }
    assert first["core"]["g4_solar_evaluator"] == {
        "candidate_id": "G3-f9c598b70a77ea54009d8f18",
        "decision": "blocked",
        "descriptor_implementation_ready": True,
        "durable_execution": {
            "decision_counts": {"blocked": 1},
            "reviewed_evaluator_invocation_count": 1,
            "task_count": 1,
            "work_state_counts": {"succeeded": 1},
        },
        "filled_registration_hash_count": 1,
        "first_missing_premise": (
            "registered_real_source_interval_and_trace_tail_prediction_bundle"
        ),
        "missing_registration_hash_count": 16,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "synthetic_GR_golden_pass_count": 5,
    }
    assert first["core"]["g4_galaxy_evaluator"] == {
        "candidate_id": "G3-f9c598b70a77ea54009d8f18",
        "decision": "blocked",
        "descriptor_implementation_ready": True,
        "durable_execution": {
            "decision_counts": {"blocked": 1},
            "reviewed_evaluator_invocation_count": 1,
            "task_count": 1,
            "work_state_counts": {"succeeded": 1},
        },
        "filled_registration_hash_count": 1,
        "first_missing_premise": "registered_action_bound_galaxy_prediction_bundle",
        "missing_registration_hash_count": 17,
        "object_specific_gravity_parameter_count": 0,
        "observational_data_opened": False,
        "prediction_bundle_registered": False,
        "primary_record_access_count": 0,
        "synthetic_control_decisions": {"covariance": "pass", "shape": "pass"},
        "forward_model": {
            "analytic_known_answer_pass_count": 3,
            "covariance_control": "pass",
            "decision": "blocked",
            "filled_registration_hash_count": 3,
            "first_missing_premise": "registered_baryonic_source_and_data_contracts",
            "missing_registration_hash_count": 15,
            "newly_filled_fields": [
                "lensing_prediction_implementation_sha256",
                "rotation_prediction_implementation_sha256",
            ],
            "object_specific_gravity_parameter_count": 0,
            "observational_data_opened": False,
            "prediction_bundle_registered": False,
        },
        "registration": {
            "branch_contract_status": "certified_exact_conditional_branch",
            "decision": "blocked",
            "distance_geometry_contract_status": ("certified_interface_no_real_values"),
            "filled_registration_hash_count": 11,
            "first_missing_premise": ("registered_real_source_manifest_and_selected_primary_roots"),
            "missing_registration_hash_count": 7,
            "newly_filled_fields": [
                "prediction_bundle_contract_sha256",
                "raw_to_calibrated_transform_sha256",
            ],
            "held_out_split_policy_registered_as_evidence": True,
            "object_specific_gravity_parameter_count": 0,
            "observational_data_opened": False,
            "prediction_bundle_registered": False,
            "real_source_geometry_registered": False,
            "real_split_commitment_registered": False,
            "real_transform_inputs_registered": False,
            "source_specific_branch_selection_proven": False,
            "manifest_bundle_tooling": {
                "decision": "blocked",
                "enabled": False,
                "filled_registration_hash_count": 11,
                "first_missing_premise": (
                    "external_registered_source_manifest_and_independent_registry_receipt"
                ),
                "missing_registration_hash_count": 7,
                "newly_filled_fields": [],
                "synthetic_bundle_registration_admissible": False,
                "synthetic_manifest_registration_admissible": False,
            },
            "source_registry_admission": {
                "decision": "blocked",
                "enabled": False,
                "filled_registration_hash_count": 11,
                "first_missing_premise": ("explicit_registered_source_opening_authorization"),
                "missing_registration_hash_count": 7,
                "newly_filled_fields": [],
                "source_opening_permission_registered": False,
                "source_records_admitted": 0,
                "target_records_opened": 0,
            },
        },
    }
    assert "C:\\" not in json.dumps(first)


def test_parallel_proof_supervisor_is_volatile_and_fail_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    body = {
        "schema_version": "sigma-quartic-tc2-mixed-third-jet-parallel-supervisor-state-1.0",
        "state": "running",
        "pid": 1234,
        "alive": True,
        "epochs_completed": 3,
        "chunks_advanced": 3,
        "next_offset": 512,
        "remaining_mixed_triples": 11_788,
        "prior_resume_sha256": "a" * 64,
        "stop_reason": None,
        "claims": {
            "full_mixed_sector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
    }
    status = {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path = root / "runs/physics-language/proof-supervisor/supervisor-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config["mixed_third_jet_supervisor_status"] = path.relative_to(root).as_posix()
    snapshot = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 20, 10, tzinfo=UTC),
        physical_cpu={"availability": "unavailable"},
        physical_gpu={"availability": "unavailable"},
    )
    runtime = snapshot["volatile"]["mixed_third_jet_supervisor"]
    assert runtime["availability"] == "available"
    assert runtime["state"] == "running"
    assert runtime["next_offset"] == 512
    assert runtime["remaining_mixed_triples"] == 11_788
    assert "mixed_third_jet_supervisor" not in snapshot["core"]

    body["claims"]["global_H7_closed"] = True
    tampered = {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not fail-closed"):
        build_unified_snapshot(root, config)


def test_future_not_before_work_is_scheduled_idle_then_stale(tmp_path: Path) -> None:
    root, config, database = _fixture(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE tasks ADD COLUMN not_before_utc TEXT")
    connection.execute(
        "UPDATE tasks SET not_before_utc = ? "
        "WHERE task_type = 'covariant_lift' AND status = 'queued'",
        ("2026-08-10T21:00:00+00:00",),
    )
    connection.commit()
    connection.close()

    scheduled = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 20, 40, tzinfo=UTC),
        physical_gpu={"availability": "unavailable"},
    )
    overdue = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 21, 40, tzinfo=UTC),
        physical_gpu={"availability": "unavailable"},
    )

    assert scheduled["core"] == overdue["core"]
    assert scheduled["core_content_sha256"] == overdue["core_content_sha256"]
    cpu_before = scheduled["volatile"]["scheduler_readiness"]["cpu_symbolic"]
    assert cpu_before == {
        "queued_total": 1,
        "runnable_now": 0,
        "delayed_until_not_before": 1,
        "earliest_future_not_before_utc": "2026-08-10T21:00:00+00:00",
    }
    freshness_before = scheduled["volatile"]["campaign_watchdog_freshness"]
    assert freshness_before["state"] == "scheduled_idle"
    assert freshness_before["stale"] is False
    assert freshness_before["expected_next_event_not_before_utc"] == ("2026-08-10T21:00:00+00:00")
    assert freshness_before["freshness_deadline_utc"] == ("2026-08-10T21:30:00+00:00")

    cpu_after = overdue["volatile"]["scheduler_readiness"]["cpu_symbolic"]
    assert cpu_after == {
        "queued_total": 1,
        "runnable_now": 1,
        "delayed_until_not_before": 0,
        "earliest_future_not_before_utc": None,
    }
    freshness_after = overdue["volatile"]["campaign_watchdog_freshness"]
    assert freshness_after["state"] == "stale"
    assert freshness_after["stale"] is True
    assert freshness_after["expected_next_event_not_before_utc"] == ("2026-08-10T21:00:00+00:00")
    assert freshness_after["freshness_deadline_utc"] == ("2026-08-10T21:30:00+00:00")
    assert freshness_after["stale_source_reason"] == ("no_event_by_2026-08-10T21:30:00+00:00")


def test_stage_counts_and_missing_evaluator_blockers_are_not_collapsed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    result = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 22, tzinfo=UTC),
        physical_gpu={"availability": "unavailable", "reason": "fixture"},
    )
    core = result["core"]
    assert core["billion_formula_streaming"]["sampled_static_stage"]["pass"] == 5855
    assert core["billion_formula_streaming"]["sampled_static_stage"]["normalized_outcomes"] == {
        "pass": 5855,
        "reject": None,
        "block": 0,
    }
    cpu_overlap = core.pop("cpu_real_formula_overlap_benchmark")
    assert cpu_overlap["decision"] == (
        "real_formula_cpu_overlap_completed_target_met_no_policy_promotion"
    )
    assert cpu_overlap["coverage"]["unique_formula_count"] == 65_536
    assert cpu_overlap["coverage"]["candidate_grid_evaluations"] == 22_478_848
    assert cpu_overlap["coverage"]["signed_term_hessian_accumulations"] == 134_873_088
    assert cpu_overlap["contract"]["worker_stages"] == [15, 16]
    assert cpu_overlap["contract"]["gpu_workers"] == 0
    assert cpu_overlap["cross_stage_replay_equal"] is True
    assert (
        cpu_overlap["stages"][0]["partition_independent_status_root_sha256"]
        == (cpu_overlap["stages"][1]["partition_independent_status_root_sha256"])
    )
    assert cpu_overlap["stages"][0]["cpu_percent_median"] == 74.3
    assert cpu_overlap["stages"][1]["cpu_percent_median"] == 89.4
    assert cpu_overlap["stages"][1]["cpu_percent_peak"] == 100.0
    assert cpu_overlap["cpu_target_met"] is True
    assert cpu_overlap["resource_backoff_triggered"] is True
    assert cpu_overlap["resource_policy_promoted"] is False
    assert cpu_overlap["scientific_pass"] is False
    assert not any(cpu_overlap["seals"].values())
    portable_formal = core.pop("formal_controls_portable_report")
    assert portable_formal["decision"] == "pass_portable_semantic_projection_118_controls"
    assert portable_formal["counts"] == {"total": 118, "passed": 118, "failed": 0}
    assert portable_formal["portability"] == {
        "absolute_windows_paths": 0,
        "absolute_wsl_paths": 0,
        "host_timestamps_in_semantic_projection": 0,
        "backend_paths_replaced_by_binary_hashes": True,
        "project_paths_replaced_by_repository_relative_paths": True,
        "same_semantics_reproduce_across_root_and_timestamp_changes": True,
    }
    assert not any(portable_formal["claim_seals"].values())
    continuous = core.pop("continuous_scientific_pipeline_admission")
    assert continuous["decision"] == (
        "admission_state_machine_ready_continuous_service_loop_not_implemented_not_started"
    )
    assert continuous["stage_order"] == [
        "generate_and_screen",
        "formal_validate",
        "rank_project",
    ]
    assert continuous["resource_contract"] == {
        "CPU_backoff_at_or_above_percent": 92,
        "CPU_generation_workers": 15,
        "GPU_lane_independent_deferred_single_owner": True,
        "maximum_actions_per_cycle": 1,
        "minimum_available_ram_mib": 32_768,
    }
    assert continuous["counts"] == {
        "GPU_owners_acquired": 0,
        "databases_created_or_opened": 0,
        "fail_closed_waits": 4,
        "formal_validation_actions": 1,
        "generation_actions": 1,
        "ranking_rebuild_actions": 1,
        "scenario_controls": 7,
        "scientific_or_ranking_passes_promoted": 0,
        "services_started": 0,
    }
    assert continuous["scenario_controls"]["formal_complete_rebuild"]["action"] == ("rank_project")
    assert (
        continuous["scenario_controls"]["formal_complete_rebuild"]["direct_rank_assignment"]
        is False
    )
    assert not any(continuous["seals"].values())
    service = core.pop("continuous_scientific_pipeline_service")
    assert service["decision"] == ("preexecution_preregistration_snapshot_no_current_runtime_claim")
    assert service["service_contract"] == {
        "single_owner_O_EXCL_PID_argv_lease": True,
        "isolated_atomic_JSON_queue": True,
        "checkpoint_resume": True,
        "external_stop_request": True,
        "maximum_actions_per_cycle": 1,
        "hard_owned_child_formal_timeout": True,
        "hard_owned_child_generation_timeout": True,
        "timeout_cleanup_is_campaign_owned_child_only": True,
        "maximum_cycles": 16,
        "maximum_service_seconds": 600,
        "maximum_action_seconds": 120,
    }
    assert service["resource_contract"] == {
        "cpu_workers": 15,
        "gpu_workers": 0,
        "CPU_backoff_percent": 92,
        "minimum_RAM_MiB": 32_768,
    }
    assert service["scientific_contract"] == {
        "candidate_manifest_reconstructed_from_ordinals": True,
        "covariant_action_health_executed_only_after_exact_mapping": True,
        "formal_backend": "candidate_bound_covariant_action_health_v1",
        "formal_backend_available": True,
        "real_formula_evaluator_allowlisted": True,
        "formal_stage_fails_closed": True,
        "ranking_is_request_only": True,
        "direct_rank_assignment": False,
    }
    assert service["execution_state"] == {
        "service_started_at_preregistration": False,
        "cycles_executed_at_preregistration": 0,
        "queue_created_at_preregistration": False,
        "current_runtime_status_claimed": False,
        "live_SQLite_accessed": False,
    }
    assert service["snapshot_scope"] == {
        "artifact_role": "preexecution_preregistration",
        "completed_execution_reported_separately": True,
        "completed_execution_result_path": (
            "runs/engine/continuous-scientific-pipeline-service-result.json"
        ),
        "runtime_status_asserted": False,
    }
    assert service["first_remaining_blocker"] == (
        "complete_candidate_specific_comparable_evidence_after_covariant_action_health_before_any_rank_rebuild_can_be_admitted"
    )
    assert not any(service["seals"].values())
    service_result = core.pop("continuous_scientific_pipeline_service_result")
    assert service_result["coverage"] == {
        "start_ordinal": 1_000_080_896,
        "stop_ordinal_exclusive": 1_004_013_056,
        "unique_formula_count": 3_932_160,
        "real_CPU_batches": 8,
        "workers_per_batch": 15,
        "formulas_per_worker": 32_768,
    }
    assert service_result["outcomes"] == {
        "sampled_static_reject_batches": 5,
        "sampled_static_pass_batches": 3,
        "formal_receipts": 3,
        "formal_blocks": 3,
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
    }
    assert len(service_result["replay_dependency_root_sha256"]) == 64
    epoch_003 = core.pop("continuous_scientific_pipeline_epoch_003_genesis")
    assert epoch_003["decision"] == (
        "disjoint_epoch_genesis_ready_for_persistent_resume_not_executed"
    )
    assert epoch_003["epoch_id"] == "continuous-scientific-pipeline-epoch-003"
    assert epoch_003["predecessor"]["stop_ordinal_exclusive"] == 1_004_013_056
    assert epoch_003["coverage"]["start_ordinal"] == 1_004_013_056
    assert epoch_003["coverage"]["stop_ordinal_exclusive"] == 1_007_945_216
    assert epoch_003["coverage"]["unique_formula_count"] == 3_932_160
    assert len(epoch_003["coverage"]["intervals"]) == 8
    assert epoch_003["execution_state"] == {
        "runtime_materialized": False,
        "formulas_evaluated": 0,
        "formal_receipts": 0,
        "epoch_complete": False,
    }
    assert not any(epoch_003["promotion_contract"].values())
    assert not any(epoch_003["seals"].values())
    epoch_003_result = core.pop("continuous_scientific_pipeline_epoch_003_result")
    assert epoch_003_result["decision"] == "bounded_epoch_complete_fail_closed_no_promotion"
    assert epoch_003_result["coverage"]["unique_formula_count"] == 3_932_160
    assert epoch_003_result["coverage"]["real_CPU_batches"] == 8
    assert epoch_003_result["outcomes"] == {
        "formal_blocks": 6,
        "formal_passes": 0,
        "formal_receipts": 6,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
        "sampled_static_pass_batches": 6,
        "sampled_static_reject_batches": 2,
    }
    assert epoch_003_result["runtime_binding"]["cycles"] == 23
    assert epoch_003_result["runtime_binding"]["terminal_state"] == "bounded_complete"
    assert not any(epoch_003_result["promotion_contract"].values())
    assert not any(epoch_003_result["seals"].values())
    candidate_followup = core.pop("continuous_scientific_pipeline_epoch_003_candidate_followup")
    assert candidate_followup["decision"] == (
        "candidate_specific_followup_blocked_no_comparable_evidence_no_promotion"
    )
    assert candidate_followup["source_pass_batch_indices"] == [1, 2, 3, 5, 6, 7]
    assert candidate_followup["counts"] == {
        "source_pass_batches": 6,
        "source_survivor_candidates": 11_439,
        "durably_recorded_candidates": 192,
        "sample_complete_batches": 0,
        "symbolic_local_preflight_passes": 26,
        "covariant_action_mapped_candidates": 0,
        "action_health_executions": 0,
        "candidate_blocks": 0,
        "candidate_rejects": 192,
        "candidate_passes": 0,
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
        "candidate_promotions": 0,
    }
    assert candidate_followup["first_remaining_blocker"] == (
        "candidate_manifest_is_bounded_not_complete"
    )
    assert candidate_followup["complete_comparable_evidence"] is False
    assert len(candidate_followup["candidate_decision_records_root_sha256"]) == 64
    assert not any(candidate_followup["promotion_contract"].values())
    assert not any(candidate_followup["seals"].values())
    assert not any(service_result["seals"].values())
    aether_boundary = core["einstein_aether_coupling_boundary_kkt"]
    assert aether_boundary["decision_counts"] == {"blocked": 1, "pass": 0, "reject": 0}
    assert aether_boundary["gate_counts"] == {
        "D_only_ambient_singular_constrained_full_rank_witnesses": 2,
        "candidate_or_theory_reject": 0,
        "five_mode_linear_positivity_chart_bindings": 1,
        "generic_symbolic_determinant_identities_pass": 5,
        "global_nonlinear_stability_pass": 0,
        "observational_pass": 0,
        "true_constrained_rank_boundary_witnesses": 3,
    }
    assert aether_boundary["symbolic_factorization"]["identity_checks"] == {
        "KKT_11x11": True,
        "KKT_equals_minus_four_tangent": True,
        "ambient_10x10": True,
        "tangent_9x9": True,
        "unit_normality_rational": True,
    }
    assert (
        aether_boundary["exact_witnesses"]["D_only_inside_five_mode_positivity_chart"]["KKT_rank"]
        == 11
    )
    assert (
        aether_boundary["exact_witnesses"]["true_constrained_boundaries"]["c14_equals_zero"][
            "KKT_rank"
        ]
        == 8
    )
    transactional = core["transactional_gravity_proposal"]
    assert transactional["decision"] == "blocked"
    assert transactional["first_blocker"] == (
        "no_paper_derivation_of_candidate_action_or_transaction_intensity_dynamics"
    )
    assert transactional["equation_preflight_counts"] == {"pass": 7, "reject": 0, "block": 1}
    assert transactional["equation_graph"]["counts"]["nodes"] == 54
    assert transactional["equation_graph"]["counts"]["edges"] == 137
    assert transactional["equation_35_normalization_gate"]["exact_ratio_middle_to_printed"] == "2"
    assert transactional["cuda_consequence_campaign"]["counts"]["poisson_samples"] == 1_572_864
    assert (
        transactional["cuda_consequence_campaign"]["counts"]["gpu_measured_consequence_evaluations"]
        == 17_179_869_184
    )
    readiness = transactional["observational_readiness"]
    assert readiness["decision"] == "blocked_registration_incomplete_observations_sealed"
    assert readiness["registration_counts"]["by_status"] == {
        "forbidden": 7,
        "missing_required": 58,
        "source_blocked": 4,
        "source_registered": 19,
    }
    assert readiness["registration_counts"]["total_fields"] == 88
    assert readiness["observational_access_count"] == 0
    assert readiness["real_data_bundle_count"] == 0
    assert readiness["real_data_pass_count"] == 0
    assert readiness["theory_or_ontology_pass_count"] == 0
    assert readiness["data_seals"]["transaction_event_observations_opened"] is False
    falsification = transactional["cuda_falsification_design"]
    assert falsification["counts"] == {
        "btfr_synthetic_residual_values": 2_097_152,
        "gpu_measured_repetitions": 16_384,
        "gpu_measured_value_evaluations": 103_079_215_104,
        "observational_records_accessed": 0,
        "poisson_synthetic_count_values": 4_194_304,
        "readiness_fields_advanced": 0,
        "scientific_tests_passed": 0,
    }
    assert falsification["poisson_power_control"]["empirical_alternative_detection_rate"] == 1.0
    assert (
        falsification["btfr_power_control"]["empirical_alternative_detection_rate"]
        == 0.999267578125
    )
    assert falsification["gpu_cpu_crosscheck"]["all_rejection_decisions_byte_equal"] is True
    assert falsification["observational_bridge"]["registration_fields_advanced"] == 0
    assert falsification["scientific_test_pass"] is False
    action_completion = transactional["candidate_action_completion"]
    assert action_completion["counts"] == {
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "normalization_branches": 2,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
        "paper_derived_actions": 0,
    }
    assert [row["beta"] for row in action_completion["completion_hypotheses"]] == [
        "1/2",
        "1/4",
    ]
    assert all(
        row["candidate_action"]["local_deterministic_action_complete"] is True
        and row["paper_authorship_or_derivation"] is False
        for row in action_completion["completion_hypotheses"]
    )
    equivalence = transactional["action_equivalence_audit"]
    assert equivalence["counts"]["canonical_dynamic_class_matches"] == 2
    assert equivalence["counts"]["new_propagating_gravity_operator_classes"] == 0
    assert equivalence["branch_comparison"]["same_propagating_operator_class"] is True
    assert equivalence["branch_comparison"]["same_constant_vacuum_energy"] is False
    formal = transactional["candidate_action_formal_admission"]
    assert formal["formal_counts"]["regular_ADM_Dirac_pass"] == 2
    assert formal["formal_counts"]["three_local_DOF_pass"] == 2
    assert formal["formal_counts"]["global_positive_energy_pass"] == 0
    assert formal["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    scalar_cuda = transactional["scalar_intensity_cuda_falsification"]
    assert scalar_cuda["counts"]["gpu_measured_scalar_consequence_evaluations"] == (137_438_953_472)
    assert scalar_cuda["counts"]["paper_qed_or_theory_passes"] == 0
    assert scalar_cuda["gpu_cpu_crosscheck"]["maximum_relative_error"] < 1e-12
    assert scalar_cuda["synthetic_only"] is True
    de_sitter = transactional["de_sitter_energy_prerequisite"]
    assert de_sitter["prerequisite_counts"]["covariant_charge_interface_pass"] == 2
    assert de_sitter["prerequisite_counts"]["fixed_background_scalar_positive_energy_pass"] == 2
    assert de_sitter["prerequisite_counts"]["nontrivial_integrable_coupled_charge_pass"] == 0
    assert de_sitter["first_blocker"] == (
        "candidate_bound_de_Sitter_boundary_conditions_zero_symplectic_flux_and_"
        "integrable_coupled_charge_not_registered"
    )
    poisson_action = transactional["poisson_action_compatibility"]
    assert poisson_action["counts"]["stationary_homogeneous_poisson_matches"] == 2
    assert poisson_action["counts"]["action_derived_point_process_measures"] == 0
    assert poisson_action["mixed_poisson_theorem"]["law_of_total_variance"] == (
        "Var(N(B))=E[mu_B]+Var(mu_B)"
    )
    assert poisson_action["exact_mixed_poisson_control"]["Fano_factor"] == "3/2"
    positivity = transactional["positive_intensity_preservation"]
    assert positivity["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert positivity["gate_counts"]["fully_coupled_constraint_satisfying_witnesses"] == 2
    assert positivity["gate_counts"]["unrestricted_positive_intensity_preservation_reject"] == 2
    assert positivity["gate_counts"]["candidate_action_reject"] == 0
    assert positivity["gate_counts"]["restricted_invariant_nonnegative_cone_pass"] == 0
    assert positivity["gate_counts"]["positive_reparameterized_action_pass"] == 0
    assert positivity["first_blocker"] == (
        "no_candidate_bound_positive_intensity_reparameterization_or_proven_invariant_"
        "nonnegative_initial_data_cone"
    )
    positive_coordinate = transactional["positive_reparameterization"]
    assert positive_coordinate["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert positive_coordinate["gate_counts"]["exact_reparameterized_action_pass"] == 2
    assert positive_coordinate["gate_counts"]["EL_equivalence_on_positive_sector_pass"] == 2
    assert positive_coordinate["gate_counts"]["regular_solution_strict_positivity_pass"] == 2
    assert positive_coordinate["gate_counts"]["paper_or_QED_positive_sector_selection_pass"] == 0
    assert positive_coordinate["gate_counts"]["action_derived_point_process_measure_pass"] == 0
    assert positive_coordinate["field_space_contract"]["map"] == "q=q0*exp(phi)"
    assert positive_coordinate["first_blocker"] == (
        "no_paper_or_QED_derived_selection_of_the_positive_field_sector_and_no_action_"
        "derived_point_process_probability_measure"
    )
    point_process = transactional["covariant_point_process_measure"]
    assert point_process["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert point_process["gate_counts"]["covariant_intensity_measure_pass"] == 2
    assert point_process["gate_counts"]["exact_covariant_nonidentifiability_witnesses"] == 2
    assert point_process["gate_counts"]["action_only_Poisson_derivation_pass"] == 0
    assert point_process["gate_counts"]["action_only_Poisson_derivation_reject"] == 2
    assert point_process["measure_domain"]["intensity_measure"] == "mu_q(B)=Integral_B q*dVol_g"
    assert (
        point_process["exact_nonidentifiability_witness"]["exact_separation"]["same_first_moment"]
        is True
    )
    assert point_process["first_blocker"] == (
        "no_registered_stochastic_generating_functional_or_QED_event_kernel_to_select_"
        "Poisson_over_a_covariant_Cox_competitor"
    )
    selector = transactional["poisson_selector_contract"]
    assert selector["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert selector["gate_counts"]["minimal_sufficient_selector_contracts"] == 3
    assert selector["gate_counts"]["registered_scalar_Poisson_PMF_assertions"] == 1
    assert selector["gate_counts"]["registered_selector_nodes"] == 0
    assert selector["gate_counts"]["registered_equations_imply_selector_reject"] == 2
    assert selector["registered_dependency_audit"]["closed_world_counts"] == {
        "edges": 137,
        "nodes": 54,
    }
    assert (
        selector["registered_dependency_audit"]["registered_equations_imply_independent_increments"]
        is False
    )
    assert selector["first_blocker"] == (
        "no_registered_derivation_of_a_set_indexed_Poisson_Laplace_functional_"
        "independent_increment_family_or_QED_counting_measure_kernel"
    )
    conditional = transactional["conditional_poisson_kernel_completion"]
    assert conditional["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert conditional["gate_counts"]["compiler_authored_conditional_kernels"] == 2
    assert conditional["gate_counts"]["conditional_Laplace_selector_pass"] == 2
    assert conditional["gate_counts"]["conditional_independent_increment_pass"] == 2
    assert conditional["gate_counts"]["paper_or_QED_actualization_derivation_pass"] == 0
    assert conditional["conditional_Poisson_kernel_contract"]["kernel"] == (
        "K[(g,phi),dN]=PRM(mu_g_phi)(dN)"
    )
    history = transactional["actualization_history_map_audit"]
    assert history["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert history["gate_counts"]["typed_map_obligations"] == 12
    assert history["gate_counts"]["paper_complete_history_to_counting_measure_maps"] == 0
    assert history["gate_counts"]["compiler_conditional_count_maps"] == 1
    assert history["compiler_conditional_count_map"]["theorem"]["countably_additive"]
    qed_poisson = transactional["qed_actualization_poisson_derivation_audit"]
    assert qed_poisson["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert qed_poisson["gate_counts"]["microscopic_derivation_obligations"] == 12
    assert qed_poisson["gate_counts"]["microscopic_obligations_closed"] == 0
    assert qed_poisson["gate_counts"]["microscopic_obligations_partial"] == 2
    assert qed_poisson["gate_counts"]["microscopic_obligations_absent"] == 10
    assert qed_poisson["gate_counts"]["compiler_conditional_sufficient_theorems"] == 1
    assert qed_poisson["gate_counts"]["exact_same_rate_non_Poisson_witnesses"] == 2
    assert qed_poisson["independent_rare_channel_Poisson_limit"]["limit_joint_PGF"] == (
        "exp(sum_i mu_i*(z_i-1))"
    )
    assert (
        qed_poisson["exact_controls"]["paired_cluster_same_rate_no_go"]["limit_variance"] == "2*mu"
    )
    assert qed_poisson["first_blocker"] == (
        "no_registered_QED_actualization_channel_probability_array_or_predictable_hazard_kernel"
    )
    compensator = transactional["deterministic_compensator_admission"]
    assert compensator["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert compensator["gate_counts"]["evidence_obligations"] == 10
    assert compensator["gate_counts"]["evidence_closed_by_compiler_hypotheses"] == 2
    assert compensator["gate_counts"]["evidence_absent"] == 8
    assert compensator["gate_counts"]["positive_candidate_mean_measures"] == 2
    assert compensator["gate_counts"]["compiler_compensator_theorem_interfaces"] == 2
    assert compensator["gate_counts"]["registered_causal_filtrations"] == 0
    assert compensator["gate_counts"]["action_or_QED_compensator_identities"] == 0
    assert compensator["gate_counts"]["exact_same_action_alternative_law_witnesses"] == 2
    assert (
        compensator["deterministic_compensator_Poisson_characterization"][
            "candidate_action_or_paper_supplies_compensator_identity"
        ]
        is False
    )
    assert (
        compensator["exact_controls"]["same_action_Poisson_Cox_nonidentifiability"][
            "Cox_variance_on_B"
        ]
        == "mu_B+mu_B^2/4"
    )
    assert compensator["first_blocker"] == (
        "no_registered_QED_probability_space_causal_filtration_or_deterministic_"
        "compensator_martingale_identity"
    )
    probability_space = transactional["canonical_probability_space"]
    assert probability_space["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert probability_space["gate_counts"]["evidence_obligations"] == 10
    assert probability_space["gate_counts"]["evidence_closed_by_compiler_construction"] == 6
    assert probability_space["gate_counts"]["evidence_absent"] == 4
    assert probability_space["gate_counts"]["compiler_canonical_configuration_spaces"] == 2
    assert probability_space["gate_counts"]["compiler_completed_causal_filtrations"] == 2
    assert probability_space["gate_counts"]["compiler_compensator_martingale_identities"] == 2
    assert probability_space["gate_counts"]["action_or_QED_stochastic_selection_rules"] == 0
    assert (
        probability_space["canonical_conditional_construction"]["probability_space"]["sample_space"]
        == "Omega=N_lf(W)"
    )
    assert (
        probability_space["exact_controls"]["same_action_Cox_nonidentifiability"][
            "Cox_variance_at_mu_2"
        ]
        == "3"
    )
    assert probability_space["first_blocker"] == (
        "no_registered_paper_QED_or_candidate_action_selection_rule_for_the_compiler_authored_"
        "Poisson_probability_space_over_same_mean_Cox_completion"
    )
    deterministic_feature = transactional["deterministic_feature_selector_no_go"]
    assert deterministic_feature["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert deterministic_feature["gate_counts"]["deterministic_feature_fibers"] == 2
    assert deterministic_feature["gate_counts"]["same_feature_Poisson_Cox_pairs"] == 2
    assert deterministic_feature["gate_counts"]["deterministic_factor_selector_no_go_theorems"] == 2
    assert deterministic_feature["gate_counts"]["registered_stochastic_features_outside_fiber"] == 0
    assert (
        "only selectors factoring through D"
        in deterministic_feature["factorization_no_go"]["scope_limit"]
    )
    assert (
        deterministic_feature["exact_controls"]["same_feature_distinct_law_witness"][
            "law_separation"
        ]["Cox_variance"]
        == "3"
    )
    second_order = transactional["second_order_selector_no_go"]
    assert second_order["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert second_order["gate_counts"]["exact_second_order_counterexamples"] == 2
    assert second_order["gate_counts"]["first_factorial_measure_matches"] == 2
    assert second_order["gate_counts"]["second_factorial_measure_matches"] == 2
    assert second_order["gate_counts"]["third_factorial_separations"] == 2
    assert second_order["second_order_no_go"]["global_pair_cumulant_measure"] == (
        "kappa_2=0, exactly as for PRM(mu)"
    )
    assert second_order["exact_controls"]["inside_count_moments"]["third_factorial_moment"] == "4"
    finite_hierarchy = transactional["finite_factorial_hierarchy_no_go"]
    assert finite_hierarchy["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert finite_hierarchy["gate_counts"]["arbitrary_finite_order_no_go_theorems"] == 1
    assert finite_hierarchy["gate_counts"]["candidate_bound_hierarchy_counterexamples"] == 2
    assert finite_hierarchy["gate_counts"]["exact_control_orders_checked"] == 6
    assert finite_hierarchy["gate_counts"]["exact_control_moment_identities_checked"] == 27
    assert finite_hierarchy["gate_counts"]["registered_nonfinite_stochastic_selectors"] == 0
    assert finite_hierarchy["finite_hierarchy_no_go"]["theorem_name"] == (
        "arbitrary_finite_factorial_hierarchy_nonidentifiability"
    )
    assert len(finite_hierarchy["exact_controls"]["orders_1_through_6"]) == 6
    countable_selector = transactional["countable_full_law_selector_admission"]
    assert countable_selector["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert countable_selector["gate_counts"]["mathematically_sufficient_countable_routes"] == 2
    assert countable_selector["gate_counts"]["compiler_route_replays"] == 4
    assert countable_selector["gate_counts"]["typed_obligations"] == 12
    assert countable_selector["gate_counts"]["closed_by_compiler_mathematics"] == 6
    assert countable_selector["gate_counts"]["absent_from_source_QED_or_action"] == 6
    assert countable_selector["gate_counts"]["source_or_QED_route_certificates"] == 0
    assert (
        countable_selector["countable_selector_admission_theorem"]["laplace_core_route"][
            "mathematically_sufficient"
        ]
        is True
    )
    assert (
        countable_selector["countable_selector_admission_theorem"]["mecke_core_route"][
            "mathematically_sufficient"
        ]
        is True
    )
    source_type_audit = transactional["source_selector_type_inhabitation_audit"]
    assert source_type_audit["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert source_type_audit["gate_counts"]["Laplace_source_complete_slots"] == 0
    assert source_type_audit["gate_counts"]["Laplace_source_partial_slots"] == 3
    assert source_type_audit["gate_counts"]["Laplace_source_absent_slots"] == 7
    assert source_type_audit["gate_counts"]["Mecke_source_complete_slots"] == 0
    assert source_type_audit["gate_counts"]["Mecke_source_partial_slots"] == 2
    assert source_type_audit["gate_counts"]["Mecke_source_absent_slots"] == 8
    assert source_type_audit["gate_counts"]["source_bound_countable_certificates"] == 0
    assert (
        source_type_audit["exact_controls"]["compiler_scalar_transform_partial_replay"][
            "qualifies_as_source_bound_core_certificate"
        ]
        is False
    )
    bridge = transactional["actualization_probability_bridge_contract"]
    assert bridge["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert bridge["gate_counts"]["primitive_source_registrations"] == 3
    assert bridge["gate_counts"]["source_complete_primitives"] == 0
    assert bridge["gate_counts"]["source_partial_primitives"] == 1
    assert bridge["gate_counts"]["source_absent_primitives"] == 2
    assert bridge["gate_counts"]["compiler_derived_bridge_objects"] == 2
    assert bridge["gate_counts"]["complete_paper_or_QED_bridges"] == 0
    assert [row["source_status"] for row in bridge["primitive_interface"]] == [
        "absent",
        "partial_semantics_only",
        "absent",
    ]
    assert bridge["compiler_derived_objects"][0]["formula"] == "P_g_phi=C_*Q_g_phi on N_lf(W)"
    assert bridge["composition_theorem"]["minimality"]["without_core_identity"] == (
        "the law is defined but Poisson versus Cox remains unresolved"
    )
    assert (
        bridge["exact_controls"]["compiler_identity_fixture_positive_control"][
            "source_or_QED_attribution"
        ]
        is False
    )
    assert [record["branch_id"] for record in bridge["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    assert all(record["candidate_decision"] == "blocked" for record in bridge["candidate_records"])
    assert not any(bridge["claim_seals"].values())
    assert not any(bridge["data_seals"].values())
    projective = transactional["history_kernel_projective_admission"]
    assert projective["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert projective["gate_counts"]["history_kernel_admission_obligations"] == 6
    assert projective["gate_counts"]["source_complete_obligations"] == 0
    assert projective["gate_counts"]["source_partial_obligations"] == 1
    assert projective["gate_counts"]["source_absent_obligations"] == 5
    assert projective["gate_counts"]["exact_same_input_distinct_history_law_witnesses"] == 2
    assert projective["extension_theorem"]["theorem_name"] == (
        "countable_projective_history_kernel_admission"
    )
    assert projective["exact_nonidentifiability_witness"]["exact_separation"] == (
        "(1+exp(-2))/2-exp(-1)=(1-exp(-1))^2/2>0"
    )
    assert [record["branch_id"] for record in projective["candidate_records"]] == [
        "eq35_middle_h",
        "eq35_printed_planck",
    ]
    assert all(
        record["candidate_decision"] == "blocked" for record in projective["candidate_records"]
    )
    assert not any(projective["claim_seals"].values())
    assert not any(projective["data_seals"].values())
    observable = transactional["transaction_event_observable_exposure"]
    assert observable["decision_counts"] == {"blocked": 2, "pass": 0, "reject": 0}
    assert observable["gate_counts"]["compiler_observation_operator_contracts"] == 1
    assert observable["gate_counts"]["exact_nonidentifiability_witnesses"] == 4
    assert observable["gate_counts"]["operational_obligations_registered"] == 0
    assert observable["gate_counts"]["real_observation_bundles"] == 0
    assert observable["identifiability_theorem"]["current_contract_satisfies_conditions"] is False
    poisson_power = transactional["poisson_cox_cuda_power"]
    assert poisson_power["counts"]["scenario_cells"] == 144
    assert poisson_power["counts"]["gpu_generated_count_values"] == 110_100_480
    assert poisson_power["counts"]["metric_replicate_values_cpu_gpu_checked"] == 1_769_472
    assert poisson_power["counts"]["scientific_tests_passed"] == 0
    assert poisson_power["counts"]["observational_records_accessed"] == 0
    assert poisson_power["gpu_cpu_crosscheck"]["all_rejection_decisions_byte_equal"]
    assert poisson_power["gpu_cpu_crosscheck"]["maximum_absolute_error"] <= 1e-10
    assert poisson_power["registered_witness_exact_sentinel"]["fano_factor"] == "3/2"
    assert poisson_power["synthetic_only"] is True
    set_indexed = transactional["set_indexed_cuda_falsification"]
    assert set_indexed["counts"]["scenario_cells"] == 48
    assert set_indexed["counts"]["gpu_generated_unique_count_values"] == 1_887_436_800
    assert set_indexed["counts"]["joint_pgf_terms_evaluated"] == 8_053_063_680
    assert set_indexed["counts"]["projection_multiply_adds"] == 322_122_547_200
    assert set_indexed["gpu_cpu_crosscheck"]["all_heldout_decisions_byte_equal"]
    assert set_indexed["exact_common_shock_sentinel"]["within_group_cross_covariance"] == "1/2"
    assert set_indexed["synthetic_only"] is True
    gpu_scheduler = transactional["set_indexed_gpu_scheduler_adapter"]
    assert gpu_scheduler["decision"] == (
        "durable_single_owner_gpu_continuous_service_ready_start_gated_not_started"
    )
    assert gpu_scheduler["scheduler_contract"]["gpu_owner_count"] == 1
    assert gpu_scheduler["scheduler_contract"]["cpu_worker_count"] == 0
    assert gpu_scheduler["scheduler_contract"]["maximum_attempts"] == 3
    assert gpu_scheduler["continuous_service_contract"]["exclusive_pid_argv_lease"] == (
        "service.lease.json"
    )
    assert gpu_scheduler["continuous_service_contract"]["maximum_service_cycles"] == 241_920
    assert gpu_scheduler["continuous_service_contract"]["gpu_start_gate"] == {
        "fails_closed_if_nvml_unavailable": True,
        "maximum_device_wide_utilization_percent": 20,
        "minimum_free_memory_mib": 8192,
    }
    assert gpu_scheduler["execution_state"] == {
        "runtime_created_by_readiness": False,
        "scheduler_started_by_readiness": False,
        "worker_result_created_by_readiness": False,
    }
    deferred_gpu = transactional["deferred_gpu_ownership"]
    assert deferred_gpu["decision"] == (
        "deferred_gpu_ownership_ready_current_device_occupied_not_started"
    )
    assert deferred_gpu["ownership_contract"]["poll_interval_seconds"] == 5
    assert deferred_gpu["ownership_contract"]["required_consecutive_safe_samples"] == 3
    assert deferred_gpu["ownership_contract"]["maximum_wait_seconds"] == 3600
    assert deferred_gpu["ownership_contract"]["maximum_polls"] == 721
    assert deferred_gpu["ownership_contract"]["maximum_gpu_utilization_percent"] == 20
    assert deferred_gpu["ownership_contract"]["minimum_free_gpu_memory_mib"] == 8192
    assert deferred_gpu["current_runtime_audit"]["ownership_reservable_now"] is False
    assert deferred_gpu["current_runtime_audit"]["nvml_sample"]["gpu_utilization_percent"] == 99
    assert deferred_gpu["current_runtime_audit"]["nvml_sample"]["memory_free_mib"] == 8083
    assert all(value is False for value in deferred_gpu["execution_state"].values())
    handoff = transactional["deferred_gpu_handoff_service"]
    assert handoff["decision"] == "deferred_handoff_ready_current_device_occupied_not_started"
    assert handoff["handoff_contract"]["required_consecutive_safe_samples"] == 3
    assert handoff["handoff_contract"]["maximum_wait_polls_per_cycle"] == 721
    assert handoff["handoff_contract"]["maximum_service_cycles"] == 24
    assert handoff["handoff_contract"]["scheduler_slice_seconds"] == 120
    assert handoff["handoff_contract"]["gpu_workers"] == 1
    assert handoff["handoff_contract"]["cpu_workers"] == 0
    assert handoff["handoff_contract"]["post_reservation_nvml_safe_recheck"] is True
    assert handoff["current_runtime_audit"]["runtime_exists"] is False
    assert handoff["current_runtime_audit"]["service_lease_exists"] is False
    assert handoff["current_runtime_audit"]["scheduler_started_by_readiness"] is False
    assert handoff["current_runtime_audit"]["nvml_sample"]["gpu_utilization_percent"] == 99
    assert handoff["current_runtime_audit"]["nvml_sample"]["memory_free_mib"] == 8088
    extended = transactional["extended_geometry_cuda_stress"]
    assert extended["counts"]["geometry_resolution_cases"] == 20
    assert extended["counts"]["gpu_measured_source_evaluation_interactions"] == 2_860_515_328
    assert extended["counts"]["extended_source_laws_registered"] == 0
    assert extended["completion_hypotheses"]["enclosed_mass"]["decision"] == (
        "blocked_not_a_registered_extended_source_law"
    )
    assert extended["completion_hypotheses"]["local_superposition"]["decision"] == (
        "hypothesis_rejected_by_exact_splitting_and_pair_balance_controls"
    )
    assert extended["lensing_rotation_consistency_gate"]["executed"] is False
    assert transactional["claim_seals"]["compiler_candidate_action_hypotheses_registered"] == 2
    assert transactional["claim_seals"]["paper_fundamental_action_registered"] is False
    assert transactional["claim_seals"]["theory_or_ontology_pass"] is False
    assert transactional["data_seals"]["synthetic_only"] is True
    assert core["promotion_overlay"]["formal"] == {"pass": 0, "reject": 70, "block": 0}
    assert core["grammar_parameter_cells"]["seed_execution"] == {
        "candidate_universe": "six reviewed deterministic seed actions",
        "deadline": "bounded_completed_artifact_no_live_deadline",
        "maximum_tasks": 6,
        "next_scaling_hook": (
            "a new hash-reviewed campaign result must register additional parameter "
            "cells before this finite range may expand beyond six"
        ),
        "normalized_scientific_outcomes": {"pass": 0, "reject": 0, "block": 6},
        "scientific_decision_counts": {"blocked": 6},
        "task_state_counts": {"succeeded": 6},
    }
    assert core["grammar_parameter_cells"]["scalable_unique_action_formal_outcomes"] == {
        "pass": 3,
        "reject": 2,
        "block": 158,
    }
    assert core["grammar_parameter_cells"]["scalable_admitted_family_formal_outcomes"] == {
        "pass": 2,
        "reject": 2,
        "block": 158,
    }
    assert core["grammar_parameter_cells"]["scalable_preflight_blocked_excluded_count"] == 1
    assert (
        core["grammar_parameter_cells"]["scalable_preflight_blocked_followup_resolved_count"] == 1
    )
    assert core["grammar_parameter_cells"]["expansion_service"] == {
        "chunk_count": 3,
        "decision_counts": {"blocked": 6},
        "parameter_cell_count": 6,
        "scientific_scope": (
            "execution scaling only; no cells beyond the reviewed manifest are inferred"
        ),
        "work_state_counts": {"succeeded": 3},
    }
    reviewed_manifest = core["grammar_parameter_cells"]["reviewed_manifest"]
    assert reviewed_manifest["parameter_cell_count"] == 256
    assert reviewed_manifest["chunk_count"] == 8
    assert reviewed_manifest["family_cell_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 32,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 64,
    }
    assert reviewed_manifest["formal_evaluation_performed"] is False
    assert reviewed_manifest["scientific_decision_counts"] == {}
    future_chunk = core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]
    assert {
        key: future_chunk[key]
        for key in (
            "input_cell_count",
            "disposition_counts",
            "preflight",
            "family_followup",
        )
    } == {
        "input_cell_count": 32,
        "disposition_counts": {
            "admitted_new_candidate": 19,
            "deduplicated_existing_candidate": 13,
        },
        "preflight": {
            "candidate_count": 19,
            "decision_counts": {"blocked": 3, "pass": 14, "reject": 2},
            "family_counts": {
                "AETHER_K1234_PARAMETER_CELL": 16,
                "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
            },
            "first_blocker_counts": {
                "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing": 3,
                "nonpositive_spin0_principal_numerator_c123": 2,
            },
            "full_candidate_specific_formal_completion_claimed": False,
            "promotion": {
                "automatic_downstream_enqueue_performed": False,
                "blocked_pending_exact_domain_registration": 3,
                "eligible_for_candidate_specific_formal_queue": 14,
                "rejected_before_candidate_specific_formal_queue": 2,
            },
        },
        "family_followup": {
            "aether": {
                "candidate_count": 14,
                "decision_counts": {"blocked": 14},
                "formal_pass_count": 0,
                "exact_negative_local_twist_witness_count": 14,
                "witness_tilt_squared_counts": {"1": 8, "2": 4, "8": 2},
                "global_tilt_strata_counts": {
                    "finite_characteristic_foliation_present": 13,
                    "globally_noncharacteristic_for_finite_unit_tilt": 1,
                },
                "explicit_affine_ansatz_constraint_reject_count": 14,
                "nonzero_Hamiltonian_constraint_residual_count": 14,
                "nonzero_momentum_constraint_residual_count": 14,
                "undefined_AE_boundary_contribution_count": 14,
                "flat_static_global_pure_twist_AE_completion_obstructed_count": 14,
                "compact_cutoff_non_pure_twist_transition_required_count": 14,
                "normalized_transition_symmetric_gradient_norm_squared_counts": {
                    "6": 8,
                    "10": 4,
                    "34": 2,
                },
                "differentiated_Killing_system": {
                    "coefficient_rank": 18,
                    "conclusion": "partial_i partial_j A_k=0",
                    "equations": "T_kij+T_kji=0",
                    "kernel_dimension": 0,
                    "second_jet": "T_ijk=partial_i partial_j A_k=T_jik",
                    "unknown_count": 18,
                },
                "constraint_satisfying_negative_total_energy_datum_count": 0,
                "weak_field_linearized_constraint_completion_count": 14,
                "strictly_positive_compact_quadratic_energy_count": 14,
                "weak_field_negative_completed_energy_direction_count": 0,
                "finite_amplitude_nonlinear_constraint_completion_count": 0,
                "compact_finite_amplitude_Aether_seed_count": 14,
                "exact_negative_static_source_monopole_count": 14,
                "frozen_source_linearized_constraint_completion_count": 14,
                "negative_linearized_completed_boundary_energy_coefficient_count": 14,
                "registered_seed_characteristic_crossing_count": 13,
                "negative_source_family_forced_characteristic_crossing_count": 11,
                "certified_negative_characteristic_free_amplitude_window_count": 2,
                "globally_noncharacteristic_candidate_count": 1,
                "regular_ADM_implicit_lift_prerequisite_pass_count": 3,
                "uniform_Aether_Legendre_block_inverse_pass_count": 3,
                "strict_negative_source_margin_pass_count": 3,
                "typed_weighted_operator_contract_complete_count": 0,
                "declared_metric_weighted_contract_count": 3,
                "metric_reference_principal_ellipticity_pass_count": 3,
                "metric_reference_trivial_kernel_pass_count": 3,
                "registered_compact_source_right_inverse_count": 3,
                "candidate_Aether_constraint_principal_block_pass_count": 0,
                "full_coupled_Fredholm_operator_defined_count": 0,
                "weighted_full_constraint_operator_isomorphism_pass_count": 0,
                "nonlinear_Frechet_remainder_bound_pass_count": 0,
                "completed_boundary_sign_persistence_count": 0,
                "fixed_free_data_constraint_variable_classification_count": 3,
                "zero_dimensional_Aether_constraint_diagonal_block_count": 3,
                "zero_Aether_second_order_off_diagonal_columns_count": 3,
                "augmented_Aether_unknown_nonelliptic_negative_control_count": 3,
                "finite_tilt_metric_York_symbol_derived_count": 3,
                "uniform_fixed_free_data_principal_ellipticity_pass_count": 1,
                "exact_nonelliptic_York_shell_count": 2,
                "York_ansatz_reject_count": 2,
                "finite_tilt_weighted_Fredholm_isomorphism_pass_count": 0,
                "uniform_principal_symbol_inverse_bound_pass_count": 1,
                "principal_elliptic_homotopy_to_reference_pass_count": 1,
                "compact_profile_C3_weighted_jet_bound_pass_count": 1,
                "lower_order_coefficient_contract_declared_count": 1,
                "full_canonical_background_point_registered_count": 1,
                "candidate_bound_flat_chart_D_residual_DAG_registered_count": 1,
                "spatially_distributed_canonical_H_core_registered_count": 0,
                "metric_covariantized_H_D_Frechet_DAG_registered_count": 0,
                "distributed_lower_order_coefficient_registry_complete_count": 0,
                "weighted_relative_lower_order_bound_pass_count": 0,
                "full_operator_inverse_norm_pass_count": 0,
                "regular_stratum_flat_chart_H_core_contract_registered_count": 1,
                "declared_profile_global_flat_chart_H_core_registered_count": 0,
                "off_flat_metric_covariantization_registered_count": 0,
                "characteristic_shell_condition": "F**2=31",
                "characteristic_shell_rank": 7,
                "characteristic_shell_nullity": 2,
                "noncrossing_control_F_squared_margin": "6",
                "missing_weighted_contract_field_counts": {
                    "codomain_space": 3,
                    "completed_boundary_first_derivative_bound": 3,
                    "completed_boundary_second_derivative_bound": 3,
                    "domain_space": 3,
                    "full_linearized_constraint_map": 3,
                    "gauge_fixing": 3,
                    "nonlinear_second_derivative_majorant": 3,
                    "operator_perturbation_norm": 3,
                    "reference_inverse_norm": 3,
                    "seed_nonlinear_constraint_residual_norm": 3,
                    "weight_delta": 3,
                },
                "c2_plus_c3_counts": {
                    "0": 2,
                    "1/16": 3,
                    "1/32": 1,
                    "1/4": 1,
                    "1/8": 3,
                    "3/16": 2,
                    "3/32": 1,
                    "5/32": 1,
                },
                "first_blocker_counts": {
                    "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
                    "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
                    "declared_compact_seed_crosses_candidate_bound_Legendre_characteristic_shell_F2_eq31": 1,
                },
                "candidate_rejection_authorized_count": 0,
            },
            "g3": {
                "candidate_count": 3,
                "decision_counts": {"blocked": 3},
                "all_direction_single_center_pass_count": 3,
                "domain_registration_filled_field_count": 36,
                "domain_registration_missing_field_count": 0,
                "full_Delta_N_derivation_pass_count": 3,
                "nonzero_componentwise_box_pass_count": 3,
                "uniform_principal_common_cone_pass_count": 3,
                "uniform_Delta_N_coercivity_pass_count": 3,
                "periodic_distributed_Dirac_pass_count": 3,
                "AF_decaying_gradient_profile_pass_count": 3,
                "AF_principal_common_cone_profile_pass_count": 3,
                "flat_reference_constraint_ansatz_reject_count": 3,
                "nonunitary_formulation_registration_pass_count": 3,
                "nonunitary_AF_principal_pass_count": 3,
                "flat_nontrivial_reference_constraint_ansatz_reject_count": 3,
                "actual_AF_vacuum_constraint_reference_pass_count": 3,
                "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": 0,
                "radial_pure_trace_momentum_constraint_reduction_pass_count": 3,
                "radial_conformal_pure_trace_ansatz_reject_count": 3,
                "radial_Lichnerowicz_BVP_registration_pass_count": 3,
                "positive_global_radial_Lichnerowicz_solution_nonexistence_count": 3,
                "nonradial_York_Hamiltonian_reduction_pass_count": 3,
                "bounded_mean_curvature_green_comparison_pass_count": 3,
                "conformally_flat_bounded_mean_curvature_York_class_reject_count": 3,
                "candidate_millicap_frontier_registration_pass_count": 3,
                "strict_extension_beyond_kappa_6_over_5_pass_count": 3,
                "expanded_nonradial_York_class_reject_count": 3,
                "next_grid_cap_inconclusive_count": 3,
                "exact_algebraic_threshold_pass_count": 3,
                "closed_threshold_endpoint_reject_count": 3,
                "above_threshold_control_inconclusive_count": 3,
                "tracefree_compensation_bound_pass_count": 3,
                "tracefree_compensated_York_class_reject_count": 3,
                "undercompensated_control_inconclusive_count": 3,
                "general_geometry_pointwise_theorem_pass_count": 3,
                "curvature_shortfall_constraint_class_reject_count": 3,
                "exact_curvature_endpoint_inconclusive_count": 3,
                "above_threshold_not_excluded_control_count": 3,
                "nonconformally_flat_metric_construction_pass_count": 0,
                "exact_surplus_identity_pass_count": 3,
                "above_threshold_surplus_mismatch_class_reject_count": 3,
                "matched_surplus_necessary_control_count": 3,
                "overcurvature_not_excluded_control_count": 3,
                "radial_momentum_leading_order_pass_count": 3,
                "flat_Hamiltonian_leading_order_pass_count": 3,
                "joint_real_asymptotic_coefficient_solution_count": 0,
                "flat_radial_matched_constraint_class_reject_count": 3,
                "registered_AF_metric_York_datum_pass_count": 0,
                "asymptotically_flat_Dirac_pass_count": 0,
                "AF_Einstein_constraint_solution_pass_count": 0,
                "global_energy_pass_count": 0,
                "full_formal_pass_count": 0,
                "first_blocker_counts": {
                    "candidate_specific_AF_metric_York_data_beyond_flat_radial_r_minus_2_asymptotic_class": 3
                },
            },
        },
    }
    future_dossiers = future_chunk["action_dossiers"]
    assert future_dossiers["candidate_count"] == 19
    assert future_dossiers["decision_counts"] == {"blocked": 17, "reject": 2}
    assert future_dossiers["ranked_candidate_count"] == 0
    assert len(future_dossiers["records"]) == 19
    assert all(
        record["comparison_contract"]["rank"] is None
        and record["comparison_contract"]["rank_eligible"] is False
        and record["action"]["human_readable_action"]["display_kind"]
        == "verbatim_ordered_covariant_density_concatenation"
        for record in future_dossiers["records"]
    )
    structural = core["grammar_parameter_cells"]["structural_metrics"]
    assert structural["candidate_count"] == 163
    assert structural["alias_count"] == 93
    assert structural["measurement_counts"] == {"measured": 163}
    assert structural["formal_decision_counts"] == {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }
    assert structural["simplicity_pareto_front"]["candidate_ids"] == [
        "G3A-2f8983c88f504150381064f2",
        "G3A-58e59412e5fe77cd54caf863",
    ]
    assert structural["scientific_validity_inference"] is False
    explanations = core["grammar_parameter_cells"]["explanation_dossiers"]
    assert explanations["candidate_count"] == 163
    assert explanations["alias_count"] == 93
    assert explanations["formal_decision_counts"] == {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }
    assert explanations["hierarchy_node_status_counts"] == {
        "blocked": 321,
        "calibration_only": 163,
        "proven": 166,
        "rejected": 2,
    }
    assert explanations["observational_data_opened"] is False
    assert reviewed_manifest["compilation"] == {
        "candidate_decision_counts": {"blocked": 0, "pass": 163, "reject": 0},
        "compiled_action_ir_count": 256,
        "equivalent_duplicate_count": 93,
        "expensive_formal_campaign_run": False,
        "formal_decision_counts": {},
        "unique_candidate_count": 163,
        "generated_action_export": {
            "candidate_count": 163,
            "action_export_counts": {
                "exact_rendered": 163,
                "rejected": 0,
                "sandbox_parsed_and_canonicalised": 163,
            },
            "metric_variation_counts": {
                "executed_by_this_campaign": 0,
                "formal_passes_inferred": 0,
                "reviewed_adapter_routes_bound": 163,
            },
            "sandbox_backend": "wsl-local",
            "network_namespace_created": True,
            "action_export_historical_first_missing_premise": (
                "candidate_specific_metric_variation_execution_from_the_generated_"
                "action_export_for_each_action_hash_and_future_operator_family"
            ),
            "first_missing_premise": (
                "metric_variation_exporters_for_future_unregistered_nonminimal_operator_families"
            ),
            "candidate_metric_specialization": {
                "candidate_count": 163,
                "counts": {
                    "aether_formal_control_bound": 128,
                    "blocked": 0,
                    "candidate_action_hashes_specialized": 163,
                    "candidate_backend_variations_executed": 0,
                    "candidate_euler_expressions_materialized": 163,
                    "candidate_specializations_symbolically_verified": 163,
                    "exact_formula_domains_validated": 163,
                    "formal_passes_inferred": 0,
                    "rejected": 0,
                    "typed_action_hashes_replayed": 163,
                },
                "first_missing_premise": (
                    "metric_variation_exporters_for_future_unregistered_nonminimal_"
                    "operator_families"
                ),
                "scope": (
                    "candidate-specific materialization and symbolic verification of exact Euler "
                    "specializations by substitution into independently executed or reviewed "
                    "generic metric-variation theorems for every current replayed action hash; "
                    "this is not 163 independent backend variations, and no formal decision, "
                    "global-energy claim, or observational gate is changed"
                ),
            },
            "gpu_synthetic_formula_stress": {
                "campaign_decision": "completed_numerical_stress_control_only",
                "counts": {
                    "candidate_count": 163,
                    "cpu_exact_rational_crosschecks": 5216,
                    "cpu_full_projection_evaluations": 5341184,
                    "family_count": 4,
                    "formal_passes_inferred": 0,
                    "gpu_measured_candidate_formula_evaluations": 87509958656,
                    "gpu_measured_repetitions": 16384,
                    "gpu_projection_dispatches": 16392,
                    "gpu_warmup_repetitions": 8,
                    "observational_records_accessed": 0,
                    "paid_llm_calls": 0,
                    "synthetic_points_per_candidate": 32768,
                    "unique_candidate_point_pairs": 5341184,
                },
                "exact_cpu_control": {
                    "bit_equal_to_converted_exact_count": 4900,
                    "candidate_count": 163,
                    "crosscheck_count": 5216,
                    "error_bound": 5e-15,
                    "exact_result_registry_root_sha256": (
                        "2f4c6b2f17f23c913dd4ed1dfb371c59bee37360a48daf99de830df49e14df2f"
                    ),
                    "float64_max_absolute_error_after_single_reference_conversion": (
                        2.220446049250313e-16
                    ),
                    "method": (
                        "Python Fraction evaluation of every candidate on the declared "
                        "dyadic sentinel points"
                    ),
                    "points_per_candidate": 32,
                    "within_bound": True,
                },
                "gpu_cpu_comparison": {
                    "absolute_error_bound": 5e-13,
                    "bound_semantics": (
                        "a point violates only when both absolute and relative bounds are exceeded"
                    ),
                    "comparison_count": 5341184,
                    "cpu_output_sha256": (
                        "51eddb89b41fd209329f3e962d884833d8cebc461f04f30ad2485493bfc62e53"
                    ),
                    "gpu_output_sha256": (
                        "28faa9d3267f2e374e172141d02c7e64ce319c8667e5a6b24771595aafab90dc"
                    ),
                    "max_absolute_error": 2.220446049250313e-16,
                    "max_relative_error": 1.2637483275749894e-13,
                    "relative_error_bound": 5e-13,
                    "relative_floor": 1e-12,
                    "violating_point_count": 0,
                    "within_bounds": True,
                },
                "runtime_measurement": {
                    "cpu_full_projection_wall_seconds": 0.011667799961287528,
                    "device": {
                        "backend": "cupy_cuda",
                        "compute_capability": "12.0",
                        "cuda_runtime_version": 12090,
                        "cupy_version": "13.5.1",
                        "device_index": 0,
                        "device_name": "NVIDIA GeForce RTX 5090",
                        "total_global_memory_mib": 32606,
                    },
                    "gpu_allocated_input_bytes": 4478616,
                    "gpu_allocated_output_bytes": 42729472,
                    "gpu_candidate_formula_evaluations_per_second": 17790117991.44622,
                    "gpu_measured_wall_seconds": 4.919020700035617,
                    "measured_utc": "2026-08-11T14:17:39.990252+00:00",
                    "timing_scope": (
                        "single measured local run; not deterministic and not a sustained-capacity guarantee"
                    ),
                    "utilization": {
                        "available": True,
                        "counter_scope": (
                            "device-wide NVML samples during measured synchronized GPU repetitions; "
                            "counters can include concurrent processes and are not a continuous or "
                            "lane-only utilization claim"
                        ),
                        "gpu_percent_max": 99,
                        "gpu_percent_mean": 92.25806451612904,
                        "memory_percent_max": 9,
                        "memory_percent_mean": 8.503225806451614,
                        "memory_used_mib_max": 2949,
                        "power_watts_max": 213.968,
                        "sample_count": 155,
                        "sample_interval_seconds": 0.02,
                    },
                },
                "synthetic_only": True,
                "formal_pass_inferred": False,
                "observations_opened": False,
                "scope": (
                    "candidate-bound numerical stress of the 163 materialized metric-Euler "
                    "formula projections on independent synthetic dyadic operator coordinates"
                ),
                "interpretation": (
                    "This is a numerical backend/throughput control. Synthetic operator "
                    "coordinates need not be realizable field jets, and agreement cannot "
                    "establish a field equation, formal pass, phenomenological fitness, or "
                    "observational support."
                ),
            },
            "generic_g4_B4_termwise_normalization": {
                "status": (
                    "pass_exact_24_term_generic_nonlinear_G4X_metric_Euler_normalization_to_KYY_B4"
                ),
                "primary_source": {
                    "arxiv_id": "1105.5723v4",
                    "authors": "Kobayashi, Yamaguchi, Yokoyama",
                    "equation": "B.4",
                    "title": (
                        "Generalized G-inflation: Inflation with the most general "
                        "second-order field equations"
                    ),
                },
                "primary_source_transcription": {
                    "path": ("formal/sources/kyy_1105.5723v4_eq_B4_canonical_coefficients.json"),
                    "file_sha256": (
                        "497042978c3c0eed8ec02b49c5ceb2c258e60416c23152e34d44cde4ae53d32f"
                    ),
                },
                "canonical_term_count": 24,
                "matched_term_count": 24,
                "nonzero_residual_count": 0,
                "metric_variation_normalization_pass": True,
                "full_candidate_formal_pass_inferred": False,
                "scope": (
                    "The independently executed Cadabra coefficient of delta g^ab matches "
                    "all 24 canonical contractions obtained from KYY equation B.4. This "
                    "closes tensor spelling and coefficient normalization only; it does not "
                    "prove global energy, nonlinear stability, observational validity, or "
                    "future unregistered operator families."
                ),
            },
        },
        "formal_preflight": {
            "candidate_count": 163,
            "decision_counts": {"blocked": 1, "pass": 162},
            "expensive_adm_or_global_energy_run": False,
            "family_decision_counts": {
                "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
                "CONFORMAL_G4_PHI_SCALAR_TENSOR": {"blocked": 1},
                "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
                "KESSENCE_G2_CONVEX": {"pass": 2},
            },
            "gate_counts": {
                "family_prerequisite": {"blocked": 1, "pass": 162},
                "receipt_binding": {"pass": 163},
            },
            "next_promotion_hook": (
                "enqueue only preflight-pass candidates into separately reviewed "
                "family-specific ADM/formal campaigns bound to "
                "candidate_id+typed_action_ir_sha256"
            ),
            "work_state_counts": {"succeeded": 163},
            "promotion_admission": {
                "decision_counts": {"pass": 162},
                "downstream_expensive_execution_started": False,
                "eligible_candidate_count": 162,
                "preflight_blocked_excluded_count": 1,
                "target_queue_counts": {
                    "grammar_v3_aether_candidate_adm_formal": 128,
                    "grammar_v3_g2_candidate_adm_formal": 2,
                    "grammar_v3_g3_candidate_adm_formal": 32,
                },
                "work_state_counts": {"succeeded": 162},
                "family_formal_execution": {
                    "aether": {
                        "candidate_count": 128,
                        "decision_counts": {"blocked": 126, "reject": 2},
                        "formal_pass_count": 0,
                        "gate_finding_counts": {
                            "finite_characteristic_slicing_present": 121,
                            "finite_negative_local_density_witness": 79,
                            "globally_noncharacteristic_for_finite_unit_tilt": 5,
                            "positive_at_every_finite_tilt_but_no_uniform_gap": 8,
                            "principal_spin0_degeneracy_reject": 2,
                            "uniform_positive_static_local_twist_gap": 39,
                        },
                    },
                    "g2": {
                        "predecessor_blocker_counts": {
                            "hash_bound_general_nonmaximal_positive_mass_theorem": 2
                        },
                        "candidate_count": 2,
                        "predecessor_decision_counts": {"blocked": 2},
                        "decision_counts": {"pass": 2},
                        "full_formal_pass_count": 2,
                        "general_nonmaximal_positive_mass_pass_count": 2,
                        "actual_initial_data_set_instantiated": False,
                        "cell_preservation_or_global_evolution_proved": False,
                        "solar_readiness": {
                            "analytic_prediction_pass_count": 2,
                            "conditional_static_source_class_pass_count": 2,
                            "decision_counts": {"blocked": 2},
                            "real_solar_bundle_count": 0,
                            "observational_data_opened": False,
                            "registration_advance": {
                                "after_missing_field_count": 4,
                                "before_missing_field_count": 10,
                                "filled_field_count": 6,
                                "filled_fields": [
                                    "candidate_specific_real_source_contract_sha256",
                                    "candidate_specific_evaluator_descriptor_sha256",
                                    "training_only_initial_state_sha256",
                                    "frozen_nuisance_likelihood_stopping_rule_sha256",
                                    "action_bound_prediction_bundle_descriptor_sha256",
                                    "action_bound_prediction_bundle_file_sha256",
                                ],
                                "remaining_fields": [
                                    "source_branch_domain_instantiation_sha256",
                                    "held_out_split_commitment_sha256",
                                    "selected_primary_record_roots_sha256",
                                    "observation_opening_authorization_sha256",
                                ],
                            },
                            "held_out_target_access_count": 0,
                            "primary_record_access_count": 0,
                            "real_data_pass_count": 0,
                            "first_missing_premise": (
                                "candidate_specific_real_source_branch_domain_"
                                "instantiation_and_metadata_only_session_split_"
                                "commitment"
                            ),
                        },
                        "work_state_counts": {"succeeded": 2},
                    },
                    "g3": {
                        "blocker_counts": {
                            "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain": 32
                        },
                        "candidate_count": 32,
                        "decision_counts": {"blocked": 32},
                        "full_formal_pass_count": 0,
                        "gate_counts": {
                            "adm_primary_degeneracy": {"pass": 32},
                            "af_Einstein_constraint_solution": {"blocked": 32},
                            "af_finite_scalar_energy_tail": {"pass": 32},
                            "af_reference_principal_common_cone": {"pass": 32},
                            "af_uniform_lapse_Dirac_invertibility": {"blocked": 32},
                            "all_spatial_covector_directions": {"pass": 32},
                            "candidate_action_preflight_admission_binding": {"pass": 32},
                            "covariant_G2_G3_variation_noether": {"pass": 32},
                            "distributed_Dirac_on_periodic_cell": {"pass": 32},
                            "exact_parameter_cell_and_weak_envelope": {"pass": 32},
                            "full_candidate_lapse_operator_derivation": {"pass": 32},
                            "full_formal_completion": {"blocked": 32},
                            "global_hamiltonian_energy": {"blocked": 32},
                            "periodic_lapse_coercivity_and_zero_mode_exclusion": {"pass": 32},
                            "uniform_local_common_time_and_BSSN_cone": {"pass": 32},
                            "uniform_local_principal_symbol": {"pass": 32},
                        },
                        "work_state_counts": {"succeeded": 32},
                    },
                    "g4_followup": {
                        "candidate_count": 1,
                        "decision_counts": {"pass": 1},
                        "equivalent_parameter_cell_alias_count": 32,
                        "formal_followup_decision": "pass",
                        "original_preflight_decision": "blocked",
                        "transfer_method": (
                            "exact_typed_density_projection_and_rational_domain_inclusion"
                        ),
                    },
                },
            },
        },
    }
    assert core["evidence_pareto"]["calibration_control_counts"] == {"pass": 13, "reject": 1}
    assert core["followup_service"]["followup_decision_counts"] == {
        "blocked": 8,
        "pass": 2,
    }
    assert core["followup_service"]["normalized_followup_outcomes"] == {
        "block": 8,
        "pass": 2,
        "reject": 0,
    }
    assert core["followup_service"]["processed"] == 10
    assert core["followup_service"]["deferred"] == 0
    assert core["followup_service"]["current_missing_evaluator_blockers"] == {}
    safety = core["continuous_dashboard"]["safety_hardening"]
    assert safety["decision"] == "hardened_service_ready_not_started"
    assert safety["service_started"] is False
    assert safety["safety_contract"]["windows_argv_list_shell_false"] is True
    assert safety["safety_contract"]["stale_projection_publication_allowed"] is False
    assert safety["safety_contract"]["legacy_worker_absence_required_before_start"] is True
    assert safety["safety_contract"]["atomic_starting_checkpoint_before_spawn"] is True
    assert safety["safety_contract"]["repeated_start_launch_allowed"] is False
    assert safety["safety_contract"]["worker_finally_releases_owned_lease"] is True
    assert (
        safety["safety_contract"]["leaderboard_history_seed_core_and_content_hash_validated"]
        is True
    )
    assert safety["safety_contract"]["leaderboard_history_seed_pre_and_post_hash_guarded"] is True
    assert safety["safety_contract"]["maximum_seed_history_entries"] == 64
    assert safety["safety_contract"]["maximum_seed_history_bytes"] == 65_536
    recovery = core["quartic_nonlinear_closure"].pop("recovery_operator_calculus")
    assert recovery["all_candidate_sets_equal"] is True
    assert len(recovery["ordered_candidate_ids"]) == 12
    assert recovery["anti_wick_composition_prerequisite"]["artifact_binding"] == {
        "path": "runs/physics-language/quartic-anti-wick-composition-campaign/campaign.json",
        "file_sha256": "9a9cb443ee86a5b5d45ba29ea1287442b101f8c675681f6eedaa927d33f41f1e",
        "content_sha256": "02c98ac16a6cd4bc3871003fb77918e21666a60fec65bc28c85484a6011c541d",
    }
    assert recovery["annular_k55_c6"]["artifact_binding"]["file_sha256"] == (
        "bcc2b4184e5bcfb64d9a8a24ca095aa4067c18502c0c2f4956dcd8ad6f7fc527"
    )
    assert recovery["bounded_frequency_defect"]["artifact_binding"]["file_sha256"] == (
        "e2dd669e0a939558d7379ac3600032eb7bca22e550d6965816ceca5e2724187a"
    )
    assert recovery["dyadic_localization"]["artifact_binding"]["file_sha256"] == (
        "859b472f666cae9175aa7da8bc90ef175ca16f1987b967d65c71f5cc14139c94"
    )
    assert recovery["anti_wick_composition_prerequisite"]["counts"] == {
        "C6_extensions_required": 12,
        "anti_wick_compositions_closed": 0,
        "exact_composition_prerequisite_audits_passed": 12,
        "rejected": 0,
        "selected": 12,
    }
    assert (
        recovery["annular_k55_c6"]["counts"]["principal_composition_constants_instantiated"] == 12
    )
    assert recovery["annular_k55_c6"]["counts"]["full_dyadic_energies_closed"] == 0
    assert (
        recovery["bounded_frequency_defect"]["counts"]["compact_frequency_defect_lemmas_passed"]
        == 12
    )
    assert recovery["dyadic_localization"]["counts"]["dyadic_local_frameworks_passed"] == 12
    assert recovery["dyadic_localization"]["counts"]["full_H7_commutators_closed"] == 0
    finite = recovery["finite_sobolev_hierarchy_no_go"]
    assert finite["artifact_binding"] == {
        "path": (
            "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-campaign/campaign.json"
        ),
        "file_sha256": "404a869cd676fb57389535c74ff4fea73ec1ae3da43cad1ff74951f817ae1308",
        "content_sha256": "a74fb10a523cef935695d99fff595de7aa51a8ead322171373827c8b6db9288a",
    }
    assert finite["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert finite["gate_counts"]["nonzero_candidate_D2_slices"] == 12
    assert finite["gate_counts"]["autonomous_finite_Sobolev_closures"] == 0
    assert finite["gate_counts"]["global_H7_closures"] == 0
    assert finite["gate_counts"]["lifespans_proved"] == 0
    assert finite["first_blocker"] == (
        "candidate_bound_full_tensor_paradifferential_cancellation_or_derivative_loss_"
        "evolution_theorem_for_the_coefficient_high_state_low_branch"
    )
    assert all(row["decision"] == "blocked" for row in finite["candidate_records"])
    assert all(not row["candidate_rejection_authorized"] for row in finite["candidate_records"])
    assert not any(finite["claim_seals"].values())
    assert not any(finite["data_seals"].values())
    reconciliation = recovery["full_tensor_good_unknown_reconciliation"]
    assert reconciliation["artifact_binding"] == {
        "path": (
            "runs/physics-language/quartic-full-tensor-good-unknown-reconciliation-gate/"
            "campaign.json"
        ),
        "file_sha256": "cf7957c2efad52a1fa91761fc6259e17a58011cc6093365f9e86e8e7eea0dfd6",
        "content_sha256": "9994df86948a4419dd999b66610e9fea847dece6d5300f68152e942ffb2b87c8",
    }
    assert reconciliation["decision"] == "representative_slice_cancelled_full_D2_identity_blocked"
    assert reconciliation["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert reconciliation["gate_counts"]["two_channel_four_entry_slice_cancellations_proved"] == 12
    assert reconciliation["gate_counts"]["representative_D2_entries_replayed"] == 48
    assert reconciliation["gate_counts"]["full_D1_entries_per_candidate"] == 1683
    assert reconciliation["gate_counts"]["closed_world_ordered_D2_target_per_candidate"] == 257499
    assert reconciliation["gate_counts"]["complete_ordered_D2_manifests_registered"] == 0
    assert reconciliation["gate_counts"]["full_high_atom_families_closed"] == 0
    assert reconciliation["first_blocker"] == (
        "complete_candidate_bound_ordered_D2F_component_manifest_and_full_high_atom_"
        "good_unknown_identity_not_registered"
    )
    assert all(
        row["two_channel_modified_slice"]["all_four_entries_cancelled"]
        and not row["two_channel_modified_slice"]["residual_entries"]
        and row["candidate_decision"] == "blocked"
        and not row["candidate_rejection_authorized"]
        for row in reconciliation["candidate_records"]
    )
    assert not any(reconciliation["claim_seals"].values())
    assert not any(reconciliation["data_seals"].values())
    scalar_d2 = recovery["scalar_hessian_d2_integrability"]
    assert scalar_d2["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert scalar_d2["gate_counts"]["ordered_D2_entries_materialized_per_candidate"] == 9_801
    assert scalar_d2["gate_counts"]["ordered_D2_entries_materialized_total"] == 117_612
    assert scalar_d2["gate_counts"]["failed_ordered_family_pairs_per_candidate"] == 24
    assert scalar_d2["gate_counts"]["nonzero_Schwarz_residuals_per_candidate"] == 30
    assert scalar_d2["gate_counts"]["full_ordered_D2_manifests_admitted"] == 0
    assert scalar_d2["first_blocker"] == (
        "typed_coordinate_to_block_Frechet_map_or_covariant_connection_terms_restoring_"
        "Schwarz_integrability_not_registered"
    )
    assert {key for key, enabled in scalar_d2["claim_seals"].items() if enabled} == {
        "naive_chunk_extension_obstructed"
    }
    assert not any(scalar_d2["data_seals"].values())
    scalar_curl = recovery["scalar_hessian_curl_invariance"]
    assert scalar_curl["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert scalar_curl["gate_counts"]["ordered_nonzero_curl_pairs_per_candidate"] == 24
    assert scalar_curl["gate_counts"]["independent_nonzero_curl_pairs_per_candidate"] == 12
    assert scalar_curl["gate_counts"]["ordered_nonzero_curl_components_per_candidate"] == 30
    assert scalar_curl["gate_counts"]["independent_nonzero_curl_components_per_candidate"] == 15
    assert scalar_curl["gate_counts"]["coordinate_only_repairs_ruled_out"] == 12
    assert scalar_curl["gate_counts"]["torsion_free_domain_connection_repairs_ruled_out"] == 12
    assert scalar_curl["gate_counts"]["corrected_source_repairs_registered"] == 0
    assert scalar_curl["gate_counts"]["full_ordered_D2_manifests_admitted"] == 0
    assert not any(scalar_curl["data_seals"].values())
    output_repair = recovery["scalar_hessian_output_bundle_repair"]
    assert output_repair["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert output_repair["gate_counts"]["registered_one_form_rank"] == 1
    assert output_repair["gate_counts"]["arbitrary_domain_torsion_pair_no_go_certificates"] == 144
    assert output_repair["gate_counts"]["output_connection_equations_per_candidate"] == 396
    assert output_repair["gate_counts"]["output_connection_unknowns_per_candidate"] == 99
    assert output_repair["gate_counts"]["output_connection_coefficient_rank"] == 88
    assert output_repair["gate_counts"]["output_connection_augmented_rank"] == 88
    assert output_repair["gate_counts"]["output_connection_affine_dimension"] == 11
    assert output_repair["gate_counts"]["sparse_output_connection_coefficients_per_candidate"] == 6
    assert output_repair["gate_counts"]["corrected_scalar_hessian_D2_entries_per_candidate"] == 891
    assert output_repair["gate_counts"]["corrected_curl_nonzero_components"] == 0
    assert output_repair["gate_counts"]["complete_ordered_D2_manifests_registered"] == 0
    assert not any(output_repair["data_seals"].values())
    full_d2f = recovery["full_d2f_high_atom_coverage"]
    assert full_d2f["artifact_binding"] == {
        "path": "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json",
        "file_sha256": "b9ce34960b766a6fe74a36a13190b0f050a1447884d599fad1eebfe189b32590",
        "content_sha256": "e7e4e4171aed90d07d68791183c58a696e77b9bed745f1018da2c5ee9438c38a",
    }
    assert full_d2f["decision"] == (
        "complete_ordered_D2F_domain_classified_values_and_full_high_atom_identity_blocked"
    )
    assert full_d2f["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert full_d2f["pair_status_counts"] == {
        "corrected_admitted": 81,
        "naive_evaluated_not_admitted": 810,
        "reverse_principal_not_registered": 810,
        "other_principal_pair_not_registered": 8_100,
        "principal_lower_not_registered": 5_346,
        "lower_principal_not_registered": 5_346,
        "lower_lower_not_registered": 2_916,
    }
    assert full_d2f["gate_counts"]["coordinate_atoms"] == 153
    assert full_d2f["gate_counts"]["ordered_pair_cells_classified"] == 23_409
    assert full_d2f["gate_counts"]["ordered_D2F_entries_in_domain"] == 257_499
    assert full_d2f["gate_counts"]["corrected_entries_admitted_per_candidate"] == 891
    assert (
        full_d2f["gate_counts"]["principal_high_atom_entries_missing_per_candidate"]
        == 106_920
    )
    assert full_d2f["gate_counts"]["full_ordered_D2F_entries_missing_per_candidate"] == 256_608
    assert full_d2f["gate_counts"]["complete_ordered_D2F_tensors_registered"] == 0
    assert full_d2f["gate_counts"]["full_high_atom_good_unknown_identities_proved"] == 0
    assert full_d2f["gate_counts"]["global_H7_closures"] == 0
    assert full_d2f["gate_counts"]["nonlinear_PDE_closures"] == 0
    assert full_d2f["gate_counts"]["lifespans_proved"] == 0
    assert full_d2f["first_blocker"] == (
        "candidate_bound_covariant_source_derivatives_and_output_bundle_connection_extension_"
        "for_remaining_106920_principal_high_atom_D2F_entries_not_registered"
    )
    assert {key for key, value in full_d2f["claim_seals"].items() if value} == {
        "complete_ordered_D2F_coverage_domain_classified",
        "corrected_scalar_hessian_high_field10_submanifest_admitted",
        "remaining_principal_high_atom_domain_exactly_classified",
    }
    assert not any(full_d2f["data_seals"].values())
    connection_extension = recovery["principal_high_atom_connection_extension"]
    assert connection_extension["artifact_binding"] == {
        "path": (
            "runs/physics-language/quartic-principal-high-atom-connection-extension-gate/"
            "campaign.json"
        ),
        "file_sha256": "e4ffc8f0d82f3c4381703f338a03cc334e15268aaf5b0c7f0dd1305ee96f8b92",
        "content_sha256": "33942664dc481ae112650c1f9ad1c4834687161601b54348b55a82139816c028",
    }
    assert connection_extension["decision"] == (
        "restricted_B10_connection_extension_exactly_ineffective_one_sided_values_"
        "materialized_cross_slice_not_admitted_candidates_blocked"
    )
    assert connection_extension["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert connection_extension["gate_counts"]["ordered_pair_cells_audited_per_candidate"] == 810
    assert (
        connection_extension["gate_counts"]["one_sided_values_materialized_per_candidate"]
        == 8_910
    )
    assert connection_extension["gate_counts"]["one_sided_nonzero_values_per_candidate"] == 93
    assert (
        connection_extension["gate_counts"][
            "restricted_connection_correction_entries_checked_per_candidate"
        ]
        == 8_910
    )
    assert connection_extension["gate_counts"]["restricted_connection_nonzero_corrections"] == 0
    assert connection_extension["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert (
        connection_extension["gate_counts"]["principal_high_atom_entries_missing_per_candidate"]
        == 106_920
    )
    assert connection_extension["first_blocker"] == (
        "reverse_Pother_by_P10_candidate_bound_source_derivatives_and_zero_corrected_curl_"
        "for_810_ordered_pairs_not_registered"
    )
    assert {key for key, value in connection_extension["claim_seals"].items() if value} == {
        "one_sided_P10_by_Pother_values_materialized",
        "other_principal_atom_subset_exactly_registered",
        "registered_B10_connection_correction_zero_on_P10_by_Pother",
        "scalar_source_row_10_zero_on_other_principal_subset",
    }
    assert not any(connection_extension["data_seals"].values())
    assert all(
        not any(lane["data_seals"].values())
        for name, lane in recovery.items()
        if name not in {"ordered_candidate_ids", "all_candidate_sets_equal"}
    )
    topology = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("topology_changing_origin_classification")
    assert topology["counts"]["direct_joint_domain_dimension"] == 2145
    assert topology["counts"]["direct_joint_cokernel_map_rank"] == 0
    assert topology["explicit_TC2_selector_classification"]["canonical_capable_indices"] == [
        21,
        44,
        48,
        51,
        53,
    ]
    assert (
        topology["explicit_TC2_selector_classification"]["registered_selector_control"][
            "target_W_in_image"
        ]
        is False
    )
    curl = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("curl_constraint_admission")
    assert curl["counts"] == {
        "candidate_reference_D4_solutions_inherited": 12,
        "curl_constraints_propagated": 33,
        "definition_constraints_propagated": 33,
        "direction_blocks": 2,
        "inferred_global_passes": 0,
        "negative_controls": 6,
        "ordered_fourth_coefficient_derivatives_checked": 256,
        "ordered_lower_coefficient_derivatives_checked": 85,
        "output_nonzero_coefficients": 6,
        "source_curl_constraints": 1,
    }
    assert curl["gauge_fixed_operator"]["direction_1_block_equals_V"] is True
    assert curl["constraint_propagation"]["definition_constraint_propagation"]["map_rank"] == 1
    assert curl["constraint_propagation"]["curl_constraint_propagation"]["map_rank"] == 3
    assert (
        curl["physical_reduction_equivalence"]["directional_operator_times_gradient_lift_zero"]
        is True
    )
    assert curl["admission_result"]["gauge_fixed_constraint_operator_constructed"] is True
    assert curl["admission_result"]["covariant_action_derived"] is False
    assert curl["admission_result"]["all_direction_Sylvester_compatibility_proved"] is False
    companion = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("curl_companion_range")
    assert companion["counts"]["reference_eigenspaces_checked"] == 7
    assert companion["counts"]["companion_compression_rank"] == 2
    assert companion["counts"]["pure_C23_effective_parameters"] == 363
    assert companion["counts"]["pure_C23_range_rank"] == 297
    assert companion["counts"]["target_augmented_rank"] == 298
    assert companion["equal_eigenspace_audit"]["sole_nonzero_eigenvalue"] == "0"
    assert companion["pure_curl_completion_range"]["exact_range_map"]["target_in_image"] is False
    assert companion["necessary_full_D4_condition"]["base_D4_RHS_computed"] is False
    axis2 = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("axis2_base_D4_RHS")
    assert axis2["counts"]["directional_evaluations"] == 15
    assert axis2["counts"]["candidate_conditions_checked"] == 12
    assert axis2["counts"]["zero_speed_cancellations_exact"] == 0
    assert axis2["counts"]["corrected_axis2_D4_obstructions"] == 12
    assert axis2["polarized_base_D4"]["RHS_base_nonzero_entries"] == 0
    assert axis2["result"]["base_D4_RHS_identically_zero"] is True
    assert axis2["result"]["corrected_axis2_D4_obstructions"] == 12
    assert axis2["claims"]["fixed_chart_curl_completion_axis2_D4_rejected"] is True
    assert axis2["claims"]["TC2_closed"] is False
    spatial_no_go = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("spatial_gradient_annihilator_no_go")
    assert spatial_no_go["counts"]["raw_affine_dimension"] == 605
    assert spatial_no_go["counts"]["effective_projected_parameters"] == 363
    assert spatial_no_go["axis2_projected_range"]["range_rank"] == 297
    assert spatial_no_go["axis2_projected_range"]["target_augmented_rank"] == 298
    assert spatial_no_go["candidate_consequence"]["candidate_no_go_results"] == 12
    full_no_go = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("full_linear_gradient_annihilator_no_go")
    assert full_no_go["counts"]["canonical_qv_selectors_checked"] == 22
    assert full_no_go["counts"]["canonical_qv_kernel_selectors"] == 11
    assert full_no_go["counts"]["canonical_qv_nonzero_incapable_selectors"] == 11
    assert full_no_go["counts"]["canonical_qv_capable_selectors"] == 0
    assert full_no_go["combined_axis2_free_B2_range"]["wedge_range_rank"] == 473
    assert full_no_go["combined_axis2_free_B2_range"]["target_augmented_rank"] == 474
    assert full_no_go["claims"]["all_operator_classes_ruled_out"] is False
    parity_cubic = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("parity_cubic_angular_escape")
    assert parity_cubic["counts"]["scalar_multiplier_degree"] == 2
    assert parity_cubic["counts"]["total_angular_polynomial_degree"] == 3
    assert parity_cubic["counts"]["reference_e1_D4_solutions_inherited"] == 12
    assert parity_cubic["counts"]["new_axis2_D4_compatibilities"] == 12
    assert parity_cubic["counts"]["new_axis2_D4_obstructions"] == 0
    assert parity_cubic["counts"]["generic_direction_D4_compatibilities_proved"] == 0
    assert parity_cubic["minimality"]["canonical_multiplier"] == "a(n)=n1^2"
    assert parity_cubic["exact_symbol"]["definition"] == ("B_cubic(n)=n1^2*(n1*V+n2*C_companion)")
    assert (
        parity_cubic["pseudodifferential_constraint_admission"]["M1_fourier_symbol"]
        == "xi1^2/|xi|^2=n1^2"
    )
    assert parity_cubic["claims"]["all_12_axis2_D4_compatibilities_proved_for_cubic_symbol"] is True
    assert parity_cubic["claims"]["generic_direction_D4_compatibility_proved"] is False
    generic_direction = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("parity_cubic_generic_direction_audit")
    assert generic_direction["counts"]["declared_rational_frames"] == 3
    assert generic_direction["counts"]["frames_evaluated"] == 1
    assert generic_direction["counts"]["frames_unevaluated_after_stop"] == 2
    assert generic_direction["counts"]["directional_recurrence_evaluations"] == 15
    assert generic_direction["counts"]["candidate_direction_compatibilities"] == 0
    assert generic_direction["counts"]["candidate_direction_obstructions"] == 12
    direction_record = generic_direction["exact_generic_direction_audit"]["direction_records"][0]
    assert direction_record["direction"] == ["3/5", "4/5", "0"]
    assert direction_record["base_D4_RHS_nonzero_entries"] == 64
    assert direction_record["cubic_correction_block_rank"] == 1
    assert direction_record["cubic_correction_skew_rank"] == 2
    assert generic_direction["claims"]["parity_cubic_all_direction_completion_rejected"] is True
    assert generic_direction["claims"]["full_generic_direction_sphere_classified"] is False
    matrix_curl = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("matrix_curl_rank_one_completion")
    assert matrix_curl["counts"]["transverse_curl_covectors"] == 22
    assert matrix_curl["counts"]["raw_matrix_parameters"] == 1210
    assert matrix_curl["counts"]["selector_projection_rank"] == 22
    assert matrix_curl["counts"]["wedge_range_rank"] == 473
    assert matrix_curl["counts"]["target_augmented_rank"] == 473
    assert matrix_curl["counts"]["constructed_block_rank"] == 1
    assert matrix_curl["counts"]["candidate_compatibilities"] == 12
    assert matrix_curl["counts"]["candidate_obstructions"] == 0
    assert matrix_curl["exact_range_classification"]["target_in_image"] is True
    assert matrix_curl["exact_range_classification"]["quotient_target_zero"] is True
    assert (
        matrix_curl["minimal_rank_one_completion"]["all_nonzero_eigenspace_compressions_zero"]
        is True
    )
    assert matrix_curl["minimal_rank_one_completion"]["zero_speed_target_cancelled_exactly"] is True
    assert matrix_curl["candidate_result"]["candidate_compatibilities"] == 12
    assert matrix_curl["claims"]["all_12_fixed_frame_D4_compatibilities_proved"] is True
    assert matrix_curl["claims"]["global_smooth_angular_extension_constructed"] is False
    sphere_extension = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("degree_three_matrix_curl_sphere_extension")
    assert sphere_extension["counts"]["minimal_total_extension_degree"] == 3
    assert sphere_extension["counts"]["preserved_direction_certificates"] == 3
    assert sphere_extension["counts"]["candidate_certificates_preserved"] == 12
    assert sphere_extension["counts"]["additional_frames_evaluated"] == 1
    assert sphere_extension["counts"]["candidate_direction_compatibilities"] == 0
    assert sphere_extension["counts"]["candidate_direction_obstructions"] == 12
    assert sphere_extension["exact_sphere_symbol"]["antipodally_odd"] is True
    assert (
        sphere_extension["exact_sphere_symbol"]["physical_gradient_lift_annihilated_identically"]
        is True
    )
    assert sphere_extension["exact_sphere_symbol"]["output_vector_sha256"] == (
        "68f42985cff4653364ddf9d0ce0a6bb1c9e84aa5c3b8998d338195220a985e7a"
    )
    assert sphere_extension["exact_sphere_symbol"]["symbol_sha256"] == (
        "965ab6cc84cfd809dede2d0b9f10a8002956e6a43dee1a4995c0fdf42c407c26"
    )
    assert sphere_extension["exact_sphere_symbol"]["gradient_lift_residual_sha256"] == (
        "4efb4f5888421b27afeb457ab5bd0f20260c86f9f4f20229e45b1baccdd89346"
    )
    assert sphere_extension["certificate_preservation"]["fixed_block_sha256"] == (
        "006aecdc99032a89a597b56e69ffed9ef35d3c9f1278b20ec96b1b0741dceb3a"
    )
    assert sphere_extension["first_additional_frame_audit"]["base_D4_RHS_sha256"] == (
        "d3ab104a0de327e978b6bbe03113b2cf883bce4b34684eed94574560388e0513"
    )
    assert (
        sphere_extension["first_additional_frame_audit"]["total_correction_block_sha256"]
        == "8dac2461183b13df9be8d92d60f3bb5926624e75ce72601c864bdddbe99db862"
    )
    assert sphere_extension["first_additional_frame_audit"]["selector"]["direction"] == [
        "3/5",
        "0",
        "4/5",
    ]
    assert sphere_extension["first_additional_frame_audit"]["selector"]["frame_name"] == (
        "xz_3_4_5"
    )
    assert (
        sphere_extension["first_additional_frame_audit"]["selector"][
            "deterministic_position_after_original_generic_frame"
        ]
        == 1
    )
    assert (
        sphere_extension["first_additional_frame_audit"]["selector"][
            "later_declared_frames_unevaluated"
        ]
        == 1
    )
    assert (
        sphere_extension["claims"][
            "canonical_degree_three_extension_rejected_as_all_direction_completion"
        ]
        is True
    )
    assert sphere_extension["claims"]["broader_matrix_curl_symbol_class_classified"] is False
    c23_escape = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("degree_three_C23_great_circle_escape")
    assert c23_escape["counts"]["minimal_total_extension_degree"] == 3
    assert c23_escape["counts"]["total_certified_directions"] == 4
    assert c23_escape["counts"]["new_candidate_direction_compatibilities"] == 12
    assert c23_escape["counts"]["new_candidate_direction_obstructions"] == 0
    assert c23_escape["counts"]["remaining_declared_frames"] == 1
    assert c23_escape["exact_sphere_symbol"]["symbol_sha256"] == (
        "9db986773ebca9a1f85eff597c3596a731d4c0ce55dd5b8c91ce74a96e2bd735"
    )
    assert c23_escape["xz_escape_audit"]["candidate_compatibilities"] == 12
    assert c23_escape["xz_escape_audit"]["candidate_obstructions"] == 0
    assert c23_escape["selector_binding"]["remaining_declared_direction"] == "xyz_1_2_2"
    assert c23_escape["claims"]["all_12_xz_D4_compatibilities_proved"] is True
    assert c23_escape["claims"]["remaining_xyz_frame_audited"] is False
    xyz = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("degree_three_rank_two_xyz_completion")
    assert xyz["counts"]["prior_candidate_obstructions"] == 12
    assert xyz["counts"]["normalized_target_rank"] == 4
    assert xyz["counts"]["transverse_selector_rank"] == 22
    assert xyz["counts"]["minimal_completion_rank"] == 2
    assert xyz["counts"]["new_curl_channels"] == 2
    assert xyz["counts"]["new_candidate_direction_compatibilities"] == 12
    assert xyz["counts"]["new_candidate_direction_obstructions"] == 0
    assert xyz["counts"]["total_certified_directions"] == 5
    assert xyz["counts"]["remaining_declared_directions"] == 0
    assert xyz["minimal_rank_two_completion"]["coordinate_pairs"] == [[11, 21], [15, 32]]
    assert xyz["exact_range_classification"]["target_in_full_transverse_curl_range"] is True
    assert xyz["corrected_xyz_result"]["candidate_compatibilities"] == 12
    assert xyz["claims"]["all_five_declared_direction_certificates_closed"] is True
    assert xyz["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    sixth = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("degree_three_sixth_frame_completion")
    assert sixth["counts"]["prior_candidate_obstructions"] == 12
    assert sixth["counts"]["normalized_target_rank"] == 4
    assert sixth["counts"]["transverse_selector_rank"] == 22
    assert sixth["counts"]["minimal_completion_rank"] == 2
    assert sixth["counts"]["new_curl_channels"] == 2
    assert sixth["counts"]["new_candidate_direction_compatibilities"] == 12
    assert sixth["counts"]["new_candidate_direction_obstructions"] == 0
    assert sixth["counts"]["prior_direction_certificates_preserved"] == 5
    assert sixth["counts"]["total_certified_directions"] == 6
    assert sixth["minimal_rank_two_completion"]["coordinate_pairs"] == [[11, 21], [15, 32]]
    assert sixth["exact_range_classification"]["normalized_target_sha256"] == (
        "8bf2ca4b022f46411344c1665879dbb60b71229572ec72d9b89867589af54abe"
    )
    assert sixth["exact_sphere_extension"]["envelope"] == ("a6(n)=(3/2)*n3*(4*n1+n2-3*n3)")
    assert sixth["corrected_result"]["candidate_compatibilities"] == 12
    assert sixth["claims"]["all_six_selector_direction_certificates_closed"] is True
    assert sixth["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    rational = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("rational_chart_determining_gate")
    assert rational["counts"]["rational_SO3_charts"] == 2
    assert rational["counts"]["real_sphere_uncovered_points"] == 0
    assert rational["counts"]["directional_recurrence_evaluations"] == 15
    assert rational["counts"]["eigenspace_compressions_checked"] == 84
    assert rational["counts"]["candidate_compatibilities"] == 0
    assert rational["counts"]["candidate_obstructions"] == 12
    assert rational["counts"]["cleared_constant_numerator_polynomials"] == 12
    assert rational["atlas"]["common_denominator"] == "1+u^2+v^2"
    assert rational["atlas"]["union_covers_real_S2"] is True
    assert rational["counterexample_selector"]["chart_coordinates"] == ["2/5", "1/5"]
    assert rational["counterexample_selector"]["direction"] == ["2/3", "2/3", "1/3"]
    assert rational["full_recurrence"]["current_global_symbol_rank"] == 5
    assert rational["exact_rational_obstruction"]["candidate_obstructions"] == 12
    assert all(
        row["zero_speed_cleared_numerator"]["numerator_polynomial_total_degree_uv"] == 0
        for row in rational["exact_rational_obstruction"]["candidate_records"]
    )
    assert (
        rational["claims"]["current_combined_symbol_full_sphere_D4_compatibility_disproved"] is True
    )
    assert rational["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    degree_five = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("degree_five_counterexample_escape")
    assert degree_five["counts"]["lower_even_envelope_degrees_rejected"] == 2
    assert degree_five["counts"]["two_monomial_supports_checked"] == 105
    assert degree_five["counts"]["two_monomial_supports_feasible"] == 1
    assert degree_five["counts"]["minimal_completion_rank"] == 2
    assert degree_five["counts"]["new_candidate_direction_compatibilities"] == 12
    assert degree_five["counts"]["new_candidate_direction_obstructions"] == 0
    assert degree_five["counts"]["total_certified_directions"] == 7
    assert (
        degree_five["exact_completion"]["minimal_preserving_envelope"]["sparsest_envelope"]
        == "a7(n)=(81/14)*n2*n3*(2*n1*n2-n3^2)"
    )
    assert degree_five["exact_completion"]["corrected_result"]["candidate_compatibilities"] == 12
    assert degree_five["claims"]["all_seven_selector_direction_certificates_closed"] is True
    assert degree_five["claims"]["revised_symbol_full_sphere_D4_compatibility_proved"] is False
    revised = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("revised_symbol_rational_counterexample")
    assert revised["counts"]["rational_SO3_charts"] == 2
    assert revised["counts"]["directional_polarization_evaluations"] == 15
    assert revised["counts"]["candidate_compatibilities"] == 0
    assert revised["counts"]["candidate_obstructions"] == 12
    assert revised["counts"]["bounded_envelope_supports_checked"] == 4_943
    assert revised["counts"]["sparsest_envelope_support"] == 5
    assert revised["counts"]["sparsest_feasible_envelopes"] == 110
    assert revised["counts"]["new_local_candidate_compatibilities"] == 12
    assert revised["counts"]["total_local_direction_certificates"] == 8
    assert revised["first_obstruction"]["selector"]["chart_coordinates"] == ["0", "1"]
    assert revised["first_obstruction"]["selector"]["direction"] == ["0", "0", "1"]
    assert (
        revised["first_obstruction"]["exact_rational_obstruction"]["eta_normalized_target_sha256"]
        == "db67ad988b50a0966f05c71ac7fbe0da460888b1baca4f36fb6f4dd9f639409f"
    )
    envelope = revised["bounded_next_escape"]["minimal_preserving_envelope"]
    assert envelope["total_supports_checked"] == 4_943
    assert envelope["sparsest_support_feasible_envelopes"] == 110
    assert revised["bounded_next_escape"]["local_completion"]["coordinate_pairs"] == [
        [11, 21],
        [15, 32],
    ]
    assert (
        revised["claims"]["revised_seven_frame_symbol_full_sphere_D4_compatibility_disproved"]
        is True
    )
    assert revised["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    revised_eight = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("revised_eight_frame_rational_counterexample")
    assert revised_eight["counts"]["rational_SO3_charts"] == 2
    assert revised_eight["counts"]["directional_polarization_evaluations"] == 15
    assert revised_eight["counts"]["candidate_compatibilities"] == 0
    assert revised_eight["counts"]["candidate_obstructions"] == 12
    assert revised_eight["counts"]["bounded_envelope_supports_checked"] == 1_940
    assert revised_eight["counts"]["sparsest_envelope_support"] == 4
    assert revised_eight["counts"]["sparsest_feasible_envelopes"] == 15
    assert revised_eight["counts"]["new_local_candidate_compatibilities"] == 12
    assert revised_eight["counts"]["total_local_direction_certificates"] == 9
    assert revised_eight["first_obstruction"]["selector"]["chart_coordinates"] == ["1", "1"]
    assert revised_eight["first_obstruction"]["selector"]["direction"] == [
        "-1/3",
        "2/3",
        "2/3",
    ]
    assert (
        revised_eight["first_obstruction"]["exact_rational_obstruction"][
            "eta_normalized_target_sha256"
        ]
        == "b696a2ec0e1e9162ab59c8be2cd688f9c808d661fc7dcad7b5f28c5c23e40f71"
    )
    revised_eight_envelope = revised_eight["bounded_next_escape"][
        "minimal_preserving_envelope"
    ]
    assert revised_eight_envelope["supports_checked_by_size"] == {
        "1": 15,
        "2": 105,
        "3": 455,
        "4": 1365,
    }
    assert revised_eight_envelope["feasible_envelopes_by_support_size"] == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 15,
    }
    assert revised_eight_envelope["deterministic_envelope"] == (
        "a9(n)=-81*n1*n2*n3**2/16 + 135*n2**3*n3/16 - "
        "567*n2**2*n3**2/32 + 189*n2*n3**3/16"
    )
    assert revised_eight["bounded_next_escape"]["exact_range_classification"][
        "transverse_selector_rank"
    ] == 22
    assert revised_eight["bounded_next_escape"]["local_completion"][
        "coordinate_pairs"
    ] == [[11, 21], [15, 32]]
    assert (
        revised_eight["claims"][
            "revised_eight_frame_symbol_full_sphere_D4_compatibility_disproved"
        ]
        is True
    )
    assert revised_eight["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    revised_nine = core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"][
        "canonical_obstruction_certificate"
    ].pop("revised_nine_frame_rational_counterexample")
    assert revised_nine["counts"]["rational_SO3_charts"] == 2
    assert revised_nine["counts"]["directional_polarization_evaluations"] == 15
    assert revised_nine["counts"]["candidate_compatibilities"] == 0
    assert revised_nine["counts"]["candidate_obstructions"] == 12
    assert revised_nine["counts"]["bounded_envelope_supports_checked"] == 1_940
    assert revised_nine["counts"]["sparsest_envelope_support"] == 4
    assert revised_nine["counts"]["sparsest_feasible_envelopes"] == 2
    assert revised_nine["counts"]["new_local_candidate_compatibilities"] == 12
    assert revised_nine["counts"]["total_local_direction_certificates"] == 10
    assert revised_nine["first_obstruction"]["selector"]["chart_coordinates"] == ["1", "-1"]
    assert revised_nine["first_obstruction"]["selector"]["direction"] == [
        "-1/3",
        "2/3",
        "-2/3",
    ]
    assert (
        revised_nine["first_obstruction"]["exact_rational_obstruction"][
            "eta_normalized_target_sha256"
        ]
        == "406cb87033eb0946d053c46d076a5e97fb9157c848e3737950c24ba7fca0369f"
    )
    revised_nine_envelope = revised_nine["bounded_next_escape"][
        "minimal_preserving_envelope"
    ]
    assert revised_nine_envelope["supports_checked_by_size"] == {
        "1": 15,
        "2": 105,
        "3": 455,
        "4": 1365,
    }
    assert revised_nine_envelope["feasible_envelopes_by_support_size"] == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 2,
    }
    assert revised_nine_envelope["deterministic_envelope"] == (
        "a10(n)=3*n1**2*n2*n3/8 - 21*n2**3*n3/16 + "
        "81*n2**2*n3**2/32 - 21*n2*n3**3/16"
    )
    assert revised_nine["bounded_next_escape"]["exact_range_classification"][
        "transverse_selector_rank"
    ] == 22
    assert revised_nine["bounded_next_escape"]["local_completion"][
        "coordinate_pairs"
    ] == [[11, 21], [15, 32]]
    assert (
        revised_nine["claims"][
            "revised_nine_frame_symbol_full_sphere_D4_compatibility_disproved"
        ]
        is True
    )
    assert revised_nine["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert core["quartic_nonlinear_closure"] == {
        "candidate_count": 12,
        "coordinate_pair_partition": {
            "canonical_active_exact_pairs": 861,
            "coverage_complete": True,
            "entrywise_zero_chain_rule_pairs": 8245,
            "excluded_exact_obligations": 2675,
            "global_pair_index_set_sha256": (
                "d300bb318a6475e88d7dfccd6ef4df9ff991e1e1d8cc535ef555c817723168ef"
            ),
            "total_unordered_coordinate_pairs": 11781,
        },
        "quadratic_deltaK_two_jet": {
            "closed_candidate_count": 12,
            "closed_derivative_orders": [0, 1, 2],
            "D2_coordinate_linf_to_Frobenius_ceiling": 16472172,
            "full_tube_Sylvester_identity_closed": False,
        },
        "diagonal_third_jet": {
            "active_direction_count": 41,
            "diagonal_triples_closed": 41,
            "candidate_direction_evaluations": 492,
            "candidate_direction_solvable": 492,
            "candidate_direction_obstructed": 0,
            "full_active_symmetric_triple_count": 12341,
            "remaining_mixed_triples": 10700,
            "mixed_third_jet_closures": 1600,
        },
        "mixed_third_jet_chunk": {
            "chunk_offset": 1536,
            "latest_chunk_processed_count": 64,
            "processed_count": 1600,
            "next_offset": 1600,
            "triple_kind_counts": {"ABB": 4, "ABC": 60},
            "symbolic_parameter_compatible": 1600,
            "latest_candidate_evaluations": 768,
            "candidate_evaluations": 19200,
            "candidate_solvable": 19200,
            "candidate_obstructed": 0,
            "remaining_mixed_triples": 10700,
            "resume_tip_sha256": (
                "f3546c475bfa7fd58443f4195111df19291ee5ea2d8f4898bcd5e3917da4a2f0"
            ),
            "service_decision": "checkpointed",
            "parallel_worker_count": 8,
            "parallel_execution_policy": (
                "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
            ),
            "sequential_predecessor_processed_count": 192,
            "full_mixed_sector_closed": True,
        },
        "mixed_third_jet_reduction": {
            "active_direction_rank": 15,
            "symmetric_cubic_dimension": 680,
            "stable_combined_evidence_rank": 233,
            "rank_gain_over_prior_reduction": 113,
            "reranked_exact_obligations": 447,
            "reranked_obligation_kind_counts": {"AAB": 77, "ABB": 81, "ABC": 289},
            "first_selector_index": 1634,
            "last_selector_index": 12269,
            "candidate_evaluation_budget": 5364,
            "brute_force_unevaluated_triples": 10700,
            "completion_rank": 680,
            "drop_final_obligation_rank": 679,
            "obligations_evaluated": 447,
            "obligations_remaining": 0,
            "candidate_evaluations": 5364,
            "candidate_solvable": 5364,
            "candidate_obstructed": 0,
            "next_obligation_offset": 447,
            "resume_tip_sha256": (
                "36338e1d76f61acbbab4927f7fd38cc116defb3a5d0ccd2a73d45faafe726e55"
            ),
            "obligations_inferred_passed": 0,
        },
        "fourth_jet_range_obligations": {
            "active_direction_rank": 15,
            "selector_obligations": 3060,
            "candidate_obligation_budget": 36720,
            "obligations_evaluated": 245,
            "obligations_closed": 244,
            "obligations_remaining": 2816,
            "candidate_evaluations": 2940,
            "candidate_solvable": 2928,
            "candidate_obstructed": 12,
            "directional_evaluations": 2918,
            "next_obligation_offset": 245,
            "resume_tip_sha256": (
                "7c309eec9d225f4c0813f0696e9806d7e5c2c9802528ade40d1d92c5f13d4c56"
            ),
            "parallel_worker_count": 8,
            "permanently_stopped": True,
            "stop_reason": "exact_obstruction",
            "first_exact_obstruction": {
                "active_indices": [0, 2, 3, 9],
                "active_positions": [0, 2, 4, 15],
                "gate": "fourth-order equal-eigenspace Sylvester compatibility",
                "obligation_offset": 244,
                "obstructed_candidate_ids": [
                    "quartic-symbol-06e267a9215345b6",
                    "quartic-symbol-076dc0ba965ab63a",
                    "quartic-symbol-317e5395817a432b",
                    "quartic-symbol-50f184dfe1a814bf",
                    "quartic-symbol-5455cad9e42a0dbc",
                    "quartic-symbol-561de1410d6cb21f",
                    "quartic-symbol-8fd254934d778c28",
                    "quartic-symbol-9e65901e5299a514",
                    "quartic-symbol-e4a6a9193316a6ff",
                    "quartic-symbol-ef832e4c3b71ee42",
                    "quartic-symbol-f31a234e2bf7b97f",
                    "quartic-symbol-fb5c20c15ce6d778",
                ],
                "record_sha256": (
                    "7c309eec9d225f4c0813f0696e9806d7e5c2c9802528ade40d1d92c5f13d4c56"
                ),
                "selector_record_sha256": (
                    "337daa86bf740ae9e66dbef0829df30297c02e22b8baeb6b90328d608fa66c87"
                ),
            },
            "canonical_obstruction_certificate": {
                "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
                "selector_obligations_classified": 1,
                "candidate_specializations_checked": 12,
                "candidate_compatibilities_certified": 0,
                "candidate_obstructions_certified": 12,
                "obligation_offset": 244,
                "active_indices": [0, 2, 3, 9],
                "zero_eigenspace_factorization": "(34816/15)*alpha^5*W",
                "zero_eigenspace_compression_rank": 2,
                "zero_eigenspace_compression_sha256": (
                    "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
                ),
                "compatibility_iff_over_Q_or_R": "alpha=0",
                "independent_of_c20": True,
                "exact_candidate_witness_gap": "[1088/15,34816/15]",
                "alternative_lower_jet_homogeneous_completion_ruled_out": True,
                "homogeneous_freedom_reduction": {
                    "status": "pass_exact_d4_obstruction_invariant_under_all_lower_homogeneous_freedom",
                    "polarization_directions_checked": 15,
                    "Taylor_orders_per_direction_checked": 5,
                    "total_exact_zero_projector_checks": 300,
                    "lower_jet_reference_kernel_slots_covered_by_identity": 20842,
                    "induced_D4_zero_eigenspace_map_rank": 0,
                    "candidate_obstructions_invariant": 12,
                    "candidate_cancellations": 0,
                    "exact_identity": "R0(Y)^T F_H(Y) R0(Y)=0 for every matrix H(Y)",
                },
                "minimal_algebraic_TC2_escape": {
                    "status": "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only",
                    "correction_basis_dimension": 1,
                    "correction_block_rank": 1,
                    "induced_cokernel_map_rank": 1,
                    "target_cokernel_line_dimension": 1,
                    "candidate_D4_solutions_after_tuning": 12,
                    "candidate_D4_obstructions_after_tuning": 0,
                    "distinct_candidate_eta_values": [
                        "-34816/15",
                        "-1088/15",
                        "1088/15",
                        "34816/15",
                    ],
                    "correction_ansatz": {
                        "definition": "K55(0)^(-1)*(e_16+e_28)*e_21^T",
                        "V_rank": 1,
                        "V_nonzero_entries": 6,
                        "energy_skew_definition": "K55(0)*V-V^T*K55(0)=W",
                        "covariant_or_action_derived": False,
                    },
                    "induced_cokernel_map": {
                        "all_other_equal_eigenspace_compressions_zero": True,
                        "canonical_obstruction_in_image": True,
                        "canonical_obstruction_line_dimension": 1,
                        "corrected_zero_eigenspace_compression": ("((34816/15)*alpha^5+eta)*W"),
                        "domain_basis": ["eta"],
                        "formula": "eta -> eta*W",
                        "image_basis": ["W"],
                        "image_dimension": 1,
                        "rank": 1,
                        "unique_solvability_condition": "eta=-(34816/15)*alpha^5",
                    },
                    "corrected_candidate_family_registered": False,
                    "correction_gauge_constraint_compatible": False,
                    "scope": (
                        "This campaign proves the smallest algebraic state-space TC2 ansatz "
                        "capable of canceling the invariant obligation-244 cokernel witness. "
                        "The tuned coefficient is candidate-specific. No covariant/action "
                        "origin, gauge compatibility, corrected candidate registration, "
                        "remaining D4 selector pass, tube theorem, CK1, CK3, TC2, B7, "
                        "global-H7, or lifespan result is inferred."
                    ),
                },
                "registered_operator_origin_no_go": {
                    "status": "pass_exact_no_go_for_registered_support_preserving_TC2_operator_class",
                    "counts": {
                        "broad_induced_cokernel_map_rank": 0,
                        "broad_support_preserving_domain_dimension": 55,
                        "inferred_global_passes": 0,
                        "negative_controls": 5,
                        "positive_controls": 1,
                        "registered_TC2_blocks_checked": 4,
                        "registered_action_terms_checked": 1,
                        "target_augmented_rank": 1,
                    },
                    "declared_operator_class": {
                        "name": "registered_support_preserving_quartic_TC2_lifts",
                        "general_block": ("B_u(Y)=u(Y)*e_54^T with arbitrary u(Y) in Q(Y)^55"),
                        "domain_dimension_at_one_jet_monomial": 55,
                        "fixed_input_state": {"index": 54, "label": "w1[10]"},
                        "scope_limit": (
                            "This class contains the currently registered linear-X "
                            "quartic-Horndeski TC2 lift and every gauge-fixed "
                            "coefficient/output-row deformation that retains its "
                            "first-order constraint topology. It does not contain a new "
                            "covariant invariant or a deformation that changes the input "
                            "selector."
                        ),
                    },
                    "induced_cokernel_map": {
                        "domain_dimension": 55,
                        "rank": 0,
                        "image_dimension": 0,
                        "augmented_rank": 1,
                        "target_W_rank": 2,
                        "target_W_nonzero_entries": 4,
                        "target_in_image": False,
                    },
                    "constraint_support_audit": {
                        "registered_right_support_columns": [54],
                        "escape_V_right_support_columns": [21],
                        "support_intersection_empty": True,
                        "zero_projector_rank": 33,
                        "interpretation": (
                            "V consumes the stationary w2[10] constraint-sector state, "
                            "whereas every registered TC2 block consumes the high w1[10] "
                            "state. Adding V therefore changes the registered "
                            "derivative-definition/constraint topology."
                        ),
                    },
                    "sharp_result": {
                        "reason": "induced map rank 0 while adjoining W raises rank to 1",
                        "registered_linear_X_quartic_Horndeski_TC2_realizes_V": False,
                        "sharpness": (
                            "The no-go permits arbitrary output u and arbitrary quartic "
                            "coefficient dependence; its only structural restriction is "
                            "the registered e54 input selector. The algebraic V succeeds "
                            "precisely by replacing that selector with e21, so changing "
                            "constraint topology remains an open escape route."
                        ),
                        "support_preserving_class_can_cancel_obligation_244_W": False,
                        "support_preserving_gauge_fixed_deformation_realizes_V": False,
                    },
                    "scope": (
                        "Exact reference-point incompatibility for the currently "
                        "registered linear-X quartic-Horndeski TC2 lift and the broader "
                        "55-dimensional class of deformations B_u=u*e54^T that preserve "
                        "its first-order input selector. No statement is made about new "
                        "covariant invariants, topology-changing gauge reductions, the "
                        "remaining D4 selector, a tube theorem, CK1, CK3, TC2, B7, "
                        "global-H7, or lifespan."
                    ),
                },
                "next_gate": (
                    "Continue the preregistered signed height-one selector at (u,v)=(-1,1) "
                    "for the revised ten-frame symbol; do not infer a finite determining "
                    "theorem or PDE/global admission."
                ),
            },
            "full_fourth_jet_range_closed": False,
        },
        "closure_counts": {
            "full_tube_Sylvester_identities": 0,
            "full_variable_CK1_closures": 0,
            "CK3_closures": 0,
            "TC2_closures": 0,
            "B7_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_missing_premise": (
            "continue_preregistered_signed_height_one_selector_at_minus_one_plus_one_for_"
            "revised_ten_frame_symbol"
        ),
    }
    assert core["cross_pipeline_total"]["status"] == "not_computed"
    assert result["volatile"]["campaign_watchdog_freshness"]["stale"] is True
    assert result["volatile"]["campaign_watchdog_freshness"]["stale_source_reason"]


def test_hash_tamper_fails_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    target = root / SOURCE_PATHS[0]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_degree_three_semantic_tamper_fails_closed_after_resealing(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "quartic_tc2_d4_degree_three_matrix_curl_sphere_extension"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["exact_extension"]["exact_sphere_symbol"]["symbol_sha256"] = "0" * 64
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    raw = target.read_bytes()
    spec["file_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="degree-three sphere extension"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_degree_three_unknown_claim_fails_closed_after_resealing(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "quartic_tc2_d4_degree_three_matrix_curl_sphere_extension"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["claims"]["theory_pass"] = True
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    raw = target.read_bytes()
    spec["file_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="degree-three sphere extension"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_corrected_record",
        "range_hash",
        "completion_pair",
        "sphere_hash",
        "full_sphere_claim",
    ],
)
def test_xyz_completion_semantic_tamper_fails_closed_after_resealing(
    tmp_path: Path, mutation: str
) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "quartic_tc2_d4_degree_three_rank_two_xyz_completion"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    completion = artifact["exact_completion"]
    if mutation == "remove_corrected_record":
        completion["corrected_xyz_result"]["candidate_records"].pop()
    elif mutation == "range_hash":
        completion["exact_range_classification"]["normalized_target_sha256"] = "0" * 64
    elif mutation == "completion_pair":
        completion["minimal_rank_two_completion"]["coordinate_pairs"][0] = [10, 21]
    elif mutation == "sphere_hash":
        completion["exact_sphere_extension"]["symbol_sha256"] = "0" * 64
    else:
        artifact["claims"]["full_direction_sphere_D4_compatibility_proved"] = True
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    spec["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="rank-two xyz completion"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_record",
        "range_hash",
        "completion_pair",
        "term_record",
        "extension_definition",
        "negative_control_key",
        "source_binding",
        "config_hash",
        "scope",
        "full_sphere_claim",
    ],
)
def test_sixth_frame_semantic_tamper_fails_closed_after_resealing(
    tmp_path: Path, mutation: str
) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "quartic_tc2_d4_degree_three_sixth_frame_completion"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    completion = artifact["exact_completion"]
    if mutation == "remove_record":
        completion["corrected_result"]["candidate_records"].pop()
    elif mutation == "range_hash":
        completion["exact_range_classification"]["normalized_target_sha256"] = "0" * 64
    elif mutation == "completion_pair":
        completion["minimal_rank_two_completion"]["coordinate_pairs"][0] = [10, 21]
    elif mutation == "term_record":
        completion["minimal_rank_two_completion"]["term_records"][0]["linear_curl_sha256"] = (
            "0" * 64
        )
    elif mutation == "extension_definition":
        completion["exact_sphere_extension"]["definition"] = "mutated"
    elif mutation == "negative_control_key":
        control = artifact["negative_controls"].pop("rank_zero_completion")
        artifact["negative_controls"]["invented_control"] = control
    elif mutation == "source_binding":
        artifact["source_bindings"]["campaign_test"]["file_sha256"] = "0" * 64
    elif mutation == "config_hash":
        artifact["config_sha256"] = "0" * 64
    elif mutation == "scope":
        artifact["scope"] = "full sphere proved"
    else:
        artifact["claims"]["full_direction_sphere_D4_compatibility_proved"] = True
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    spec["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="sixth-frame"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


@pytest.mark.parametrize(
    "mutation",
    ["candidate_id", "compression_hash", "numerator_hash", "symbolic_reduction", "scope"],
)
def test_rational_chart_semantic_tamper_fails_closed_after_resealing(
    tmp_path: Path, mutation: str
) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "quartic_tc2_d4_rational_chart_determining_gate"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    record = artifact["exact_gate"]["exact_rational_obstruction"]["candidate_records"][0]
    if mutation == "candidate_id":
        record["candidate_id"] = "invented-candidate"
    elif mutation == "compression_hash":
        record["nonzero_equal_eigenspace_compressions"]["0"]["sha256"] = "0" * 64
    elif mutation == "numerator_hash":
        record["zero_speed_cleared_numerator"]["numerator_sha256"] = "0" * 64
    elif mutation == "symbolic_reduction":
        artifact["exact_gate"]["symbolic_chart_reduction"][
            "full_polynomial_identity_reduction_required_after_counterexample"
        ] = True
    else:
        artifact["scope"] = "all PDE and global claims proved"
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    spec["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="rational-chart"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_claim_key",
        "extra_data_key",
        "missing_source_binding",
        "predecessor_binding",
        "primitive_properties",
        "theorem_minimality",
        "fixture_purpose",
        "candidate_identity",
    ],
)
def test_probability_bridge_semantic_tamper_fails_closed_after_resealing(
    tmp_path: Path, mutation: str
) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "kastner_schlatter_actualization_probability_bridge_contract"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "missing_claim_key":
        artifact["claim_seals"].pop("theory_validity_claimed")
    elif mutation == "extra_data_key":
        artifact["data_seals"]["invented_false_key"] = False
    elif mutation == "missing_source_binding":
        artifact["source_bindings"].pop("test")
    elif mutation == "predecessor_binding":
        artifact["source_bindings"]["source_type_audit"]["path"] = "runs/engine/mutated.json"
        artifact["source_bindings"]["source_type_audit"]["file_sha256"] = "0" * 64
    elif mutation == "primitive_properties":
        artifact["primitive_interface"][0]["required_properties"] = ["normalization only"]
    elif mutation == "theorem_minimality":
        artifact["composition_theorem"]["minimality"]["without_core_identity"] = "resolved"
    elif mutation == "fixture_purpose":
        artifact["exact_controls"]["compiler_identity_fixture_positive_control"]["purpose"] = (
            "physical attribution"
        )
    else:
        artifact["candidate_records"][0]["branch_id"] = "mutated_branch"
        artifact["candidate_records"][0]["first_blocker"] = "mutated_blocker"
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    spec["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="actualization probability bridge"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_history_kernel_projective_tamper_fails_closed_after_resealing(
    tmp_path: Path,
) -> None:
    root, config, _ = _fixture(tmp_path)
    label = "kastner_schlatter_history_kernel_projective_admission"
    spec = next(source for source in config["sources"] if source["label"] == label)
    target = root / spec["path"]
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["scope"] = "proves the physical actualization law"
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    spec["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    spec["content_sha256"] = artifact["content_sha256"]

    with pytest.raises(ValueError, match="history-kernel projective"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_portable_artifact_core_and_config_are_hash_bound() -> None:
    config = load_config(REPO / "configs/unified_engine_status.json")
    artifact = json.loads((REPO / "runs/engine/unified-engine-status.json").read_text())
    assert config["schema_version"] == "sigma-unified-engine-status-config-1.0"
    assert artifact["core"]["schema_version"] == "sigma-unified-engine-status-1.0"
    assert (
        hashlib.sha256(_canonical(artifact["core"])).hexdigest() == artifact["core_content_sha256"]
    )
    assert artifact["core"]["data_seals"] == {
        "dark_matter_or_halo_inputs": False,
        "observations_opened": False,
        "paid_llm_in_streaming_promotion_grammar": False,
        "redshift_distance_inputs": False,
    }
    live = json.loads((REPO / "runs/engine/unified-engine-status-live-refresh.json").read_text())
    dashboard = (REPO / "runs/engine/unified-engine-dashboard.html").read_text(encoding="utf-8")
    assert hashlib.sha256(_canonical(live["core"])).hexdigest() == live["core_content_sha256"]
    assert (REPO / "runs/engine/unified-engine-status.json").stat().st_size + (
        REPO / "runs/engine/unified-engine-dashboard.html"
    ).stat().st_size < DEFAULT_MAXIMUM_OUTPUT_BYTES
    assert live["core_content_sha256"] in dashboard
    static_recovery = artifact["core"]["quartic_nonlinear_closure"]["recovery_operator_calculus"]
    live_recovery = live["core"]["quartic_nonlinear_closure"]["recovery_operator_calculus"]
    assert static_recovery == live_recovery
    assert static_recovery["all_candidate_sets_equal"] is True
    assert len(static_recovery["ordered_candidate_ids"]) == 12
    assert (
        static_recovery["anti_wick_composition_prerequisite"]["artifact_binding"]["content_sha256"]
        == "02c98ac16a6cd4bc3871003fb77918e21666a60fec65bc28c85484a6011c541d"
    )
    assert (
        static_recovery["annular_k55_c6"]["counts"]["principal_composition_constants_instantiated"]
        == 12
    )
    assert static_recovery["annular_k55_c6"]["counts"]["full_dyadic_energies_closed"] == 0
    assert static_recovery["dyadic_localization"]["counts"]["full_H7_commutators_closed"] == 0
    assert static_recovery["finite_sobolev_hierarchy_no_go"]["decision_counts"] == {
        "pass": 0,
        "reject": 0,
        "blocked": 12,
    }
    assert (
        static_recovery["finite_sobolev_hierarchy_no_go"]["gate_counts"][
            "autonomous_finite_Sobolev_closures"
        ]
        == 0
    )
    assert "Quartic operator recovery" in dashboard
    assert "Scalar-Hessian output-bundle repair" in dashboard
    assert "396 exact equations in 99 unknowns" in dashboard
    assert "symmetric 891-entry high-field-10 D2 subslice only" in dashboard
    assert "Continuous CPU Epoch 003 terminal result" in dashboard
    assert "six survivor batches" in dashboard
    assert live["core"]["followup_service"]["followup_decision_counts"] == {
        "blocked": 8,
        "pass": 2,
    }
    assert live["core"]["followup_service"]["deferred"] == 0
    assert live["core"]["followup_service"]["current_missing_evaluator_blockers"] == {}
    leaderboards = live["core"]["scientific_leaderboards"]
    assert len(leaderboards["categories"]) == 9
    assert len(leaderboards["history"]) >= 1
    assert all(
        "top10" in category and "full_ranked" in category
        for category in leaderboards["categories"].values()
    )
    assert "Theory formula" in dashboard
    assert "Conformal scalar–tensor gravity" in dashboard
    assert "φ²/100" in dashboard
    assert "Derived operator terms / evidence scope" in dashboard
    assert "Proof and test hierarchy" in dashboard
    assert "1 rejected, 1 blocked, 1 calibration-only" in dashboard
    assert "Formal decision: pass" in dashboard
    assert "Overall: pass" not in dashboard
    assert "How to read a candidate theory" in dashboard
    assert "Notation guide for the displayed actions" in dashboard
    assert "The exact ordered covariant densities remain available" in dashboard
    assert "compact master formula" in dashboard
    assert "G3A-e0eff4150989e3522dc6ba03" in dashboard
    assert "current exact formal tally is 3 pass, 2 reject, and 158 blocked" in dashboard
    assert "G2 formal passes" in dashboard
    assert "G2 Solar fields remaining" in dashboard
    assert "Future preflight passes" in dashboard
    assert "Sandboxed actions" in dashboard
    assert "Independent backend variations" in dashboard
    assert "Euler specializations" in dashboard
    assert "Future Aether blocked" in dashboard
    assert "Negative finite seeds" in dashboard
    assert "Forced characteristic crossings" in dashboard
    assert "Regular-ADM prerequisites" in dashboard
    assert "Legendre inverse margins" in dashboard
    assert "Negative source margins" in dashboard
    assert "Weighted contracts complete" in dashboard
    assert "Aether metric weighted contracts" in dashboard
    assert "reference spectrum (2, 2, 8/3, 4)" in dashboard
    assert "off-diagonal principal symbol remain missing" in dashboard
    assert "Future G3 uniform boxes" in dashboard
    assert "Future G3 AF profiles" in dashboard
    assert "Radial BVP no-go" in dashboard
    assert "Nonradial York no-go" in dashboard
    assert "York cap extensions" in dashboard
    assert "Next caps inconclusive" in dashboard
    assert "G3 analytic York thresholds" in dashboard
    assert "closed class |K| &lt;= kappa_star*v is excluded" in dashboard
    assert "Nontrivial AF solutions" in dashboard
    assert "11 are forced across an ADM characteristic shell" in dashboard
    assert "exact uniform Legendre-sector inverse bounds" in dashboard
    assert "candidate caps 1.211, 1.211, and 1.210" in dashboard
    assert "Staged future candidate formulas (unranked)" in dashboard
    assert "Current exact Aether and G3 boundary" in dashboard
    assert "Aether canonical backgrounds" in dashboard
    assert "Local regular-stratum H-core" in dashboard
    assert "Global declared-profile H-core" in dashboard
    assert "Characteristic shell" in dashboard
    assert "Shell rank" in dashboard
    assert "Shell nullity" in dashboard
    assert "Aether covariant H/D DAGs" in dashboard
    assert "candidate-bound flat-chart canonical seed" in dashboard
    assert "off-flat metric-covariantized" in dashboard
    assert "G3 radial momentum audits" in dashboard
    assert "Real joint coefficients" in dashboard
    assert "<code>1+2k^2=0</code>" in dashboard
    assert "Live dashboard refresh service" in dashboard
    assert "never overwrites the immutable checked snapshot" in dashboard
    assert "Safety readiness artifact" in dashboard
    assert "hardened_service_ready_not_started" in dashboard
    assert "hardened_safety" in dashboard
    assert "<span>Alive</span><strong>True</strong>" in dashboard
    assert "PID identity" in dashboard
    assert "Config current" in dashboard
    assert "individually atomic, not one cross-file transaction" in dashboard
    assert "History seed cap" in dashboard
    assert "seeds compatible scientific-leaderboard history" in dashboard
    assert "incompatible, tampered, or oversized histories fail closed" in dashboard
    assert "These master actions are recompiled from the exact typed cells" in dashboard
    assert "G3A-8555e529226d13e2e9dacad5" in dashboard
    assert "S = integral d^4x" in dashboard
    assert "blocked and rejected staged actions never enter a scientific ranking" in dashboard
    assert "Future reviewed cells" in dashboard
    assert "Future new candidates" in dashboard
    assert "Every action has an exact human-readable master formula" in dashboard
    assert "Quartic nonlinear closure" in dashboard
    assert "Quartic operator recovery" in dashboard
    assert "Op_h^AW(K)=Op_h^W(exp((h/4)Delta)K)" in dashboard
    assert "1/(8*pi)" in dashboard
    assert "compact-frequency defect coefficient <code>4/3</code>" in dashboard
    assert "2^-15" in dashboard
    assert "targeted principal/local results only" in dashboard
    assert "exact finite-hierarchy witness" in dashboard
    assert "a_N=N^(1-s) exp(iNx_1)a_0" in dashboard
    assert "unmodified variable" in dashboard
    assert "Two-channel slice cancellations" in dashboard
    assert "s01[10]/H_01" in dashboard
    assert "257499" in dashboard
    assert "four-entry principal slice" in dashboard
    assert "Diagonal third jets" in dashboard
    assert "Reference mixed sector" in dashboard
    assert "Evidence rank" in dashboard
    assert "Reduced obligations" in dashboard
    assert "Reduced obligations evaluated" in dashboard
    assert "Reduced obligations remaining" in dashboard
    assert "Twenty-five exact mixed chunks supply 1,600 stable records" in dashboard
    assert "closed all 447/447 obligations and all 5,364/5,364 candidate systems" in dashboard
    assert "full 12,300-entry reference mixed third-jet sector is closed" in dashboard
    assert "other 10,700 lexicographic triples" in dashboard
    assert "CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed" in dashboard
    assert "Fourth-order range closure" in dashboard
    assert "Generic G4 equation B.4 normalization" in dashboard
    assert "Exact matches" in dashboard
    assert "24/24" in dashboard
    assert "coefficient normalization only" in dashboard
    assert "CPU real-formula overlap benchmark" in dashboard
    assert "Portable formal-control evidence" in dashboard
    assert "118 / 118" in dashboard
    assert "host-local execution receipt remains separate" in dashboard
    assert "Continuous scientific pipeline admission" in dashboard
    assert "generate → formal → rank" in dashboard
    assert "readiness only; no service or database was started" in dashboard
    assert "Bounded continuous CPU service" in dashboard
    assert "preexecution snapshot" in dashboard
    assert "cleanup-inclusive 120-second deadline" in dashboard
    assert "eight contiguous ordinal intervals" in dashboard
    assert "exactly 15 owned real-formula workers" in dashboard
    assert "Survivors are re-derived from ordinals" in dashboard
    assert "Continuous CPU Epoch 003 genesis" in dashboard
    assert "genesis / not started" in dashboard
    assert "no formulas were evaluated" in dashboard
    assert "preregistered future coverage only" in dashboard
    assert "Epoch 003 candidate-specific survivor follow-up" in dashboard
    assert "11,439 survivors" in dashboard
    assert "192 exact ordinals" in dashboard
    assert "forbidden baryonic atom" in dashboard
    assert "other 11,247 survivors" in dashboard
    assert "resumable paginated survivor-manifest contract" in dashboard
    assert "Scalar-Hessian curl invariance gate" in dashboard
    assert "12 independent antisymmetric pairs" in dashboard
    assert "corrected source Jacobian" in dashboard
    assert "Full ordered D2F coverage ledger" in dashboard
    assert "153x153 ordered atom pairs" in dashboard
    assert "106,920 principal high-atom entries" in dashboard
    assert "256,608 total entries" in dashboard
    assert "Principal high-atom connection extension audit" in dashboard
    assert "8,910 one-sided values" in dashboard
    assert "93 are nonzero" in dashboard
    assert "reverse <code>Pother x P10</code> derivatives" in dashboard
    assert "not a general connection no-go" in dashboard
    assert "TC2 revised-symbol e3 counterexample and bounded escape" in dashboard
    assert "(u,v)=(0,1)" in dashboard
    assert "4,943" in dashboard
    assert "110 five-support envelopes" in dashboard
    assert "not a finite determining theorem" in dashboard
    assert "TC2 revised-eight-frame (1,1) counterexample and bounded escape" in dashboard
    assert "(u,v)=(1,1)" in dashboard
    assert "n=(-1/3,2/3,2/3)" in dashboard
    assert "1,940" in dashboard
    assert "15 feasible four-support envelopes" in dashboard
    assert "nine exact direction certificates" in dashboard
    assert "TC2 revised-nine-frame (1,-1) counterexample and bounded escape" in dashboard
    assert "(u,v)=(1,-1)" in dashboard
    assert "n=(-1/3,2/3,-2/3)" in dashboard
    assert "two feasible four-support envelopes" in dashboard
    assert "a10(n)=3*n1**2*n2*n3/8" in dashboard
    assert "ten exact direction certificates" in dashboard
    assert "remaining signed selector points" in dashboard
    assert "never assign rank directly" in dashboard
    assert "22,478,848 unique candidate-grid evaluations" in dashboard
    assert "89.4% median and 100% peak device-wide CPU" in dashboard
    assert "triggering backoff and no persistent resource-policy change" in dashboard
    assert "sampled-static screen reject, not a theory rejection" in dashboard
    assert "RTX 5090 synthetic formula stress" in dashboard
    assert "Einstein-Aether constrained coupling boundaries" in dashboard
    assert "Ambient-only witnesses" in dashboard
    assert "tangent rank nine and KKT rank eleven" in dashboard
    assert "generic aligned coupling-boundary classification only" in dashboard
    assert "87.5-billion-evaluation timing loop" in dashboard
    assert "device-wide and can include concurrent processes" in dashboard
    assert "GPU/CPU violations" in dashboard
    assert "transactional-gravity proposal" in dashboard
    assert "Graph nodes" in dashboard
    assert "Compiler action hypotheses" in dashboard
    assert "Paper-derived actions" in dashboard
    assert "Registered fields" in dashboard
    assert "Missing fields" in dashboard
    assert "Synthetic power evals" in dashboard
    assert "103079215104" in dashboard
    assert "Poisson alt detection" in dashboard
    assert "BTFR alt detection" in dashboard
    assert "Sufficient selector contracts" in dashboard
    assert "Registered selector nodes" in dashboard
    assert "RTX generated counts" in dashboard
    assert "110100480" in dashboard
    assert "n=64 likelihood power" in dashboard
    assert "n=1024 factorial power" in dashboard
    assert "full set-indexed Laplace functional" in dashboard
    assert "single scalar PMF node" in dashboard
    assert "finite-sample separability" in dashboard
    assert "Conditional Poisson kernel completion" in dashboard
    assert "Compiler kernels" in dashboard
    assert "N|(g,phi)~PRM(q0*exp(phi)*dVol_g)" in dashboard
    assert "Actualization map and set-indexed RTX falsification" in dashboard
    assert "QED actualization-to-Poisson derivation audit" in dashboard
    assert "Microscopic obligations" in dashboard
    assert "Same-rate non-Poisson witnesses" in dashboard
    assert "variance <code>2 mu</code>" in dashboard
    assert (
        "registered paper/QED evidence closes none of those twelve microscopic premises"
        in dashboard
    )
    assert "Deterministic compensator admission boundary" in dashboard
    assert "Positive mean measures" in dashboard
    assert "deterministic predictable compensator" in dashboard
    assert "mu_B+mu_B^2/4" in dashboard
    assert "does not select a physical stochastic law" in dashboard
    assert "Canonical conditional probability-space boundary" in dashboard
    assert "Omega=N_lf(W)" in dashboard
    assert "six of ten typed obligations" in dashboard
    assert "same-action Cox completion" in dashboard
    assert "Deterministic-feature selector no-go" in dashboard
    assert "selectors factoring through" in dashboard
    assert "exp(-2) cosh(1)" in dashboard
    assert "Second-order stochastic selector no-go" in dashboard
    assert "P(N_B=0)=1/3" in dashboard
    assert "full second factorial measure" in dashboard
    assert "Finite factorial-hierarchy selector no-go" in dashboard
    assert "Arbitrary-order theorems" in dashboard
    assert "no fixed finite hierarchy uniquely selects Poisson" in dashboard
    assert "an infinite hierarchy would still require a determinacy premise" in dashboard
    assert "Countable full-law selector admission" in dashboard
    assert "Sufficient routes" in dashboard
    assert "rational simple functions" in dashboard
    assert "zero Laplace-core certificates" in dashboard
    assert "Source-domain selector type audit" in dashboard
    assert "Laplace partial" in dashboard
    assert "Mecke absent" in dashboard
    assert "inhabits neither source-bound certificate" in dashboard
    assert "Minimal actualization-probability bridge" in dashboard
    assert "Projective actualization-history kernel admission" in dashboard
    assert "K=N_A+N_B~Pois(2)" in dashboard
    assert "e^-1" in dashboard
    assert "(1+e^-2)/2" in dashboard
    assert "compiler-only two-cell control" in dashboard
    assert "Q:(g,phi)-&gt;Prob(H,Sigma_H)" in dashboard
    assert "P=C_*Q" in dashboard
    assert "compiler-only" in dashboard
    assert "Operational event exposure and continuous RTX execution" in dashboard
    assert "one GPU owner" in dashboard
    assert "Service cycles" in dashboard
    assert "GPU start ceiling" in dashboard
    assert "8087 MiB free" in dashboard
    assert "no runtime, queue, lease, or worker was created" in dashboard
    assert "Deferred GPU ownership" in dashboard
    assert "Safe samples required" in dashboard
    assert "Maximum wait" in dashboard
    assert "registered readiness sample was 99% utilized with 8083 MiB free" in dashboard
    assert "no CUDA context" in dashboard
    assert "Deferred RTX scheduler handoff" in dashboard
    assert "Wait polls/cycle" in dashboard
    assert "24-cycle/24-hour bound" in dashboard
    assert "volatile runtime panel" in dashboard
    assert "PID identity" in dashboard
    assert "Consecutive safe" in dashboard
    assert "device-wide runtime measurements" in dashboard
    assert "isolated queue SQLite" in dashboard
    assert "TC2 spatial-gradient completion no-go" in dashboard
    assert "TC2 full fixed-B1 linear completion no-go" in dashboard
    assert "11 v columns project to zero" in dashboard
    assert "11 q columns are nonzero but incapable" in dashboard
    assert "wedge range rank 473" in dashboard
    assert "not every operator class" in dashboard
    assert "TC2 minimal parity-cubic angular escape" in dashboard
    assert "a(n)=n1^2" in dashboard
    assert "all 12 previous axis-two obstructions become exact compatibilities" in dashboard
    assert "xi1^2/|xi|^2" in dashboard
    assert "Generic-direction D4 compatibility" in dashboard
    assert "TC2 parity-cubic generic-direction audit" in dashboard
    assert "n=(3/5,4/5,0)" in dashboard
    assert "15 exact polarization evaluations" in dashboard
    assert "zero compatibilities and 12 exact obstructions" in dashboard
    assert "not the full direction sphere" in dashboard
    assert "TC2 fixed-frame matrix-curl completion" in dashboard
    assert "1,210-parameter matrix-valued wedge range has rank 473" in dashboard
    assert "single combined-curl, rank-one block" in dashboard
    assert "makes all 12 candidate D4 systems compatible" in dashboard
    assert "exact positive result at one fixed frame only" in dashboard
    assert "TC2 degree-three matrix-curl sphere extension" in dashboard
    assert "DeltaB(n)=(25/12)n1*n2*w*(n1 e21-n2 e54)^T" in dashboard
    assert "n=(3/5,0,4/5)" in dashboard
    assert "All 12 candidates retain rank-two zero-speed obstructions" in dashboard
    assert "not broader matrix-curl channel or envelope classes" in dashboard
    assert "TC2 degree-three C23 great-circle escape" in dashboard
    assert "TC2 rank-two XYZ completion" in dashboard
    assert "n=(1/3,2/3,2/3)" in dashboard
    assert "Rank one is impossible" in dashboard
    assert "two-wedge, rank-two" in dashboard
    assert "five-direction predecessor" in dashboard
    assert "TC2 sixth rational-frame completion" in dashboard
    assert "n=(2/3,1/3,2/3)" in dashboard
    assert "a6(n)=(3/2)n3(4n1+n2-3n3)" in dashboard
    assert "Six exact directions are now certified" in dashboard
    assert "does not determine the full sphere" in dashboard
    assert "TC2 rational-chart full-sphere counterexample" in dashboard
    assert "TC2 degree-five counterexample escape" in dashboard
    assert "a7(n)=(81/14)n2*n3(2*n1*n2-n3^2)" in dashboard
    assert "Seven exact directions are certified only" in dashboard
    assert "(u,v)=(2/5,1/5)" in dashboard
    assert "n=(2/3,2/3,1/3)" in dashboard
    assert "rank-four, 56-entry zero-speed compression" in dashboard
    assert "disproves full-sphere D4 compatibility of the current combined symbol" in dashboard
    assert "does not rule out a new preserving topology-changing angular correction" in dashboard
    assert "DeltaB23(n)=(25/16)n3^2*w23*(n3 e21-n2 e32)^T" in dashboard
    assert "At this predecessor milestone four directions were certified" in dashboard
    assert "the next certificate resolves that frame" in dashboard
    assert "Paper-complete count maps" in dashboard
    assert "1887436800" in dashboard
    assert "8053063680" in dashboard
    assert "322122547200" in dashboard
    assert "block common-shock alternatives preserve every one-cell Poisson marginal" in dashboard
    assert "Readiness advanced" in dashboard
    assert "Positive coordinate actions" in dashboard
    assert "Strict-positive solution gates" in dashboard
    assert "q=q0*exp(phi)" in dashboard
    assert "No paper or QED derivation selecting that positive sector is registered" in dashboard
    assert "Covariant intensity measures" in dashboard
    assert "Poisson/Cox witnesses" in dashboard
    assert "mu_q(B)=Integral_B q*dVol_g" in dashboard
    assert "mu+epsilon^2*mu^2" in dashboard
    assert "Scientific tests" in dashboard
    assert "Extended-source laws" in dashboard
    assert "Geometry cases" in dashboard
    assert "Source interactions" in dashboard
    assert "Lensing cases" in dashboard
    assert "beta=1/2 matches Equation 35" in dashboard
    assert "Canonical dynamic-class matches" in dashboard
    assert "New gravity operator classes" in dashboard
    assert "Global positive energy" in dashboard
    assert "Scalar CUDA evals" in dashboard
    assert "existing canonical Einstein-scalar control" in dashboard
    assert "identical linear dynamics" in dashboard
    assert "Transactional action boundary and point-process gate" in dashboard
    assert "Charge interfaces" in dashboard
    assert "Var(N)=E(mu)+Var(mu)" in dashboard
    assert "not a unique probability law" in dashboard
    assert "Coupled crossing witnesses" in dashboard
    assert "Unrestricted positivity rejects" in dashboard
    assert "0&lt;Tau_cross&lt;=q0/v" in dashboard
    assert "not either action" in dashboard
    assert "naive local-superposition completion is rejected as a hypothesis" in dashboard
    assert "do not establish the transactional ontology" in dashboard
    assert "Exact selector" in dashboard
    assert "Obligations closed" in dashboard
    assert "Polarization directions" in dashboard
    assert "Lower-jet slots covered" in dashboard
    assert "Algebraic correction basis" in dashboard
    assert "Correction-map rank" in dashboard
    assert "Tuned D4 solutions" in dashboard
    assert "Covariant origin" in dashboard
    assert "Registered-origin map rank" in dashboard
    assert "Registered TC2 blocks" in dashboard
    assert "TC2 topology-changing origin classification" in dashboard
    assert "Capable selectors" in dashboard
    assert "21, 44, 48, 51, and 53" in dashboard
    assert "R0^T(HP-P^T H)R0=0" in dashboard
    assert "no homogeneous lower-jet completion can cancel" in dashboard
    assert "eta=-(34816/15) alpha^5" in dashboard
    assert "This remains an algebraic escape, not a physical correction" in dashboard
    assert "support-preserving gauge deformation" in dashboard
    assert "changing constraint topology" in dashboard
    assert "TC2 fixed-gauge curl-constraint admission" in dashboard
    assert "Source curl constraints" in dashboard
    assert "Definition constraints" in dashboard
    assert "Curl constraints" in dashboard
    assert "eta(Y) u C_12^[10]" in dashboard
    assert "fixed-gauge constraint-surface admission" in dashboard
    assert "TC2 companion-direction range audit" in dashboard
    assert "Eigenspaces audited" in dashboard
    assert "Pure-curl range rank" in dashboard
    assert "Augmented rank" in dashboard
    assert "rank 297" in dashboard
    assert "raises the rank to 298" in dashboard
    assert "TC2 axis-two base D4 obstruction test" in dashboard
    assert "Axis-two obstructions" in dashboard
    assert "zero cancellations, zero compatibilities" in dashboard
    assert "12 exact axis-two obstructions" in dashboard
    assert "proposed fixed-chart C12 curl completion" in dashboard
    assert "not yet an axis-two obstruction theorem" not in dashboard
    assert "-eta*C_companion" in dashboard
    assert "No full formal pass is inferred" not in dashboard
    assert "class #1" in dashboard
    assert "g4_global_positive_energy: 1" not in dashboard
    assert "completed in separate evidence classes" in dashboard
    assert "solar_prediction_obligation" in dashboard
    assert "LLM budget and proposal quarantine" in dashboard
    assert "quarantine_until_downstream_validation" in dashboard
    assert len(dashboard.encode()) < 524288
    assert "C:\\" not in dashboard


def _write_fixture_config(root: Path, config: dict[str, object]) -> Path:
    body = {
        "schema_version": "sigma-unified-engine-status-config-1.0",
        **config,
    }
    body["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path = root / "configs/unified_engine_status.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def test_standalone_refresh_and_dashboard_keep_watchdog_database_read_only(
    tmp_path: Path,
) -> None:
    root, config, database = _fixture(tmp_path)
    _write_fixture_config(root, config)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert (
        main(
            [
                "refresh",
                "--project-root",
                str(root),
                "--output",
                "runs/engine/refreshed.json",
                "--dashboard-output",
                "runs/engine/dashboard.html",
                "--maximum-output-bytes",
                "1200000",
                "--disable-leaderboards",
                "--disable-gpu-sample",
                "--sampled-at-utc",
                "2026-08-10T20:10:00+00:00",
            ]
        )
        == 0
    )
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert before == after

    snapshot_path = root / "runs/engine/refreshed.json"
    dashboard_path = root / "runs/engine/dashboard.html"
    snapshot = json.loads(snapshot_path.read_text())
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert snapshot["volatile"]["physical_gpu"] == {
        "availability": "disabled",
        "source": "disabled_by_operator",
    }
    assert len(snapshot_path.read_bytes()) < 1048576
    assert len(dashboard_path.read_bytes()) < 1048576
    assert "Scheduler lanes" in dashboard
    assert "Physical hardware sample" in dashboard
    if snapshot["volatile"]["physical_cpu"]["availability"] == "available":
        assert "Physical CPU utilization" in dashboard
        assert "CPU topology" in dashboard
    else:
        assert "Physical CPU telemetry" in dashboard
        assert snapshot["volatile"]["physical_cpu"]["reason"] in dashboard
    assert "C:\\" not in dashboard

    assert (
        main(
            [
                "export-dashboard",
                "--project-root",
                str(root),
                "--snapshot",
                "runs/engine/refreshed.json",
                "--output",
                "runs/engine/dashboard-replay.html",
                "--maximum-output-bytes",
                "1048576",
            ]
        )
        == 0
    )
    assert (root / "runs/engine/dashboard-replay.html").read_bytes() == dashboard_path.read_bytes()


def test_standalone_output_budget_and_path_escape_fail_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    _write_fixture_config(root, config)
    output = root / "runs/engine/too-large.json"
    with pytest.raises(RuntimeError, match="bounded JSON output"):
        main(
            [
                "refresh",
                "--project-root",
                str(root),
                "--output",
                "runs/engine/too-large.json",
                "--maximum-output-bytes",
                "4096",
                "--disable-gpu-sample",
            ]
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="escapes project root"):
        main(
            [
                "refresh",
                "--project-root",
                str(root),
                "--output",
                "../escaped.json",
                "--disable-gpu-sample",
            ]
        )
