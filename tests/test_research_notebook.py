from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.research_notebook import (
    NotebookValidationError,
    build_action_jet_nonidentifiability_notebook,
    build_component_map_schema_ambiguity_notebook,
    build_flat_factorized_leaf_jet_d2_notebook,
    build_natural_sum_notebook,
    build_ordered_mixed_d2_differentiability_notebook,
    build_p10_arbitrary_background_leaf_derivative_notebook,
    build_p10_inverse_product_replay_notebook,
    build_pother_arbitrary_background_leaf_derivative_notebook,
    build_quartic_survivor_notebook,
    build_registered_variation_selection_notebook,
    materialize_example_notebooks,
    render_ipynb,
    render_markdown,
    validate_notebook,
)

ROOT = Path(__file__).resolve().parents[1]
ACTION_JET_RECEIPT = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json"
)
REGISTERED_VARIATION_RECEIPT = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-registered-variation-selection-audit/campaign.json"
)
COMPONENT_MAP_RECEIPT = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json"
)
ORDERED_MIXED_D2_RECEIPT = (
    "runs/physics-language/"
    "quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json"
)
FLAT_FACTORIZED_D2_RECEIPT = (
    "runs/physics-language/quartic-flat-factorized-leaf-jet-d2-specialization-gate/campaign.json"
)
P10_ARBITRARY_BACKGROUND_RECEIPT = (
    "runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json"
)
P10_REPLAY_RECEIPT = (
    "runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json"
)
POTHER_LEAF_RECEIPT = (
    "runs/physics-language/quartic-pother-arbitrary-background-leaf-derivative-gate/campaign.json"
)
EXISTING_NOTEBOOK_SHA256 = {
    "natural-sum-rediscovery.ipynb": (
        "63cb0b063c0bce8cf26ecf2b3fcaf035424a3307f5d44875ff9a423e9b309154"
    ),
    "natural-sum-rediscovery.md": (
        "4953a9679b766f1211bdeaa3eace191851032196abae41db968c82e78c83ff6d"
    ),
    "quartic-local-survivor.ipynb": (
        "19392a677da4a4a27d09fcbf26294f0e48099e13be0bb596663f35dce8bee8f8"
    ),
    "quartic-local-survivor.md": (
        "d4f09ae04dad567f001e871ccaf587125282ffd560ebfa58c515269dc67632af"
    ),
    "quartic-action-jet-nonidentifiability.ipynb": (
        "6330f947b56625566b5d98016ae71f4f2551e48ea874c9d540939e6ed8f37098"
    ),
    "quartic-action-jet-nonidentifiability.md": (
        "5e2180b69ec5cd9a72a45c6f17f2b0f67af757224d73e9250915195f097aaeca"
    ),
    "quartic-registered-variation-selection-audit.ipynb": (
        "84c988dfe42f82f5ecaec6a302c2d8f276d1ff21251020e42ce3111db68e36f2"
    ),
    "quartic-registered-variation-selection-audit.md": (
        "75b918bfc2d7598f4b76c59af5eec118a04bcd78715f5968eed9631c0557cba6"
    ),
    "quartic-component-map-schema-ambiguity.ipynb": (
        "d8bd61397bd8bb0b20af5fb8b7e986e57b1ba28494ecef036f4d4f207f370e1c"
    ),
    "quartic-component-map-schema-ambiguity.md": (
        "4bc82c9ecfc46b3f0df6dcead9eef51b90f575432607c86148918598108cb90a"
    ),
    "quartic-ordered-mixed-d2-differentiability.ipynb": (
        "088962cc63283ab0f28f9f3d7c452fb1047b81a020f3d28a85ea2cee00e53add"
    ),
    "quartic-ordered-mixed-d2-differentiability.md": (
        "b2128f2c12d2e6ee4d8b048a866b0c0a677f878d7ac0276f1710fca3c505e366"
    ),
    "quartic-flat-factorized-leaf-jet-d2.ipynb": (
        "dac43280b961ab73878336b0f741a4f88058f34a5649f59097f763da04c4c97a"
    ),
    "quartic-flat-factorized-leaf-jet-d2.md": (
        "f2ed5111a1b888b446b71407692ce60acbd2e99fddcf0cd03f02b8f00d0fc4c6"
    ),
    "quartic-p10-arbitrary-background-leaf-derivatives.ipynb": (
        "338ed5643c459599dc42afd4e68cc7ef3d1a68328bf0fa8f82a6422f79bb17ef"
    ),
    "quartic-p10-arbitrary-background-leaf-derivatives.md": (
        "be34aa6eb7e3af80942db64a92bc1644f44fb0dfcb536bca0efef03132fbeb52"
    ),
    "quartic-p10-inverse-product-d2-replay.ipynb": (
        "5372e1079ebc68f3227842e37539ed7f2e6bcd054a610144c69742cc06383e16"
    ),
    "quartic-p10-inverse-product-d2-replay.md": (
        "9cf0ecbd0b42508af860918241f566085f486bfcd098b73310164e9a0074dfe1"
    ),
}


def test_natural_sum_notebook_reads_like_a_proof_without_overclaiming() -> None:
    notebook = build_natural_sum_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "q(n)=\\frac{n^2+n}{2}" in markdown
    assert "q(n+1)-q(n)" in markdown
    assert "every nonnegative integer" in markdown
    assert "46,656 raw coefficient triples" in markdown
    assert "59 additional" in markdown
    assert "not a novelty claim" in markdown
    assert "private model reasoning" in markdown


def test_quartic_notebook_preserves_local_global_boundary() -> None:
    notebook = build_quartic_survivor_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "formal_local_survivor"
    assert "rank 6 and nullity 1" in markdown
    assert "N_{\\rm dof}=\\frac{20-2(6)-2}{2}=3" in markdown
    assert "257{,}499" in markdown
    assert "256,608 entries remain" in markdown
    assert "106,920 principal high-atom" in markdown
    assert "not yet an admitted global theory" in markdown
    assert "lifespan remain unproved" in markdown


def test_action_jet_notebook_proves_only_finite_data_nonidentifiability() -> None:
    notebook = build_action_jet_nonidentifiability_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "p(g)&=(g+1)(g+\\tfrac{1}{2})(g-\\tfrac{1}{2})(g-1)" in markdown
    assert "g^4-\\tfrac{5}{4}g^2+\\tfrac{1}{4}" in markdown
    assert "| $-1$ | $0$ | $-3/2$ | $19/2$ |" in markdown
    assert "| $-1/2$ | $0$ | $3/4$ | $1/2$ |" in markdown
    assert "| $1/2$ | $0$ | $-3/4$ | $1/2$ |" in markdown
    assert "| $1$ | $0$ | $3/2$ | $19/2$ |" in markdown
    assert "22-parameter family" in markdown
    assert "all 88 recorded first-jet samples" in markdown
    assert "all 88 second-jet samples unidentified" in markdown
    assert "not** a no-go theorem for a covariant action derivation" in markdown
    assert "complete ordered $D^2F$" in markdown
    assert "global $H^7$" in markdown
    assert "nonlinear PDE closure, and lifespan all remain open" in markdown
    assert (
        "registered_local_covariant_variation_rule_or_corrected_second_source_jet_values_"
        "required_to_select_one_extension_from_the_exact_22_parameter_jet_ambiguity_family"
        in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": ACTION_JET_RECEIPT,
            "file_sha256": ("e0b87eb270d73f1fa7acb1ff31e0f234a545cf80c383fac21ffa0abc390a902b"),
            "content_sha256": ("b73d3bb175cf008f080ac900c0aed7f463f341d8efc8ebd4cdc4a8fbc3b6de21"),
        }
    ]
    rendered = render_ipynb(notebook)
    assert all(cell["cell_type"] == "markdown" for cell in rendered["cells"])


def test_registered_variation_notebook_proves_only_closed_inventory_rank_zero() -> None:
    notebook = build_registered_variation_selection_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "The four registered evidence bundles" in markdown
    assert "| Generic $G_4$ metric variation | 24 | no | no | 0 |" in markdown
    assert "| Generated metric variations | 163 | no | no | 0 |" in markdown
    assert "| Universal source DAG | 1,056 | yes | no | 0 |" in markdown
    assert "| Full source $D^1$ | 20,196 | yes | no | 0 |" in markdown
    assert "\\sum_{i=0}^{21} a_{ri}\\lambda_i=b_r" in markdown
    assert "A\\in\\mathbb K^{0\\times22}" in markdown
    assert "\\operatorname{rank}(A)=0" in markdown
    assert "\\operatorname{nullity}(A)=22-0=22" in markdown
    assert "zero parameters are selected and all 22 remain free" in markdown
    assert "absence of a row does not justify setting $\\lambda_i=0$" in markdown
    assert "not a physical no-go" in markdown
    assert "All 12 downstream candidates remain blocked, not rejected" in markdown
    assert (
        "candidate_bound_component_map_from_the_registered_G4_variation_or_source_DAG_into_"
        "the_22_output_connection_coordinates_or_exact_corrected_second_source_jet_values_"
        "required" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": REGISTERED_VARIATION_RECEIPT,
            "file_sha256": ("dfc8940a6f092de73da5641afd95c6cbf997b73ad63f8fa6f4ea3eaa8f395a20"),
            "content_sha256": ("6de93ca6700b21ff9f858a2b7f01d1a9d103271de1dde3f75385faaaa4a377d6"),
        }
    ]
    rendered = render_ipynb(notebook)
    assert all(cell["cell_type"] == "markdown" for cell in rendered["cells"])


def test_component_map_notebook_constructs_only_schema_ambiguity() -> None:
    notebook = build_component_map_schema_ambiguity_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "22-by-24 projection problem" in markdown
    assert "M\\in\\mathbb K^{22\\times24}" in markdown
    assert "22\\cdot24=528" in markdown
    assert "constraint rank is exactly 22" in markdown
    assert "\\dim\\{M:Mc=\\beta\\}=528-22=506" in markdown
    assert "M_{00}=1/2" in markdown
    assert "M_{01}=-1" in markdown
    assert "\\tfrac12(1)+(-1)(-\\tfrac12)=1=\\beta_0" in markdown
    assert "two distinct exact $22\\times24$ maps" in markdown
    assert "D^2_{\\mathrm{mixed}}F_i=\\mu_i" in markdown
    assert "22 unit vectors $e_0,\\ldots,e_{21}$" in markdown
    assert "23 explicit, pairwise distinct completions" in markdown
    assert "22 independent parameters" in markdown
    assert "not** a physical no-go" in markdown
    assert "not certified covariant physical maps" in markdown
    assert "complete $D^2F$" in markdown
    assert "global $H^7$, nonlinear PDE closure, or lifespan" in markdown
    assert (
        "register_the_typed_generic_term_to_source_component_projection_P10_Pother_state_"
        "tangent_embedding_and_22_ordered_mixed_D2F_roots" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": COMPONENT_MAP_RECEIPT,
            "file_sha256": ("0256f64acb53f38c0cada5e43a58c974b7f9bebe2529bdf7c3f08e65b9d2563f"),
            "content_sha256": ("3a3da9ecef30e596ae18cb8e76687338a9fe1bf8e7284ee009287420ce5613ec"),
        }
    ]
    rendered = render_ipynb(notebook)
    assert all(cell["cell_type"] == "markdown" for cell in rendered["cells"])


def test_ordered_mixed_d2_notebook_marks_missing_leaf_jets_unknown() -> None:
    notebook = build_ordered_mixed_d2_differentiability_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "13,983 arithmetic-DAG nodes" in markdown
    assert "341 distinct component-input labels" in markdown
    assert "exactly 132 component-input leaves" in markdown
    assert "D(c)=0" in markdown
    assert "D(xy)=D(x)y+xD(y)" in markdown
    assert "D(x)y-xD(y)" in markdown
    assert "20\\cdot132=2{,}640" in markdown
    assert "12\\cdot2{,}640=31{,}680" in markdown
    assert "34,848" in markdown
    assert "zero registered leaf-derivative roots" in markdown
    assert "zero registered ordered mixed-$D^2$ roots out of 264 targets" in markdown
    assert "not the equation $D(x)=0$" in markdown
    assert "not a vanishing theorem" in markdown
    assert "do not prove a physical no-go" in markdown
    assert "complete $D^2F$" in markdown
    assert "global $H^7$, nonlinear PDE closure, or lifespan" in markdown
    assert (
        "register_candidate_bound_coordinate_derivatives_for_the_31680_reachable_A_B_C_"
        "component_input_leaf_obligations" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": ORDERED_MIXED_D2_RECEIPT,
            "file_sha256": ("2992571c544846efc96142e2e4a74efe280a7bb025efadb1ff945ab9515bafcc"),
            "content_sha256": ("d8afd9f91c090ad1c07e4bb22257baa8c61c095f8d434e02a27082b5591abb6a"),
        }
    ]
    rendered = render_ipynb(notebook)
    assert all(cell["cell_type"] == "markdown" for cell in rendered["cells"])


def test_flat_factorized_d2_notebook_preserves_specialization_boundary() -> None:
    notebook = build_flat_factorized_leaf_jet_d2_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "31,680 candidate-bound derivatives" in markdown
    assert "20 target formulas per candidate" in markdown
    assert "D_{s_i}F=\\sum_a\\frac{\\partial F}{\\partial J_a}" in markdown
    assert "12\\cdot22=264" in markdown
    assert "| $0$ | 192 |" in markdown
    assert "| $-1$ | 18 |" in markdown
    assert "| $-1/2$ | 18 |" in markdown
    assert "| $1/2$ | 18 |" in markdown
    assert "| $1$ | 18 |" in markdown
    assert "72 values are nonzero" in markdown
    assert "\\{-1,-\\tfrac12,0,\\tfrac12,1\\}" in markdown
    assert "five-node DAG" in markdown
    assert "zero of 264 general-background $D^2$ roots" in markdown
    assert "not a nonlinear arbitrary-background chain rule" in markdown
    assert "complete $D^2F$" in markdown
    assert "global $H^7$, nonlinear PDE closure, lifespan" in markdown
    assert (
        "register_the_nonlinear_arbitrary_background_coordinate_to_covariant_jet_map_and_"
        "candidate_bound_A_B_C_leaf_derivatives" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": FLAT_FACTORIZED_D2_RECEIPT,
            "file_sha256": ("7f433906323391a8b84179d2abb63b0b107fadad2667a8f6350d3add357a7d1c"),
            "content_sha256": ("be94d39348864e642a0b4460c35f845e21f7cd093792f0ce97eab152505bfd2a"),
        }
    ]
    assert all(cell["cell_type"] == "markdown" for cell in render_ipynb(notebook)["cells"])


def test_p10_arbitrary_background_notebook_stops_before_d2_propagation() -> None:
    notebook = build_p10_arbitrary_background_leaf_derivative_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "H_{ij}=\\nabla_i\\nabla_j\\phi" in markdown
    assert "\\frac{\\partial H_{ij}}{\\partial s_{ij}[10]}=1" in markdown
    for tangent in ("H_{11}", "H_{12}", "H_{13}", "H_{22}", "H_{23}"):
        assert tangent in markdown
    assert "5\\cdot132=660" in markdown
    assert "5\\cdot4=20\\text{ nonzero}" in markdown
    assert "5\\cdot128=640\\text{ zero}" in markdown
    assert "7,920 roots: 240 nonzero and 7,680 zero" in markdown
    assert "exactly eleven constant nodes" in markdown
    assert "zero of 84 P10 ordered-$D^2$ targets" in markdown
    assert "constructive partial progress" in markdown
    assert "23,760 leaf roots remain unregistered" in markdown
    assert "complete $D^2F$" in markdown
    assert "global $H^7$, nonlinear PDE closure, lifespan" in markdown
    assert (
        "differentiate_and_replay_the_bound_inverse_product_D1_DAG_using_the_7920_registered_"
        "P10_leaf_roots_then_register_Pother_leaf_derivatives" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": P10_ARBITRARY_BACKGROUND_RECEIPT,
            "file_sha256": ("c74171c48d7fc4f80de8f0c51b2b2700a1ce33de8795c3a999cee7c957b35869"),
            "content_sha256": ("51f76fa7ebc81ab2f570bfe5ad920215420e005687d0c861b24ea6da766c37e0"),
        }
    ]
    assert all(cell["cell_type"] == "markdown" for cell in render_ipynb(notebook)["cells"])


def test_p10_replay_notebook_closes_subset_without_promoting_full_d2() -> None:
    notebook = build_p10_inverse_product_replay_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "all 7,920 bound input roots" in markdown
    assert "811,296 nodes" in markdown
    assert "786,396 exact derived Merkle nodes" in markdown
    assert "\\dot{(xy)}=\\dot x\\,y+x\\,\\dot y" in markdown
    assert "\\dot x\\,y-x\\,\\dot y" in markdown
    assert "\\det(A)$ is nonzero" in markdown
    assert "12\\cdot5=60" in markdown
    assert "12\\cdot7=84" in markdown
    assert "84 of 84 arbitrary-background P10" in markdown
    assert "180 ordered targets remain blocked" in markdown
    assert "84 of 264 targets" in markdown
    assert "not a complete ordered $D^2F$ tensor" in markdown
    assert "global $H^7$, nonlinear PDE closure, lifespan" in markdown
    assert (
        "register_candidate_bound_Pother_A_B_C_leaf_derivatives_then_replay_the_remaining_"
        "180_ordered_mixed_D2_roots" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": P10_REPLAY_RECEIPT,
            "file_sha256": ("2a9814a27123099b9e942bde72fa45fe8783e3ddde0743d080b17008dbb9318c"),
            "content_sha256": ("e02949cb28f43851483d2b0b6cb06c6710ac53a16f210150449d85ceb0ec92ba"),
        }
    ]
    assert all(cell["cell_type"] == "markdown" for cell in render_ipynb(notebook)["cells"])


def test_pother_leaf_notebook_marks_records_ready_but_not_replayed() -> None:
    notebook = build_pother_arbitrary_background_leaf_derivative_notebook(ROOT)
    value = notebook.to_dict()
    validate_notebook(value)
    markdown = render_markdown(notebook)
    assert notebook.verdict == "proved"
    assert "15 Pother coordinate-second-metric directions" in markdown
    assert "15\\cdot132=1{,}980" in markdown
    assert "12\\cdot1{,}980=23{,}760" in markdown
    assert "156 nonzero and 23,604 zero" in markdown
    assert "s11[4] and s22[7]" in markdown
    assert "all 132 reachable leaf derivatives are zero" in markdown
    assert "all 180 records globally—are replay-ready" in markdown
    assert "zero of the 180 Pother ordered-$D^2$ roots is emitted" in markdown
    assert "Readiness certifies the composition input domain" in markdown
    assert "84 admitted P10 roots and 180 blocked Pother roots" in markdown
    assert (
        "replay_the_bound_inverse_product_D1_DAG_along_the_23760_registered_Pother_leaf_"
        "roots_to_seal_the_remaining_180_ordered_D2_roots" in markdown
    )
    assert value["source_bindings"] == [
        {
            "path": POTHER_LEAF_RECEIPT,
            "file_sha256": ("c687e15839628dcca3480740ea8ee568576461c200ce8937ab5499a71e9e49c2"),
            "content_sha256": ("b2f4eacd73026bc92a057be0ad5340d487ef0ff3b18353e3412d7aea5475b670"),
        }
    ]
    assert all(cell["cell_type"] == "markdown" for cell in render_ipynb(notebook)["cells"])


def test_preexisting_checked_notebooks_remain_byte_identical() -> None:
    checked = ROOT / "docs/notebooks/generated"
    observed = {
        name: hashlib.sha256((checked / name).read_bytes()).hexdigest()
        for name in EXISTING_NOTEBOOK_SHA256
    }
    assert observed == EXISTING_NOTEBOOK_SHA256


def test_ipynb_is_standard_markdown_only_derived_view() -> None:
    notebook = build_natural_sum_notebook(ROOT)
    value = render_ipynb(notebook)
    assert value["nbformat"] == 4
    assert value["metadata"]["invariant"]["authority"] == "derived_view_only"
    assert all(cell["cell_type"] == "markdown" for cell in value["cells"])
    assert value == render_ipynb(notebook)


def test_notebook_validation_rejects_resealed_semantic_tampering() -> None:
    value = build_quartic_survivor_notebook(ROOT).to_dict()
    tampered = copy.deepcopy(value)
    tampered["verdict"] = "proved"
    with pytest.raises(NotebookValidationError, match="content seal mismatch"):
        validate_notebook(tampered)


def test_materialization_is_deterministic(tmp_path: Path) -> None:
    first = materialize_example_notebooks(ROOT, tmp_path)
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = materialize_example_notebooks(ROOT, tmp_path)
    assert first == second
    assert first_bytes == {path.name: path.read_bytes() for path in second}
    for path in second:
        if path.suffix == ".ipynb":
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value["nbformat"] == 4


def test_checked_notebooks_match_materialized_receipts(tmp_path: Path) -> None:
    generated = materialize_example_notebooks(ROOT, tmp_path)
    checked = ROOT / "docs/notebooks/generated"
    for path in generated:
        assert path.read_bytes() == (checked / path.name).read_bytes()


def test_bound_receipt_content_tamper_fails_closed(tmp_path: Path) -> None:
    relative = Path("runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    value["enumeration"]["raw_cartesian_candidates"] += 1
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(NotebookValidationError, match="content seal mismatch"):
        build_natural_sum_notebook(tmp_path)
