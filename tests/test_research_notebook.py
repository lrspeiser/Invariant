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
    build_natural_sum_notebook,
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
