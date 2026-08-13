from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.research_notebook import (
    NotebookValidationError,
    build_action_jet_nonidentifiability_notebook,
    build_natural_sum_notebook,
    build_quartic_survivor_notebook,
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
