from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.research_notebook import (
    NotebookValidationError,
    build_natural_sum_notebook,
    build_quartic_survivor_notebook,
    materialize_example_notebooks,
    render_ipynb,
    render_markdown,
    validate_notebook,
)

ROOT = Path(__file__).resolve().parents[1]


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
