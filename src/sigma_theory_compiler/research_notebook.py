"""Deterministic, human-readable views of sealed discovery receipts.

The notebook is a derived presentation artifact, never the authority for a claim.
Every substantive cell names the immutable receipt(s) from which it was built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "invariant-research-notebook-1.0"
RECONSTRUCTION_NOTICE = (
    "Historical-style reconstruction generated from sealed machine receipts. "
    "It is not an authentic historical document, private model reasoning, or a "
    "replacement for the cited receipts."
)


class NotebookValidationError(ValueError):
    """Raised when a notebook or a bound receipt is malformed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_sealed_receipt(root: Path, relative_path: str) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / relative_path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NotebookValidationError(f"cannot read receipt {relative_path}: {error}") from error
    if not isinstance(value, dict) or "content_sha256" not in value:
        raise NotebookValidationError(f"receipt {relative_path} has no content seal")
    body = dict(value)
    claimed = body.pop("content_sha256")
    computed = _sha256_bytes(_canonical_bytes(body))
    if claimed != computed:
        raise NotebookValidationError(f"receipt {relative_path} content seal mismatch")
    return value, {
        "path": relative_path,
        "file_sha256": _file_sha256(path),
        "content_sha256": str(claimed),
    }


@dataclass(frozen=True)
class NotebookClaim:
    status: str
    statement: str
    evidence_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"proved", "certified_local", "blocked", "scope_limit"}:
            raise NotebookValidationError(f"unsupported claim status {self.status!r}")
        if not self.statement or not self.evidence_paths:
            raise NotebookValidationError("claim statement and evidence paths are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "statement": self.statement,
            "evidence_paths": list(self.evidence_paths),
        }


@dataclass(frozen=True)
class NotebookCell:
    title: str
    source: str
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title or not self.source:
            raise NotebookValidationError("cell title and source are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "title": self.title,
            "source": self.source,
            "evidence_paths": list(self.evidence_paths),
        }


@dataclass(frozen=True)
class ResearchNotebook:
    notebook_id: str
    title: str
    verdict: str
    source_bindings: tuple[Mapping[str, str], ...]
    claims: tuple[NotebookClaim, ...]
    cells: tuple[NotebookCell, ...]
    limits: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verdict not in {"proved", "formal_local_survivor"}:
            raise NotebookValidationError(f"unsupported notebook verdict {self.verdict!r}")
        paths = [binding.get("path") for binding in self.source_bindings]
        if len(paths) != len(set(paths)) or not paths:
            raise NotebookValidationError("source binding paths must be nonempty and unique")
        known = set(paths)
        for item in (*self.claims, *self.cells):
            if not set(item.evidence_paths).issubset(known):
                raise NotebookValidationError("cell or claim cites an unbound receipt")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": SCHEMA_VERSION,
            "notebook_id": self.notebook_id,
            "title": self.title,
            "verdict": self.verdict,
            "reconstruction_notice": RECONSTRUCTION_NOTICE,
            "source_bindings": [dict(binding) for binding in self.source_bindings],
            "claims": [claim.to_dict() for claim in self.claims],
            "cells": [cell.to_dict() for cell in self.cells],
            "limits": list(self.limits),
        }
        return {**body, "content_sha256": _sha256_bytes(_canonical_bytes(body))}


def validate_notebook(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "notebook_id",
        "title",
        "verdict",
        "reconstruction_notice",
        "source_bindings",
        "claims",
        "cells",
        "limits",
        "content_sha256",
    }
    if set(value) != expected:
        raise NotebookValidationError("research notebook schema changed")
    body = dict(value)
    claimed = body.pop("content_sha256")
    if claimed != _sha256_bytes(_canonical_bytes(body)):
        raise NotebookValidationError("research notebook content seal mismatch")
    if value["schema_version"] != SCHEMA_VERSION:
        raise NotebookValidationError("research notebook schema version changed")
    if value["reconstruction_notice"] != RECONSTRUCTION_NOTICE:
        raise NotebookValidationError("research notebook disclosure changed")
    bindings = value["source_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise NotebookValidationError("research notebook must bind receipts")
    paths: set[str] = set()
    for binding in bindings:
        if set(binding) != {"path", "file_sha256", "content_sha256"}:
            raise NotebookValidationError("receipt binding schema changed")
        if binding["path"] in paths:
            raise NotebookValidationError("duplicate receipt binding")
        paths.add(binding["path"])
        for key in ("file_sha256", "content_sha256"):
            digest = binding[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise NotebookValidationError("invalid receipt hash")
    for claim in value["claims"]:
        NotebookClaim(
            status=claim["status"],
            statement=claim["statement"],
            evidence_paths=tuple(claim["evidence_paths"]),
        )
        if not set(claim["evidence_paths"]).issubset(paths):
            raise NotebookValidationError("claim cites an unbound receipt")
    for cell in value["cells"]:
        if set(cell) != {"cell_type", "title", "source", "evidence_paths"}:
            raise NotebookValidationError("notebook cell schema changed")
        if cell["cell_type"] != "markdown":
            raise NotebookValidationError("only deterministic markdown cells are allowed")
        NotebookCell(cell["title"], cell["source"], tuple(cell["evidence_paths"]))
        if not set(cell["evidence_paths"]).issubset(paths):
            raise NotebookValidationError("cell cites an unbound receipt")


def render_markdown(notebook: ResearchNotebook) -> str:
    value = notebook.to_dict()
    validate_notebook(value)
    lines = [
        f"# {notebook.title}",
        "",
        f"> **Verdict:** `{notebook.verdict}`",
        f"> **Disclosure:** {RECONSTRUCTION_NOTICE}",
        "",
    ]
    for cell in notebook.cells:
        lines.extend((f"## {cell.title}", "", cell.source, ""))
        if cell.evidence_paths:
            lines.extend(
                (
                    "Evidence: " + ", ".join(f"`{path}`" for path in cell.evidence_paths),
                    "",
                )
            )
    lines.extend(("## Claim ledger", ""))
    for claim in notebook.claims:
        lines.append(f"- **{claim.status}:** {claim.statement}")
    lines.extend(("", "## Receipt bindings", ""))
    for binding in notebook.source_bindings:
        lines.append(
            f"- `{binding['path']}` — file `{binding['file_sha256']}`, "
            f"content `{binding['content_sha256']}`"
        )
    lines.extend(("", "## Limits", ""))
    lines.extend(f"- {limit}" for limit in notebook.limits)
    lines.extend(("", f"Notebook content seal: `{value['content_sha256']}`", ""))
    return "\n".join(lines)


def render_ipynb(notebook: ResearchNotebook) -> dict[str, Any]:
    value = notebook.to_dict()
    validate_notebook(value)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {"invariant_role": "disclosure"},
            "source": [
                f"# {notebook.title}\n",
                f"\n**Verdict:** `{notebook.verdict}`\n",
                f"\n> {RECONSTRUCTION_NOTICE}\n",
            ],
        }
    ]
    for cell in notebook.cells:
        source = f"## {cell.title}\n\n{cell.source}\n"
        if cell.evidence_paths:
            source += "\nEvidence: " + ", ".join(f"`{p}`" for p in cell.evidence_paths)
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {
                    "evidence_paths": list(cell.evidence_paths),
                    "invariant_role": "derived_presentation",
                },
                "source": [source + "\n"],
            }
        )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {"invariant_role": "receipt_ledger"},
            "source": [
                "## Receipt ledger\n\n"
                + "\n".join(
                    f"- `{b['path']}` — content `{b['content_sha256']}`"
                    for b in notebook.source_bindings
                )
                + f"\n\nNotebook content seal: `{value['content_sha256']}`\n"
            ],
        }
    )
    return {
        "cells": cells,
        "metadata": {
            "invariant": {
                "notebook_id": notebook.notebook_id,
                "verdict": notebook.verdict,
                "content_sha256": value["content_sha256"],
                "authority": "derived_view_only",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_natural_sum_notebook(root: Path) -> ResearchNotebook:
    receipt_path = "runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json"
    result, binding = _load_sealed_receipt(root, receipt_path)
    counts = result["enumeration"]
    coefficients = result["winner"]["coefficients"]
    proof = result["induction_proof"]
    claims = result["claims"]
    if (
        result["decision"]
        != "pass_blind_bounded_grammar_rediscovery_independently_proved_before_unseal"
        or coefficients
        != {
            "square": {"numerator": 1, "denominator": 2},
            "linear": {"numerator": 1, "denominator": 2},
            "constant": {"numerator": 0, "denominator": 1},
        }
        or proof["successor_identity"] != "candidate(n+1)-candidate(n)=n+1"
        or not claims["universal_identity_proved_by_induction"]
    ):
        raise NotebookValidationError("natural-sum proof receipt boundary changed")
    evidence = (receipt_path,)
    cells = (
        NotebookCell(
            "Problem stated without the answer",
            "Let $S(0)=0$ and let the anonymous sequence satisfy\n\n"
            "$$S(n+1)-S(n)=n+1.\n$$\n\n"
            "We seek a closed form using only the declared quadratic rational grammar. "
            "The familiar name of the theorem is withheld until the end.",
            evidence,
        ),
        NotebookCell(
            "Finite discovery experiment",
            f"Start with $q(n)=an^2+bn+c$. The bounded search examined "
            f"{counts['raw_cartesian_candidates']:,} raw coefficient triples, reduced them "
            f"to {counts['canonical_coefficient_classes']:,} canonical classes, and left "
            f"{counts['exact_example_survivors']} survivor on the public examples. The "
            f"survivor then passed {counts['counterexample_tests_per_survivor']} additional "
            "exact counterexample points. These tests identify a candidate; they do not prove it.",
            evidence,
        ),
        NotebookCell(
            "The conjectured formula",
            "The surviving coefficients are $a=1/2$, $b=1/2$, and $c=0$, hence\n\n"
            "$$q(n)=\\frac{n^2+n}{2}=\\frac{n(n+1)}{2}.\n$$",
            evidence,
        ),
        NotebookCell(
            "Proof",
            "**Base case.** $q(0)=0=S(0)$.\n\n"
            "**Successor step.** Exact polynomial arithmetic gives\n\n"
            "$$q(n+1)-q(n)="
            "\\frac{(n+1)^2+(n+1)-n^2-n}{2}=n+1.\n$$\n\n"
            "Assume $q(n)=S(n)$. The defining recurrence and the displayed identity imply\n\n"
            "$$q(n+1)=q(n)+(n+1)=S(n)+(n+1)=S(n+1).\n$$\n\n"
            "Therefore $q(n)=S(n)$ for every nonnegative integer $n$ by induction.",
            evidence,
        ),
        NotebookCell(
            "Chronological unsealing",
            "The candidate catalog, counterexample record, and induction proof were sealed "
            "before the withheld reference was read. Only afterward was the result compared "
            "with the conventional natural-sum identity, and the forms matched exactly. "
            "This demonstrates bounded rediscovery mechanics; it is not a novelty claim.",
            evidence,
        ),
    )
    return ResearchNotebook(
        notebook_id="natural-sum-chronological-reconstruction-001",
        title="Chronological rediscovery of an anonymous finite-sum formula",
        verdict="proved",
        source_bindings=(binding,),
        claims=(
            NotebookClaim(
                "proved",
                "The discovered quadratic equals the anonymous recurrence-defined sequence for every nonnegative integer.",
                evidence,
            ),
            NotebookClaim(
                "scope_limit",
                "Only the declared finite quadratic rational grammar was exhausted; no unbounded formula-space or novelty claim follows.",
                evidence,
            ),
        ),
        cells=cells,
        limits=tuple(result["limits"]),
    )


def _matching_certificate(result: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    matches = [c for c in result["certificates"] if c["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise NotebookValidationError(f"expected one certificate for {candidate_id}")
    return matches[0]


def build_quartic_survivor_notebook(root: Path) -> ResearchNotebook:
    paths = {
        "dirac": "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json",
        "symmetrizer": "runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json",
        "cauchy": "runs/physics-language/quartic-nonquasilinear-pde-campaign/campaign.json",
        "d2f": "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json",
    }
    loaded = {name: _load_sealed_receipt(root, path) for name, path in paths.items()}
    dirac = loaded["dirac"][0]
    symmetrizer = loaded["symmetrizer"][0]
    cauchy = loaded["cauchy"][0]
    d2f = loaded["d2f"][0]
    if any(result["counts"]["selected"] != 12 for result in (dirac, symmetrizer, cauchy)):
        raise NotebookValidationError("quartic selected-candidate count changed")
    if d2f["gate_counts"] != {
        "selected": 12,
        "coordinate_atoms": 153,
        "ordered_pair_cells_classified": 23409,
        "ordered_D2F_entries_in_domain": 257499,
        "corrected_entries_admitted_per_candidate": 891,
        "principal_high_atom_entries_missing_per_candidate": 106920,
        "full_ordered_D2F_entries_missing_per_candidate": 256608,
        "complete_ordered_D2F_tensors_registered": 0,
        "full_high_atom_good_unknown_identities_proved": 0,
        "global_H7_closures": 0,
        "nonlinear_PDE_closures": 0,
        "lifespans_proved": 0,
    }:
        raise NotebookValidationError("quartic D2F coverage boundary changed")
    candidate_id = min(c["candidate_id"] for c in dirac["certificates"])
    dc = _matching_certificate(dirac, candidate_id)
    sc = _matching_certificate(symmetrizer, candidate_id)
    pc = _matching_certificate(cauchy, candidate_id)
    hessian = dc["adm_hessian_and_primary_constraint"]
    constraints = dc["dirac_chain"]["constraint_count"]
    hamiltonian = dc["on_shell_quadratic_physical_hamiltonian"]
    if (
        (hessian["rank"], hessian["nullity"]) != (6, 1)
        or constraints["physical_configuration_dof"] != 3
        or not hamiltonian["strictly_positive"]
        or sc["status"] != "pass_uniform_local_jet_strong_hyperbolicity"
        or pc["status"] != "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift"
    ):
        raise NotebookValidationError("quartic local certificate boundary changed")
    all_paths = tuple(paths.values())
    local_paths = (paths["dirac"], paths["symmetrizer"], paths["cauchy"])
    cells = (
        NotebookCell(
            "Question and candidate",
            f"Consider the exact candidate `{candidate_id}` with\n\n"
            f"- $G_2={dc['covariant_action_specialization']['G2']}$,\n"
            f"- $G_3={dc['covariant_action_specialization']['G3']}$,\n"
            f"- $G_4={dc['covariant_action_specialization']['G4']}$,\n"
            f"- $G_5={dc['covariant_action_specialization']['G5']}$.\n\n"
            "The scientific question is not merely whether this expression is compact, but "
            "whether its constrained dynamics, local PDE symbol, and nonlinear energy structure survive exact checks.",
            (paths["dirac"],),
        ),
        NotebookCell(
            "1. Exhibit an exact local solution",
            "At the certified FLRW point the lapse, scale, and scalar equation residuals are "
            f"all exactly zero. The witness uses $X={dc['on_shell_local_flrw_witness']['X']}$ "
            f"and $A_\\star={dc['on_shell_local_flrw_witness']['A_star']}$. Every recorded "
            "jet component lies strictly inside the local hyperbolicity box. This establishes a "
            "nonempty on-shell patch, not a global spacetime solution theorem.",
            (paths["dirac"], paths["symmetrizer"]),
        ),
        NotebookCell(
            "2. Count the physical modes",
            f"The velocity Hessian in the ordered variables "
            f"{', '.join(hessian['velocity_order'])} has rank {hessian['rank']} and nullity "
            f"{hessian['nullity']}. Its null direction yields the primary constraint "
            f"`${hessian['primary_constraint']}`. The closed Dirac chain records "
            f"{constraints['first_class']} first-class and {constraints['second_class']} "
            f"second-class constraints on an extended phase space of dimension "
            f"{constraints['extended_phase_dimension']}:\n\n"
            "$$N_{\\rm dof}=\\frac{20-2(6)-2}{2}=3.\n$$\n\n"
            "Thus the local constrained system propagates three configuration degrees of freedom.",
            (paths["dirac"],),
        ),
        NotebookCell(
            "3. Check local energy and hyperbolicity",
            "The reduced quadratic Hamiltonian has the form\n\n"
            "$$H_k=\\tfrac12[P^T K^{-1}P+k^2Q^TFQ],\n$$\n\n"
            "with the recorded kinetic and gradient matrices strictly positive. Independently, "
            "the complete 22-by-22 directional symbol is strongly hyperbolic throughout the "
            "declared local-jet box, and its symmetrizer lifts to the full 55-state first-order "
            "system. For compatible vacuum initial data in a compact subset of that box, the "
            "local Cauchy theorem applies for some unspecified $T>0$.",
            local_paths,
        ),
        NotebookCell(
            "4. Locate the unresolved obstruction",
            "The second-derivative source domain contains $153^2=23{,}409$ ordered atom "
            "pairs and $11\\times153^2=257{,}499$ output entries per candidate. Only 891 "
            "entries have admitted corrected values. Therefore 256,608 entries remain "
            "unregistered, including 106,920 principal high-atom entries. The first blocker is "
            f"`{d2f['first_blocker']}`. Consequently the complete high-atom identity, global "
            "$H^7$ closure, nonlinear global PDE theorem, and lifespan remain unproved.",
            (paths["d2f"],),
        ),
        NotebookCell(
            "Scientific conclusion",
            "This is a **formal local survivor**: it has an exact local on-shell witness, the "
            "expected three-mode constraint count, positive reduced quadratic energy, and local "
            "strong hyperbolicity. It is not yet an admitted global theory. A mathematician's "
            "notebook should preserve both halves of that sentence.",
            all_paths,
        ),
    )
    return ResearchNotebook(
        notebook_id="quartic-local-survivor-reconstruction-001",
        title="Local viability analysis of a quartic scalar–tensor candidate",
        verdict="formal_local_survivor",
        source_bindings=tuple(loaded[name][1] for name in paths),
        claims=(
            NotebookClaim(
                "certified_local",
                "The selected candidate has a three-mode local constrained Hamiltonian, positive reduced quadratic energy, and a conditional local vacuum Cauchy certificate.",
                local_paths,
            ),
            NotebookClaim(
                "blocked",
                "The complete D2F/high-atom identity, global H7 estimate, nonlinear global PDE closure, and lifespan are not proved.",
                (paths["d2f"],),
            ),
            NotebookClaim(
                "scope_limit",
                "No observational or universal-matter conclusion follows from these local vacuum certificates.",
                all_paths,
            ),
        ),
        cells=cells,
        limits=(
            "the presentation is derived from sealed receipts and is not an independent proof kernel",
            "the local Cauchy time is existential and has no numerical lower bound",
            "preservation of the local box under general inhomogeneous evolution is not proved",
            "matter evolution, boundary estimates, global H7 closure, lifespan, and observations are outside the certified result",
        ),
    )


def write_notebook_pair(notebook: ResearchNotebook, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = output_stem.with_suffix(".md")
    ipynb_path = output_stem.with_suffix(".ipynb")
    markdown_path.write_text(render_markdown(notebook), encoding="utf-8", newline="\n")
    ipynb_path.write_text(
        json.dumps(render_ipynb(notebook), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return markdown_path, ipynb_path


def materialize_example_notebooks(root: Path, output: Path) -> tuple[Path, ...]:
    notebooks: Sequence[tuple[str, ResearchNotebook]] = (
        ("natural-sum-rediscovery", build_natural_sum_notebook(root)),
        ("quartic-local-survivor", build_quartic_survivor_notebook(root)),
    )
    paths: list[Path] = []
    for name, notebook in notebooks:
        paths.extend(write_notebook_pair(notebook, output / name))
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("docs/notebooks/generated"))
    args = parser.parse_args()
    for path in materialize_example_notebooks(args.root.resolve(), args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
