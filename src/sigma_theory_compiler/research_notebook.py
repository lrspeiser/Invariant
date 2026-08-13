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


def build_action_jet_nonidentifiability_notebook(root: Path) -> ResearchNotebook:
    receipt_path = (
        "runs/physics-language/"
        "quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json"
    )
    result, binding = _load_sealed_receipt(root, receipt_path)
    expected_counts = {
        "complete_ordered_D2F_tensors_registered": 0,
        "cross_slice_D2F_entries_admitted": 0,
        "fitted_connection_coordinates": 22,
        "full_high_atom_good_unknown_identities_proved": 0,
        "global_H7_closures": 0,
        "independent_ambiguity_parameters": 22,
        "lifespans_proved": 0,
        "nonidentified_first_jet_samples": 88,
        "nonidentified_second_jet_samples": 88,
        "nonlinear_PDE_closures": 0,
        "registered_G4_X_grid_points": 4,
        "registered_corrected_second_source_jet_entries": 0,
        "registered_covariant_derivation_functors": 0,
        "registered_value_equalities_replayed": 88,
        "selected": 12,
    }
    expected_samples = [
        {
            "G4_X": "-1",
            "first_jet_lambda_coefficient": "-3/2",
            "null_value": "0",
            "second_jet_lambda_coefficient": "19/2",
        },
        {
            "G4_X": "-1/2",
            "first_jet_lambda_coefficient": "3/4",
            "null_value": "0",
            "second_jet_lambda_coefficient": "1/2",
        },
        {
            "G4_X": "1/2",
            "first_jet_lambda_coefficient": "-3/4",
            "null_value": "0",
            "second_jet_lambda_coefficient": "1/2",
        },
        {
            "G4_X": "1",
            "first_jet_lambda_coefficient": "3/2",
            "null_value": "0",
            "second_jet_lambda_coefficient": "19/2",
        },
    ]
    certificate = result["null_polynomial_certificate"]
    records = result["coordinate_ambiguity_records"]
    claim_seals = result["claim_seals"]
    if (
        result["decision"] != "pass_exact_finite_grid_first_and_second_jet_nonidentifiability"
        or result["decision_counts"] != {"blocked": 0, "pass": 12, "reject": 0}
        or result["downstream_admission_counts"] != {"blocked": 12, "pass": 0, "reject": 0}
        or result["gate_counts"] != expected_counts
        or certificate
        != {
            "expanded_null_polynomial": "g^4-5/4*g^2+1/4",
            "first_derivative": "4*g^3-5/2*g",
            "grid_samples": expected_samples,
            "second_derivative": "12*g^2-5/2",
        }
        or len(records) != 22
        or [record["coordinate_ordinal"] for record in records] != list(range(22))
        or any(
            record["registered_value_equalities"] != 4
            or record["first_jet_ambiguities"] != 4
            or record["second_jet_ambiguities"] != 4
            or record["jet_identified"] is not False
            for record in records
        )
        or {key for key, value in claim_seals.items() if value}
        != {
            "all_22_coordinate_value_extensions_nonunique",
            "all_88_first_jet_samples_nonidentified",
            "all_88_second_jet_samples_nonidentified",
            "degree_four_null_polynomial_constructed",
            "finite_grid_value_factorization_bound",
            "independent_22_parameter_ambiguity_family_constructed",
        }
    ):
        raise NotebookValidationError("action-jet nonidentifiability receipt boundary changed")
    broad_false = {
        "candidate_theory_rejected",
        "complete_ordered_D2F_tensor_registered",
        "corrected_second_source_jet_registered",
        "covariant_output_connection_derivation_registered",
        "cross_slice_D2F_entries_admitted",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "observational_claim_made",
    }
    if any(claim_seals[key] for key in broad_false):
        raise NotebookValidationError("action-jet broad false-claim boundary changed")
    evidence = (receipt_path,)
    table = (
        "| $g$ | $p(g)$ | $p'(g)$ | $p''(g)$ |\n"
        "|---:|---:|---:|---:|\n"
        "| $-1$ | $0$ | $-3/2$ | $19/2$ |\n"
        "| $-1/2$ | $0$ | $3/4$ | $1/2$ |\n"
        "| $1/2$ | $0$ | $-3/4$ | $1/2$ |\n"
        "| $1$ | $0$ | $3/2$ | $19/2$ |"
    )
    cells = (
        NotebookCell(
            "The finite-data question",
            "For each of 22 fitted connection coordinates, the receipt registers only four "
            "values of a function of $g=G_{4,X}$, at\n\n"
            "$$g\\in\\{-1,-\\tfrac{1}{2},\\tfrac{1}{2},1\\},\\qquad "
            "f_i(g)=\\beta_i g.\n$$\n\n"
            "Can these four values determine the first and second $g$ derivatives of the "
            "underlying extension? No derivative functor or polynomial degree bound below four "
            "is among the registered premises.",
            evidence,
        ),
        NotebookCell(
            "Derive the null polynomial",
            "A polynomial that vanishes at all four registered values is obtained by taking "
            "one factor for each point:\n\n"
            "$$\\begin{aligned}\n"
            "p(g)&=(g+1)(g+\\tfrac{1}{2})(g-\\tfrac{1}{2})(g-1)\\\\\n"
            "&=(g^2-1)(g^2-\\tfrac{1}{4})\\\\\n"
            "&=g^4-\\tfrac{5}{4}g^2+\\tfrac{1}{4}.\n"
            "\\end{aligned}$$\n\n"
            "Exact differentiation gives\n\n"
            "$$p'(g)=4g^3-\\tfrac{5}{2}g,\\qquad "
            "p''(g)=12g^2-\\tfrac{5}{2}.$$\n",
            evidence,
        ),
        NotebookCell(
            "Evaluate values and jets at all four points",
            table + "\n\nThus $p$ is invisible to every registered value sample, while both "
            "$p'$ and $p''$ are nonzero at every sampled point. The table is exact rational "
            "arithmetic, not a numerical fit.",
            evidence,
        ),
        NotebookCell(
            "The 22-parameter ambiguity family",
            "For coordinate $i\\in\\{0,\\ldots,21\\}$ introduce an independent parameter "
            "$\\lambda_i$ and set\n\n"
            "$$F_i(g)=\\beta_i g+\\lambda_i p(g).$$\n\n"
            "At each registered grid point, $F_i(g)=\\beta_i g$ for every $\\lambda_i$. "
            "But\n\n"
            "$$F_i'(g)=\\beta_i+\\lambda_i p'(g),\\qquad "
            "F_i''(g)=\\lambda_i p''(g),$$\n\n"
            "and the table shows that both jets vary nontrivially with $\\lambda_i$ at all four "
            "points. Because the parameters are coordinate-wise independent, their product "
            "gives a 22-parameter family. Equivalently, the four values leave all 88 recorded "
            "first-jet samples and all 88 second-jet samples unidentified.",
            evidence,
        ),
        NotebookCell(
            "What is proved",
            "The registered four-point value map is not injective on first or second action "
            "feature jets within the displayed degree-four extension class. Therefore those "
            "finite values alone cannot select the affine extension $\\beta_i g$ over the "
            "alternatives $\\beta_i g+\\lambda_i p(g)$. This is an exact identifiability "
            "obstruction, and it holds independently in all 22 coordinates.",
            evidence,
        ),
        NotebookCell(
            "What remains open",
            "This obstruction is **not** a no-go theorem for a covariant action derivation. A "
            "registered local variation rule, derivative samples, a justified degree bound, or "
            "corrected second-source jet values could select one extension. In the sealed "
            "receipt, zero corrected second-source entries and zero cross-slice $D^2F$ entries "
            "are admitted; complete ordered $D^2F$, the high-atom identity, global $H^7$, "
            "nonlinear PDE closure, and lifespan all remain open. All 12 downstream candidates "
            "remain blocked rather than rejected. The first blocker is\n\n"
            f"`{result['first_blocker']}`.",
            evidence,
        ),
    )
    return ResearchNotebook(
        notebook_id="quartic-action-jet-nonidentifiability-reconstruction-001",
        title="Exact action-jet nonidentifiability from four registered values",
        verdict="proved",
        source_bindings=(binding,),
        claims=(
            NotebookClaim(
                "proved",
                "Four registered values do not identify the first or second G4_X jets in the displayed degree-four extension class.",
                evidence,
            ),
            NotebookClaim(
                "proved",
                "The product construction supplies 22 independent ambiguity parameters and leaves 88 first-jet and 88 second-jet samples unidentified.",
                evidence,
            ),
            NotebookClaim(
                "blocked",
                "No covariant variation functor, corrected second-source jet, cross-slice D2F admission, complete D2F tensor, H7 closure, nonlinear PDE closure, or lifespan is established.",
                evidence,
            ),
            NotebookClaim(
                "scope_limit",
                "The proved finite-data obstruction neither rejects the 12 candidates nor proves that no covariant action derivation exists.",
                evidence,
            ),
        ),
        cells=cells,
        limits=(
            "the notebook is a derived presentation of one sealed receipt, not an independent proof kernel",
            "the obstruction concerns the registered four-point value data and displayed degree-four null direction",
            "a registered variation rule, derivative evidence, or corrected second-source jet may remove the ambiguity",
            "complete covariant D2F, high-atom, global H7, nonlinear PDE, lifespan, and observational claims remain fail-closed",
        ),
    )


def build_registered_variation_selection_notebook(root: Path) -> ResearchNotebook:
    receipt_path = (
        "runs/physics-language/"
        "quartic-fitted-output-connection-registered-variation-selection-audit/"
        "campaign.json"
    )
    result, binding = _load_sealed_receipt(root, receipt_path)
    expected_capabilities = [
        {
            "candidate_bound_to_quartic_grid": False,
            "corrected_second_source_jet_values": 0,
            "evidence": "generic_G4_metric_variation",
            "map_to_22_output_connection_coordinates": False,
            "registered_unit_type": "generic_metric_Euler_tensor_contractions",
            "registered_units": 24,
            "selector_equations_contributed": 0,
        },
        {
            "candidate_backend_variations_executed": 0,
            "candidate_bound_to_quartic_grid": False,
            "corrected_second_source_jet_values": 0,
            "evidence": "generated_candidate_metric_variation",
            "map_to_22_output_connection_coordinates": False,
            "quartic_G4_X_grid_candidate_overlap": 0,
            "registered_unit_type": "candidate_specialized_metric_Euler_expressions",
            "registered_units": 163,
            "selector_equations_contributed": 0,
        },
        {
            "candidate_bound_to_quartic_grid": True,
            "candidate_count": 12,
            "corrected_second_source_jet_values": 0,
            "evidence": "universal_source_DAG",
            "full_component_Frechet_tensors_complete": False,
            "map_to_22_output_connection_coordinates": False,
            "registered_unit_type": "pure_derivative_component_roots",
            "registered_units": 1056,
            "selector_equations_contributed": 0,
        },
        {
            "candidate_bound_to_quartic_grid": True,
            "candidate_count": 12,
            "complete_orders_2_to_4": False,
            "corrected_second_source_jet_values": 0,
            "evidence": "full_source_D1",
            "map_to_22_output_connection_coordinates": False,
            "registered_unit_type": "first_source_Jacobian_entries",
            "registered_units": 20196,
            "selector_equations_contributed": 0,
            "source_Jacobian_shape": [11, 153],
        },
    ]
    expected_counts = {
        "ambiguity_parameters_remaining": 22,
        "ambiguity_parameters_selected": 0,
        "complete_ordered_D2F_tensors_registered": 0,
        "cross_slice_D2F_entries_admitted": 0,
        "eligible_selector_equations_registered": 0,
        "full_high_atom_good_unknown_identities_proved": 0,
        "full_source_D1_entries": 20196,
        "generated_metric_variation_candidates": 163,
        "generated_quartic_G4_X_grid_candidate_overlap": 0,
        "generic_G4_metric_Euler_terms": 24,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "nonlinear_PDE_closures": 0,
        "registered_corrected_second_source_jet_entries": 0,
        "registered_evidence_bundles": 4,
        "selected_candidates": 12,
        "universal_source_DAG_pure_derivative_roots": 1056,
    }
    theorem = result["registered_selection_theorem"]
    records = result["coordinate_selection_records"]
    claim_seals = result["claim_seals"]
    exact_controls = result["exact_controls"]
    if (
        result["decision"]
        != "registered_variation_and_source_inventory_has_rank_zero_for_22_jet_selectors"
        or result["predecessor_decision"]
        != "pass_exact_finite_grid_first_and_second_jet_nonidentifiability"
        or result["decision_counts"] != {"blocked": 0, "pass": 12, "reject": 0}
        or result["downstream_admission_counts"] != {"blocked": 12, "pass": 0, "reject": 0}
        or result["evidence_capabilities"] != expected_capabilities
        or result["gate_counts"] != expected_counts
        or result["selection_matrix"]
        != {
            "columns": 22,
            "nullity": 22,
            "rank": 0,
            "rows": 0,
            "selected_parameters": 0,
            "unselected_parameters": 22,
        }
        or len(records) != 22
        or [record["coordinate_ordinal"] for record in records] != list(range(22))
        or [record["ambiguity_parameter"] for record in records]
        != [f"lambda_{ordinal}" for ordinal in range(22)]
        or any(
            record["eligible_selector_equations_registered"] != 0
            or record["parameter_selected"] is not False
            for record in records
        )
        or {key for key, value in claim_seals.items() if value}
        != {
            "all_22_ambiguity_parameters_remain_unselected",
            "eligible_selector_schema_applied",
            "registered_selector_rank_zero",
            "registered_variation_source_inventory_bound",
        }
        or not exact_controls
        or any(control != {"rejected": True} for control in exact_controls.values())
    ):
        raise NotebookValidationError("registered variation selection receipt boundary changed")
    expected_theorem = {
        "boundary": (
            "This is a closed-world result for the four explicitly bound registered evidence "
            "bundles. It is neither a physical no-go nor evidence that a covariant variation "
            "rule cannot exist; adding a candidate-bound component map or corrected second "
            "source-jet values invalidates the premise and requires a new gate."
        ),
        "exact_result": (
            "The generic metric theorem has no output-coordinate map; the 163 generated "
            "specializations have zero overlap with the quartic G4_X grid; the source DAGs "
            "leave complete component Frechet tensors open; and the materialized source "
            "tensors are D1 only. Thus the registered selector matrix has shape 0-by-22, "
            "rank zero, and nullity 22. No lambda_i is selected."
        ),
        "name": "closed_inventory_rank_zero_for_action_jet_ambiguity_selection",
        "premises": (
            "The declared registered inventory contains the 24-term generic G4 metric Euler "
            "normalization, 163 generated candidate metric-Euler specializations, twelve "
            "candidate-aligned universal source DAGs, and twelve complete first source "
            "Jacobians. Eligible selectors must be candidate-bound first/second G4_X jet "
            "values or explicit maps into the matching 22 output-connection coordinates."
        ),
    }
    broad_false = {
        "candidate_theory_rejected",
        "complete_ordered_D2F_tensor_registered",
        "corrected_second_source_jet_registered",
        "covariant_output_connection_derivation_registered",
        "cross_slice_D2F_entries_admitted",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "observational_claim_made",
        "physical_covariant_variation_no_go_proved",
    }
    if theorem != expected_theorem or any(claim_seals[key] for key in broad_false):
        raise NotebookValidationError("registered variation theorem boundary changed")
    evidence = (receipt_path,)
    bundle_table = (
        "| Registered evidence bundle | Units | Candidate-bound | Matching 22-coordinate map | "
        "Eligible rows | Recorded limitation |\n"
        "|---|---:|:---:|:---:|---:|---|\n"
        "| Generic $G_4$ metric variation | 24 | no | no | 0 | generic Euler contractions |\n"
        "| Generated metric variations | 163 | no | no | 0 | zero quartic-grid overlap |\n"
        "| Universal source DAG | 1,056 | yes | no | 0 | component Fréchet tensors incomplete |\n"
        "| Full source $D^1$ | 20,196 | yes | no | 0 | first Jacobian only; no corrected second jet |"
    )
    cells = (
        NotebookCell(
            "The selection question",
            "The preceding receipt exhibited 22 independent ambiguity parameters "
            "$\\lambda=(\\lambda_0,\\ldots,\\lambda_{21})$. This audit asks a narrower "
            "question: does the declared, sealed inventory contain an equation that selects "
            "any of them? The answer applies only to the four registered bundles below.",
            evidence,
        ),
        NotebookCell(
            "The four registered evidence bundles",
            bundle_table
            + "\n\nEvery bundle is substantive evidence, but eligibility requires more than "
            "quantity: it must connect candidate-bound first/second $G_{4,X}$ jet data to "
            "the matching fitted output-connection coordinate.",
            evidence,
        ),
        NotebookCell(
            "Define an eligible selector equation",
            "Over the exact coefficient field $\\mathbb K$, an eligible row has the form\n\n"
            "$$\\sum_{i=0}^{21} a_{ri}\\lambda_i=b_r,$$\n\n"
            "where the coefficients and right-hand side come from candidate-bound first- or "
            "second-$G_{4,X}$ jet values, or from an explicit component map into the same 22 "
            "output-connection coordinates. Generic contractions, unmatched candidates, pure "
            "DAG roots without the component map, and $D^1$ source entries are not silently "
            "promoted into selector rows.",
            evidence,
        ),
        NotebookCell(
            "Assemble and reduce the exact system",
            "All four row counts are zero, so stacking the eligible equations gives\n\n"
            "$$A\\lambda=b,\\qquad A\\in\\mathbb K^{0\\times22},\\qquad "
            "b\\in\\mathbb K^0.$$\n\n"
            "The empty matrix has no pivots. Hence\n\n"
            "$$\\operatorname{rank}(A)=0,\\qquad "
            "\\operatorname{nullity}(A)=22-0=22,$$\n\n"
            "and $\\ker A=\\mathbb K^{22}$. Thus zero parameters are selected and all 22 "
            "remain free in this inventory. In particular, absence of a row does not justify "
            "setting $\\lambda_i=0$.",
            evidence,
        ),
        NotebookCell(
            "The exact closed-inventory conclusion",
            "The registered selector matrix is exactly $0\\times22$, rank zero, and nullity "
            "22. The result is an inventory obstruction: none of the four bound evidence "
            "bundles supplies an eligible equation under the declared schema. All 12 "
            "downstream candidates remain blocked, not rejected.",
            evidence,
        ),
        NotebookCell(
            "Why this is not a physical no-go",
            "The conclusion quantifies over four registered bundles, not over all possible "
            "covariant variations. A candidate-bound component map from the $G_4$ variation "
            "or source DAG into the 22 coordinates, or exact corrected second-source jet "
            "values, would add rows and require a new rank audit. Therefore no physical "
            "covariant-variation no-go, candidate rejection, complete $D^2F$ tensor, "
            "high-atom identity, global $H^7$ estimate, nonlinear PDE closure, or lifespan "
            "follows here. The first blocker is\n\n"
            f"`{result['first_blocker']}`.",
            evidence,
        ),
    )
    return ResearchNotebook(
        notebook_id="quartic-registered-variation-selection-audit-reconstruction-001",
        title="A rank-zero audit of registered action-jet selectors",
        verdict="proved",
        source_bindings=(binding,),
        claims=(
            NotebookClaim(
                "proved",
                "The four registered evidence bundles contribute zero eligible equations to the 22-column selector system.",
                evidence,
            ),
            NotebookClaim(
                "proved",
                "The exact registered matrix is 0-by-22 with rank zero and nullity 22, so no ambiguity parameter is selected.",
                evidence,
            ),
            NotebookClaim(
                "blocked",
                "A candidate-bound component map or corrected second-source jet is still required before a nonempty selector system can be audited.",
                evidence,
            ),
            NotebookClaim(
                "scope_limit",
                "Closed-inventory rank zero is not a physical covariant-variation no-go and does not reject any candidate.",
                evidence,
            ),
        ),
        cells=cells,
        limits=(
            "the notebook is a derived presentation of one sealed receipt, not an independent proof kernel",
            "the rank computation ranges only over the four explicitly registered evidence bundles",
            "new candidate-bound component maps or corrected second-source jets invalidate the empty-row premise",
            "covariant no-go, complete D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, and observational claims remain fail-closed",
        ),
    )


def build_component_map_schema_ambiguity_notebook(root: Path) -> ResearchNotebook:
    receipt_path = (
        "runs/physics-language/"
        "quartic-fitted-output-connection-component-map-schema-ambiguity-gate/"
        "campaign.json"
    )
    result, binding = _load_sealed_receipt(root, receipt_path)
    expected_counts = {
        "complete_ordered_D2F_tensors_registered": 0,
        "cross_slice_D2F_entries_admitted": 0,
        "distinct_projection_witnesses": 2,
        "full_high_atom_good_unknown_identities_proved": 0,
        "generic_projection_affine_dimension": 506,
        "generic_projection_unknowns": 528,
        "generic_projection_value_constraints": 22,
        "generic_term_basis_dimension": 24,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "mixed_D2_extension_ambiguity_parameters": 22,
        "mixed_multi_index_components_completed": 0,
        "nonlinear_PDE_closures": 0,
        "output_connection_basis_dimension": 22,
        "registered_corrected_second_source_jet_entries": 0,
        "selected_candidates": 12,
        "target_D1_memberships_found": 22,
        "target_direction_tangent_embeddings_registered": 0,
        "target_ordered_mixed_D2F_roots_registered": 0,
        "target_pure_DAG_checkpoint_overlaps": 0,
        "unique_target_D1_row_atom_entries": 20,
    }
    projection = result["generic_term_projection_ambiguity"]
    mixed = result["mixed_D2_extension_ambiguity"]
    records = result["coordinate_records"]
    claim_seals = result["claim_seals"]
    exact_controls = result["exact_controls"]
    beta = projection["target_beta_vector"]
    base = projection["base_sparse_entries"]
    alternate = projection["alternate_sparse_entries"]
    expected_base = [
        {"generic_term": 0, "output_coordinate": ordinal, "value": value}
        for ordinal, value in enumerate(beta)
    ]
    expected_alternate = [
        {"generic_term": 0, "output_coordinate": 0, "value": "1/2"},
        {"generic_term": 1, "output_coordinate": 0, "value": "-1"},
        *expected_base[1:],
    ]
    if (
        result["decision"] != "pass_constructive_component_map_and_mixed_D2_schema_ambiguity"
        or result["predecessor_decision"]
        != "registered_variation_and_source_inventory_has_rank_zero_for_22_jet_selectors"
        or result["decision_counts"] != {"blocked": 0, "pass": 12, "reject": 0}
        or result["downstream_admission_counts"] != {"blocked": 12, "pass": 0, "reject": 0}
        or result["gate_counts"] != expected_counts
        or projection["matrix_shape"] != [22, 24]
        or projection["unknown_entries"] != 528
        or projection["value_constraints"] != 22
        or projection["constraint_rank"] != 22
        or projection["affine_solution_dimension"] != 506
        or projection["base_residual_nonzero_count"] != 0
        or projection["alternate_residual_nonzero_count"] != 0
        or projection["maps_distinct"] is not True
        or projection["covariance_or_index_equivariance_certified"] is not False
        or len(projection["generic_term_ids"]) != 24
        or len(projection["generic_coefficient_vector"]) != 24
        or projection["generic_coefficient_vector"][:2] != ["1", "-1/2"]
        or len(beta) != 22
        or base != expected_base
        or alternate != expected_alternate
        or len(records) != 22
        or mixed["target_coordinate_records"] != records
        or [record["coordinate_ordinal"] for record in records] != list(range(22))
        or any(
            record["both_extensions_preserve_registered_D1_value"] is not True
            or record["direction_state_tangent_registered"] is not False
            or record["ordered_mixed_D2F_root_registered"] is not False
            or record["zero_extension_D2_value"] != "0"
            or record["unit_extension_D2_value"] != "1"
            for record in records
        )
    ):
        raise NotebookValidationError("component-map schema ambiguity receipt boundary changed")
    expected_mixed = {
        "candidate_count": 12,
        "coordinate_atom_basis_sha256": (
            "cdb30c510a24bc6e64bc78245ac6f69d9dfc207e7812fd2d8abeba8e03cb2525"
        ),
        "direction_tangent_embeddings_registered": 0,
        "explicit_witness_completions": 23,
        "full_component_Frechet_tensors_complete": False,
        "independent_mixed_D2_extension_parameters": 22,
        "mixed_multi_index_components_completed": 0,
        "pure_checkpoint_atoms": ["q[0]", "p0[10]"],
        "pure_derivative_roots": 1056,
        "target_D1_memberships_found": 22,
        "target_atom_overlap_with_pure_DAG_checkpoints": 0,
        "target_coordinates": 22,
        "target_ordered_mixed_D2F_roots_registered": 0,
        "unique_target_D1_row_atom_entries": 20,
    }
    if (
        {key: value for key, value in mixed.items() if key != "target_coordinate_records"}
        != expected_mixed
        or result["missing_schema"]
        != {
            "P10_Pother_direction_to_153_state_tangent": False,
            "corrected_source_jet_to_output_bundle_connection": False,
            "generic_term_id_to_source_component": False,
            "ordered_mixed_D2F_root_for_each_output_coordinate": False,
            "required_fields": [
                "generic_term_id",
                "source_row",
                "coordinate_atom",
                "direction_tangent_coefficients_in_153_basis",
                "ordered_D2_arithmetic_root",
                "ordered_D2_arithmetic_dag_sha256",
                "output_bundle_projection_rule_id",
                "candidate_id",
            ],
        }
        or {key for key, value in claim_seals.items() if value}
        != {
            "all_22_target_D1_row_atom_entries_registered",
            "mixed_D2_22_parameter_ambiguity_constructed",
            "term_projection_affine_dimension_506_proved",
            "two_exact_term_projection_witnesses_constructed",
        }
        or not exact_controls
        or any(control != {"rejected": True} for control in exact_controls.values())
    ):
        raise NotebookValidationError("component-map schema completion boundary changed")
    theorem = result["component_map_schema_theorem"]
    broad_false = {
        "candidate_theory_rejected",
        "complete_ordered_D2F_tensor_registered",
        "corrected_second_source_jet_registered",
        "covariant_output_connection_derivation_registered",
        "cross_slice_D2F_entries_admitted",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "observational_claim_made",
        "physical_covariant_component_map_no_go_proved",
        "registered_cross_registry_component_map_unique",
    }
    if (
        theorem["name"]
        != "registered_tensor_and_index_conventions_do_not_determine_the_cross_registry_map"
        or "affine dimension 506" not in theorem["exact_result"]
        or "22-parameter source-jet ambiguity" not in theorem["exact_result"]
        or "not a no-go" not in theorem["boundary"]
        or any(claim_seals[key] for key in broad_false)
    ):
        raise NotebookValidationError("component-map schema theorem boundary changed")
    evidence = (receipt_path,)
    cells = (
        NotebookCell(
            "The cross-registry question",
            "The registered generic $G_4$ variation has 24 exact abstract term coefficients, "
            "while the fitted output connection has 22 coordinates. The source inventory also "
            "contains the target $D^1$ row-atom values. The question is whether those registered "
            "values uniquely determine a component projection and the corresponding mixed "
            "second jets.",
            evidence,
        ),
        NotebookCell(
            "Set up the 22-by-24 projection problem",
            "Let $c\\in\\mathbb K^{24}$ be the generic coefficient vector over "
            "$\\mathbb K=\\mathbb Q(\\sqrt2)$, let "
            "$M\\in\\mathbb K^{22\\times24}$ be a proposed cross-registry map, and let "
            "$\\beta\\in\\mathbb K^{22}$ be the fitted value vector. Registered value "
            "agreement imposes\n\n"
            "$$Mc=\\beta.$$\n\n"
            "There are $22\\cdot24=528$ entries of $M$. Because $c_0=1\\ne0$, each output "
            "row contributes one independent scalar equation, and different rows use disjoint "
            "unknowns. Therefore the constraint rank is exactly 22 and\n\n"
            "$$\\dim\\{M:Mc=\\beta\\}=528-22=506.$$",
            evidence,
        ),
        NotebookCell(
            "Construct two maps with identical registered values",
            "A base witness puts $\\beta_j$ in column zero of row $j$ and zeros elsewhere. "
            "Since $c_0=1$, it sends $c$ to $\\beta$. The alternate witness changes only row "
            "zero: it sets $M_{00}=1/2$ and $M_{01}=-1$. Since $c_1=-1/2$,\n\n"
            "$$M_{00}c_0+M_{01}c_1=\\tfrac12(1)+(-1)(-\\tfrac12)=1=\\beta_0.$$\n\n"
            "Every other row is unchanged. Thus two distinct exact $22\\times24$ maps have "
            "zero residual against the same registered value vector. This is a constructive "
            "failure of schema identification, not an approximate fit.",
            evidence,
        ),
        NotebookCell(
            "Construct the mixed-second-jet ambiguity",
            "For each typed target coordinate introduce an independent exact parameter "
            "$\\mu_i$ and retain its registered first derivative:\n\n"
            "$$D^1F_i=\\beta_i,\\qquad D^2_{\\mathrm{mixed}}F_i=\\mu_i,\\qquad "
            "i=0,\\ldots,21.$$\n\n"
            "The receipt registers every target $D^1$ membership but no direction-to-state "
            "tangent embedding and no ordered mixed-$D^2F$ root. Hence changing any $\\mu_i$ "
            "preserves all registered values. The zero vector and the 22 unit vectors "
            "$e_0,\\ldots,e_{21}$ give 23 explicit, pairwise distinct completions. More "
            "generally, the mixed-jet ambiguity has 22 independent parameters.",
            evidence,
        ),
        NotebookCell(
            "What identical values do not determine",
            "The first construction holds $Mc=\\beta$ fixed while changing the projection "
            "schema through a 506-dimensional affine family. The second holds all 22 registered "
            "$D^1$ values fixed while changing 22 mixed-$D^2$ entries. Together they show "
            "constructively that equality of the registered values alone does not select the "
            "map or its action jets.",
            evidence,
        ),
        NotebookCell(
            "The scientific boundary",
            "These witnesses are schema completions, not certified covariant physical maps. "
            "Tensor equivariance, the typed generic-term-to-source-component projection, the "
            "$P10/Pother$ state-tangent embedding, and the 22 ordered mixed-$D^2F$ roots remain "
            "unregistered. Therefore schema nonidentifiability is **not** a physical no-go, a "
            "candidate rejection, or an admission of corrected source jets, complete $D^2F$, "
            "the high-atom identity, global $H^7$, nonlinear PDE closure, or lifespan. All 12 "
            "candidates remain blocked. The first blocker is\n\n"
            f"{result['first_blocker']}.",
            evidence,
        ),
    )
    return ResearchNotebook(
        notebook_id="quartic-component-map-schema-ambiguity-reconstruction-001",
        title="Constructive nonidentifiability of a quartic component-map schema",
        verdict="proved",
        source_bindings=(binding,),
        claims=(
            NotebookClaim(
                "proved",
                "The exact 22-by-24 registered value system has rank 22 and a 506-dimensional affine family of projection completions.",
                evidence,
            ),
            NotebookClaim(
                "proved",
                "Twenty-two independent mixed-D2 parameters admit at least 23 explicit completions preserving every registered target D1 value.",
                evidence,
            ),
            NotebookClaim(
                "blocked",
                "The typed cross-registry projection, state-tangent embedding, and 22 ordered mixed-D2F roots remain unregistered.",
                evidence,
            ),
            NotebookClaim(
                "scope_limit",
                "Schema nonidentifiability is not a physical component-map no-go, D2F admission, global theorem, or candidate rejection.",
                evidence,
            ),
        ),
        cells=cells,
        limits=(
            "the notebook is a derived presentation of one sealed receipt, not an independent proof kernel",
            "the two projection witnesses satisfy registered values but are not certified covariant maps",
            "the 23 mixed-D2 witnesses are schema completions, not admitted corrected second-source jets",
            "physical no-go, D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, and observational claims remain fail-closed",
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
        (
            "quartic-action-jet-nonidentifiability",
            build_action_jet_nonidentifiability_notebook(root),
        ),
        (
            "quartic-registered-variation-selection-audit",
            build_registered_variation_selection_notebook(root),
        ),
        (
            "quartic-component-map-schema-ambiguity",
            build_component_map_schema_ambiguity_notebook(root),
        ),
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
