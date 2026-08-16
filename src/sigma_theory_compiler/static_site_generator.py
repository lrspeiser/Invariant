"""Receipts-driven static site generator for the public Invariant site.

The site is a rendering of the repository's sealed evidence, never a parallel set of
claims.  Three rules keep it honest.

**Every quantitative claim is read from a repo artifact at build time.**  Counts,
margins, throughputs, formulas, and decisions come from the sealed receipts under
``runs/``, the sealed problem queue under ``configs/``, and the measured-outcome
documents under ``docs/``.  This module contains none of those numerals as source
literals; delete a receipt and its numbers disappear from the site rather than being
remembered.

**No scalar score of any kind is rendered.**  A result either passes an exact gate,
fails it, or is explicitly undecided.  Failures get the same billing as successes:
the sealed cluster negative, the blind-guessing failure, and the formal-rejection
history are headline content, not footnotes.

**Missing evidence is declared, never papered over.**  An absent optional artifact
produces an explicit "Evidence not yet published" block in place of its section; the
build never crashes on absence and never silently omits the section.

Determinism: ``render_site`` is a pure function of the artifact bytes and the commit
argument — no timestamps, no randomness, no environment reads — so the same inputs
produce byte-identical output and ``--validate`` can re-run the build and compare.
The commit is passed in by the caller (``--commit``) rather than queried here, so the
build itself stays deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .problem_queue import MACHINE_FORM_KINDS, SYSTEM_CAPS, ProblemQueueError, load_queue

SITE_TITLE = "Invariant — an exact, auditable discovery engine"

#: Required on every page, verbatim (tests pin it).
FOOTER_CREED = (
    "Every number on this page is read from a sealed receipt at build time. "
    "Corpus absence is never novelty; survival is never proof; "
    "negative receipts are deliverables."
)

#: Required opening of every fail-soft block (tests pin it).
MISSING_NOTE = "Evidence not yet published"

SUBMIT_NOTICE = (
    "Submissions are reviewed by a human and, if accepted, sealed into the queue with a "
    "new content hash. Nothing on this site writes to the repository."
)

GITHUB_BLOB = "https://github.com/lrspeiser/Invariant/blob/main/"
GITHUB_NEW_ISSUE = "https://github.com/lrspeiser/Invariant/issues/new"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")

#: Repo-relative paths of every artifact the site consumes.  All are optional at
#: build time (fail-soft); the committed site is built with all of them present.
ARTIFACT_PATHS = {
    "queue": "configs/problem_queue_v1.json",
    "billion": "runs/gpu-baryonic-screen/billion-v1.json",
    "lensing": "runs/gpu-baryonic-screen/lensing-cluster-v1.json",
    "sweep": "runs/math/counterexample-sweeps/collatz-halving-1e8.json",
    "lean": "formal/lean/CollatzHalvingRelation.lean",
    "goals_doc": "docs/GOALS_AND_MEASURED_OUTCOMES.md",
    "idt_doc": "docs/INDEPENDENT_DISCOVERY_TRIAL.md",
    "roadmap_doc": "docs/CONTINUOUS_DISCOVERY_ROADMAP.md",
}

_DOC_NAMES = {
    "goals_doc": "GOALS_AND_MEASURED_OUTCOMES.md",
    "idt_doc": "INDEPENDENT_DISCOVERY_TRIAL.md",
    "roadmap_doc": "CONTINUOUS_DISCOVERY_ROADMAP.md",
}

#: The claim-discipline list, binding on every task (CONTINUOUS_DISCOVERY_ROADMAP.md).
CLAIM_DISCIPLINE = (
    "Corpus absence is never novelty.",
    "Survival is never proof.",
    "A restricted domain is never a global claim.",
    "Kernel verification happens only in the kernel.",
    "No invisible mass as target or rescue.",
    "Sealed data opens once, no refit.",
    "Negative receipts are deliverables.",
)

NAV = (
    ("index", "/", "Overview"),
    ("problems", "/problems", "Problems"),
    ("gravity", "/gravity", "Gravity"),
    ("collatz", "/collatz", "Collatz"),
    ("evidence", "/evidence", "Evidence"),
    ("method", "/method", "Method"),
    ("submit", "/submit", "Submit"),
)

#: Which sealed receipts / notebook pages reference which queue problems.
EVIDENCE_TIMELINES: dict[str, tuple[tuple[str, str], ...]] = {
    "collatz_stopping_time": (
        ("Case study: engine silence, two exact identities, conditional Lean proof", "/collatz"),
        ("Counterexample-sweep receipt for the halving relation", "/collatz#sweep"),
        ("Formal source: the conditional halving relation in Lean", "/collatz#lean"),
    ),
    "baryonic_rotation_law": (
        ("Billion-candidate GPU screen of the nu(y) family", "/gravity#base-screen"),
        ("Screen Pareto front: formulas with measured margins", "/gravity#pareto"),
    ),
    "lensing_dynamics_consistency": (
        ("Lensing gate (P1) over the full family, with named controls", "/gravity#lensing-cluster"),
        ("Near-miss formulas with their P1/P2 margins", "/gravity#near-misses"),
    ),
    "cluster_missing_mass": (
        ("The sealed negative cluster verdict, with exact margins", "/gravity#sealed-negative"),
    ),
}

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0b0e14; color: #c9d3e0;
  font: 15px/1.55 system-ui, "Segoe UI", Roboto, sans-serif; }
a { color: #6cb1e1; text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre, .mono { font-family: ui-monospace, "Cascadia Code", Consolas, Menlo, monospace; }
header { border-bottom: 1px solid #1e2633; padding: 0.65rem 1.25rem; display: flex;
  flex-wrap: wrap; gap: 0.4rem 1.5rem; align-items: baseline; }
header .brand a { color: #e8eef6; font-weight: 700; letter-spacing: 0.02em; }
header .tag { color: #7f8b9e; font-size: 0.8rem; }
nav a { color: #7f8b9e; margin-right: 0.9rem; font-size: 0.86rem; }
nav a.on { color: #e8eef6; }
main { max-width: 1100px; margin: 0 auto; padding: 1.4rem 1.25rem 3rem; }
h1 { font-size: 1.45rem; margin: 0.4rem 0 0.6rem; }
h2 { font-size: 1.12rem; margin: 2rem 0 0.5rem; border-bottom: 1px solid #1e2633;
  padding-bottom: 0.25rem; }
h3 { font-size: 0.98rem; margin: 1.2rem 0 0.4rem; }
p { margin: 0.5rem 0; }
.sub { color: #7f8b9e; }
.small { font-size: 0.82rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 0.6rem; margin: 0.9rem 0; }
.tile { background: #10151f; border: 1px solid #1e2633; border-radius: 6px;
  padding: 0.6rem 0.75rem; }
.tile .v { font-family: ui-monospace, Consolas, monospace; font-size: 1.3rem;
  font-weight: 600; line-height: 1.25; }
.tile .k { color: #aeb9c9; font-size: 0.8rem; margin-top: 0.1rem; }
.tile .r { color: #7f8b9e; font-family: ui-monospace, Consolas, monospace;
  font-size: 0.68rem; margin-top: 0.3rem; word-break: break-all; }
.tile.neg .v { color: #d8a03d; }
.tile.fail .v { color: #e0655a; }
.tile.pass .v { color: #4cb964; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; margin: 0.7rem 0; }
th, td { border: 1px solid #1e2633; padding: 0.26rem 0.5rem; text-align: left;
  vertical-align: top; }
th { background: #10151f; font-weight: 600; }
td.num, th.num { text-align: right; font-family: ui-monospace, Consolas, monospace;
  white-space: nowrap; }
td.mono { font-family: ui-monospace, Consolas, monospace; white-space: nowrap; }
pre { background: #0e1219; border: 1px solid #1e2633; border-radius: 6px;
  padding: 0.75rem 0.9rem; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; }
blockquote { border-left: 3px solid #2a3648; margin: 0.7rem 0; padding: 0.15rem 0.9rem;
  color: #9daabb; font-size: 0.88rem; }
.missing { border: 1px dashed #8a6d1f; background: rgba(216, 160, 61, 0.07);
  border-radius: 6px; padding: 0.65rem 0.9rem; color: #d8b04a; margin: 0.8rem 0;
  font-size: 0.88rem; }
.neg-block { border: 1px solid #8a6d1f; background: rgba(216, 160, 61, 0.06);
  border-radius: 6px; padding: 0.8rem 1rem; margin: 0.9rem 0; }
.badge { display: inline-block; border: 1px solid; border-radius: 4px;
  font: 600 0.68rem ui-monospace, Consolas, monospace; padding: 0.06rem 0.4rem;
  margin-left: 0.4rem; vertical-align: middle; }
.badge.pass { color: #4cb964; border-color: #4cb964; }
.badge.fail { color: #e0655a; border-color: #e0655a; }
.badge.neg { color: #d8a03d; border-color: #d8a03d; }
.badge.ctl { color: #8caff5; border-color: #8caff5; }
details.card { background: #10151f; border: 1px solid #1e2633; border-radius: 6px;
  margin: 0.45rem 0; }
details.card > summary { cursor: pointer; padding: 0.5rem 0.8rem; font-size: 0.82rem;
  font-family: ui-monospace, Consolas, monospace; }
details.card > div { padding: 0.1rem 0.9rem 0.7rem; border-top: 1px solid #1e2633; }
ul { margin: 0.4rem 0 0.4rem 1.2rem; padding: 0; }
li { margin: 0.22rem 0; }
.form-grid { display: grid; grid-template-columns: 13rem 1fr; gap: 0.5rem 0.8rem;
  align-items: start; margin: 0.8rem 0; }
.form-grid label { color: #aeb9c9; font-size: 0.84rem; padding-top: 0.3rem; }
input[type="text"], select, textarea { width: 100%; background: #0e1219;
  color: #c9d3e0; border: 1px solid #2a3648; border-radius: 4px;
  padding: 0.35rem 0.5rem; font: 0.84rem ui-monospace, Consolas, monospace; }
textarea { min-height: 4.4rem; resize: vertical; }
button { background: #1b2536; color: #dfe7f1; border: 1px solid #2a3648;
  border-radius: 4px; padding: 0.4rem 0.9rem; font-size: 0.85rem; cursor: pointer; }
button:hover { background: #24314a; }
#errors { color: #e0655a; font-size: 0.85rem; }
fieldset { border: 1px solid #1e2633; border-radius: 6px; margin: 0.6rem 0;
  padding: 0.5rem 0.9rem 0.8rem; }
legend { color: #aeb9c9; font-size: 0.8rem; padding: 0 0.4rem; }
footer { border-top: 1px solid #1e2633; margin-top: 2.5rem; padding: 1.1rem 1.25rem 2rem;
  color: #7f8b9e; font-size: 0.79rem; text-align: center; }
footer p { max-width: 62rem; margin: 0.3rem auto; }
"""


class SiteGenerationError(ValueError):
    """Raised for caller errors (bad commit, undecodable output tree)."""


# ---------------------------------------------------------------------------
# Artifact loading (fail-soft: absence is declared, never fatal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    key: str
    path: str
    data: Any
    text: str | None
    sha256: str | None
    sealed: bool
    missing_reason: str | None

    @property
    def present(self) -> bool:
        return self.missing_reason is None


def _load_artifact(root: Path, key: str, rel_path: str) -> Artifact:
    absent = Artifact(key, rel_path, None, None, None, False, "missing from the repository")
    try:
        raw = (root / rel_path).read_bytes()
    except OSError:
        return absent
    file_hash = hashlib.sha256(raw).hexdigest()
    if rel_path.endswith(".json"):
        if key == "queue":
            try:
                data = load_queue(root / rel_path)
            except ProblemQueueError:
                return Artifact(key, rel_path, None, None, None, False, "present but fails seal validation")
        else:
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return Artifact(key, rel_path, None, None, None, False, "present but is not valid JSON")
        seal = data.get("content_sha256") if isinstance(data, dict) else None
        if isinstance(seal, str):
            return Artifact(key, rel_path, data, None, seal, True, None)
        return Artifact(key, rel_path, data, None, file_hash, False, None)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return Artifact(key, rel_path, None, None, None, False, "present but is not valid UTF-8")
    return Artifact(key, rel_path, None, text, file_hash, False, None)


def _load_artifacts(root: Path) -> dict[str, Artifact]:
    return {key: _load_artifact(root, key, rel) for key, rel in ARTIFACT_PATHS.items()}


# ---------------------------------------------------------------------------
# Documented history claims, parsed from the docs (never source literals)
# ---------------------------------------------------------------------------

#: fact -> (artifact key, regex).  Regexes match measured statements in the docs;
#: a failed match renders as not-yet-published rather than a stale number.
_DOC_FACT_PATTERNS = {
    "blind_pass_reject": ("idt_doc", r"Blind semantic formula guessing \| (\d+) PASS / (\d+) REJECT"),
    "conditioned": ("goals_doc", r"(\d+)/(\d+) candidates passed with exact certificates"),
    "formal_rejections": ("idt_doc", r"(\d+)/(\d+) formal rejections"),
    "curriculum": ("goals_doc", r"registers (\d+)/(\d+) slots"),
    "formal_controls": ("goals_doc", r"(\d+)/(\d+) portable formal controls pass"),
    "halving_holdout": ("idt_doc", r"(\d+) holdout confirmations"),
}


def _doc_facts(artifacts: dict[str, Artifact]) -> dict[str, tuple[int, ...] | None]:
    facts: dict[str, tuple[int, ...] | None] = {}
    for fact, (doc_key, pattern) in _DOC_FACT_PATTERNS.items():
        text = artifacts[doc_key].text if artifacts[doc_key].present else None
        match = re.search(pattern, text) if text is not None else None
        facts[fact] = tuple(int(group) for group in match.groups()) if match else None
    return facts


def _blind_ratio(facts: dict[str, tuple[int, ...] | None]) -> str | None:
    """PASS/REJECT counts -> the documented 'passes out of attempts' form."""

    pair = facts.get("blind_pass_reject")
    if pair is None:
        return None
    passed, rejected = pair
    return f"{passed}/{passed + rejected}"


def _plain_ratio(facts: dict[str, tuple[int, ...] | None], fact: str) -> str | None:
    pair = facts.get(fact)
    if pair is None:
        return None
    return f"{pair[0]}/{pair[1]}"


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _cell(value: Any) -> str:
    """Escape a receipt value for a table cell; absent values render as an em dash."""

    return "&mdash;" if value is None else _esc(value)


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _dec(text: str, places: int | None = None) -> str:
    """Exact decimal rendering of a receipt's decimal-string value (no floats)."""

    value = Decimal(text)
    if places is not None:
        value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(value, "f")


def _pow10_exponent(value: int) -> int | None:
    digits = str(value)
    if digits.startswith("1") and len(digits) > 1 and set(digits[1:]) == {"0"}:
        return len(digits) - 1
    return None


def _tile(
    value: str, label: str, receipt: str, *, kind: str = "", data_key: str = "", data_value: str = ""
) -> str:
    cls = f"tile {kind}".strip()
    attrs = ""
    if data_key:
        attrs = f' data-key="{_esc(data_key)}" data-value="{_esc(data_value)}"'
    return (
        f'<div class="{cls}"{attrs}><div class="v">{value}</div>'
        f'<div class="k">{_esc(label)}</div><div class="r">{_esc(receipt)}</div></div>'
    )


def _missing_block(artifact: Artifact, section: str) -> str:
    return (
        f'<div class="missing"><strong>{_esc(MISSING_NOTE)}.</strong> '
        f"The {_esc(section)} section renders from <code>{_esc(artifact.path)}</code>, "
        f"which is {_esc(artifact.missing_reason or 'unavailable')}. "
        "Nothing is substituted in its place; the numbers appear when the artifact does.</div>"
    )


def _doc_gap(fact_name: str) -> str:
    return (
        f'<span class="sub">[{_esc(MISSING_NOTE)}: the documented {_esc(fact_name)} value '
        "was not found in the repository docs]</span>"
    )


def _badge(passed: bool, yes: str = "PASS", no: str = "FAIL") -> str:
    if passed:
        return f'<span class="badge pass">{_esc(yes)}</span>'
    return f'<span class="badge fail">{_esc(no)}</span>'


def _github(rel_path: str, label: str | None = None) -> str:
    return (
        f'<a href="{_esc(GITHUB_BLOB + rel_path)}" rel="noopener">'
        f"{_esc(label if label is not None else rel_path)}</a>"
    )


def _sha_abbrev(digest: str | None) -> str:
    if not digest:
        return "&mdash;"
    return f'<code title="{_esc(digest)}">{_esc(digest[:16])}&hellip;</code>'


def _page(slug: str, title: str, body: str, commit: str) -> bytes:
    nav_links = "".join(
        (
            f'<a href="{_esc(href)}" class="on">{_esc(label)}</a>'
            if key == slug
            else f'<a href="{_esc(href)}">{_esc(label)}</a>'
        )
        for key, href, label in NAV
    )
    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        '<header><div class="brand"><a href="/">Invariant</a> '
        '<span class="tag">an exact, auditable discovery engine</span></div>'
        f"<nav>{nav_links}</nav></header>\n<main>\n{body}\n</main>\n"
        f"<footer><p>{_esc(FOOTER_CREED)}</p>"
        f"<p>Site content as of <code>{_esc(commit)}</code>. Generated deterministically by "
        "<code>src/sigma_theory_compiler/static_site_generator.py</code>; "
        "re-run with <code>--validate</code> to byte-compare.</p></footer>\n"
        "</body>\n</html>\n"
    )
    return doc.encode("utf-8")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def _index_tiles(artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None]) -> str:
    tiles: list[str] = []
    billion = artifacts["billion"]
    if billion.present:
        counts = billion.data["counts"]
        tiles.append(
            _tile(
                _esc(_fmt_int(counts["processed"])),
                "candidate laws processed — the complete declared family",
                billion.path,
                data_key="candidates_processed",
                data_value=str(counts["processed"]),
            )
        )
        tiles.append(
            _tile(
                _esc(_fmt_int(billion.data["throughput_candidates_per_second"])) + "/s",
                f"screen throughput on {billion.data['device']}",
                billion.path,
                data_key="throughput_candidates_per_second",
                data_value=str(billion.data["throughput_candidates_per_second"]),
            )
        )
    else:
        tiles.append(_tile("&mdash;", "candidate laws processed", billion.path, kind="neg"))
    lensing = artifacts["lensing"]
    if lensing.present:
        counts = lensing.data["counts"]
        tiles.append(
            _tile(
                _esc(_fmt_int(counts["lensing_pass"])),
                "candidates passing the lensing gate (P1)",
                lensing.path,
                data_key="lensing_pass",
                data_value=str(counts["lensing_pass"]),
            )
        )
        tiles.append(
            _tile(
                _esc(_fmt_int(counts["cluster_pass"])),
                "candidates passing the cluster gate (P2) — a sealed negative",
                lensing.path,
                kind="neg",
                data_key="cluster_pass",
                data_value=str(counts["cluster_pass"]),
            )
        )
    else:
        tiles.append(_tile("&mdash;", "lensing / cluster campaign", lensing.path, kind="neg"))
    sweep = artifacts["sweep"]
    if sweep.present:
        hi = sweep.data["range"]["hi"]
        exponent = _pow10_exponent(hi)
        shown = f"10^{exponent}" if exponent is not None else _fmt_int(hi)
        tiles.append(
            _tile(
                _esc(shown),
                "Collatz halving relation swept to this bound: "
                + str(sweep.data["decision"]).lower().replace("_", " "),
                sweep.path,
                data_key="sweep_hi",
                data_value=str(hi),
            )
        )
    else:
        tiles.append(_tile("&mdash;", "counterexample sweep", sweep.path, kind="neg"))
    queue = artifacts["queue"]
    if queue.present:
        entries = queue.data["entries"]
        tiles.append(
            _tile(
                _esc(_fmt_int(len(entries))),
                "declared open problems in the sealed intake queue",
                queue.path,
                data_key="problem_count",
                data_value=str(len(entries)),
            )
        )
    else:
        tiles.append(_tile("&mdash;", "sealed problem queue", queue.path, kind="neg"))
    blind = _blind_ratio(facts)
    tiles.append(
        _tile(
            _esc(blind) if blind else "&mdash;",
            "blind semantic formula guessing — an honest failure, kept on the books",
            f"documented in repository docs ({_DOC_NAMES['idt_doc']})",
            kind="fail",
        )
    )
    rejections = _plain_ratio(facts, "formal_rejections")
    tiles.append(
        _tile(
            _esc(rejections) if rejections else "&mdash;",
            "formal rejections in one production epoch, with zero rotation curves computed",
            f"documented in repository docs ({_DOC_NAMES['idt_doc']})",
            kind="fail",
        )
    )
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _index_page(
    artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None], commit: str
) -> bytes:
    body: list[str] = []
    body.append(f"<h1>{_esc(SITE_TITLE)}</h1>")
    body.append(
        "<p>Invariant generates candidate laws and mathematical statements at GPU scale, then"
        " subjects them to fail-closed verification: exact-arithmetic replays, screen-versus-exact"
        " crosschecks, formal proof kernels, and sealed hash-bound receipts. What survives and what"
        " dies are published with the same ceremony — a sealed negative is a deliverable, not an"
        " embarrassment. Every quantitative claim on these pages is read from an artifact committed"
        " to the repository at the commit named in the footer; the site generator contains no"
        " numbers of its own.</p>"
    )
    body.append("<h2>Headline counts, each labeled with its source artifact</h2>")
    body.append(_index_tiles(artifacts, facts))
    body.append(
        '<p class="small sub">Doc-sourced history values predate sealed receipts and are cited to'
        " their document filenames; everything else is read from the named receipt at build"
        " time.</p>"
    )
    body.append("<h2>The honesty manifesto</h2>")
    body.append(
        "<ul>"
        + "".join(f"<li>{_esc(line)}</li>" for line in CLAIM_DISCIPLINE)
        + "<li>No scalar score of any kind appears in the receipts or on this site: a statement"
        " passes an exact gate, fails it, or is explicitly undecided.</li>"
        "<li>Every failed campaign stays on the books with the same billing as a success.</li>"
        "</ul>"
    )
    body.append(
        f'<p class="small sub">The first {len(CLAIM_DISCIPLINE)} lines are the claim-discipline'
        " list, binding on every task; documented in repository docs"
        f" ({_esc(_DOC_NAMES['roadmap_doc'])}).</p>"
    )
    body.append("<h2>Sections</h2>")
    body.append(
        "<ul>"
        '<li><a href="/problems">Problems</a> — the sealed intake queue: every declared target,'
        " its citation, why it is believed open, and what counts as progress.</li>"
        '<li><a href="/gravity">Gravity</a> — the physics notebook: the billion-candidate screen,'
        " the lensing gate, and the sealed negative cluster verdict with exact margins.</li>"
        '<li><a href="/collatz">Collatz</a> — the math notebook: engine silence, two discovered'
        " identities, the exhaustive sweep, and the conditional Lean proof.</li>"
        '<li><a href="/evidence">Evidence</a> — every artifact this site consumed, with content'
        " hashes and links to the repository.</li>"
        '<li><a href="/method">Method</a> — the validation ladder, the claim discipline, and the'
        " measured failures that made both necessary.</li>"
        '<li><a href="/submit">Submit</a> — propose a problem for the queue; reviewed by a human,'
        " sealed on acceptance.</li>"
        "</ul>"
    )
    return _page("index", SITE_TITLE, "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _flag_badges(entry: dict[str, Any]) -> str:
    badges = []
    if entry["control_rediscovery"]:
        badges.append('<span class="badge ctl">REDISCOVERY CONTROL</span>')
    if entry["synthetic"]:
        badges.append('<span class="badge ctl">SYNTHETIC</span>')
    if not badges:
        badges.append('<span class="badge neg">BELIEVED OPEN</span>')
    return "".join(badges)


def _machine_form_block(machine_form: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td class=\"mono\">{_esc(key)}</td><td class=\"mono\">{_esc(machine_form[key])}</td></tr>"
        for key in sorted(machine_form)
    )
    return (
        '<div class="scroll"><table><thead><tr><th>machine_form key</th><th>value</th></tr>'
        f"</thead><tbody>{rows}</tbody></table></div>"
    )


def _problem_detail_page(entry: dict[str, Any], queue: Artifact, commit: str) -> bytes:
    entry_id = entry["id"]
    body: list[str] = []
    body.append(f"<h1><code>{_esc(entry_id)}</code> {_flag_badges(entry)}</h1>")
    body.append(
        f'<p class="sub small">Domain <code>{_esc(entry["domain"])}</code> &middot; sealed in'
        f" <code>{_esc(queue.path)}</code> under queue seal"
        f" {_sha_abbrev(queue.sha256)}</p>"
    )
    body.append("<h2>Statement</h2>")
    body.append(f"<blockquote>{_esc(entry['statement'])}</blockquote>")
    body.append("<h2>Source citation</h2>")
    body.append(f"<p>{_esc(entry['source_citation'])}</p>")
    body.append("<h2>Why it is believed open</h2>")
    body.append(f"<p>{_esc(entry['believed_open_because'])}</p>")
    if entry["control_rediscovery"]:
        body.append(
            '<p class="small sub">Schema-labeled rediscovery control: the answer is known, the'
            " label is a validated boolean, and rediscovery is calibration evidence only — it is"
            " never reported as a discovery.</p>"
        )
    if entry["synthetic"]:
        body.append(
            '<p class="small sub">Schema-labeled synthetic world: openness here is operational'
            " (a sealed holdout), not epistemic, and the label cannot be dropped in prose.</p>"
        )
    body.append("<h2>What counts as progress</h2>")
    body.append(f"<p>{_esc(entry['progress_definition'])}</p>")
    body.append("<h2>Machine form</h2>")
    body.append(_machine_form_block(entry["machine_form"]))
    body.append("<h2>Evidence timeline</h2>")
    timeline = EVIDENCE_TIMELINES.get(entry_id, ())
    if timeline:
        body.append(
            "<ul>"
            + "".join(
                f'<li><a href="{_esc(href)}">{_esc(label)}</a></li>' for label, href in timeline
            )
            + "</ul>"
        )
    else:
        body.append(
            "<p>No sealed receipt in this repository references this problem yet. When one lands,"
            ' it will be linked here and listed on the <a href="/evidence">evidence</a> page.</p>'
        )
    body.append('<p class="small"><a href="/problems">&larr; all problems</a></p>')
    title = f"{entry_id} · {SITE_TITLE}"
    return _page("problems", title, "\n".join(body), commit)


def _problems_index_page(queue: Artifact, commit: str) -> bytes:
    body: list[str] = []
    body.append("<h1>The sealed problem queue</h1>")
    body.append(
        "<p>The intake queue is a hash-bound registry of declared targets. Every entry cites a"
        " source, states in prose why it is believed open, and defines in advance what counts as"
        " progress. Rediscovery controls and synthetic sealed worlds carry their labels as"
        " validated schema booleans, so a calibration entry can never be reported as a discovery"
        " by dropping a sentence. Queue membership asserts provenance — not importance,"
        " tractability, or solvability.</p>"
    )
    if not queue.present:
        body.append(_missing_block(queue, "problem queue"))
        return _page("problems", f"Problems · {SITE_TITLE}", "\n".join(body), commit)
    entries = queue.data["entries"]
    counts = {
        "entries": len(entries),
        "math": sum(1 for e in entries if e["domain"].split("/", 1)[0] == "math"),
        "physics": sum(1 for e in entries if e["domain"].split("/", 1)[0] == "physics"),
        "controls": sum(1 for e in entries if e["control_rediscovery"]),
        "synthetic": sum(1 for e in entries if e["synthetic"]),
    }
    tiles = [
        _tile(
            _esc(_fmt_int(counts["entries"])),
            "sealed entries",
            queue.path,
            data_key="problem_count",
            data_value=str(counts["entries"]),
        ),
        _tile(_esc(_fmt_int(counts["math"])), "math domain", queue.path),
        _tile(_esc(_fmt_int(counts["physics"])), "physics domain", queue.path),
        _tile(
            _esc(_fmt_int(counts["controls"])),
            "rediscovery controls (labeled, never discoveries)",
            queue.path,
        ),
        _tile(
            _esc(_fmt_int(counts["synthetic"])),
            "synthetic sealed worlds (labeled)",
            queue.path,
        ),
    ]
    body.append('<div class="tiles">' + "".join(tiles) + "</div>")
    body.append(
        f'<p class="small sub">Queue schema <code>{_esc(queue.data["schema_version"])}</code>,'
        f" content seal {_sha_abbrev(queue.sha256)} &middot; "
        + _github(queue.path, "view the sealed file on GitHub")
        + "</p>"
    )
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f'<td class="mono"><a href="/problems/{_esc(entry["id"])}">{_esc(entry["id"])}</a></td>'
            f'<td class="mono">{_esc(entry["domain"])}</td>'
            f'<td class="mono">{_esc(entry["machine_form"]["kind"])}</td>'
            f"<td>{_flag_badges(entry)}</td>"
            f"<td>{_esc(entry['statement'])}</td>"
            "</tr>"
        )
    body.append("<h2>Entries</h2>")
    body.append(
        '<div class="scroll"><table><thead><tr><th>id</th><th>domain</th><th>machine form</th>'
        "<th>flags</th><th>statement</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return _page("problems", f"Problems · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------


def _thresholds_table(config: dict[str, Any]) -> str:
    fp32 = config.get("fp32_thresholds", {})
    fp64 = config.get("fp64_thresholds", {})
    names = sorted(set(fp32) | set(fp64))
    rows = "".join(
        "<tr>"
        f'<td class="mono">{_esc(name)}</td>'
        f'<td class="num">{_cell(fp32.get(name))}</td>'
        f'<td class="num">{_cell(fp64.get(name))}</td>'
        "</tr>"
        for name in names
    )
    return (
        '<div class="scroll"><table><thead><tr><th>gate threshold</th><th class="num">fp32'
        ' screen</th><th class="num">fp64 confirm</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _base_screen_section(billion: Artifact) -> str:
    parts: list[str] = ['<h2 id="base-screen">Base screen: the complete declared family</h2>']
    if not billion.present:
        parts.append(_missing_block(billion, "base screen"))
        return "\n".join(parts)
    data = billion.data
    counts = data["counts"]
    tiles = [
        _tile(
            _esc(_fmt_int(counts["processed"])),
            "candidates processed (equals the declared family size)",
            billion.path,
            data_key="candidates_processed",
            data_value=str(counts["processed"]),
        ),
        _tile(
            _esc(_fmt_int(counts["fp64_survivors"])),
            "fp64 survivors of all five gates",
            billion.path,
            data_key="fp64_survivors",
            data_value=str(counts["fp64_survivors"]),
        ),
        _tile(
            _esc(_fmt_int(counts["exact_confirmed"])) + " / " + _esc(_fmt_int(counts["exact_refuted"])),
            "Pareto entries exact-confirmed / refuted at high-precision replay",
            billion.path,
        ),
        _tile(
            _esc(_fmt_int(data["throughput_candidates_per_second"])) + "/s",
            f"throughput, {data['device']}, {data['elapsed_seconds']} s wall",
            billion.path,
        ),
    ]
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")
    parts.append(
        f"<p>Decision: <code>{_esc(data['decision'])}</code>. The family is an ordinal-indexed"
        " grammar of universal baryonic acceleration laws with one shared constant and zero"
        " per-galaxy freedom, screened on frozen synthetic disk controls for Newtonian recovery,"
        " monotone dynamics, flat outer rotation curves, and the baryonic Tully-Fisher slope."
        " Survivors are search priorities, not validated theories — the receipt says so in its"
        " own scope field, quoted at the bottom of this page.</p>"
    )
    parts.append("<h3>Gate thresholds (from the receipt config)</h3>")
    parts.append(_thresholds_table(data.get("config", {})))
    pareto = data.get("pareto_front", [])
    parts.append(f'<h2 id="pareto">The {len(pareto)}-entry Pareto front</h2>')
    parts.append(
        "<p>Axes: simplicity (grammar term count), Solar-System convergence"
        " (<code>newton_error</code>), and outer-curve flatness. Formulas are rendered verbatim"
        " from the receipt; every entry was exact-confirmed at high-precision replay.</p>"
    )
    rows = []
    for index, entry in enumerate(pareto):
        rows.append(
            "<tr>"
            f'<td class="num">{index + 1}</td>'
            f'<td class="num">{_esc(entry["simplicity"])}</td>'
            f'<td class="mono">{_esc(entry["formula"])}</td>'
            f'<td class="num">{_esc(entry["newton_error"])}</td>'
            f'<td class="num">{_esc(entry["flatness"])}</td>'
            f'<td class="num">{_esc(entry["ordinal"])}</td>'
            f"<td>{_badge(bool(entry.get('exact_confirmed')), 'CONFIRMED', 'REFUTED')}</td>"
            "</tr>"
        )
    parts.append(
        '<div class="scroll"><table><thead><tr><th class="num">#</th><th class="num">simplicity'
        '</th><th>formula</th><th class="num">newton_error</th><th class="num">flatness</th>'
        '<th class="num">ordinal</th><th>exact replay</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return "\n".join(parts)


def _control_card(name: str, control: dict[str, Any], note: str) -> str:
    lensing = control.get("lensing", {})
    cluster = control.get("cluster", {})
    per_mass = "".join(
        "<tr>"
        f'<td class="mono">{_esc(row["mass_text"])}</td>'
        f'<td class="num">{_esc(row["flatness"])}</td>'
        f'<td class="num">{_esc(row["v_flat"])}</td>'
        "</tr>"
        for row in lensing.get("per_mass", [])
    )
    return (
        '<details class="card"><summary>'
        f"{_esc(control.get('formula', name))} &nbsp;"
        + _badge(bool(lensing.get("passes")), "P1 PASS", "P1 FAIL")
        + _badge(bool(cluster.get("passes")), "P2 PASS", "P2 FAIL")
        + "</summary><div>"
        f"<p>{note}</p>"
        f'<p class="small">Control key <code>{_esc(name)}</code>, ordinal'
        f' <code>{_cell(control.get("ordinal"))}</code>. Worst lensing consistency'
        f' <code>{_cell(lensing.get("worst_consistency"))}</code>, worst flatness'
        f' <code>{_cell(lensing.get("worst_flatness"))}</code>; cluster max deviation'
        f' <code>{_cell(cluster.get("max_deviation"))}</code>, closest probe deviation'
        f' <code>{_cell(cluster.get("closest_probe_deviation"))}</code>.</p>'
        '<div class="scroll"><table><thead><tr><th>disk mass</th><th class="num">flatness</th>'
        '<th class="num">v_flat</th></tr></thead><tbody>' + per_mass + "</tbody></table></div>"
        "</div></details>"
    )


def _lensing_cluster_section(lensing: Artifact) -> str:
    parts: list[str] = ['<h2 id="lensing-cluster">Lensing and cluster campaign</h2>']
    if not lensing.present:
        parts.append(_missing_block(lensing, "lensing and cluster campaign"))
        return "\n".join(parts)
    data = lensing.data
    counts = data["counts"]
    tiles = [
        _tile(
            _esc(_fmt_int(counts["processed"])),
            "candidates re-processed against both gates",
            lensing.path,
        ),
        _tile(
            _esc(_fmt_int(counts["fp32_union_survivors"])),
            "fp32 union survivors carried to fp64",
            lensing.path,
        ),
        _tile(
            _esc(_fmt_int(counts["lensing_pass"])),
            "pass the lensing gate (P1)",
            lensing.path,
            data_key="lensing_pass",
            data_value=str(counts["lensing_pass"]),
        ),
        _tile(
            _esc(_fmt_int(counts["cluster_pass"])),
            "pass the cluster gate (P2)",
            lensing.path,
            kind="neg",
            data_key="cluster_pass",
            data_value=str(counts["cluster_pass"]),
        ),
        _tile(
            _esc(_fmt_int(counts["both_pass"])),
            "pass both gates jointly",
            lensing.path,
            kind="neg",
            data_key="both_pass",
            data_value=str(counts["both_pass"]),
        ),
    ]
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")
    parts.append(
        f"<p>Throughput {_esc(_fmt_int(data['throughput_candidates_per_second']))}/s on"
        f" {_esc(data['device'])}, {_esc(data['elapsed_seconds'])} s wall. P1 tests deflection"
        " flatness and dynamics&ndash;lensing consistency on spherical equivalents of the screen"
        " masses under a declared factor-two lensing prescription; P2 is a hydrostatic isothermal"
        " gas cluster whose dynamical field the gas alone cannot supply under Newton. One"
        " universal constant, zero per-object freedom.</p>"
    )
    cluster_negative = data.get("cluster_negative", {})
    closest = cluster_negative.get("closest_cluster_approach", {})
    tolerance = (
        data.get("config", {}).get("cluster", {}).get("fp64_thresholds", {}).get("consistency")
    )
    parts.append('<h2 id="sealed-negative">The headline result is a sealed negative</h2>')
    negative_lines = [
        '<div class="neg-block">',
        (
            "<p><strong>Decision, verbatim from the receipt:</strong> "
            f"<code>{_esc(data['decision'])}</code></p>"
        ),
    ]
    if closest and tolerance:
        margin = closest.get("max_deviation", "")
        negative_lines.append(
            "<p>The closest any candidate came to carrying the cluster control: max deviation"
            f' <code data-key="closest_cluster_max_deviation" data-value="{_esc(margin)}">'
            f"{_esc(_dec(margin, 4))}</code> against tolerance"
            f' <code data-key="cluster_tolerance" data-value="{_esc(tolerance)}">'
            f"{_esc(_dec(tolerance))}</code>"
            f" (receipt strings <code>{_esc(margin)}</code> vs <code>{_esc(tolerance)}</code>)."
            " The nearest miss:</p>"
        )
        negative_lines.append(
            '<details class="card"><summary>'
            + _esc(closest.get("formula", ""))
            + " &nbsp;"
            + _badge(bool(closest.get("lensing_passes")), "P1 PASS", "P1 FAIL")
            + '<span class="badge neg">P2 NEAREST MISS</span>'
            "</summary><div>"
            f'<p class="small">Ordinal <code>{_esc(closest.get("ordinal", ""))}</code>;'
            f" located by {_esc(closest.get('located_by', ''))}. Max cluster deviation"
            f" <code>{_esc(closest.get('max_deviation', ''))}</code>.</p>"
            "</div></details>"
        )
    statement = cluster_negative.get("statement")
    if statement:
        negative_lines.append(
            f"<p><strong>Sealed statement:</strong> {_esc(statement)}.</p>"
        )
    negative_lines.append(
        "<p>This is the deliverable. A grammar-wide zero, with exact margins, is a scientific"
        " result about this grammar — it is published with the same ceremony as any success,"
        " and it is what the queue's cluster entry predicted as the expected outcome.</p>"
    )
    negative_lines.append("</div>")
    parts.extend(negative_lines)
    parts.append('<h3 id="controls">Named controls</h3>')
    controls = data.get("controls", {})
    notes = {
        "linear_u": (
            "The named &ldquo;flattens curves but fails lensing&rdquo; control: nu = 1 + u"
            " flattens all three synthetic rotation curves (flatness column below) yet fails the"
            " P1 dynamics&ndash;lensing consistency gate — exactly the family the roadmap said"
            " this gate must name. Documented in repository docs"
            f" ({_esc(_DOC_NAMES['roadmap_doc'])}, Track C)."
        ),
        "newton_nu1": (
            "Pure Newton on baryons alone (nu = 1): fails flatness outright — the control that"
            " restates the missing-mass problem the campaign is probing."
        ),
        "sqrt_family": (
            "A known interpolating shape: passes the lensing gate and still fails the cluster"
            " gate — the historically expected MOND-like behavior, measured here per family."
        ),
    }
    for name in sorted(controls):
        parts.append(_control_card(name, controls[name], notes.get(name, "Receipt control.")))
    parts.append('<h3 id="near-misses">Example near-misses with their P1/P2 margins</h3>')
    parts.append(
        "<p>Exact-verified candidates ordered by cluster max deviation (P2 margin); the P1 column"
        " is the worst dynamics&ndash;lensing consistency across the three disk masses.</p>"
    )
    verified = [
        entry
        for entry in data.get("exact_verification", [])
        if isinstance(entry.get("cluster"), dict)
        and isinstance(entry["cluster"].get("max_deviation"), str)
    ]
    verified.sort(key=lambda e: (Decimal(e["cluster"]["max_deviation"]), e.get("ordinal", 0)))
    rows = []
    for entry in verified[:8]:
        lensing_block = entry.get("lensing", {})
        cluster_block = entry.get("cluster", {})
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(entry.get("formula", ""))}</td>'
            f'<td class="num">{_esc(entry.get("ordinal", ""))}</td>'
            f'<td class="num">{_cell(lensing_block.get("worst_consistency"))}</td>'
            f"<td>{_badge(bool(lensing_block.get('passes')), 'P1 PASS', 'P1 FAIL')}</td>"
            f'<td class="num">{_cell(cluster_block.get("max_deviation"))}</td>'
            f"<td>{_badge(bool(cluster_block.get('passes')), 'P2 PASS', 'P2 FAIL')}</td>"
            "</tr>"
        )
    parts.append(
        '<div class="scroll"><table><thead><tr><th>formula</th><th class="num">ordinal</th>'
        '<th class="num">P1 worst consistency</th><th>P1</th><th class="num">P2 max deviation'
        "</th><th>P2</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    assumptions = data.get("assumptions", {})
    if assumptions:
        parts.append("<h3>Declared assumptions (not derivations)</h3>")
        parts.append(
            "<ul>"
            + "".join(
                f"<li><code>{_esc(key)}</code>: {_esc(assumptions[key])}</li>"
                for key in sorted(assumptions)
            )
            + "</ul>"
        )
    return "\n".join(parts)


def _crosscheck_section(billion: Artifact, lensing: Artifact) -> str:
    parts = ['<h2 id="crosscheck">Screen-versus-exact crosschecks</h2>']
    wrote_any = False
    if billion.present:
        crosscheck = billion.data.get("crosscheck", {})
        parts.append(
            f"<p>Base screen: {_esc(_fmt_int(crosscheck.get('disagreements', 0)))} disagreements"
            f" between the GPU screen and the exact CPU replay over a"
            f" {_esc(_fmt_int(crosscheck.get('sample', 0)))}-candidate sample"
            f" (<code>{_esc(billion.path)}</code>).</p>"
        )
        wrote_any = True
    if lensing.present:
        crosscheck = lensing.data.get("crosscheck", {})
        parts.append(
            f"<p>Lensing/cluster campaign: {_esc(_fmt_int(crosscheck.get('lensing_disagreements', 0)))}"
            f" lensing disagreements and {_esc(_fmt_int(crosscheck.get('cluster_disagreements', 0)))}"
            f" cluster disagreements over a {_esc(_fmt_int(crosscheck.get('sample', 0)))}-candidate"
            f" sample (<code>{_esc(lensing.path)}</code>).</p>"
        )
        wrote_any = True
    if not wrote_any:
        parts.append(
            f'<p class="sub">[{_esc(MISSING_NOTE)}: no crosscheck receipt is available]</p>'
        )
    return "\n".join(parts)


def _scope_section(billion: Artifact, lensing: Artifact) -> str:
    parts = ['<h2 id="scope">What this does and does not establish</h2>']
    parts.append(
        "<p>Quoted verbatim from the receipts&rsquo; own <code>scope</code> fields — the scope"
        " travels with the evidence, not with the marketing:</p>"
    )
    for artifact in (billion, lensing):
        if artifact.present:
            parts.append(
                f'<p class="small sub"><code>{_esc(artifact.path)}</code>:</p>'
                f"<blockquote>{_esc(artifact.data.get('scope', ''))}</blockquote>"
            )
        else:
            parts.append(_missing_block(artifact, "scope"))
    return "\n".join(parts)


def _gravity_page(artifacts: dict[str, Artifact], commit: str) -> bytes:
    billion = artifacts["billion"]
    lensing = artifacts["lensing"]
    body: list[str] = []
    body.append("<h1>Gravity: a physics notebook with sealed margins</h1>")
    body.append(
        "<p>Three queue problems drive this campaign:"
        ' <a href="/problems/baryonic_rotation_law"><code>baryonic_rotation_law</code></a>,'
        ' <a href="/problems/lensing_dynamics_consistency"><code>lensing_dynamics_consistency'
        "</code></a>, and"
        ' <a href="/problems/cluster_missing_mass"><code>cluster_missing_mass</code></a>. The'
        " campaign searches a declared grammar of universal acceleration laws on synthetic"
        " analytic controls. No observational data has been opened; no invisible mass appears"
        " anywhere in the grammar; and the expected cluster outcome — a sealed negative — is"
        " exactly what was found.</p>"
    )
    body.append(_base_screen_section(billion))
    body.append(_lensing_cluster_section(lensing))
    body.append(_crosscheck_section(billion, lensing))
    body.append(_scope_section(billion, lensing))
    return _page("gravity", f"Gravity · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Collatz
# ---------------------------------------------------------------------------


def _collatz_page(
    artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None], commit: str
) -> bytes:
    sweep = artifacts["sweep"]
    lean = artifacts["lean"]
    idt_name = _DOC_NAMES["idt_doc"]
    body: list[str] = []
    body.append("<h1>Collatz: a math notebook that starts with silence</h1>")
    body.append(
        "<p>Queue problem:"
        ' <a href="/problems/collatz_stopping_time"><code>collatz_stopping_time</code></a> — the'
        " total stopping time sigma(n) of the Collatz map, a documented open problem. No progress"
        " on the conjecture itself is claimed anywhere on this page, and nothing below says"
        " anything about termination.</p>"
    )
    body.append('<h2 id="silence">The engine-silence story</h2>')
    body.append(
        "<p>Pointed at the raw stopping times with no declared target, the engine&rsquo;s first"
        " run failed completely: the basis-synthesis and structural-repair stages blocked, and"
        " the conjecture generator proposed nothing at all — zero conjectures, zero refutations —"
        " while two elementary true statements sat unseen in the same data. The silence was the"
        " finding. It exposed two structural blind spots, both general rather than"
        " Collatz-specific: sub-domains were dense-only, so structure living in a transformed"
        " index was invisible; and no statement kind could relate a(n) to a(c&middot;n). Both"
        " were fixed as general capabilities, and the same data then yielded two exact"
        f" identities. Documented in repository docs ({_esc(idt_name)}).</p>"
    )
    body.append('<h2 id="identities">The two discovered identities</h2>')
    holdout = facts.get("halving_holdout")
    halving_confirmations = (
        f"{_fmt_int(holdout[0])} holdout confirmations" if holdout else _doc_gap("holdout count")
    )
    body.append(
        '<details class="card" open><summary>sigma(2^k) = k &nbsp;'
        '<span class="badge ctl">RESTRICTED INDEX FAMILY</span></summary><div>'
        "<p>On the reindexed sub-domain n = 2^k the stopping time is exactly k: pure halving"
        " descent. Found only after sparse geometric reindexing existed; holdout-confirmed on"
        f" held-out rows. Documented in repository docs ({_esc(idt_name)}).</p></div></details>"
    )
    body.append(
        '<details class="card" open><summary>sigma(2n) = sigma(n) + 1 &nbsp;'
        '<span class="badge ctl">INDEX SCALING RELATION</span></summary><div>'
        f"<p>The halving relation, proposed from a prefix and confirmed on {halving_confirmations}"
        " — the discovery that closed the loop from raw data to a kernel-checkable statement."
        f" Documented in repository docs ({_esc(idt_name)}).</p></div></details>"
    )
    body.append('<h2 id="sweep">The exhaustive counterexample sweep</h2>')
    if sweep.present:
        data = sweep.data
        lo = data["range"]["lo"]
        hi = data["range"]["hi"]
        exponent = _pow10_exponent(hi)
        hi_shown = f"{_fmt_int(hi)}" + (f" (10^{exponent})" if exponent is not None else "")
        tiles = [
            _tile(
                _esc(data["decision"]).replace("_", " "),
                "decision, verbatim from the receipt",
                sweep.path,
                kind="pass",
                data_key="sweep_decision",
                data_value=str(data["decision"]),
            ),
            _tile(
                f"[{_esc(_fmt_int(lo))}, {_esc(hi_shown)})",
                "half-open range swept",
                sweep.path,
                data_key="sweep_hi",
                data_value=str(hi),
            ),
            _tile(
                _esc(_fmt_int(data["counts"]["checked"])),
                "instances checked",
                sweep.path,
                data_key="sweep_checked",
                data_value=str(data["counts"]["checked"]),
            ),
            _tile(
                _esc(_fmt_int(data["undecided"]["count"])),
                "undecided (step-cap bucket — fail-closed, never pass or fail)",
                sweep.path,
                data_key="sweep_undecided",
                data_value=str(data["undecided"]["count"]),
            ),
            _tile(
                _esc(_fmt_int(data["throughput_per_second"])) + "/s",
                f"throughput on {data['device']}, {data['elapsed_seconds']} s wall",
                sweep.path,
            ),
        ]
        body.append('<div class="tiles">' + "".join(tiles) + "</div>")
        body.append(
            "<p>Statement swept, verbatim:"
            f" <code>{_esc(data['statement']['text'])}</code></p>"
        )
        body.append(
            "<p>A finite sweep proves nothing outside its range — the receipt&rsquo;s scope"
            " field says so itself:</p>"
            f"<blockquote>{_esc(data['scope'])}</blockquote>"
        )
    else:
        body.append(_missing_block(sweep, "counterexample sweep"))
    body.append('<h2 id="lean">The formal artifact, in full</h2>')
    if lean.present:
        body.append(
            "<p>The halving relation&rsquo;s honest conditional form is provable with no"
            " termination assumption: if n reaches 1 in exactly k steps, then 2n reaches 1 in"
            " exactly k + 1 steps, with reachability stated inductively. The file below is"
            f" rendered verbatim from <code>{_esc(lean.path)}</code>"
            f" ({_github(lean.path, 'view on GitHub')}).</p>"
        )
        body.append(f"<pre><code>{_esc(lean.text)}</code></pre>")
        body.append(
            "<p><strong>Claim boundary.</strong> The empirical identity presupposes both stopping"
            " times exist, which is the open part of the Collatz conjecture and is not claimed."
            " The Lean theorem proves only the conditional statement; it proves nothing about"
            " whether any n reaches 1 at all, and the case-study tests include a check that no"
            " statement the engine emits can even express a termination claim.</p>"
        )
    else:
        body.append(_missing_block(lean, "formal Lean source"))
    return _page("collatz", f"Collatz · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _artifact_summary(artifact: Artifact) -> str:
    if not artifact.present:
        return f"{MISSING_NOTE} ({artifact.missing_reason})"
    if artifact.key == "queue":
        return f"VALID — {len(artifact.data['entries'])} sealed entries"
    if artifact.key == "lean":
        return "Conditional halving relation, Std-only tactics; proves nothing about termination"
    if artifact.key in _DOC_NAMES:
        titles = {
            "goals_doc": "Release-facing goal registry: measured outcomes, including failures",
            "idt_doc": "Independent Discovery Trial: what counts as discovery, and what failed",
            "roadmap_doc": "Continuous discovery roadmap and the binding claim-discipline list",
        }
        return titles[artifact.key]
    decision = artifact.data.get("decision") if isinstance(artifact.data, dict) else None
    return str(decision) if decision else "sealed receipt"


def _evidence_page(artifacts: dict[str, Artifact], commit: str) -> bytes:
    body: list[str] = []
    body.append("<h1>Evidence: every artifact this site consumed</h1>")
    body.append(
        "<p>The generator reads exactly the artifacts below — nothing else — and every number on"
        " every page traces to one of them. JSON receipts carry their own"
        " <code>content_sha256</code> seal over the canonical body; for plain-text artifacts the"
        " hash shown is the SHA-256 of the file bytes (marked &dagger;). Links go to the same"
        " paths on GitHub.</p>"
    )
    rows = []
    for key in sorted(artifacts):
        artifact = artifacts[key]
        schema = "&mdash;"
        if artifact.present and isinstance(artifact.data, dict):
            schema = _cell(artifact.data.get("schema_version"))
        hash_cell = _sha_abbrev(artifact.sha256)
        if artifact.sha256 and not artifact.sealed:
            hash_cell += "&dagger;"
        rows.append(
            "<tr>"
            f'<td class="mono">{_github(artifact.path)}</td>'
            f'<td class="mono">{schema}</td>'
            f"<td>{_esc(_artifact_summary(artifact))}</td>"
            f'<td class="mono">{hash_cell}</td>'
            "</tr>"
        )
    body.append(
        '<div class="scroll"><table><thead><tr><th>path (links to GitHub)</th>'
        "<th>schema_version</th><th>decision / summary</th><th>content_sha256</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    body.append(
        '<p class="small sub">&dagger; file-byte hash (the artifact is not a sealed JSON'
        " receipt). Hover a hash for the full digest. Doc-sourced history numbers"
        " (blind-guessing, formal-rejection, curriculum, and control counts) are cited on each"
        " page to their document filename because they predate sealed receipts.</p>"
    )
    return _page("evidence", f"Evidence · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

_SUBMIT_JS_TEMPLATE = """
function byId(id) { return document.getElementById(id); }
var ID_RE = /^[a-z][a-z0-9_.-]{0,127}$/;
var SUB_RE = /^[a-z][a-z0-9_]*$/;
var INT_RE = /^\\d+$/;
function sortKeys(obj) {
  var out = {};
  Object.keys(obj).sort().forEach(function (key) { out[key] = obj[key]; });
  return out;
}
function showKind() {
  var kind = byId("f-kind").value;
  var blocks = document.querySelectorAll(".mf-fields");
  for (var i = 0; i < blocks.length; i += 1) {
    blocks[i].hidden = blocks[i].getAttribute("data-kind") !== kind;
  }
}
function collect() {
  var errors = [];
  var idv = byId("f-id").value.trim();
  if (!ID_RE.test(idv)) {
    errors.push("id must be lowercase kebab/snake case matching " + String(ID_RE) + ".");
  }
  var sub = byId("f-subdomain").value.trim();
  if (sub && !SUB_RE.test(sub)) {
    errors.push("subdomain must match " + String(SUB_RE) + ".");
  }
  var domain = byId("f-domain").value + (sub ? "/" + sub : "");
  var textFields = {
    believed_open_because: "f-open",
    progress_definition: "f-progress",
    source_citation: "f-citation",
    statement: "f-statement"
  };
  var texts = {};
  Object.keys(textFields).sort().forEach(function (key) {
    var value = byId(textFields[key]).value.trim();
    if (!value) { errors.push(key + " must be nonempty."); }
    if (value.length > CAPS.max_text_chars) {
      errors.push(key + " exceeds the " + CAPS.max_text_chars + "-character cap.");
    }
    texts[key] = value;
  });
  var kind = byId("f-kind").value;
  var machineForm = { kind: kind };
  Object.keys(KINDS[kind]).sort().forEach(function (name) {
    var raw = byId("f-mf-" + kind + "-" + name).value.trim();
    if (KINDS[kind][name] === "int") {
      if (!INT_RE.test(raw)) {
        errors.push("machine_form." + name +
          " must be a nonnegative integer; floats are forbidden everywhere in the queue.");
        return;
      }
      var num = parseInt(raw, 10);
      if (num > CAPS.max_machine_form_int) {
        errors.push("machine_form." + name + " exceeds the integer cap " +
          CAPS.max_machine_form_int + ".");
      }
      machineForm[name] = num;
    } else {
      if (!raw) { errors.push("machine_form." + name + " must be nonempty."); }
      machineForm[name] = raw;
    }
  });
  var entry = sortKeys({
    id: idv,
    domain: domain,
    statement: texts.statement,
    source_citation: texts.source_citation,
    believed_open_because: texts.believed_open_because,
    machine_form: sortKeys(machineForm),
    progress_definition: texts.progress_definition,
    control_rediscovery: byId("f-control").checked,
    synthetic: byId("f-synthetic").checked
  });
  return { errors: errors, entry: entry };
}
function buildEntry() {
  var result = collect();
  var list = byId("errors");
  list.textContent = "";
  if (result.errors.length) {
    result.errors.forEach(function (message) {
      var item = document.createElement("li");
      item.textContent = message;
      list.appendChild(item);
    });
    byId("result").hidden = true;
    return;
  }
  var jsonText = JSON.stringify(result.entry, null, 1);
  byId("out").value = jsonText;
  var title = "Problem-queue submission: " + result.entry.id;
  var fence = String.fromCharCode(96, 96, 96);
  var issueBody = "Proposed A2 problem-queue entry (client-validated; floats forbidden; " +
    "the seal is recomputed on acceptance):\\n\\n" + fence + "json\\n" + jsonText +
    "\\n" + fence + "\\n";
  byId("issue-link").href = ISSUE_URL + "?title=" + encodeURIComponent(title) +
    "&body=" + encodeURIComponent(issueBody);
  byId("result").hidden = false;
}
function copyOut() {
  var out = byId("out");
  out.select();
  if (navigator.clipboard) { navigator.clipboard.writeText(out.value); }
}
document.addEventListener("DOMContentLoaded", function () {
  byId("f-kind").addEventListener("change", showKind);
  byId("build").addEventListener("click", buildEntry);
  byId("copy").addEventListener("click", copyOut);
  showKind();
});
"""


def _submit_page(commit: str) -> bytes:
    kinds_json = json.dumps(
        {
            kind: {name: ("int" if field_type is int else "str") for name, field_type in fields.items()}
            for kind, fields in MACHINE_FORM_KINDS.items()
        },
        sort_keys=True,
    )
    caps_json = json.dumps(dict(sorted(SYSTEM_CAPS.items())), sort_keys=True)
    schema_keys = (
        "id",
        "domain",
        "statement",
        "source_citation",
        "believed_open_because",
        "machine_form",
        "progress_definition",
        "control_rediscovery",
        "synthetic",
    )
    body: list[str] = []
    body.append("<h1>Submit a problem for the sealed queue</h1>")
    body.append(f"<p><strong>{_esc(SUBMIT_NOTICE)}</strong></p>")
    body.append(
        f"<p>The queue schema has exactly {len(schema_keys)} keys per entry: "
        + ", ".join(f"<code>{_esc(key)}</code>" for key in schema_keys)
        + ". Floats are forbidden everywhere. &ldquo;Believed open&rdquo; must point at cited"
        " literature — or say plainly that the entry is a control or synthetic world; those two"
        " labels are schema booleans, not prose. Define progress in advance, including what a"
        " sealed negative would look like.</p>"
    )
    body.append('<div class="form-grid">')
    body.append('<label for="f-id">id</label>')
    body.append(
        '<input type="text" id="f-id" placeholder="lowercase-kebab-or_snake"'
        ' pattern="[a-z][a-z0-9_.-]{0,127}">'
    )
    body.append('<label for="f-domain">domain</label>')
    body.append(
        '<select id="f-domain"><option value="math">math</option>'
        '<option value="physics">physics</option></select>'
    )
    body.append('<label for="f-subdomain">subdomain (optional)</label>')
    body.append('<input type="text" id="f-subdomain" placeholder="number_theory">')
    body.append('<label for="f-statement">statement</label>')
    body.append('<textarea id="f-statement"></textarea>')
    body.append('<label for="f-citation">source_citation</label>')
    body.append('<textarea id="f-citation"></textarea>')
    body.append('<label for="f-open">believed_open_because</label>')
    body.append('<textarea id="f-open"></textarea>')
    body.append('<label for="f-progress">progress_definition</label>')
    body.append('<textarea id="f-progress"></textarea>')
    body.append('<label for="f-kind">machine_form.kind</label>')
    body.append(
        '<select id="f-kind">'
        + "".join(
            f'<option value="{_esc(kind)}">{_esc(kind)}</option>'
            for kind in sorted(MACHINE_FORM_KINDS)
        )
        + "</select>"
    )
    body.append('<label for="f-control">control_rediscovery</label>')
    body.append(
        '<div><input type="checkbox" id="f-control"> <span class="small sub">the answer is'
        " known; this entry calibrates the method (default false)</span></div>"
    )
    body.append('<label for="f-synthetic">synthetic</label>')
    body.append(
        '<div><input type="checkbox" id="f-synthetic"> <span class="small sub">a sealed'
        " synthetic world; openness is operational, not epistemic (default false)</span></div>"
    )
    body.append("</div>")
    for kind in sorted(MACHINE_FORM_KINDS):
        fields = MACHINE_FORM_KINDS[kind]
        inputs = []
        for name in sorted(fields):
            type_label = "integer" if fields[name] is int else "string"
            inputs.append(
                f'<label for="f-mf-{_esc(kind)}-{_esc(name)}">{_esc(name)}'
                f' <span class="sub">({type_label})</span></label>'
                f'<input type="text" id="f-mf-{_esc(kind)}-{_esc(name)}">'
            )
        body.append(
            f'<fieldset class="mf-fields" data-kind="{_esc(kind)}" hidden>'
            f"<legend>machine_form: {_esc(kind)}</legend>"
            f'<div class="form-grid">{"".join(inputs)}</div></fieldset>'
        )
    body.append(
        '<p><button type="button" id="build">Validate and build the entry</button></p>'
        '<ul id="errors"></ul>'
        '<div id="result" hidden>'
        "<h2>Entry JSON</h2>"
        '<textarea id="out" readonly rows="18"></textarea>'
        '<p><button type="button" id="copy">Copy JSON</button> '
        f'<a id="issue-link" href="{_esc(GITHUB_NEW_ISSUE)}" rel="noopener">'
        "Open a prefilled GitHub issue with this entry</a></p>"
        f"<p class=\"small sub\">{_esc(SUBMIT_NOTICE)}</p>"
        "</div>"
    )
    script = (
        "var KINDS = " + kinds_json + ";\n"
        "var CAPS = " + caps_json + ";\n"
        'var ISSUE_URL = "' + GITHUB_NEW_ISSUE + '";\n' + _SUBMIT_JS_TEMPLATE
    )
    body.append(f"<script>{script}</script>")
    return _page("submit", f"Submit · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------


def _method_page(facts: dict[str, tuple[int, ...] | None], commit: str) -> bytes:
    goals_name = _DOC_NAMES["goals_doc"]
    idt_name = _DOC_NAMES["idt_doc"]
    roadmap_name = _DOC_NAMES["roadmap_doc"]
    blind = _blind_ratio(facts)
    conditioned = _plain_ratio(facts, "conditioned")
    rejections = _plain_ratio(facts, "formal_rejections")
    curriculum = _plain_ratio(facts, "curriculum")
    controls = _plain_ratio(facts, "formal_controls")
    body: list[str] = []
    body.append("<h1>Method: the validation ladder and why it exists</h1>")
    body.append("<h2>The ladder</h2>")
    body.append(
        "<ol>"
        "<li><strong>Declared grammars, ordinal-indexed.</strong> Every candidate family is"
        " enumerable and finite; a survivor is an address in a declared space, never a free-form"
        " artifact.</li>"
        "<li><strong>GPU screen.</strong> Cheap physics- or math-informative gates over the whole"
        " family, fp32 then fp64, with thresholds recorded in the receipt config.</li>"
        "<li><strong>Exact confirmation.</strong> Screen survivors on the Pareto front are"
        " replayed in high-precision or exact integer arithmetic; screen-versus-exact"
        " crosschecks are receipted with their disagreement counts.</li>"
        "<li><strong>Sealed receipts.</strong> Results are canonical JSON, hash-bound"
        " (<code>content_sha256</code>), float-free, and carry their own scope and claims"
        " fields. Negative verdicts are sealed with the same ceremony.</li>"
        "<li><strong>Formal kernel.</strong> What admits a proof is proved in Lean; layer-one"
        " exact checking and kernel verification are never merged, and every receipt says which"
        " one it has.</li>"
        "<li><strong>Sealed observational ladder.</strong> Real data opens once, under a"
        " committed protocol, with no refit. To date zero real observations have been opened —"
        f" documented in repository docs ({_esc(goals_name)}).</li>"
        "</ol>"
    )
    body.append("<h2>The history that made the discipline necessary</h2>")
    tiles = [
        _tile(
            _esc(blind) if blind else "&mdash;",
            "blind semantic formula guessing: hidden formulas were not learnable from public"
            " rows alone — an honest failure",
            f"documented in repository docs ({idt_name}; {goals_name})",
            kind="fail",
        ),
        _tile(
            _esc(conditioned) if conditioned else "&mdash;",
            "the same worlds with public constraints: recovery belongs to the solver plus the"
            " constraints, not to independent discovery",
            f"documented in repository docs ({goals_name})",
            kind="pass",
        ),
        _tile(
            _esc(rejections) if rejections else "&mdash;",
            "one production epoch died at the formal tier with zero rotation curves computed —"
            " the GPU screen tier exists because of this receipt",
            f"documented in repository docs ({idt_name}; {roadmap_name})",
            kind="fail",
        ),
        _tile(
            _esc(curriculum) if curriculum else "&mdash;",
            "blind benchmark curriculum slots registered and sealed before one atomic opening",
            f"documented in repository docs ({goals_name})",
        ),
        _tile(
            _esc(controls) if controls else "&mdash;",
            "known-answer portable formal controls pass",
            f"documented in repository docs ({goals_name})",
        ),
    ]
    body.append('<div class="tiles">' + "".join(tiles) + "</div>")
    blind_text = _esc(blind) if blind else _doc_gap("blind-guessing")
    conditioned_text = _esc(conditioned) if conditioned else _doc_gap("conditioned-recovery")
    rejections_text = _esc(rejections) if rejections else _doc_gap("formal-rejection")
    body.append(
        f"<p>The {blind_text} versus {conditioned_text} contrast is the founding measurement:"
        " the same engine, on the same worlds, failed every blind guess and then succeeded"
        " exactly when public constraints made the answer unique. The honest conclusion is"
        " recorded in the trial document: that success belongs to the solver plus the"
        " constraints, and calling it discovery would be recall wearing a costume. Independently,"
        f" the {rejections_text} epoch showed a funnel that let candidates die at the expensive"
        " formal tier without ever being asked a cheap physics question. Both failures are kept"
        " on the books, and both dictated the architecture: cheap falsification first, exact"
        " confirmation second, formal proof third, sealed data last — with every claim carrying"
        " its boundary.</p>"
    )
    body.append("<h2>The claim discipline (binding on every task)</h2>")
    body.append(
        "<ul>" + "".join(f"<li>{_esc(line)}</li>" for line in CLAIM_DISCIPLINE) + "</ul>"
    )
    body.append(
        f'<p class="small sub">Quoted from the roadmap&rsquo;s binding list; documented in'
        f" repository docs ({_esc(roadmap_name)}).</p>"
    )
    body.append("<h2>The documents</h2>")
    body.append(
        "<ul>"
        f"<li>{_github(ARTIFACT_PATHS['goals_doc'])} — the goal registry: every measured outcome,"
        " including the blocked and failed ones.</li>"
        f"<li>{_github(ARTIFACT_PATHS['idt_doc'])} — what would count as independent discovery,"
        " and why nothing yet does.</li>"
        f"<li>{_github(ARTIFACT_PATHS['roadmap_doc'])} — the continuous-discovery roadmap and"
        " the claim-discipline list.</li>"
        "</ul>"
    )
    return _page("method", f"Method · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Assembly, CLI
# ---------------------------------------------------------------------------


def render_site(root: Path | str, commit: str) -> dict[str, bytes]:
    """Render every page as bytes, purely from artifacts under ``root`` and ``commit``."""

    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise SiteGenerationError("commit must be a 40-character lowercase hex sha")
    artifacts = _load_artifacts(Path(root))
    facts = _doc_facts(artifacts)
    pages: dict[str, bytes] = {
        "index.html": _index_page(artifacts, facts, commit),
        "problems.html": _problems_index_page(artifacts["queue"], commit),
        "gravity.html": _gravity_page(artifacts, commit),
        "collatz.html": _collatz_page(artifacts, facts, commit),
        "evidence.html": _evidence_page(artifacts, commit),
        "submit.html": _submit_page(commit),
        "method.html": _method_page(facts, commit),
    }
    queue = artifacts["queue"]
    if queue.present:
        for entry in queue.data["entries"]:
            pages[f"problems/{entry['id']}.html"] = _problem_detail_page(entry, queue, commit)
    return pages


def _write_pages(output: Path, pages: dict[str, bytes]) -> None:
    if output.exists():
        shutil.rmtree(output)
    for rel_path in sorted(pages):
        target = output / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pages[rel_path])


def build_site(root: Path | str, output_dir: Path | str, commit: str) -> dict[str, bytes]:
    """Render and write the site, replacing ``output_dir`` entirely.  Returns the pages."""

    pages = render_site(root, commit)
    _write_pages(Path(output_dir), pages)
    return pages


def _existing_pages(output: Path) -> dict[str, bytes]:
    if not output.is_dir():
        return {}
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(output.rglob("*.html"))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Receipts-driven static site generator for the public Invariant site."
    )
    parser.add_argument("--root", default=".", help="repository root to read artifacts from")
    parser.add_argument("--output", default="public", help="directory to write the site into")
    parser.add_argument(
        "--commit",
        required=True,
        help="content-snapshot sha for the footer (pass `git rev-parse HEAD`; never invoked here)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="re-render and byte-compare against --output instead of writing",
    )
    args = parser.parse_args(argv)
    try:
        rendered = render_site(Path(args.root), args.commit)
    except SiteGenerationError as error:
        print(f"ERROR: {error}")
        return 2
    output = Path(args.output)
    if args.validate:
        existing = _existing_pages(output)
        changed = {
            path
            for path in set(existing) & set(rendered)
            if existing[path] != rendered[path]
        }
        mismatched = sorted((set(existing) ^ set(rendered)) | changed)
        if mismatched:
            for path in mismatched:
                print(f"MISMATCH {path}")
            return 1
        print(f"VALIDATED pages={len(rendered)} output={output.as_posix()}")
        return 0
    _write_pages(output, rendered)
    print(f"WROTE pages={len(rendered)} output={output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
