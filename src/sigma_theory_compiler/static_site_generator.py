"""Receipts-driven static site generator for the public Invariant site.

The site is a rendering of the repository's sealed evidence, never a parallel set of
claims, presented as a grid-paper journal an educated non-specialist can read.  Four
rules keep it honest.

**Every quantitative claim is read from a repo artifact at build time.**  Counts,
margins, throughputs, formulas, and decisions come from the sealed receipts under
``runs/``, the sealed problem queue under ``configs/``, and the measured-outcome
documents under ``docs/``.  This module contains none of those numerals as source
literals; delete a receipt and its numbers disappear from the site rather than being
remembered.

**Every problem page and case study opens with a status banner** drawn from a fixed
five-word vocabulary (OPEN / REDISCOVERED / RANGE-VERIFIED / PROVED (CONDITIONAL) /
SEALED NEGATIVE), defined on every page that uses it, so a reader can always tell
whether the famous question was settled (it was not) and what, exactly, the machine
established.  Banners are derived from queue flags and receipt presence, never typed
free-hand per page.

**No scalar score of any kind is rendered.**  A result either passes an exact gate,
fails it, or is explicitly undecided.  Failures get the same billing as successes:
the sealed cluster negative, the blind-guessing failure, and the formal-rejection
history are headline content, not footnotes.

**Missing evidence is declared, never papered over.**  An absent optional artifact
produces an explicit "Evidence not yet published" block in place of its section; the
build never crashes on absence and never silently omits the section.

Mathematics is typeset properly: receipt formula strings are translated by a small
deterministic ASCII-to-LaTeX translator and rendered to MathML at build time via
``latex2mathml`` — no client-side JavaScript for math, no external requests.  A
formula outside the receipts' grammar, or an unusable converter, fails the build
loudly; there is no silent ASCII fallback.

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

import latex2mathml.converter as _latex2mathml

from .problem_queue import MACHINE_FORM_KINDS, SYSTEM_CAPS, ProblemQueueError, load_queue

SITE_TITLE = "Invariant — a pattern-hunting machine and its public logbook"

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

#: The homepage opens with this passage, verbatim (tests pin it).
GARDNER_OPENING = (
    "Pick a number, any positive whole number. If it is even, cut it in half. If it is odd,"
    " triple it and add one. Now repeat, and keep repeating. Start with 7 and you get 22, 11,"
    " 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1. Every number anyone has ever tried"
    " comes tumbling down to 1 eventually — and yet, after nearly a century, nobody can prove"
    " that every number must. That is the kind of question this site is about: easy to state,"
    " maddening to settle. What lives here is the logbook of a machine built to work on such"
    " questions the only way a machine honestly can — by guessing patterns, trying its hardest"
    " to break them, and filing paperwork for every attempt, the failures included."
)

#: The Collatz case-study banner body, verbatim after "STATUS: OPEN." (tests pin it).
COLLATZ_STATUS_TEXT = (
    "The Collatz conjecture is unsolved — by us and by everyone else since 1937. What follows"
    " is what our machine established about it, which is far short of a solution. The two"
    " identities below are well known to mathematicians; the result is that a machine found"
    " them unaided."
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
    "dozen": "runs/math/solved-dozen/campaign.json",
    "goals_doc": "docs/GOALS_AND_MEASURED_OUTCOMES.md",
    "idt_doc": "docs/INDEPENDENT_DISCOVERY_TRIAL.md",
    "roadmap_doc": "docs/CONTINUOUS_DISCOVERY_ROADMAP.md",
}

#: The unsolved-dozen campaign is expected but may not exist yet; it is loaded
#: fail-soft under its own key and is deliberately not in ARTIFACT_PATHS so that
#: fixtures which copy the artifact set do not require a file that may be absent.
UNSOLVED_CAMPAIGN_PATH = "runs/math/unsolved-dozen/campaign.json"

#: Prefix for the dynamically loaded per-world rediscovery receipts.
DOZEN_WORLD_PREFIX = "dozen_world:"

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
    ("index", "/", "Home"),
    ("paper", "/paper", "The write-up"),
    ("problems", "/problems", "Problems"),
    ("papers", "/papers", "Papers"),
    ("gravity", "/gravity", "Gravity"),
    ("collatz", "/collatz", "Collatz"),
    ("evidence", "/evidence", "Evidence"),
    ("method", "/method", "Method"),
    ("submit", "/submit", "Submit"),
)

#: Required, verbatim, in every rediscovery paper's abstract (tests pin it).
REDISCOVERY_SENTENCE = (
    "This is a rediscovery of a classical result under a sealed blind protocol; "
    "no novelty is claimed."
)

#: Caption under every anonymized-rows table (tests pin it).
ROWS_CAPTION = "The machine saw only this."

#: Second sentence of every elimination-funnel caption (tests pin it).
FUNNEL_FAMILY_NOTE = (
    "Within each family, exact linear algebra jumps to the single best member in one step "
    "— the funnel counts families examined, not formulas tried."
)

#: Every funnel's first bar carries this phrase after its count (tests pin it);
#: the first bar never counts candidates, because a family is not a candidate.
DECLARED_FAMILIES_PHRASE = "declared families (each an infinite space of formulas)"

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

#: The five status words.  (word, css class suffix, definition) — the definition is
#: printed wherever the word is stamped, so the vocabulary travels with its use.
STATUS_DEFINITIONS = (
    (
        "OPEN",
        "open",
        "the famous question is unsolved, by us and by everyone.",
    ),
    (
        "REDISCOVERED",
        "rediscovered",
        ("the engine independently found facts already known to mathematics"
        " (the news is the machine, not the math)."),
    ),
    (
        "RANGE-VERIFIED",
        "range",
        "checked exhaustively up to an exact bound; says nothing beyond the bound.",
    ),
    (
        "PROVED (CONDITIONAL)",
        "proved",
        "machine-checked proof of a limited statement, stated exactly.",
    ),
    (
        "SEALED NEGATIVE",
        "negative",
        "an exhaustive search that found nothing, published with margins.",
    ),
)

_STATUS_CSS = {word: css for word, css, _definition in STATUS_DEFINITIONS}

#: Standing formulas, rendered to MathML at build time wherever they appear.
LATEX_RAR_LAW = r"g_{\mathrm{obs}} = \nu\!\left(g_{\mathrm{bar}}/a_{0}\right)\,g_{\mathrm{bar}}"
LATEX_BTFR = r"v_{\mathrm{flat}}^{4} \propto M\,a_{0}"
LATEX_SIGMA_HALVING = r"\sigma(2n) = \sigma(n) + 1"
LATEX_SIGMA_POW2 = r"\sigma(2^{k}) = k"
LATEX_NU_FAMILY = r"\nu(y) = \left[\frac{P(u)}{Q(u)}\right]^{\beta}, \quad u = y^{-1/2}"

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; color: #2a2418; font-size: 17px; line-height: 1.65;
  font-family: "Courier Prime","American Typewriter","Courier New",Courier,monospace;
  background-color: #fbfaf6;
  background-image:
    repeating-linear-gradient(0deg, rgba(148,160,178,0.42) 0, rgba(148,160,178,0.42) 1px,
      transparent 1px, transparent 95px),
    repeating-linear-gradient(90deg, rgba(148,160,178,0.42) 0, rgba(148,160,178,0.42) 1px,
      transparent 1px, transparent 95px),
    repeating-linear-gradient(0deg, rgba(215,219,226,0.5) 0, rgba(215,219,226,0.5) 1px,
      transparent 1px, transparent 19px),
    repeating-linear-gradient(90deg, rgba(215,219,226,0.5) 0, rgba(215,219,226,0.5) 1px,
      transparent 1px, transparent 19px); }
a { color: #27506d; text-decoration: underline; text-underline-offset: 2px; }
a:hover { color: #8b3a2e; }
code, pre { font-family: inherit; }
code { font-size: 0.92em; background: rgba(42,36,24,0.06); padding: 0 0.14em; }
.sheet { max-width: 76ch; margin: 2.1rem auto 2.6rem; background: #fffefa;
  border: 1px solid #d9d3c4; padding: 2.2rem clamp(1.1rem, 4.5vw, 3.1rem) 1.9rem;
  box-shadow: 0 1px 3px rgba(64,58,42,0.18), 0 12px 30px rgba(64,58,42,0.12); }
header { text-align: center; border-bottom: 3px double #2a2418; padding-bottom: 0.9rem; }
header .brand a { font-size: 1.5rem; font-weight: 700; letter-spacing: 0.42em;
  text-transform: uppercase; color: inherit; text-decoration: none; padding-left: 0.42em; }
header .tag { margin: 0.35rem auto 0; max-width: 56ch; font-size: 0.76rem;
  font-style: italic; color: #6b6353; }
nav { margin-top: 0.7rem; font-size: 0.72rem; letter-spacing: 0.16em;
  text-transform: uppercase; }
nav a { color: #6b6353; margin: 0 0.5rem; text-decoration: none; white-space: nowrap; }
nav a.on { color: #2a2418; border-bottom: 2px solid #8b3a2e; }
nav a:hover { color: #2a2418; }
main { margin-top: 1.7rem; }
h1 { font-size: 1.42rem; line-height: 1.32; margin: 0.3rem 0 1rem; }
h2 { font-size: 0.94rem; letter-spacing: 0.18em; text-transform: uppercase;
  border-bottom: 3px double #2a2418; padding-bottom: 0.3rem; margin: 2.3rem 0 0.9rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 0.5rem; }
p { margin: 0.55rem 0; }
.kicker { font-size: 0.72rem; letter-spacing: 0.24em; text-transform: uppercase;
  color: #8b3a2e; margin: 0 0 0.1rem; }
.lede { font-size: 1.01rem; }
.byline { letter-spacing: 0.1em; }
.sub { color: #6b6353; }
.small { font-size: 0.79rem; }
.status-banner { border: 2px solid #2a2418; padding: 0.8rem 1rem; margin: 1.3rem 0 0.9rem;
  transform: rotate(-0.5deg); font-size: 0.9rem; background: rgba(255,255,255,0.6);
  box-shadow: 2px 2px 0 rgba(42,36,24,0.08); }
.status-banner .stamp { font-weight: 700; letter-spacing: 0.13em; }
.status-open { border-color: #8b3a2e; }
.status-open .stamp { color: #8b3a2e; }
.status-rediscovered { border-color: #31577e; }
.status-rediscovered .stamp { color: #31577e; }
.status-range, .status-proved { border-color: #2e6b3a; }
.status-range .stamp, .status-proved .stamp { color: #2e6b3a; }
.status-negative { border-color: #44505e; }
.status-negative .stamp { color: #44505e; }
.status-key { border: 1px dashed #a99f8a; padding: 0.55rem 0.85rem 0.6rem;
  margin: 0 0 1.5rem; font-size: 0.74rem; color: #57503f; }
.status-key p { margin: 0 0 0.25rem; letter-spacing: 0.12em; text-transform: uppercase; }
.status-key dl { margin: 0; }
.status-key dt { font-weight: 700; float: left; clear: left; margin-right: 0.45rem; }
.status-key dd { margin: 0 0 0.12rem 0; overflow: hidden; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 0.7rem; margin: 1rem 0; }
.tile { background: #fdfcf6; border: 1px solid #cfc8b6; padding: 0.55rem 0.7rem;
  box-shadow: 1px 2px 0 rgba(64,58,42,0.08); }
.tile .v { font-size: 1.22rem; font-weight: 700; line-height: 1.3; }
.tile .k { font-size: 0.76rem; color: #57503f; margin-top: 0.15rem; }
.tile .r { font-size: 0.64rem; color: #8a8171; margin-top: 0.35rem; word-break: break-all; }
.tile.neg .v { color: #7a5b21; }
.tile.fail .v { color: #8b3a2e; }
.tile.pass .v { color: #2e6b3a; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.76rem; margin: 0.8rem 0; }
th, td { border: 1px solid #cfc8b6; padding: 0.28rem 0.5rem; text-align: left;
  vertical-align: top; }
th { background: #f6f3ea; border-bottom: 3px double #2a2418; font-weight: 700;
  letter-spacing: 0.05em; }
td.num, th.num { text-align: right; white-space: nowrap; }
td.mono { white-space: nowrap; }
pre { background: #f6f3ea; border: 1px solid #cfc8b6; padding: 0.75rem 0.9rem;
  overflow-x: auto; font-size: 0.72rem; line-height: 1.5; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #a99f8a; margin: 0.7rem 0; padding: 0.15rem 0.9rem;
  color: #57503f; font-size: 0.86rem; font-style: italic; }
.missing { border: 1px dashed #8a6d1f; background: rgba(158,120,32,0.07);
  padding: 0.65rem 0.9rem; color: #7a5b21; margin: 0.8rem 0; font-size: 0.85rem; }
.neg-block { border: 2px solid #44505e; background: rgba(68,80,94,0.05);
  padding: 0.8rem 1rem; margin: 0.9rem 0; }
.badge { display: inline-block; border: 1px solid; font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.08em; padding: 0.05rem 0.4rem; margin-left: 0.4rem;
  vertical-align: middle; }
.badge.pass { color: #2e6b3a; border-color: #2e6b3a; }
.badge.fail { color: #8b3a2e; border-color: #8b3a2e; }
.badge.neg { color: #7a5b21; border-color: #7a5b21; }
.badge.ctl { color: #31577e; border-color: #31577e; }
details.card { background: #fdfcf6; border: 1px solid #cfc8b6; margin: 0.5rem 0; }
details.card > summary { cursor: pointer; padding: 0.5rem 0.8rem; font-size: 0.8rem; }
details.card > div { padding: 0.15rem 0.9rem 0.7rem; border-top: 1px solid #e2dccc; }
details.methods { background: #f9f7f0; border: 1px solid #cfc8b6; margin: 0.6rem 0 1rem; }
details.methods > summary { cursor: pointer; padding: 0.6rem 0.9rem; font-size: 0.74rem;
  letter-spacing: 0.14em; text-transform: uppercase; }
details.methods > div { padding: 0.15rem 1rem 0.9rem; border-top: 1px solid #e2dccc; }
ul, ol { margin: 0.5rem 0 0.5rem 1.3rem; padding: 0; }
li { margin: 0.28rem 0; }
.math math { font-size: 1.08em; }
.math-block { display: block; margin: 0.75rem 0 0.75rem 1.4rem; }
.receipt-form { display: block; font-size: 0.66rem; color: #8a8171; background: none;
  padding: 0; margin-top: 0.2rem; }
.form-grid { display: grid; grid-template-columns: 13rem 1fr; gap: 0.5rem 0.8rem;
  align-items: start; margin: 0.8rem 0; }
.form-grid label { color: #57503f; font-size: 0.8rem; padding-top: 0.3rem; }
input[type="text"], select, textarea { width: 100%; background: #fffefa; color: #2a2418;
  border: 1px solid #a99f8a; padding: 0.35rem 0.5rem; font-size: 0.8rem;
  font-family: inherit; }
textarea { min-height: 4.4rem; resize: vertical; }
button { background: #f6f3ea; color: #2a2418; border: 1px solid #6b6353;
  padding: 0.4rem 0.9rem; font-size: 0.82rem; font-family: inherit; cursor: pointer;
  box-shadow: 1px 2px 0 rgba(64,58,42,0.15); }
button:hover { background: #efeadb; }
#errors { color: #8b3a2e; font-size: 0.82rem; }
fieldset { border: 1px solid #cfc8b6; margin: 0.6rem 0; padding: 0.5rem 0.9rem 0.8rem; }
legend { color: #57503f; font-size: 0.76rem; padding: 0 0.4rem; letter-spacing: 0.1em; }
footer { border-top: 3px double #2a2418; margin-top: 2.6rem; padding-top: 0.9rem;
  text-align: center; font-size: 0.7rem; color: #6b6353; }
footer .orn { letter-spacing: 0.7em; margin: 0 0 0.4rem; }
footer p { max-width: 62ch; margin: 0.3rem auto; }
:root { --funnel-bar: rgba(49,87,126,0.30); --funnel-hit: rgba(46,107,58,0.38);
  --funnel-edge: #31577e; --funnel-hit-edge: #2e6b3a; }
figure.funnel-figure { margin: 1.1rem 0 1.3rem; padding: 0; }
figure.funnel-figure svg { width: 100%; height: auto; display: block; }
svg.funnel text { font-family: inherit; font-size: 12px; fill: currentColor; }
svg.funnel text.num { font-weight: 700; }
svg.funnel text.lbl { fill-opacity: 0.82; }
svg.funnel rect.bar { fill: var(--funnel-bar); stroke: var(--funnel-edge);
  stroke-width: 1; }
svg.funnel rect.bar.hit { fill: var(--funnel-hit); stroke: var(--funnel-hit-edge); }
svg.funnel line.axis { stroke: currentColor; stroke-opacity: 0.5; stroke-width: 1; }
svg.funnel line.rule { stroke: currentColor; stroke-opacity: 0.16;
  stroke-dasharray: 2 3; }
figure.funnel-figure figcaption { font-size: 0.74rem; color: #57503f;
  margin-top: 0.35rem; }
.duo { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.7rem; margin: 0.9rem 0; }
.duo .half { background: #fdfcf6; border: 1px solid #cfc8b6; padding: 0.6rem 0.8rem;
  box-shadow: 1px 2px 0 rgba(64,58,42,0.08); }
.duo .half .hd { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: #57503f; margin-bottom: 0.35rem; }
.aside-box { border: 2px solid #31577e; background: rgba(49,87,126,0.06);
  padding: 0.75rem 1rem; margin: 1rem 0; }
.aside-box .hd { font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  font-size: 0.78rem; color: #31577e; margin-bottom: 0.3rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 0.7rem; margin: 1rem 0; }
.cards a.card-link { display: block; background: #fdfcf6; border: 1px solid #cfc8b6;
  padding: 0.55rem 0.75rem; text-decoration: none; color: inherit;
  box-shadow: 1px 2px 0 rgba(64,58,42,0.08); }
.cards a.card-link:hover { border-color: #8b3a2e; color: inherit; }
.cards a.card-link .t { font-weight: 700; }
.cards a.card-link .d { font-size: 0.74rem; color: #57503f; margin-top: 0.2rem; }
.cards a.card-link .f { font-size: 0.68rem; color: #8a8171; margin-top: 0.3rem;
  word-break: break-all; }
@media (max-width: 640px) {
  .sheet { margin: 0 auto; border-left: 0; border-right: 0; }
}
@media (prefers-color-scheme: dark) {
  :root { color-scheme: dark; }
  body { color: #d8d5c8; background-color: #1e1f22;
    background-image:
      repeating-linear-gradient(0deg, rgba(150,160,178,0.11) 0, rgba(150,160,178,0.11) 1px,
        transparent 1px, transparent 95px),
      repeating-linear-gradient(90deg, rgba(150,160,178,0.11) 0, rgba(150,160,178,0.11) 1px,
        transparent 1px, transparent 95px),
      repeating-linear-gradient(0deg, rgba(150,160,178,0.055) 0, rgba(150,160,178,0.055) 1px,
        transparent 1px, transparent 19px),
      repeating-linear-gradient(90deg, rgba(150,160,178,0.055) 0, rgba(150,160,178,0.055) 1px,
        transparent 1px, transparent 19px); }
  a { color: #9db8d6; }
  a:hover { color: #d99a82; }
  code { background: rgba(216,213,200,0.09); }
  .sheet { background: #26272b; border-color: #3a3b41;
    box-shadow: 0 1px 3px rgba(0,0,0,0.55), 0 12px 30px rgba(0,0,0,0.35); }
  header, h2, footer { border-color: #d8d5c8; }
  header .tag, nav a, .sub, .status-key, footer { color: #a49f90; }
  nav a.on { color: #d8d5c8; border-bottom-color: #d99a82; }
  nav a:hover { color: #d8d5c8; }
  .kicker { color: #d99a82; }
  .status-banner { background: rgba(0,0,0,0.25); box-shadow: 2px 2px 0 rgba(0,0,0,0.35); }
  .status-open { border-color: #d99a82; }
  .status-open .stamp { color: #d99a82; }
  .status-rediscovered { border-color: #8fb3d9; }
  .status-rediscovered .stamp { color: #8fb3d9; }
  .status-range, .status-proved { border-color: #86c493; }
  .status-range .stamp, .status-proved .stamp { color: #86c493; }
  .status-negative { border-color: #a7b4c2; }
  .status-negative .stamp { color: #a7b4c2; }
  .status-key { border-color: #57503f; }
  .tile, details.card, details.methods, pre { background: #2b2c31; border-color: #3f4046; }
  .tile .k { color: #a49f90; }
  .tile .r { color: #7b776b; }
  .tile.neg .v { color: #cfa658; }
  .tile.fail .v { color: #d99a82; }
  .tile.pass .v { color: #86c493; }
  th { background: #2c2d33; border-color: #3f4046; border-bottom-color: #d8d5c8; }
  td { border-color: #3f4046; }
  details.card > div, details.methods > div { border-top-color: #3f4046; }
  blockquote { border-left-color: #57503f; color: #a49f90; }
  .missing { border-color: #8a6d1f; background: rgba(207,166,88,0.08); color: #cfa658; }
  .neg-block { border-color: #a7b4c2; background: rgba(167,180,194,0.06); }
  .badge.pass { color: #86c493; border-color: #86c493; }
  .badge.fail { color: #d99a82; border-color: #d99a82; }
  .badge.neg { color: #cfa658; border-color: #cfa658; }
  .badge.ctl { color: #8fb3d9; border-color: #8fb3d9; }
  input[type="text"], select, textarea { background: #1e1f22; color: #d8d5c8;
    border-color: #57503f; }
  button { background: #2b2c31; color: #d8d5c8; border-color: #57503f;
    box-shadow: 1px 2px 0 rgba(0,0,0,0.4); }
  button:hover { background: #33343a; }
  #errors { color: #d99a82; }
  .receipt-form { color: #7b776b; }
  fieldset { border-color: #3f4046; }
  legend { color: #a49f90; }
  :root { --funnel-bar: rgba(143,179,217,0.28); --funnel-hit: rgba(134,196,147,0.30);
    --funnel-edge: #8fb3d9; --funnel-hit-edge: #86c493; }
  figure.funnel-figure figcaption { color: #a49f90; }
  .duo .half { background: #2b2c31; border-color: #3f4046; }
  .duo .half .hd { color: #a49f90; }
  .aside-box { border-color: #8fb3d9; background: rgba(143,179,217,0.07); }
  .aside-box .hd { color: #8fb3d9; }
  .cards a.card-link { background: #2b2c31; border-color: #3f4046; }
  .cards a.card-link:hover { border-color: #d99a82; }
  .cards a.card-link .d { color: #a49f90; }
  .cards a.card-link .f { color: #7b776b; }
}
"""


class SiteGenerationError(ValueError):
    """Raised for caller errors (bad commit, bad formula grammar, broken math renderer)."""


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
                return Artifact(
                    key, rel_path, None, None, None, False, "present but fails seal validation"
                )
        else:
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return Artifact(
                    key, rel_path, None, None, None, False, "present but is not valid JSON"
                )
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
    artifacts = {key: _load_artifact(root, key, rel) for key, rel in ARTIFACT_PATHS.items()}
    artifacts["unsolved"] = _load_artifact(root, "unsolved", UNSOLVED_CAMPAIGN_PATH)
    dozen = artifacts["dozen"]
    if dozen.present:
        for world in dozen.data.get("world_results", []):
            key = DOZEN_WORLD_PREFIX + str(world.get("classical_id", ""))
            artifact = _load_artifact(root, key, str(world.get("world_receipt_path", "")))
            expected_seal = world.get("world_receipt_sha256")
            if (
                artifact.present
                and isinstance(expected_seal, str)
                and artifact.sha256 != expected_seal
            ):
                artifact = Artifact(
                    key,
                    artifact.path,
                    None,
                    None,
                    None,
                    False,
                    "present but does not match the campaign receipt's recorded seal",
                )
            artifacts[key] = artifact
    return artifacts


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
    "prior_art_records": ("goals_doc", r"Prior-art audit: (\d+) records"),
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


def _plain_count(facts: dict[str, tuple[int, ...] | None], fact: str) -> int | None:
    values = facts.get(fact)
    if values is None:
        return None
    return values[0]


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


# ---------------------------------------------------------------------------
# Mathematics: deterministic ASCII -> LaTeX -> MathML, all at build time
# ---------------------------------------------------------------------------


def latex_to_mathml(latex: str, display: str = "inline") -> str:
    """Convert a LaTeX string to MathML markup, failing loudly on any breakage.

    There is deliberately no ASCII fallback: a page with broken math must never
    build, because a silently degraded formula is a misstated formula.
    """

    try:
        rendered = _latex2mathml.convert(latex, display=display)
    except Exception as error:
        raise SiteGenerationError(f"latex2mathml failed on {latex!r}: {error}") from error
    if "<math" not in rendered:
        raise SiteGenerationError(f"latex2mathml produced no <math> element for {latex!r}")
    return rendered


#: Shape of every candidate-law formula string sealed in the gravity receipts:
#: ``nu(y) = [(P) / (Q)]^E,  u = y^(-1/2)`` with integer-coefficient polynomials.
_FORMULA_SHAPE = re.compile(
    r"^nu\(y\) = \[\((?P<num>[0-9u^ +\-]+)\) / \((?P<den>[0-9u^ +\-]+)\)\]"
    r"\^(?P<exp>\d+(?:/\d+)?),  u = y\^\(-1/2\)$"
)

_FORMULA_TERM = re.compile(r"^(?:\d+|\d*u(?:\^\d+)?)$")


def _polynomial_to_latex(polynomial: str) -> str:
    pieces = polynomial.split(" ")
    if len(pieces) % 2 == 0:
        raise SiteGenerationError(f"polynomial outside the receipt grammar: {polynomial!r}")
    rendered: list[str] = []
    for index, piece in enumerate(pieces):
        if index % 2:
            if piece not in {"+", "-"}:
                raise SiteGenerationError(f"polynomial outside the receipt grammar: {polynomial!r}")
            rendered.append(piece)
        else:
            if _FORMULA_TERM.fullmatch(piece) is None:
                raise SiteGenerationError(f"polynomial outside the receipt grammar: {polynomial!r}")
            rendered.append(re.sub(r"\^(\d+)", r"^{\1}", piece))
    return " ".join(rendered)


def formula_ascii_to_latex(formula: str) -> str:
    """Translate a receipt's ASCII formula string into LaTeX, deterministically.

    Anything outside the sealed grammar raises: the translator would rather kill
    the build than render a formula it does not fully understand.
    """

    match = _FORMULA_SHAPE.fullmatch(formula)
    if match is None:
        raise SiteGenerationError(f"formula outside the receipt grammar: {formula!r}")
    numerator = _polynomial_to_latex(match["num"])
    denominator = _polynomial_to_latex(match["den"])
    return (
        "\\nu(y) = \\left[\\frac{" + numerator + "}{" + denominator + "}\\right]^{"
        + match["exp"] + "}, \\quad u = y^{-1/2}"
    )


def _math(latex: str) -> str:
    return f'<span class="math">{latex_to_mathml(latex)}</span>'


def _math_block(latex: str) -> str:
    return f'<span class="math math-block">{latex_to_mathml(latex, display="block")}</span>'


def _formula_html(formula: str) -> str:
    """A receipt formula as proper mathematics, with the receipt string underneath."""

    rendered = latex_to_mathml(formula_ascii_to_latex(formula))
    return (
        f'<span class="math">{rendered}</span>'
        f'<code class="receipt-form">{_esc(formula)}</code>'
    )


# ---------------------------------------------------------------------------
# The status system: five words, defined wherever they are stamped
# ---------------------------------------------------------------------------


def _status_key() -> str:
    items = "".join(
        f"<dt>{_esc(word)}</dt><dd>{_esc(definition)}</dd>"
        for word, _css, definition in STATUS_DEFINITIONS
    )
    return (
        '<div class="status-key"><p>The five status words used on this site</p>'
        f"<dl>{items}</dl></div>"
    )


def _status_banner(word: str, text: str) -> str:
    css = _STATUS_CSS[word]
    return (
        f'<div class="status-banner status-{css}">'
        f'<span class="stamp">STATUS: {_esc(word)}.</span> {_esc(text)}</div>'
    )


def _problem_status(entry: dict[str, Any], artifacts: dict[str, Artifact]) -> tuple[str, str]:
    """Derive (status word, banner text) for a queue entry from flags and receipts.

    The derivation is honest by construction: a receipt is referenced only when it
    is actually present, and calibration entries always say what they are.
    """

    entry_id = entry["id"]
    lensing = artifacts["lensing"]
    billion = artifacts["billion"]
    if entry_id == "collatz_stopping_time":
        return (
            "OPEN",
            ("The Collatz conjecture is unsolved — by us and by everyone else since 1937. The"
            " machine's work on it — two rediscovered identities, one bounded exhaustive check,"
            " one machine-checked conditional statement — falls far short of a solution, and the"
            " identities themselves are well known to mathematicians. The full account is on the"
            " Collatz case-study page."),
        )
    if entry_id == "erdos_straus":
        return (
            "OPEN",
            ("The Erdős–Straus conjecture is unsolved — by us and by everyone. No sealed receipt"
            " in this repository records progress on it yet; this entry declares the target, the"
            " source, and what progress would have to look like before any work is claimed."),
        )
    if entry_id == "aliquot_276":
        return (
            "OPEN",
            ("Whether the aliquot sequence starting at 276 stays bounded is unknown — to us and to"
            " everyone. No sealed receipt in this repository records progress on it yet; the"
            " entry exists so that any future claim has a declared target to answer to."),
        )
    if entry_id == "prime_gap_polynomial":
        return (
            "OPEN",
            ("No closed form for prime gaps is known — to us or to anyone — and the cited"
            " literature expects refutations, not formulas. No sealed receipt in this repository"
            " records progress here yet; a machine claiming a global closed form would be treated"
            " as defective until externally verified."),
        )
    if entry_id == "baryonic_rotation_law":
        if billion.present:
            return (
                "OPEN",
                ("Whether one baryons-only law can generate flat rotation curves and the"
                " Tully–Fisher relation is unresolved — by us and by everyone. What our machine"
                " did is narrower: it screened one declared family of candidate laws against"
                " synthetic controls. Survivors are search priorities, not validated theories,"
                " and no telescope data has ever been opened. Counts and margins are on the"
                " Gravity page."),
            )
        return (
            "OPEN",
            ("Whether one baryons-only law can generate flat rotation curves and the Tully–Fisher"
            " relation is unresolved — by us and by everyone. The screen receipt this entry"
            " points to is not yet published in this repository."),
        )
    if entry_id == "lensing_dynamics_consistency":
        if lensing.present:
            return (
                "OPEN",
                ("Whether a single law can match both galaxy dynamics and gravitational lensing"
                " is unresolved — by us and by everyone. Within our one declared family, many"
                " candidates pass the synthetic lensing gate alone, but the joint verdict with"
                " the cluster gate is a sealed negative: nothing passed both. Margins are on the"
                " Gravity page; no observational data has been opened."),
            )
        return (
            "OPEN",
            ("Whether a single law can match both galaxy dynamics and gravitational lensing is"
            " unresolved — by us and by everyone. The campaign receipt this entry points to is"
            " not yet published in this repository."),
        )
    if entry_id == "cluster_missing_mass":
        if lensing.present:
            return (
                "SEALED NEGATIVE",
                ("The cluster missing-mass problem itself remains open — nobody has a baryons-only"
                " answer. What our machine established is exact and negative: in the one declared"
                " family it searched end to end, no candidate law carried the synthetic cluster"
                " test, and that zero is published with exact margins on the Gravity page. A zero"
                " for one family says nothing about laws outside it — and this is the outcome the"
                " entry predicted in advance."),
            )
        return (
            "OPEN",
            ("The cluster missing-mass problem is open — for us and for everyone. The expected"
            " outcome recorded in this entry is a sealed negative, and its receipt is not yet"
            " published in this repository."),
        )
    if entry_id == "quantified_inequality_families":
        return (
            "OPEN",
            ("Open only relative to this system's own proof coverage. This is an internal"
            " engineering target — proving machine-generated inequality families in a proof"
            " kernel — not a named question from the literature, and nothing here bears on"
            " mathematics at large. Receipts will count families proved and families still open."),
        )
    if entry["control_rediscovery"]:
        return (
            "REDISCOVERED",
            ("Nothing here is unsolved: the cited literature settled this question long ago. The"
            " entry is a labeled calibration control — the machine's job is to find the known"
            " answer unaided, and if it does, the news is the machine, not the math. No"
            " rediscovery receipt for this entry has been published yet."),
        )
    if entry["synthetic"]:
        return (
            "OPEN",
            ("Open only in the operational sense, and only for the machine. This is a"
            " manufactured world: the generating rule exists and sits sealed away from the"
            " discovery engine, so nothing here is unsolved mathematics. Recovering the rule —"
            " or failing to — is evidence about the machine alone, and no unseal receipt has"
            " been published yet."),
        )
    return (
        "OPEN",
        ("This question is recorded as open in the cited literature — unsolved by us and by"
        " everyone. No sealed receipt in this repository records progress on it yet."),
    )


def _gravity_status(lensing: Artifact) -> tuple[str, str]:
    if lensing.present:
        return (
            "SEALED NEGATIVE",
            ("The missing-mass problem — galaxies spin and lens as if they hold far more matter"
            " than we can see — is unsolved, by us and by everyone. What follows is what our"
            " machine established, which is much narrower: one declared family of candidate"
            " laws, searched in full against synthetic controls, produced not a single law that"
            " passes the galaxy and cluster tests together. That zero is published below with"
            " exact margins. A zero for one family is a fact about the family, not about nature."),
        )
    return (
        "OPEN",
        ("The missing-mass problem is unsolved — by us and by everyone. The campaign receipts"
        " this page renders are not all published in this repository yet; each missing one is"
        " declared exactly where its numbers would appear."),
    )


# ---------------------------------------------------------------------------
# Page skeleton: one typed sheet on grid paper, with a colophon
# ---------------------------------------------------------------------------


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
        '<div class="sheet">\n'
        '<header><div class="brand"><a href="/">Invariant</a></div>'
        '<p class="tag">a machine that guesses patterns, tries its hardest to break them, and'
        " files paperwork for every attempt &mdash; the failures included</p>"
        f"<nav>{nav_links}</nav></header>\n<main>\n{body}\n</main>\n"
        '<footer><p class="orn">* * *</p>'
        f"<p>{_esc(FOOTER_CREED)}</p>"
        f"<p>Site content as of <code>{_esc(commit)}</code>. Composed deterministically by "
        "<code>src/sigma_theory_compiler/static_site_generator.py</code>; "
        "re-run with <code>--validate</code> to byte-compare. Set in typewriter type on grid"
        " paper; no page requests any external resource.</p></footer>\n"
        "</div>\n</body>\n</html>\n"
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
                "candidate gravity laws tested — the complete declared family, no sampling",
                billion.path,
                data_key="candidates_processed",
                data_value=str(counts["processed"]),
            )
        )
        tiles.append(
            _tile(
                _esc(_fmt_int(billion.data["throughput_candidates_per_second"])) + "/s",
                f"laws tested per second on {billion.data['device']}",
                billion.path,
                data_key="throughput_candidates_per_second",
                data_value=str(billion.data["throughput_candidates_per_second"]),
            )
        )
    else:
        tiles.append(_tile("&mdash;", "candidate gravity laws tested", billion.path, kind="neg"))
    lensing = artifacts["lensing"]
    if lensing.present:
        counts = lensing.data["counts"]
        tiles.append(
            _tile(
                _esc(_fmt_int(counts["lensing_pass"])),
                "laws passing the synthetic lensing test (P1) — survivors, not theories",
                lensing.path,
                data_key="lensing_pass",
                data_value=str(counts["lensing_pass"]),
            )
        )
        tiles.append(
            _tile(
                _esc(_fmt_int(counts["cluster_pass"])),
                "laws passing the synthetic cluster test (P2) — a sealed negative",
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
                "Collatz halving relation checked to this bound: "
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
                "declared problems in the sealed intake queue",
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
            "blind formula guessing — an honest failure, kept on the books",
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
    body.append("<h1>Easy to state, maddening to settle</h1>")
    body.append(f'<p class="lede">{_esc(GARDNER_OPENING)}</p>')
    body.append(
        "<p>The machine is called Invariant, and it does three things, in a loop. It"
        " <strong>guesses</strong>: given raw numbers — stopping times, rotation speeds — it"
        " proposes exact patterns that might govern them. It <strong>attacks</strong>: every"
        " guess is handed to checkers that hunt for a counterexample in exact arithmetic, with"
        " no rounding to hide behind. And it <strong>files</strong>: whatever happens, triumph"
        " or wreck, goes into a receipt — a tamper-evident file recording exactly what was"
        " computed, sealed with a hash so it cannot be quietly edited later. Every number on"
        " this site is read out of those receipts at the moment the page is built; the site's"
        " own code contains none of them.</p>"
    )
    body.append(
        "<p>Two warnings before you wander in. First, nothing here settles a famous problem."
        " Each problem page opens with a stamped status, and OPEN means exactly what it says:"
        " unsolved, by us and by everyone. Second, when this machine does find something true,"
        " the fact has usually been known to mathematicians for a long time. The news is that a"
        " machine found it unaided — and the pages say so, every time, in the same breath.</p>"
    )
    body.append("<h2>Where to begin</h2>")
    body.append(
        "<ul>"
        '<li><a href="/paper">The write-up</a> — the whole story as one traditional paper:'
        " question, method, numbers, and a frank account of why this is not a journal"
        " submission.</li>"
        '<li><a href="/collatz">Collatz</a> — start here for the story behind the puzzle above:'
        " a machine that first said nothing at all, then found two small truths every number"
        " theorist already knew.</li>"
        '<li><a href="/gravity">Gravity</a> — a billion-law search whose headline result is a'
        " carefully measured zero, published with margins like any triumph.</li>"
        '<li><a href="/problems">Problems</a> — the ledger of declared targets: what each one'
        " is, why it is believed open, and what would count as progress.</li>"
        '<li><a href="/papers">Papers</a> — twelve blinded rediscoveries of long-settled'
        " results, written up one by one, each with the elimination funnel its receipts"
        " actually record.</li>"
        '<li><a href="/evidence">Evidence</a> — every file this site reads, with its seal.</li>'
        '<li><a href="/method">Method</a> — the assembly line of checks, and the measured'
        " failures that made each one necessary.</li>"
        '<li><a href="/submit">Submit</a> — propose a problem for the queue; a human reviews'
        " it, and acceptance means sealing, not publicity.</li>"
        "</ul>"
    )
    body.append("<h2>The ledger at a glance</h2>")
    body.append(_index_tiles(artifacts, facts))
    body.append(
        '<p class="small sub">Doc-sourced history values predate sealed receipts and are cited'
        " to their document filenames; everything else is read from the named receipt at build"
        " time.</p>"
    )
    body.append("<h2>House rules</h2>")
    body.append(
        "<p>Seven sentences bind every page of this logbook. They exist because each one was"
        " once violated somewhere, by someone, in some published claim about machines doing"
        " science — and the receipts are built so this project cannot repeat the habit.</p>"
    )
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
        f'<tr><td class="mono">{_esc(key)}</td><td class="mono">{_esc(machine_form[key])}</td></tr>'
        for key in sorted(machine_form)
    )
    return (
        '<div class="scroll"><table><thead><tr><th>machine_form key</th><th>value</th></tr>'
        f"</thead><tbody>{rows}</tbody></table></div>"
    )


def _problem_detail_page(
    entry: dict[str, Any], queue: Artifact, artifacts: dict[str, Artifact], commit: str
) -> bytes:
    entry_id = entry["id"]
    body: list[str] = []
    body.append('<p class="kicker">From the sealed problem ledger</p>')
    body.append(f"<h1><code>{_esc(entry_id)}</code> {_flag_badges(entry)}</h1>")
    body.append(
        f'<p class="sub small">Domain <code>{_esc(entry["domain"])}</code> &middot; sealed in'
        f" <code>{_esc(queue.path)}</code> under queue seal"
        f" {_sha_abbrev(queue.sha256)}</p>"
    )
    word, text = _problem_status(entry, artifacts)
    body.append(_status_banner(word, text))
    body.append(_status_key())
    body.append("<h2>The question</h2>")
    body.append(f"<blockquote>{_esc(entry['statement'])}</blockquote>")
    body.append("<h2>Where it comes from</h2>")
    body.append(f"<p>{_esc(entry['source_citation'])}</p>")
    body.append("<h2>Why it sits in the queue</h2>")
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
    if (entry["control_rediscovery"] or entry["synthetic"]) and artifacts["dozen"].present:
        body.append(
            '<p class="small sub">Related reading — separate from this entry: a sealed'
            " campaign of twelve blinded rediscoveries of the same calibration character is"
            ' written up, funnel by funnel, under <a href="/papers">Papers</a>.</p>'
        )
    body.append("<h2>What counts as progress</h2>")
    body.append(f"<p>{_esc(entry['progress_definition'])}</p>")
    body.append("<h2>Machine form</h2>")
    body.append(
        "<p>The exact, validated form the engine consumes — no floats anywhere, by rule:</p>"
    )
    body.append(_machine_form_block(entry["machine_form"]))
    body.append("<h2>Evidence in this repository</h2>")
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
    body.append('<p class="small"><a href="/problems">&larr; back to the full ledger</a></p>')
    title = f"{entry_id} · {SITE_TITLE}"
    return _page("problems", title, "\n".join(body), commit)


def _problems_index_page(queue: Artifact, artifacts: dict[str, Artifact], commit: str) -> bytes:
    body: list[str] = []
    body.append("<h1>The problem ledger</h1>")
    body.append(
        "<p>Every target this machine works on is declared here first, in a sealed queue — a"
        " hash-bound registry that cannot be edited without changing its fingerprint. Each entry"
        " cites a source, says in plain prose why the question is believed open, and defines in"
        " advance what would count as progress, so no result can quietly move its own"
        " goalposts. Calibration entries — questions with known answers, or manufactured worlds"
        " — carry their labels as validated schema fields, which means a rediscovery can never"
        " be dressed up as a discovery by dropping a sentence. Queue membership asserts"
        " provenance, nothing more: not importance, not tractability, not solvability.</p>"
    )
    if artifacts["dozen"].present:
        body.append(
            "<p>Calibration of exactly this character has been run at scale once already: a"
            " sealed campaign of twelve blinded rediscoveries, written up paper by paper —"
            ' each with its receipt-derived elimination funnel — under <a href="/papers">'
            "Papers</a>.</p>"
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
    body.append("<h2>Entries</h2>")
    body.append(
        "<p>The status column uses the five-word vocabulary defined on each entry's page; click"
        " through for the full banner, the citation, and the evidence trail.</p>"
    )
    rows = []
    for entry in entries:
        word, _text_unused = _problem_status(entry, artifacts)
        rows.append(
            "<tr>"
            f'<td class="mono"><a href="/problems/{_esc(entry["id"])}">{_esc(entry["id"])}</a></td>'
            f'<td class="mono">{_esc(word)}</td>'
            f'<td class="mono">{_esc(entry["domain"])}</td>'
            f'<td class="mono">{_esc(entry["machine_form"]["kind"])}</td>'
            f"<td>{_flag_badges(entry)}</td>"
            f"<td>{_esc(entry['statement'])}</td>"
            "</tr>"
        )
    body.append(
        '<div class="scroll"><table><thead><tr><th>id</th><th>status</th><th>domain</th>'
        "<th>machine form</th><th>flags</th><th>statement</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return _page("problems", f"Problems · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Gravity case study
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


def _gravity_abstract(billion: Artifact, lensing: Artifact) -> str:
    if billion.present and lensing.present:
        processed = billion.data["counts"]["processed"]
        survivors = billion.data["counts"]["fp64_survivors"]
        return (
            "<p>We wrote down, ahead of time, one exact family of candidate gravity laws —"
            f" {_esc(_fmt_int(processed))} of them, each a small formula with one shared constant"
            " and no per-galaxy knobs. A graphics card tested every single one against synthetic"
            f" galaxy controls, and {_esc(_fmt_int(survivors))} survived that first screen. Two"
            " harder tests followed: a gravitational-lensing gate and a model galaxy cluster."
            " Not one candidate passed both, and that zero is published below with exact"
            " margins. No telescope data was used anywhere; the machinery, the margins, and the"
            " near-misses follow.</p>"
        )
    return (
        "<p>We declared one exact family of candidate gravity laws, each a small formula with"
        " one shared constant and no per-galaxy knobs, and set out to test every single one"
        " against synthetic galaxy, lensing, and cluster controls. The campaign's receipts are"
        " not all published in this repository yet, so this page declares each gap exactly where"
        " the numbers would appear. What is present is rendered from the receipts alone. No"
        " telescope data has been opened at any point.</p>"
    )


def _base_screen_section(billion: Artifact) -> str:
    parts: list[str] = ['<h3 id="base-screen">The base screen</h3>']
    if not billion.present:
        parts.append(_missing_block(billion, "base screen"))
        return "\n".join(parts)
    data = billion.data
    counts = data["counts"]
    tiles = [
        _tile(
            _esc(_fmt_int(counts["processed"])),
            "candidates tested (equals the declared family size — no sampling)",
            billion.path,
            data_key="candidates_processed",
            data_value=str(counts["processed"]),
        ),
        _tile(
            _esc(_fmt_int(counts["fp64_survivors"])),
            "survivors of all five gates at double precision",
            billion.path,
            data_key="fp64_survivors",
            data_value=str(counts["fp64_survivors"]),
        ),
        _tile(
            _esc(_fmt_int(counts["exact_confirmed"]))
            + " / "
            + _esc(_fmt_int(counts["exact_refuted"])),
            "shortlist entries confirmed / refuted on exact high-precision replay",
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
        f"<p>Decision, verbatim from the receipt: <code>{_esc(data['decision'])}</code>. Five"
        " gates ran on frozen synthetic disk-galaxy controls: the law must vanish into ordinary"
        " Newtonian gravity where gravity is strong (our solar system is the sanity check), must"
        " keep accelerations positive and orderly, must flatten the outer rotation curve, and"
        " must reproduce the measured slope of the Tully&ndash;Fisher relation. Survivors are"
        " search priorities, not validated theories — the receipt says so in its own scope"
        " field, quoted under &ldquo;What this does not show.&rdquo;</p>"
    )
    return "\n".join(parts)


def _lensing_campaign_section(lensing: Artifact) -> str:
    parts: list[str] = ['<h3 id="lensing-cluster">Lensing and cluster campaign</h3>']
    if not lensing.present:
        parts.append(_missing_block(lensing, "lensing and cluster campaign"))
        return "\n".join(parts)
    data = lensing.data
    counts = data["counts"]
    tiles = [
        _tile(
            _esc(_fmt_int(counts["processed"])),
            "candidates re-tested against both new gates",
            lensing.path,
        ),
        _tile(
            _esc(_fmt_int(counts["fp32_union_survivors"])),
            "single-precision survivors carried to double precision",
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
        f" {_esc(data['device'])}, {_esc(data['elapsed_seconds'])} s wall. P1 asks whether the"
        " same law that flattens rotation curves also bends light by the right amount, using"
        " spherical equivalents of the screen&rsquo;s galaxies under a declared lensing"
        " prescription. P2 is the historically lethal test: a model galaxy cluster of hot gas"
        " whose measured pull the gas alone cannot supply under Newton. One universal constant,"
        " zero per-object freedom, everywhere.</p>"
    )
    return "\n".join(parts)


def _sealed_negative_section(lensing: Artifact) -> str:
    parts: list[str] = ['<h3 id="sealed-negative">The headline is a sealed negative</h3>']
    if not lensing.present:
        parts.append(_missing_block(lensing, "sealed negative"))
        return "\n".join(parts)
    data = lensing.data
    cluster_negative = data.get("cluster_negative", {})
    closest = cluster_negative.get("closest_cluster_approach", {})
    tolerance = (
        data.get("config", {}).get("cluster", {}).get("fp64_thresholds", {}).get("consistency")
    )
    lines = [
        '<div class="neg-block">',
        (
            "<p><strong>Decision, verbatim from the receipt:</strong> "
            f"<code>{_esc(data['decision'])}</code></p>"
        ),
    ]
    if closest and tolerance:
        margin = closest.get("max_deviation", "")
        lines.append(
            "<p>The closest any candidate came to carrying the cluster control: max deviation"
            f' <code data-key="closest_cluster_max_deviation" data-value="{_esc(margin)}">'
            f"{_esc(_dec(margin, 4))}</code> against tolerance"
            f' <code data-key="cluster_tolerance" data-value="{_esc(tolerance)}">'
            f"{_esc(_dec(tolerance))}</code>"
            f" (receipt strings <code>{_esc(margin)}</code> vs <code>{_esc(tolerance)}</code>)."
            " The nearest miss:</p>"
        )
        lines.append(
            '<details class="card"><summary>'
            + _esc(closest.get("formula", ""))
            + " &nbsp;"
            + _badge(bool(closest.get("lensing_passes")), "P1 PASS", "P1 FAIL")
            + '<span class="badge neg">P2 NEAREST MISS</span>'
            "</summary><div>"
            + _formula_html(closest.get("formula", ""))
            + f'<p class="small">Ordinal <code>{_esc(closest.get("ordinal", ""))}</code>;'
            f" located by {_esc(closest.get('located_by', ''))}. Max cluster deviation"
            f" <code>{_esc(closest.get('max_deviation', ''))}</code>.</p>"
            "</div></details>"
        )
    statement = cluster_negative.get("statement")
    if statement:
        lines.append(f"<p><strong>Sealed statement:</strong> {_esc(statement)}.</p>")
    lines.append(
        "<p>This is the deliverable. A family-wide zero, with exact margins, is a scientific"
        " result about this family — it is published with the same ceremony as any success,"
        " and it is what the queue&rsquo;s cluster entry predicted as the expected outcome.</p>"
    )
    lines.append("</div>")
    parts.extend(lines)
    return "\n".join(parts)


def _pareto_section(billion: Artifact) -> str:
    if not billion.present:
        return _missing_block(billion, "Pareto front")
    pareto = billion.data.get("pareto_front", [])
    parts = [f'<h3 id="pareto">The {len(pareto)}-entry Pareto front</h3>']
    parts.append(
        "<p>The shortlist below is what mathematicians call a Pareto front: the candidates no"
        " rival beats on every axis at once. The axes here are simplicity (how few terms the"
        " formula carries), how fast the law disappears into Newton&rsquo;s where it must"
        " (<code>newton_error</code>, smaller is better), and how flat it holds the outer"
        " rotation curve. Formulas are typeset from the receipt strings, which appear underneath"
        " each one; every entry was re-confirmed in high-precision arithmetic.</p>"
    )
    rows = []
    for index, entry in enumerate(pareto):
        rows.append(
            "<tr>"
            f'<td class="num">{index + 1}</td>'
            f'<td class="num">{_esc(entry["simplicity"])}</td>'
            f"<td>{_formula_html(entry['formula'])}</td>"
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
        + _formula_html(control.get("formula", ""))
        + f"<p>{note}</p>"
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


def _controls_section(lensing: Artifact) -> str:
    parts: list[str] = ['<h3 id="controls">Named controls: the tests discriminate</h3>']
    if not lensing.present:
        parts.append(_missing_block(lensing, "named controls"))
        return "\n".join(parts)
    parts.append(
        "<p>Three known laws ride along as controls, so a reader can see the gates rejecting"
        " and accepting for the right reasons rather than waving everything through:</p>"
    )
    controls = lensing.data.get("controls", {})
    notes = {
        "linear_u": (
            "The named &ldquo;flattens curves but fails lensing&rdquo; control: nu = 1 + u"
            " flattens all three synthetic rotation curves (flatness column below) yet fails the"
            " P1 dynamics&ndash;lensing consistency gate — exactly the family the roadmap said"
            " this gate must name. Documented in repository docs"
            f" ({_esc(_DOC_NAMES['roadmap_doc'])}, Track C)."
        ),
        "newton_nu1": (
            "Pure Newton on visible matter alone (nu = 1): fails flatness outright — the"
            " control that restates the missing-mass problem the campaign is probing."
        ),
        "sqrt_family": (
            "A known interpolating shape: passes the lensing gate and still fails the cluster"
            " gate — the historically expected MOND-like behavior, measured here per family."
        ),
    }
    for name in sorted(controls):
        parts.append(_control_card(name, controls[name], notes.get(name, "Receipt control.")))
    return "\n".join(parts)


def _near_miss_section(lensing: Artifact) -> str:
    parts: list[str] = ['<h3 id="near-misses">Near misses, with their exact margins</h3>']
    if not lensing.present:
        parts.append(_missing_block(lensing, "near misses"))
        return "\n".join(parts)
    parts.append(
        "<p>Exact-verified candidates ordered by cluster max deviation (the P2 margin); the P1"
        " column is the worst dynamics&ndash;lensing consistency across the three disk masses."
        " Close is not a pass, and no partial credit is awarded anywhere in the pipeline.</p>"
    )
    verified = [
        entry
        for entry in lensing.data.get("exact_verification", [])
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
            f"<td>{_formula_html(entry.get('formula', ''))}</td>"
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
    return "\n".join(parts)


def _gravity_methods_section(billion: Artifact, lensing: Artifact) -> str:
    parts: list[str] = []
    parts.append(
        "<p>The screen runs in two precisions: a fast single-precision (fp32) pass over the"
        " whole family, then a double-precision (fp64) confirmation of everything that"
        " survives, with the shortlist replayed once more in 50-digit arithmetic. The gate"
        " thresholds below are copied from each receipt&rsquo;s config block, not from this"
        " page&rsquo;s author.</p>"
    )
    if billion.present:
        parts.append("<h3>Base-screen gate thresholds</h3>")
        parts.append(_thresholds_table(billion.data.get("config", {})))
    else:
        parts.append(_missing_block(billion, "base-screen thresholds"))
    if lensing.present:
        cluster_config = lensing.data.get("config", {}).get("cluster", {})
        parts.append("<h3>Cluster-gate thresholds</h3>")
        parts.append(_thresholds_table(cluster_config))
        assumptions = lensing.data.get("assumptions", {})
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
    else:
        parts.append(_missing_block(lensing, "cluster thresholds and assumptions"))
    parts.append('<h3 id="crosscheck">Screen-versus-exact crosschecks</h3>')
    parts.append(
        "<p>A fast screen is only trustworthy if a slow, exact replay agrees with it, so both"
        " receipts carry sampled crosschecks:</p>"
    )
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
            f"<p>Lensing/cluster campaign:"
            f" {_esc(_fmt_int(crosscheck.get('lensing_disagreements', 0)))}"
            f" lensing disagreements and"
            f" {_esc(_fmt_int(crosscheck.get('cluster_disagreements', 0)))}"
            f" cluster disagreements over a {_esc(_fmt_int(crosscheck.get('sample', 0)))}-candidate"
            f" sample (<code>{_esc(lensing.path)}</code>).</p>"
        )
        wrote_any = True
    if not wrote_any:
        parts.append(
            f'<p class="sub">[{_esc(MISSING_NOTE)}: no crosscheck receipt is available]</p>'
        )
    return "\n".join(parts)


def _gravity_references(queue: Artifact, billion: Artifact, lensing: Artifact) -> str:
    parts: list[str] = []
    parts.append("<h3>Receipts rendered on this page</h3>")
    items = []
    for artifact in (billion, lensing):
        if artifact.present:
            items.append(
                f"<li>{_github(artifact.path)} &mdash; seal {_sha_abbrev(artifact.sha256)}</li>"
            )
        else:
            items.append(
                f"<li><code>{_esc(artifact.path)}</code> &mdash; {_esc(MISSING_NOTE)}</li>"
            )
    parts.append("<ul>" + "".join(items) + "</ul>")
    parts.append("<h3>Literature cited by the driving queue entries</h3>")
    if queue.present:
        wanted = ("baryonic_rotation_law", "lensing_dynamics_consistency", "cluster_missing_mass")
        rows = []
        for entry in queue.data["entries"]:
            if entry["id"] in wanted:
                rows.append(
                    f'<li><a href="/problems/{_esc(entry["id"])}"><code>{_esc(entry["id"])}'
                    f"</code></a>: {_esc(entry['source_citation'])}</li>"
                )
        parts.append("<ul>" + "".join(rows) + "</ul>")
    else:
        parts.append(_missing_block(queue, "queue citations"))
    parts.append("<h3>Background documents</h3>")
    parts.append(
        "<ul>"
        f"<li>{_github(ARTIFACT_PATHS['idt_doc'])} &mdash; the grammar, the screen history, and"
        " what would count as independent discovery.</li>"
        f"<li>{_github(ARTIFACT_PATHS['roadmap_doc'])} &mdash; why lensing and clusters come"
        " before heavy formal work.</li>"
        "</ul>"
    )
    return "\n".join(parts)


def _gravity_page(artifacts: dict[str, Artifact], commit: str) -> bytes:
    billion = artifacts["billion"]
    lensing = artifacts["lensing"]
    queue = artifacts["queue"]
    body: list[str] = []
    body.append('<p class="kicker">Case study II &middot; physics</p>')
    body.append("<h1>A billion guesses at gravity, and the zero that matters</h1>")
    word, text = _gravity_status(lensing)
    body.append(_status_banner(word, text))
    body.append(_status_key())
    body.append("<h2>Abstract</h2>")
    body.append(_gravity_abstract(billion, lensing))
    body.append("<h2>The question</h2>")
    body.append(
        "<p>Point a telescope at a spiral galaxy and clock its outer stars. They orbit far too"
        " fast for the gravity of the matter you can see — and instead of slowing with"
        " distance, their speeds level off onto a plateau. Astronomers call the plotted speeds"
        " a rotation curve, and the mystery is why the curve goes flat. The standard answer is"
        " unseen mass; a rival tradition asks instead whether gravity itself follows a"
        " different law when it gets very weak. In 2016, measurements across a large sample of"
        " galaxies (the citation is under References) sharpened the riddle into an equation:"
        " the gravity you observe,"
        " g<sub>obs</sub>, tracks the gravity of visible matter alone, g<sub>bar</sub>,"
        " through a single smooth function &nu; and one universal acceleration constant"
        " a<sub>0</sub>:</p>"
    )
    body.append(_math_block(LATEX_RAR_LAW))
    body.append(
        "<p>Any such law also fixes a second observed regularity, the baryonic"
        " Tully&ndash;Fisher relation, which ties the plateau speed to the visible mass:</p>"
    )
    body.append(_math_block(LATEX_BTFR))
    body.append(
        "<p>Nobody has derived a unique &nu; from first principles, and nobody knows whether"
        " any single choice can survive every test at once. Three entries in our problem ledger"
        " frame the campaign:"
        ' <a href="/problems/baryonic_rotation_law"><code>baryonic_rotation_law</code></a>,'
        ' <a href="/problems/lensing_dynamics_consistency"><code>lensing_dynamics_consistency'
        "</code></a>, and"
        ' <a href="/problems/cluster_missing_mass"><code>cluster_missing_mass</code></a> —'
        " the last of which predicted, in writing, that the expected outcome was a sealed"
        " negative.</p>"
    )
    body.append("<h2>What we did</h2>")
    body.append(
        "<p>Rather than fit one law to data, we enumerated a whole family of laws and tried to"
        " kill every member. Each candidate is a ratio of two small polynomials raised to a"
        " power:</p>"
    )
    body.append(_math_block(LATEX_NU_FAMILY))
    body.append(
        "<p>Here y is the visible-matter gravity in units of a<sub>0</sub>, and u is its"
        " reciprocal square root, so u grows exactly where gravity grows weak. Integer"
        " coefficients, one shared constant, zero per-galaxy freedom: every candidate is an"
        " address in a declared list, which is what makes &ldquo;we tested them all&rdquo; a"
        " checkable sentence rather than a boast. The family is enumerated so that shapes known"
        " to fail are included on purpose — the gates must be seen rejecting them. Documented"
        f" in repository docs ({_esc(_DOC_NAMES['idt_doc'])}).</p>"
    )
    body.append(_base_screen_section(billion))
    body.append(_lensing_campaign_section(lensing))
    body.append("<h2>What we found</h2>")
    body.append(_sealed_negative_section(lensing))
    body.append(_pareto_section(billion))
    body.append(_controls_section(lensing))
    body.append(_near_miss_section(lensing))
    body.append("<h2>What this does not show</h2>")
    body.append(
        "<ul>"
        "<li>Nothing here validates any law of gravity. Survivors of the galaxy gates are"
        " search priorities awaiting harder tests, and every one of them failed the cluster"
        " gate anyway.</li>"
        "<li>Nothing here touches real observations. Every control is synthetic and analytic;"
        " the sealed protocol for opening real data exists, and it has never been opened.</li>"
        "<li>The zero is family-shaped. Laws outside this declared grammar — including"
        " environment-dependent laws the roadmap names as the next grammar — are simply not"
        " addressed.</li>"
        "<li>No dark-matter conclusion follows in either direction. The campaign measures one"
        " grammar against declared controls, nothing more.</li>"
        "</ul>"
    )
    body.append(
        "<p>Quoted verbatim from the receipts&rsquo; own <code>scope</code> fields — the scope"
        " travels with the evidence, not with the marketing:</p>"
    )
    for artifact in (billion, lensing):
        if artifact.present:
            body.append(
                f'<p class="small sub"><code>{_esc(artifact.path)}</code>:</p>'
                f"<blockquote>{_esc(artifact.data.get('scope', ''))}</blockquote>"
            )
        else:
            body.append(_missing_block(artifact, "scope"))
    body.append("<h2>Methods</h2>")
    body.append(
        '<details class="methods"><summary>Technical detail &mdash; open to expand</summary>'
        "<div>" + _gravity_methods_section(billion, lensing) + "</div></details>"
    )
    body.append("<h2>References</h2>")
    body.append(_gravity_references(queue, billion, lensing))
    return _page("gravity", f"Gravity · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Collatz case study
# ---------------------------------------------------------------------------


def _collatz_abstract(sweep: Artifact) -> str:
    if sweep.present:
        hi = sweep.data["range"]["hi"]
        bound = _fmt_int(hi)
        checked = (
            f"It then checked one of them against every starting number below {_esc(bound)}"
            " without finding a counterexample, and proved a careful conditional version of it"
            " in a proof checker."
        )
    else:
        checked = (
            "It then set out to check one of them exhaustively up to an exact bound — that"
            " receipt is not yet published here — and proved a careful conditional version of"
            " it in a proof checker."
        )
    return (
        "<p>We pointed our pattern-hunting machine at a famous unsolved problem, expecting"
        " nothing, to watch precisely how it fails. On the first attempt it proposed nothing at"
        " all, and that silence located two specific blind spots in its design. After both were"
        " repaired as general abilities, the machine found two exact facts about Collatz"
        " stopping times — facts mathematicians have known for a long time. "
        + checked
        + " The conjecture itself is exactly as unsolved as it was in 1937.</p>"
    )


def _collatz_identities(facts: dict[str, tuple[int, ...] | None], idt_name: str) -> str:
    holdout = facts.get("halving_holdout")
    halving_confirmations = (
        f"{_fmt_int(holdout[0])} holdout confirmations" if holdout else _doc_gap("holdout count")
    )
    parts: list[str] = []
    parts.append("<h3>Two identities, found unaided</h3>")
    parts.append(
        '<details class="card" open><summary>'
        + _math(LATEX_SIGMA_POW2)
        + ' &nbsp;<span class="badge ctl">RESTRICTED INDEX FAMILY</span></summary><div>'
        + '<code class="receipt-form">sigma(2^k) = k</code>'
        "<p>On the powers of two the stopping time is exactly the exponent: 8 takes three"
        " halvings, 16 takes four, and so on — pure descent, no tripling ever triggered. Obvious"
        " once said, and known to anyone who has played with the problem; the machine could not"
        " see it until it learned to relabel the data along a sparse index. Holdout-confirmed on"
        f" rows it had never seen. Documented in repository docs ({_esc(idt_name)}).</p>"
        "</div></details>"
    )
    parts.append(
        '<details class="card" open><summary>'
        + _math(LATEX_SIGMA_HALVING)
        + ' &nbsp;<span class="badge ctl">INDEX SCALING RELATION</span></summary><div>'
        + '<code class="receipt-form">sigma(2n) = sigma(n) + 1</code>'
        "<p>Doubling any number costs exactly one extra step, because the first move from 2n is"
        " a halving back to n. This is the halving relation: proposed from a data prefix and"
        f" confirmed on {halving_confirmations} — the find that closed the loop from raw data"
        " to a kernel-checkable statement. It, too, is well known to mathematicians."
        f" Documented in repository docs ({_esc(idt_name)}).</p>"
        "</div></details>"
    )
    return "\n".join(parts)


def _collatz_sweep_section(sweep: Artifact) -> str:
    parts: list[str] = ['<h3 id="sweep">The exhaustive sweep</h3>']
    if not sweep.present:
        parts.append(_missing_block(sweep, "counterexample sweep"))
        return "\n".join(parts)
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
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")
    parts.append(
        "<p>Statement swept, verbatim:"
        f" <code>{_esc(data['statement']['text'])}</code></p>"
    )
    return "\n".join(parts)


def _collatz_methods(sweep: Artifact, lean: Artifact, idt_name: str) -> str:
    parts: list[str] = []
    parts.append("<h3>How the blind spots were fixed</h3>")
    parts.append(
        "<p>Both repairs were built as general abilities, not Collatz patches. First, sparse"
        " reindexing: sub-domains had been dense-only (every n past a point, or by parity), so"
        " structure living along a transformed index — such as n = 2<sup>k</sup> — was"
        " invisible; restrictions now carry explicit reindexing maps, and a reindexed result"
        " always declares its index variable. Second, a new statement kind,"
        " <code>index_scaling_relation</code>, of the shape a(c&middot;n) ="
        " &alpha;&middot;a(n) + &beta;, with vacuous-point discipline: rows whose scaled"
        " partner is absent contribute no support. Documented in repository docs"
        f" ({_esc(idt_name)}).</p>"
    )
    parts.append("<h3>How the sweep stays honest</h3>")
    if sweep.present:
        caps = sweep.data.get("system_caps", {})
        cap_value = caps.get("collatz_step_cap_default")
        cap_text = (
            f" Each instance runs under a declared step cap ({_esc(_fmt_int(cap_value))} steps"
            " here);" if isinstance(cap_value, int) else " Each instance runs under a declared"
            " step cap;"
        )
        parts.append(
            "<p>The GPU layer is a screen only: any reported witness would be re-verified in"
            " exact integer arithmetic before entering the receipt, and a screen violation the"
            " exact layer cannot reproduce raises instead of being dropped."
            + cap_text
            + " any lane that exceeds the cap lands in a fail-closed"
            " <code>UNDECIDED</code> bucket, never in pass or fail. The undecided count on this"
            " page is part of the receipt, not a footnote.</p>"
        )
    else:
        parts.append(_missing_block(sweep, "sweep methods"))
    parts.append('<h3 id="lean">The formal artifact, in full</h3>')
    if lean.present:
        parts.append(
            "<p>The halving relation&rsquo;s honest conditional form is provable with no"
            " termination assumption: if n reaches 1 in exactly k steps, then 2n reaches 1 in"
            " exactly k + 1 steps, with reachability stated inductively. The file below is"
            f" rendered verbatim from <code>{_esc(lean.path)}</code>"
            f" ({_github(lean.path, 'view on GitHub')}); it uses only the standard tactic"
            " library, and its trailer prints its own dependency audit.</p>"
        )
        parts.append(f"<pre><code>{_esc(lean.text)}</code></pre>")
    else:
        parts.append(_missing_block(lean, "formal Lean source"))
    return "\n".join(parts)


def _collatz_references(sweep: Artifact, lean: Artifact, queue: Artifact) -> str:
    parts: list[str] = []
    parts.append("<h3>Receipts and formal sources rendered on this page</h3>")
    items = []
    if sweep.present:
        items.append(f"<li>{_github(sweep.path)} &mdash; seal {_sha_abbrev(sweep.sha256)}</li>")
    else:
        items.append(f"<li><code>{_esc(sweep.path)}</code> &mdash; {_esc(MISSING_NOTE)}</li>")
    if lean.present:
        items.append(
            f"<li>{_github(lean.path)} &mdash; file hash {_sha_abbrev(lean.sha256)}&dagger;</li>"
        )
    else:
        items.append(f"<li><code>{_esc(lean.path)}</code> &mdash; {_esc(MISSING_NOTE)}</li>")
    parts.append("<ul>" + "".join(items) + "</ul>")
    parts.append("<h3>Literature cited by the driving queue entry</h3>")
    if queue.present:
        rows = []
        for entry in queue.data["entries"]:
            if entry["id"] == "collatz_stopping_time":
                rows.append(
                    f'<li><a href="/problems/{_esc(entry["id"])}"><code>{_esc(entry["id"])}'
                    f"</code></a>: {_esc(entry['source_citation'])}</li>"
                )
        parts.append("<ul>" + "".join(rows) + "</ul>")
    else:
        parts.append(_missing_block(queue, "queue citation"))
    parts.append("<h3>Background documents</h3>")
    parts.append(
        "<ul>"
        f"<li>{_github(ARTIFACT_PATHS['idt_doc'])} &mdash; the silence, the two repairs, and"
        " the claim boundary, as first documented.</li>"
        "</ul>"
    )
    parts.append(
        '<p class="small sub">&dagger; file-byte hash (the artifact is not a sealed JSON'
        " receipt).</p>"
    )
    return "\n".join(parts)


def _collatz_page(
    artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None], commit: str
) -> bytes:
    sweep = artifacts["sweep"]
    lean = artifacts["lean"]
    queue = artifacts["queue"]
    idt_name = _DOC_NAMES["idt_doc"]
    body: list[str] = []
    body.append('<p class="kicker">Case study I &middot; mathematics</p>')
    body.append("<h1>The machine meets Collatz: silence, two small truths, a bounded check</h1>")
    body.append(_status_banner("OPEN", COLLATZ_STATUS_TEXT))
    body.append(_status_key())
    body.append("<h2>Abstract</h2>")
    body.append(_collatz_abstract(sweep))
    body.append("<h2>The question</h2>")
    body.append(
        "<p>The rule from the home page: halve an even number, triple an odd number and add"
        " one, repeat. Write &sigma;(n) — sigma of n — for how many steps it takes n to come"
        " down to 1, if it ever does. Nobody has proved that &sigma;(n) is finite for every n,"
        " and nobody has a formula for it; that double ignorance, standing since 1937, is the"
        " Collatz problem. Its ledger entry, with citation and progress rules, is"
        ' <a href="/problems/collatz_stopping_time"><code>collatz_stopping_time</code></a>.'
        " Everything below is about the fine structure of &sigma;, and nothing below says"
        " anything about whether the process always terminates.</p>"
    )
    body.append("<h2>What we did</h2>")
    body.append(
        "<p>We handed the engine the raw stopping times — just the numbers, no target, no"
        " hint — and asked it to propose patterns. The first run failed completely. The"
        " basis-synthesis and structural-repair stages blocked, and the conjecture generator"
        " proposed nothing at all: zero conjectures, zero refutations, while two elementary"
        " true statements sat unseen in the very same data. That silence was the useful part."
        " It located two structural blind spots, both general rather than Collatz-specific:"
        " the machine could not relabel data along a sparse index, and it had no statement"
        " kind relating a(n) to a(c&middot;n). We fixed both as general abilities and ran"
        f" again; the same data then yielded the two identities below. Documented in"
        f" repository docs ({_esc(idt_name)}).</p>"
    )
    body.append(
        "<p>Then we attacked the second identity two more ways: an exhaustive"
        " counterexample sweep on a graphics card, and a machine-checked proof of its honest"
        " conditional form in the Lean proof checker — each with its scope stated exactly.</p>"
    )
    body.append("<h2>What we found</h2>")
    body.append(_collatz_identities(facts, idt_name))
    body.append(_collatz_sweep_section(sweep))
    body.append("<h2>What this does not show</h2>")
    body.append("<h3>Claim boundary</h3>")
    body.append(
        "<p>The empirical identity presupposes that both stopping times exist — which is the"
        " open part of the Collatz conjecture, and is not claimed. The Lean theorem proves only"
        " the conditional statement; it proves nothing about whether any n reaches 1 at all,"
        " and the case-study tests include a check that no statement the engine emits can even"
        " express a termination claim. A finite sweep, likewise, proves nothing outside its"
        " range. The receipt&rsquo;s scope field says so itself:</p>"
    )
    if sweep.present:
        body.append(f"<blockquote>{_esc(sweep.data['scope'])}</blockquote>")
    else:
        body.append(_missing_block(sweep, "sweep scope"))
    body.append(
        "<p>And the two identities, while found unaided, are well known to mathematicians —"
        " the news here is the finding machine, not the found mathematics.</p>"
    )
    body.append("<h2>Methods</h2>")
    body.append(
        '<details class="methods"><summary>Technical detail &mdash; open to expand</summary>'
        "<div>" + _collatz_methods(sweep, lean, idt_name) + "</div></details>"
    )
    body.append("<h2>References</h2>")
    body.append(_collatz_references(sweep, lean, queue))
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
    if artifact.key.startswith(DOZEN_WORLD_PREFIX):
        verdict = artifact.data.get("unseal", {}).get("verdict", "sealed receipt")
        return f"{verdict} — blinded rediscovery world receipt"
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
        "<p>This page is the site&rsquo;s bibliography of itself. The generator reads exactly"
        " the files below — nothing else — and every number on every page traces to one of"
        " them. JSON receipts carry their own <code>content_sha256</code> seal, a fingerprint"
        " of the canonical content: change one character and the fingerprint changes, which is"
        " what &ldquo;sealed&rdquo; means throughout this site. For plain-text artifacts the"
        " hash shown is the SHA-256 of the file bytes (marked &dagger;). Links go to the same"
        " paths on GitHub, so you can diff what you read here against the source of record.</p>"
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
            kind: {
                name: ("int" if field_type is int else "str")
                for name, field_type in fields.items()
            }
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
    body.append("<h1>Propose a problem for the ledger</h1>")
    body.append(f"<p><strong>{_esc(SUBMIT_NOTICE)}</strong></p>")
    body.append(
        "<p>Anyone can nominate a question. The form below builds a queue entry in the exact"
        f" sealed format — the schema has exactly {len(schema_keys)} keys per entry: "
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
        f'<p class="small sub">{_esc(SUBMIT_NOTICE)}</p>'
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
    body.append("<h1>Method: how the machine keeps itself honest</h1>")
    body.append(
        "<p>Every claim on this site rode down the same assembly line, and every station on"
        " that line exists because some earlier version of this project measured itself doing"
        " something embarrassing without it. First the line, then the embarrassments.</p>"
    )
    body.append("<h2>The assembly line</h2>")
    body.append(
        "<ol>"
        "<li><strong>Declared families, numbered exhaustively.</strong> Every candidate family"
        " is enumerable and finite; a survivor is an address in a declared list, never a"
        " free-form artifact that wandered in.</li>"
        "<li><strong>GPU screen.</strong> Cheap, physics- or math-informative tests over the"
        " whole family — single precision first, double precision to confirm — with every"
        " threshold recorded in the receipt&rsquo;s config block.</li>"
        "<li><strong>Exact confirmation.</strong> Shortlisted survivors are replayed in"
        " high-precision or exact integer arithmetic, and screen-versus-exact crosschecks are"
        " receipted with their disagreement counts.</li>"
        "<li><strong>Sealed receipts.</strong> Results are canonical JSON, hash-bound"
        " (<code>content_sha256</code>), float-free, and carry their own scope and claims"
        " fields. Negative verdicts are sealed with the same ceremony.</li>"
        "<li><strong>Formal kernel.</strong> What admits a proof is proved in Lean — a proof"
        " kernel, meaning a small trusted program that checks every logical step. Exact"
        " checking and kernel verification are never merged, and every receipt says which one"
        " it has.</li>"
        "<li><strong>Sealed observational ladder.</strong> Real data opens once, under a"
        " committed protocol, with no refit. To date zero real observations have been opened —"
        f" documented in repository docs ({_esc(goals_name)}).</li>"
        "</ol>"
    )
    body.append("<h2>The failures that shaped it</h2>")
    tiles = [
        _tile(
            _esc(blind) if blind else "&mdash;",
            "blind formula guessing: hidden formulas were not learnable from public rows"
            " alone — an honest failure",
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
        f"<p>The {blind_text} versus {conditioned_text} contrast is the founding measurement."
        " The same engine, on the same hidden worlds, failed every blind guess — and then"
        " succeeded exactly when public constraints made the answer unique. The honest reading"
        " is recorded in the trial document: that success belongs to the solver plus the"
        " constraints, and calling it discovery would be recall wearing a costume."
        f" Independently, the {rejections_text} epoch showed a funnel that let candidates die"
        " at the expensive formal tier without ever being asked a cheap physics question. Both"
        " failures stay on the books, and both dictated the architecture: cheap falsification"
        " first, exact confirmation second, formal proof third, sealed data last — with every"
        " claim carrying its boundary.</p>"
    )
    body.append("<h2>The claim discipline</h2>")
    body.append(
        "<p>Binding on every task, quoted from the roadmap&rsquo;s list:</p>"
        "<ul>" + "".join(f"<li>{_esc(line)}</li>" for line in CLAIM_DISCIPLINE) + "</ul>"
    )
    body.append(
        f'<p class="small sub">Documented in repository docs ({_esc(roadmap_name)}).</p>'
    )
    body.append("<h2>The documents</h2>")
    body.append(
        "<ul>"
        f"<li>{_github(ARTIFACT_PATHS['goals_doc'])} — the goal registry: every measured"
        " outcome, including the blocked and failed ones.</li>"
        f"<li>{_github(ARTIFACT_PATHS['idt_doc'])} — what would count as independent"
        " discovery, and why nothing yet does.</li>"
        f"<li>{_github(ARTIFACT_PATHS['roadmap_doc'])} — the continuous-discovery roadmap and"
        " the claim-discipline list.</li>"
        "</ul>"
    )
    return _page("method", f"Method · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# The write-up: one traditional paper covering both campaigns
# ---------------------------------------------------------------------------


def _paper_math_results(
    artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None]
) -> str:
    sweep = artifacts["sweep"]
    idt_name = _DOC_NAMES["idt_doc"]
    holdout = facts.get("halving_holdout")
    holdout_text = (
        f"{_fmt_int(holdout[0])} held-out rows" if holdout else _doc_gap("holdout count")
    )
    parts: list[str] = ["<h3>Campaign one: the Collatz probe</h3>"]
    parts.append(
        "<p>Handed the raw stopping times with no declared target, the engine&rsquo;s first"
        " run proposed nothing: zero conjectures, zero refutations. The silence exposed two"
        " general capability gaps — no sparse reindexing, no statement kind relating a(n) to"
        " a(c&middot;n) — and after both were built as general abilities, the same data yielded"
        " two exact identities, " + _math(LATEX_SIGMA_POW2) + " on powers of two and the"
        " halving relation " + _math(LATEX_SIGMA_HALVING) + ", the latter confirmed on "
        + holdout_text
        + f". Both are well known to mathematicians. Documented in repository docs"
        f" ({_esc(idt_name)}).</p>"
    )
    if sweep.present:
        data = sweep.data
        hi = data["range"]["hi"]
        exponent = _pow10_exponent(hi)
        hi_shown = _fmt_int(hi) + (f" (10^{exponent})" if exponent is not None else "")
        parts.append(
            "<p>The halving relation was then attacked exhaustively on the GPU:"
            f" {_esc(_fmt_int(data['counts']['checked']))} instances checked over"
            f" [{_esc(_fmt_int(data['range']['lo']))}, {_esc(hi_shown)}), decision"
            f" <code>{_esc(data['decision'])}</code>, with"
            f" {_esc(_fmt_int(data['undecided']['count']))} instances in the fail-closed"
            " undecided bucket, at"
            f" {_esc(_fmt_int(data['throughput_per_second']))} instances per second on"
            f" {_esc(data['device'])}. A finite sweep says nothing beyond its bound, and this"
            " one claims nothing beyond it.</p>"
        )
    else:
        parts.append(_missing_block(sweep, "sweep results"))
    lean = artifacts["lean"]
    if lean.present:
        parts.append(
            "<p>Finally, the relation&rsquo;s honest conditional form — if n reaches 1 in"
            " exactly k steps, then 2n reaches 1 in exactly k + 1 — was proved in the Lean"
            " proof checker with no termination assumption, and the source is rendered in full"
            ' on the <a href="/collatz">Collatz page</a>. The proof says nothing about whether'
            " any number reaches 1 at all.</p>"
        )
    else:
        parts.append(_missing_block(lean, "formal proof"))
    return "\n".join(parts)


def _paper_gravity_results(artifacts: dict[str, Artifact]) -> str:
    billion = artifacts["billion"]
    lensing = artifacts["lensing"]
    parts: list[str] = ["<h3>Campaign two: the gravity screen</h3>"]
    if billion.present:
        data = billion.data
        counts = data["counts"]
        parts.append(
            "<p>The declared family of candidate acceleration laws — "
            + _math(LATEX_NU_FAMILY)
            + " with small integer coefficients and one shared constant — contains"
            f" {_esc(_fmt_int(counts['processed']))} members, and every one was tested: five"
            " synthetic-galaxy gates at"
            f" {_esc(_fmt_int(data['throughput_candidates_per_second']))} candidates per second"
            f" on {_esc(data['device'])}, {_esc(data['elapsed_seconds'])} seconds wall clock."
            f" {_esc(_fmt_int(counts['fp64_survivors']))} candidates survived at double"
            f" precision; the {_esc(_fmt_int(counts['pareto_front']))}-entry shortlist was"
            " re-confirmed in 50-digit arithmetic with"
            f" {_esc(_fmt_int(counts['exact_refuted']))} refutations. Survivors are search"
            " priorities, not validated theories.</p>"
        )
    else:
        parts.append(_missing_block(billion, "base screen results"))
    if lensing.present:
        data = lensing.data
        counts = data["counts"]
        closest = data.get("cluster_negative", {}).get("closest_cluster_approach", {})
        tolerance = (
            data.get("config", {}).get("cluster", {}).get("fp64_thresholds", {}).get("consistency")
        )
        margin = closest.get("max_deviation")
        margin_text = ""
        if margin and tolerance:
            margin_text = (
                f" The nearest miss deviated by {_esc(_dec(margin, 4))} against a tolerance of"
                f" {_esc(_dec(tolerance))} (receipt strings <code>{_esc(margin)}</code> vs"
                f" <code>{_esc(tolerance)}</code>)."
            )
        parts.append(
            "<p>Two harder gates followed: P1, dynamics&ndash;lensing consistency, and P2, a"
            " hydrostatic model cluster."
            f" {_esc(_fmt_int(counts['lensing_pass']))} candidates passed P1 alone;"
            f" {_esc(_fmt_int(counts['cluster_pass']))} passed P2, and"
            f" {_esc(_fmt_int(counts['both_pass']))} passed both — the sealed negative the"
            " cluster queue entry predicted in advance, decision"
            f" <code>{_esc(data['decision'])}</code>."
            + margin_text
            + " No observational data was opened at any point in either campaign.</p>"
        )
    else:
        parts.append(_missing_block(lensing, "lensing and cluster results"))
    return "\n".join(parts)


def _paper_history_results(facts: dict[str, tuple[int, ...] | None]) -> str:
    goals_name = _DOC_NAMES["goals_doc"]
    idt_name = _DOC_NAMES["idt_doc"]
    blind = _blind_ratio(facts)
    conditioned = _plain_ratio(facts, "conditioned")
    rejections = _plain_ratio(facts, "formal_rejections")
    curriculum = _plain_ratio(facts, "curriculum")
    controls = _plain_ratio(facts, "formal_controls")
    blind_text = _esc(blind) if blind else _doc_gap("blind-guessing")
    conditioned_text = _esc(conditioned) if conditioned else _doc_gap("conditioned-recovery")
    rejections_text = _esc(rejections) if rejections else _doc_gap("formal-rejection")
    curriculum_text = _esc(curriculum) if curriculum else _doc_gap("curriculum")
    controls_text = _esc(controls) if controls else _doc_gap("formal-control")
    return (
        "<h3>The measured record behind the design</h3>"
        f"<p>Blind formula guessing scored {blind_text}: hidden formulas were not learnable"
        " from public rows alone. The same worlds with public constraints scored"
        f" {conditioned_text} — which is why recovery under constraints is credited to the"
        " solver plus the constraints, never to discovery. One production epoch died"
        f" {rejections_text} at the formal tier with zero rotation curves computed, which is"
        f" why the cheap screen tier exists. A {curriculum_text} blind benchmark curriculum"
        f" was registered and sealed before one atomic opening, and {controls_text}"
        " known-answer formal controls pass. Documented in repository docs"
        f" ({_esc(idt_name)}; {_esc(goals_name)}).</p>"
    )


def _paper_references(artifacts: dict[str, Artifact]) -> str:
    queue = artifacts["queue"]
    parts: list[str] = []
    parts.append("<h3>Receipts and formal sources</h3>")
    items = []
    for key in ("billion", "lensing", "sweep", "lean", "queue"):
        artifact = artifacts[key]
        if artifact.present:
            dagger = "" if artifact.sealed else "&dagger;"
            items.append(
                f"<li>{_github(artifact.path)} &mdash; {_sha_abbrev(artifact.sha256)}{dagger}</li>"
            )
        else:
            items.append(
                f"<li><code>{_esc(artifact.path)}</code> &mdash; {_esc(MISSING_NOTE)}</li>"
            )
    parts.append("<ul>" + "".join(items) + "</ul>")
    parts.append("<h3>Literature cited by the queue entries behind both campaigns</h3>")
    if queue.present:
        wanted = (
            "collatz_stopping_time",
            "baryonic_rotation_law",
            "lensing_dynamics_consistency",
            "cluster_missing_mass",
        )
        rows = []
        for entry in queue.data["entries"]:
            if entry["id"] in wanted:
                rows.append(f"<li>{_esc(entry['source_citation'])}</li>")
        parts.append("<ul>" + "".join(rows) + "</ul>")
    else:
        parts.append(_missing_block(queue, "queue citations"))
    parts.append("<h3>Repository documents</h3>")
    parts.append(
        "<ul>"
        f"<li>{_github(ARTIFACT_PATHS['goals_doc'])}</li>"
        f"<li>{_github(ARTIFACT_PATHS['idt_doc'])}</li>"
        f"<li>{_github(ARTIFACT_PATHS['roadmap_doc'])}</li>"
        "</ul>"
    )
    parts.append(
        '<p class="small sub">&dagger; file-byte hash (the artifact is not a sealed JSON'
        " receipt).</p>"
    )
    return "\n".join(parts)


def _paper_page(
    artifacts: dict[str, Artifact], facts: dict[str, tuple[int, ...] | None], commit: str
) -> bytes:
    prior_art = _plain_count(facts, "prior_art_records")
    goals_name = _DOC_NAMES["goals_doc"]
    idt_name = _DOC_NAMES["idt_doc"]
    body: list[str] = []
    body.append('<p class="kicker">The write-up</p>')
    body.append("<h1>Guess, attack, file: what a discovery machine actually established</h1>")
    body.append('<p class="byline">The Invariant Project</p>')
    body.append(
        '<p class="sub small">Dated by content, not by clock: this article describes the'
        f" repository exactly as of commit <code>{_esc(commit)}</code>, and it changes only"
        " when the repository does.</p>"
    )
    body.append("<h2>Abstract</h2>")
    body.append(
        "<p>We describe a machine that proposes exact mathematical patterns, attacks its own"
        " proposals with counterexample searches and proof checkers, and publishes a sealed,"
        " tamper-evident receipt for every attempt — including the failures. We report two"
        " campaigns in full. In mathematics, the machine was pointed at the Collatz problem:"
        " its first run produced silence, the silence located two design gaps, and after"
        " general repairs it found two classical identities unaided, checked one exhaustively"
        " to an exact bound, and proved a conditional form in a proof kernel. In physics, it"
        " enumerated over a billion candidate gravity laws and tested every one against"
        " synthetic galaxy, lensing, and cluster gates; none survived all of them — a sealed"
        " negative, published with exact margins, that was predicted in advance. No famous problem was settled, no"
        " observational data was opened, and no result here is new to mathematics; the"
        " contribution is the auditable machinery, demonstrated end to end.</p>"
    )
    body.append("<h2>Introduction</h2>")
    body.append(
        "<p>Machines that claim to do science have an honesty problem: it is cheap to"
        " announce a discovery and expensive to check one. This project inverts the expense."
        " Every claim is born inside a receipt — a hash-sealed file recording exactly what was"
        " computed, under what thresholds, with what outcome — and the public site, including"
        " this article, is generated by reading those receipts at build time. The generator"
        " contains none of the numbers you are about to read. If a receipt were deleted, its"
        " numbers would vanish from this page rather than be remembered.</p>"
    )
    body.append(
        "<p>Results are described with a five-word status vocabulary — OPEN, REDISCOVERED,"
        " RANGE-VERIFIED, PROVED (CONDITIONAL), SEALED NEGATIVE — each defined where used, so"
        " that a reader can always distinguish an unsolved question from a bounded check and a"
        " rediscovered classic from news. Two further rules bind every page: no scalar scores"
        " of any kind, and equal billing for failures.</p>"
    )
    body.append("<h2>Methods</h2>")
    body.append(
        "<p>Candidates are never free-form. Each campaign declares a finite, numbered family"
        " in advance — polynomial-ratio acceleration laws in the gravity campaign; statement"
        " kinds over integer sequences in the mathematics campaign — so &ldquo;we tested them"
        " all&rdquo; is a checkable sentence. Testing proceeds down a ladder: a fast GPU screen"
        " in single precision, a double-precision confirmation, an exact or 50-digit replay of"
        " every shortlisted survivor, and, where a statement admits one, a machine-checked"
        " proof in the Lean kernel. Screen-versus-exact crosschecks ride in every receipt with"
        " their disagreement counts. Bounded iteration runs under declared step caps, and a"
        " lane that hits its cap lands in a fail-closed undecided bucket — never in pass or"
        " fail. Receipts are canonical JSON, float-free, sealed with"
        " <code>content_sha256</code>, and carry their own scope and claims fields so the"
        " boundary travels with the evidence.</p>"
    )
    body.append(
        "<p>The site itself is part of the method. Pages are rendered deterministically from"
        " the receipts and the sealed problem queue; a validation mode rebuilds everything and"
        " byte-compares. Headline numbers are annotated in the HTML with their receipt keys,"
        " and the test suite greps the generator source to prove the numerals are not"
        " hard-coded.</p>"
    )
    body.append("<h2>Results</h2>")
    body.append(_paper_math_results(artifacts, facts))
    body.append(_paper_gravity_results(artifacts))
    body.append(_paper_history_results(facts))
    body.append("<h2>Limitations</h2>")
    body.append(
        "<ul>"
        "<li>Every control in both campaigns is synthetic and analytic. Zero real observations"
        " have ever been opened, and the sealed protocol for opening one remains unused —"
        f" documented in repository docs ({_esc(goals_name)}).</li>"
        "<li>The exhaustive sweep is bounded, and claims nothing beyond its bound. The Lean"
        " theorem is conditional, and claims nothing about termination.</li>"
        "<li>The gravity zero is family-shaped: laws outside the declared grammar are not"
        " addressed, and no dark-matter conclusion follows in either direction.</li>"
        "<li>Both identities the engine found are classical. Machine performance outside the"
        " declared families and registered curricula is unclaimed.</li>"
        "</ul>"
    )
    body.append("<h3>Why this is not a mathematics-journal submission</h3>")
    prior_art_text = (
        f"{_fmt_int(prior_art)}-record" if prior_art is not None else _doc_gap("prior-art size")
    )
    body.append(
        "<p>Because its mathematical findings are not new. The two Collatz identities are"
        " classical facts, stated in textbooks and known to any specialist; what is new is"
        " only the provenance — a machine reached them unaided from raw data. Novelty could"
        " not honestly be claimed even if we suspected it: our prior-art corpus is a"
        f" {prior_art_text} snapshot, far too small to support any absence-based statement,"
        " and the house rules forbid treating corpus absence as novelty. Documented in"
        f" repository docs ({_esc(goals_name)}; {_esc(idt_name)}).</p>"
    )
    body.append(
        "<p>What would have to change first: targets authored outside this system, certified"
        " in advance to lie outside the machine&rsquo;s declared basis; a prior-art corpus"
        " grown by orders of magnitude with externally checked equivalence controls; and then"
        " a genuinely new theorem — kernel-checked, surviving expert review — rather than a"
        " rediscovered one. That ladder is specified in the Independent Discovery Trial"
        " document, and none of its upper rungs has been climbed. Until they are, the right"
        " venue for this work is exactly this: a public logbook with its receipts showing.</p>"
    )
    body.append("<h2>References</h2>")
    body.append(_paper_references(artifacts))
    return _page("paper", f"The write-up · {SITE_TITLE}", "\n".join(body), commit)


# ---------------------------------------------------------------------------
# Rediscovery papers: deterministic statement mathematics
# ---------------------------------------------------------------------------

#: Tokenizer for the sealed integer-sequence statement grammar.  The alternation
#: order matters: the summation head and ``**`` must win before letters and ``*``.
_SEQ_TOKEN = re.compile(r"\s*(sum_\(i<=n\)|\*\*|\d+|[A-Za-z]+|[()+\-*/^,])")

_INT_LATEX = re.compile(r"^-?\d+$")


class _SeqExprParser:
    """Recursive-descent parser for receipt sequence expressions -> LaTeX.

    Nodes are ``(kind, latex, inner)`` tuples; ``inner`` carries the unwrapped
    body of parenthesized groups so fractions and exponents can drop redundant
    parentheses without ever guessing.  Anything outside the grammar raises.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens: list[str] = []
        position = 0
        while position < len(text):
            match = _SEQ_TOKEN.match(text, position)
            if match is None:
                if text[position:].strip() == "":
                    break
                raise SiteGenerationError(f"statement outside the receipt grammar: {text!r}")
            self.tokens.append(match.group(1))
            position = match.end()
        self.index = 0

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise SiteGenerationError(f"statement outside the receipt grammar: {self.text!r}")
        self.index += 1
        return token

    def _expect(self, token: str) -> None:
        if self._take() != token:
            raise SiteGenerationError(f"statement outside the receipt grammar: {self.text!r}")

    def parse(self) -> str:
        node = self._add()
        if self._peek() is not None:
            raise SiteGenerationError(f"statement outside the receipt grammar: {self.text!r}")
        return node[1]

    def _add(self) -> tuple[str, str, str]:
        leading = ""
        if self._peek() == "-":
            self._take()
            leading = "-"
        first = self._mul()
        rendered = leading + first[1]
        count = 1
        while self._peek() in {"+", "-"}:
            operator = self._take()
            term = self._mul()
            rendered += f" {operator} {term[1]}"
            count += 1
        if count == 1 and leading == "":
            return first
        if count == 1 and _INT_LATEX.fullmatch(rendered):
            return ("neg", rendered, rendered)
        return ("add", rendered, rendered)

    def _mul(self) -> tuple[str, str, str]:
        factors = [self._pow()]
        while self._peek() in {"*", "/"}:
            operator = self._take()
            right = self._pow()
            if operator == "*":
                factors.append(right)
            else:
                numerator = self._join(factors)
                rendered = (
                    "\\frac{" + self._unwrap(numerator) + "}{" + self._unwrap(right) + "}"
                )
                factors = [("frac", rendered, rendered)]
        return self._join(factors)

    @staticmethod
    def _unwrap(node: tuple[str, str, str]) -> str:
        return node[2] if node[0] == "paren" else node[1]

    @staticmethod
    def _join(factors: list[tuple[str, str, str]]) -> tuple[str, str, str]:
        if len(factors) == 1:
            return factors[0]
        rendered = factors[0][1]
        for factor in factors[1:]:
            separator = " \\cdot " if factor[1][:1].isdigit() else "\\,"
            rendered += separator + factor[1]
        return ("mul", rendered, rendered)

    def _pow(self) -> tuple[str, str, str]:
        base = self._atom()
        if self._peek() in {"**", "^"}:
            self._take()
            exponent = self._pow()
            rendered = base[1] + "^{" + self._unwrap(exponent) + "}"
            return ("pow", rendered, rendered)
        return base

    def _atom(self) -> tuple[str, str, str]:
        token = self._take()
        if token == "sum_(i<=n)":
            return ("sum", "\\sum_{i \\le n}", "\\sum_{i \\le n}")
        if token.isdigit():
            return ("int", token, token)
        if token == "(":
            inner = self._add()
            self._expect(")")
            if inner[0] == "int":
                return inner
            rendered = "\\left(" + inner[1] + "\\right)"
            return ("paren", rendered, inner[1])
        if token.isalpha():
            if self._peek() == "(":
                self._take()
                arguments = [self._add()]
                while self._peek() == ",":
                    self._take()
                    arguments.append(self._add())
                self._expect(")")
                if token in {"C", "binomial"} and len(arguments) == 2:
                    rendered = "\\binom{" + arguments[0][1] + "}{" + arguments[1][1] + "}"
                    return ("call", rendered, rendered)
                if token == "a" and len(arguments) == 1:
                    rendered = "a(" + arguments[0][1] + ")"
                    return ("call", rendered, rendered)
                raise SiteGenerationError(
                    f"statement outside the receipt grammar: {self.text!r}"
                )
            if token in {"n", "i"}:
                return ("name", token, token)
        raise SiteGenerationError(f"statement outside the receipt grammar: {self.text!r}")


def _seq_expr_to_latex(text: str) -> str:
    return _SeqExprParser(text).parse()


_DIVIDES_SHAPE = re.compile(r"^(\d+) divides (.+)$")
_CONGRUENCE_SHAPE = re.compile(r"^(.+?) = (.+?) \(mod (\d+)\)$")
_SUM_HEAD = "sum_(i<=n) "


def _relation_to_latex(text: str) -> str:
    match = _DIVIDES_SHAPE.fullmatch(text)
    if match is not None:
        return match.group(1) + " \\mid " + _seq_expr_to_latex(match.group(2))
    match = _CONGRUENCE_SHAPE.fullmatch(text)
    if match is not None:
        return (
            _seq_expr_to_latex(match.group(1))
            + " \\equiv "
            + _seq_expr_to_latex(match.group(2))
            + " \\pmod{"
            + match.group(3)
            + "}"
        )
    for operator, symbol in ((" = ", "="), (" < ", "<"), (" > ", ">")):
        if operator in text:
            sides = text.split(operator)
            if len(sides) != 2:
                raise SiteGenerationError(f"statement outside the receipt grammar: {text!r}")
            left, right = sides
            if left.startswith(_SUM_HEAD):
                left_latex = "\\sum_{i \\le n} " + _seq_expr_to_latex(left[len(_SUM_HEAD):])
            else:
                left_latex = _seq_expr_to_latex(left)
            return left_latex + f" {symbol} " + _seq_expr_to_latex(right)
    raise SiteGenerationError(f"statement outside the receipt grammar: {text!r}")


def statement_ascii_to_latex(statement: str) -> str:
    """Translate a sealed sequence-statement string into LaTeX, deterministically.

    Handles every statement shape the solved-dozen receipts emit: closed forms,
    recurrences (with optional seed clauses after ``;``), ratio laws, partial
    sums, congruences, divisibility, sign, and monotonicity.  Anything outside
    the grammar raises rather than degrading silently.
    """

    if "; " in statement:
        main, seeds = statement.split("; ", 1)
        seed_latexes = [_relation_to_latex(seed) for seed in seeds.split(", ")]
        return _relation_to_latex(main) + ",\\quad " + ",\\; ".join(seed_latexes)
    return _relation_to_latex(statement)


def _statement_html(statement: str) -> str:
    """A sequence statement as proper mathematics, with the receipt string underneath."""

    rendered = latex_to_mathml(statement_ascii_to_latex(statement))
    return (
        f'<span class="math">{rendered}</span>'
        f'<code class="receipt-form">{_esc(statement)}</code>'
    )


# ---------------------------------------------------------------------------
# Rediscovery papers: the elimination funnel, derived only from the trail
# ---------------------------------------------------------------------------


def _reason_breakdown(entries: Any, reason_scope: str) -> list[tuple[str, int]]:
    """Sorted (reason, count) pairs over a minimality certificate's rejections."""

    if not isinstance(entries, list):
        raise SiteGenerationError(f"minimality certificate without rejections: {reason_scope}")
    counts: dict[str, int] = {}
    for entry in entries:
        reason = str(entry.get("reason", "unrecorded"))
        counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items())


def _funnel_model(data: dict[str, Any]) -> dict[str, Any]:
    """Derive the elimination-funnel bars for one world from its receipt trail.

    A stage whose count the receipt does not record is omitted — never estimated
    — and the omission is footnoted.  Bar counts are the receipt values; the one
    derived value (``family accepted``) is the receipt's examined-minus-rejected
    difference, which the tests re-derive independently.
    """

    phase = data.get("phase_a", {})
    bars: list[dict[str, Any]] = []
    reason_tables: list[tuple[str, str, list[tuple[str, int]]]] = []
    footnotes: list[str] = []

    def bar(key: str, label: str, count: Any, display: str | None = None, accent: bool = False):
        if not isinstance(count, int):
            footnotes.append(f"{label}: stage counts not recorded in receipt")
            return
        bars.append(
            {
                "key": key,
                "label": label,
                "count": count,
                "display": display if display is not None else str(count),
                "accent": accent,
            }
        )

    stages = phase.get("stages", [])
    b1_passed = False
    for stage in stages:
        stage_id = stage.get("stage_id")
        receipt = stage.get("receipt") or {}
        counts = receipt.get("counts", {})
        if stage_id == "b1_basis_synthesis":
            if stage.get("decision") == "PASS":
                b1_passed = True
                bar("declared_families", DECLARED_FAMILIES_PHRASE, counts.get("ladder_entries"))
                bar(
                    "families_examined",
                    "families examined before acceptance",
                    counts.get("entries_examined"),
                )
                bar(
                    "families_rejected",
                    "families rejected on the way",
                    counts.get("entries_rejected_before_acceptance"),
                )
                examined = counts.get("entries_examined")
                rejected = counts.get("entries_rejected_before_acceptance")
                accepted = (
                    examined - rejected
                    if isinstance(examined, int) and isinstance(rejected, int)
                    else None
                )
                bar("family_accepted", "family accepted", accepted, accent=True)
                result = receipt.get("result") or {}
                bar(
                    "holdout_confirmations",
                    "holdout confirmations of the accepted form",
                    result.get("confirmations"),
                )
                certificate = receipt.get("minimality_certificate") or {}
                reason_tables.append(
                    (
                        "b1_reason",
                        "Why the simpler families died",
                        _reason_breakdown(
                            certificate.get("strictly_simpler_entries_rejected"), "b1"
                        ),
                    )
                )
            else:
                bar(
                    "declared_families",
                    DECLARED_FAMILIES_PHRASE + " — all rejected",
                    counts.get("ladder_entries"),
                )
        elif stage_id == "b7_structural_repair":
            attempted = counts.get("strategies_attempted")
            strategy_word = (
                _plural(attempted, "repair strategy", "repair strategies")
                if isinstance(attempted, int)
                else "repair strategies"
            )
            bar(
                "repair_strategies",
                f"{strategy_word} attempted — all rejected",
                attempted,
            )
        elif stage_id == "b3_conjecture_generation" and not b1_passed:
            bar(
                "statement_kinds",
                "statement kinds considered",
                counts.get("declared_statement_kinds"),
            )
            proposed = counts.get("proposed")
            survived = counts.get("survived")
            proposed_word = (
                _plural(proposed, "statement", "statements")
                if isinstance(proposed, int)
                else "statements"
            )
            survived_word = (
                _plural(survived, "statement", "statements")
                if isinstance(survived, int)
                else "statements"
            )
            bar("proposed", f"{proposed_word} proposed from the prefix", proposed)
            bar("survived", f"{survived_word} surviving the holdout", survived)
        elif stage_id == "b2_nonlinear_coefficient_search" and stage.get("decision") == "PASS":
            bar(
                "ratio_models",
                "declared ratio-law models over the derived rows",
                counts.get("declared_models"),
            )
            bar(
                "ratio_models_rejected",
                "models rejected on the way",
                counts.get("models_rejected_before_acceptance"),
            )
            result = receipt.get("result") or {}
            bar(
                "ratio_model_accepted",
                "model accepted",
                1 if result.get("status") == "PASS" else None,
                accent=True,
            )
            bar(
                "ratio_confirmations",
                "holdout confirmations of the accepted law",
                result.get("confirmations"),
            )
            certificate = receipt.get("minimality_certificate") or {}
            reason_tables.append(
                (
                    "b2_reason",
                    "Why the simpler ratio models died",
                    _reason_breakdown(
                        certificate.get("strictly_simpler_models_rejected"), "b2"
                    ),
                )
            )
    if b1_passed:
        route = next(
            (
                route
                for route in phase.get("prover_routes", [])
                if route.get("route") == "b5_lemma_decomposition"
                and isinstance(route.get("receipt"), dict)
            ),
            None,
        )
        if route is not None:
            route_counts = route["receipt"].get("counts", {})
            obligations = route_counts.get("obligations")
            failing = route_counts.get("obligations_failing_exact_local_check")
            if isinstance(obligations, int) and isinstance(failing, int):
                discharged = obligations - failing
                bar(
                    "proof_obligations",
                    "proof obligations discharged",
                    discharged,
                    display=f"{discharged}/{obligations}",
                )
            else:
                footnotes.append(
                    "proof obligations discharged: stage counts not recorded in receipt"
                )
    else:
        bar(
            "principal_result",
            "principal result",
            1 if isinstance(phase.get("candidate"), dict) else None,
            accent=True,
        )
    return {"bars": bars, "reason_tables": reason_tables, "footnotes": footnotes}


def _funnel_svg(bars: list[dict[str, Any]]) -> str:
    top, row_pitch, bar_height, left, plot = 10, 46, 16, 16, 452
    max_count = max(entry["count"] for entry in bars)
    height = top + row_pitch * len(bars) + 4
    parts = [
        (
            f'<svg class="funnel" viewBox="0 0 640 {height}" role="img"'
            ' aria-label="Elimination funnel: one bar per recorded stage, widths linear in the'
            ' receipt counts">'
        )
    ]
    axis_bottom = top + row_pitch * len(bars) - 10
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + 4}" x2="{left}" y2="{axis_bottom}"></line>'
    )
    for index, entry in enumerate(bars):
        y = top + index * row_pitch
        width = max((entry["count"] * plot) // max_count, 1)
        css = "bar hit" if entry["accent"] else "bar"
        parts.append(
            f'<text class="lbl" x="{left + 4}" y="{y + 10}">'
            f"{_esc(entry['display'])} {_esc(entry['label'])}</text>"
        )
        parts.append(
            f'<rect class="{css}" x="{left}" y="{y + 15}" width="{width}"'
            f' height="{bar_height}" data-key="funnel_{_esc(entry["key"])}"'
            f' data-value="{_esc(entry["display"])}"></rect>'
        )
        parts.append(
            f'<line class="rule" x1="{left}" y1="{y + 15 + bar_height}" x2="{left + plot}"'
            f' y2="{y + 15 + bar_height}"></line>'
        )
        parts.append(
            f'<text class="num" x="{left + width + 8}" y="{y + 15 + 13}">'
            f"{_esc(entry['display'])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _funnel_figure(model: dict[str, Any]) -> str:
    caption = (
        "Every count is read from the sealed receipt trail, and bar widths are linear in the"
        " counts, narrowing from the declared families (each an infinite space of formulas). "
        + FUNNEL_FAMILY_NOTE
    )
    footnote_html = ""
    if model["footnotes"]:
        notes = "; ".join(sorted(set(model["footnotes"])))
        footnote_html = f' <span class="sub">[{_esc(notes)}]</span>'
    tables: list[str] = []
    for key_prefix, heading, breakdown in model["reason_tables"]:
        rows = "".join(
            "<tr>"
            f'<td class="mono">{_esc(reason)}</td>'
            f'<td class="num" data-key="funnel_{_esc(key_prefix)}_{_esc(reason)}"'
            f' data-value="{count}">{count}</td>'
            "</tr>"
            for reason, count in breakdown
        )
        tables.append(
            f"<h3>{_esc(heading)}</h3>"
            '<div class="scroll"><table><thead><tr><th>rejection reason, verbatim</th>'
            '<th class="num">count</th></tr></thead><tbody>' + rows + "</tbody></table></div>"
        )
    return (
        '<figure class="funnel-figure">'
        + _funnel_svg(model["bars"])
        + f"<figcaption>{_esc(caption)}{footnote_html}</figcaption></figure>"
        + "".join(tables)
    )


# ---------------------------------------------------------------------------
# Rediscovery papers: prose registry (words only — every number is receipt-read)
# ---------------------------------------------------------------------------

#: Hand-written prose per world: names and framing sentences, no quantities.
WORLD_PROSE: dict[str, dict[str, str]] = {
    "gauss_triangular": {
        "name": "the triangular numbers",
        "hidden": "the running totals of the counting numbers — the triangular numbers",
        "boundary": (
            "The schoolroom sum is among the oldest facts in arithmetic; finding it is"
            " evidence about the finder, not the found."
        ),
    },
    "square_pyramidal": {
        "name": "the square-pyramidal numbers",
        "hidden": "the totals of stacked squares — cannonball-pyramid counts",
        "boundary": (
            "The closed form has been known since antiquity; nothing about the mathematics"
            " is new here."
        ),
    },
    "nicomachus_cubes": {
        "name": "the sum of cubes",
        "hidden": "the running totals of the cubes",
        "boundary": (
            "That the cubes sum to a perfect square is a two-thousand-year-old theorem;"
            " the machine merely met it blind."
        ),
    },
    "hanoi_mersenne": {
        "name": "the Tower of Hanoi count",
        "hidden": "the minimal move counts of the Tower of Hanoi puzzle",
        "boundary": (
            "The doubling-less-one law is the textbook first example of a settled recurrence;"
            " rediscovery is the entire content of the result."
        ),
    },
    "handshake": {
        "name": "the handshake numbers",
        "hidden": "the handshake counts — how many pairs a room of people can form",
        "boundary": (
            "Counting pairs is elementary combinatorics; the point is the sealed protocol,"
            " not the theorem."
        ),
    },
    "hockey_stick_c3": {
        "name": "the tetrahedral numbers",
        "hidden": "the tetrahedral numbers — the count of ways to choose three things",
        "boundary": (
            "The binomial identity behind this world is classical; no combinatorial novelty"
            " is on offer."
        ),
    },
    "fibonacci": {
        "name": "the Fibonacci rule",
        "hidden": "the rabbit-problem numbers, each term the sum of the previous two",
        "boundary": (
            "The recurrence is eight centuries old, and the machine's refusal to name a"
            " closed form is a statement about its declared grammar, not about the"
            " mathematics."
        ),
        "abstain": (
            "The true closed form here is Binet's: a growth law built from the square root"
            " of five, an irrational number that no family on the declared rational ladder"
            " can express — so declining to claim a closed form was the correct answer,"
            " not a failure."
        ),
    },
    "lucas": {
        "name": "the Lucas sequence",
        "hidden": "Lucas's companion sequence, the Fibonacci rule started from different doors",
        "boundary": (
            "The recurrence and its lineage are nineteenth-century classics; only the blind"
            " provenance is new."
        ),
        "abstain": (
            "The exact closed form again runs through the square root of five — the same"
            " irrational growth law as Binet's formula — which no rational-coefficient"
            " family on the declared ladder can express, so abstaining from a closed-form"
            " claim was correct."
        ),
    },
    "pell": {
        "name": "the Pell sequence",
        "hidden": "the Pell numbers, the arithmetic of approximating the square root of two",
        "boundary": (
            "Pell arithmetic long predates this machine; the sealed protocol is the only"
            " novelty."
        ),
        "abstain": (
            "The exact closed form is built from powers of one plus the square root of two"
            " — an irrational number outside every family on the declared rational ladder —"
            " so refusing to claim a closed form was the correct move."
        ),
    },
    "catalan_ratio": {
        "name": "the Catalan ratio",
        "hidden": "the Catalan numbers, which count balanced arrangements of many kinds",
        "boundary": (
            "The term-to-term ratio of the Catalan numbers is a classical identity; the"
            " derived-row route to it is declared in the public config, not improvised."
        ),
    },
    "geometric_sum3": {
        "name": "the geometric sum",
        "hidden": "the running totals of the powers of three",
        "boundary": (
            "Summing a geometric series is ancient; the receipt trail, not the theorem,"
            " is the exhibit."
        ),
    },
    "weighted_geometric": {
        "name": "the weighted geometric sum",
        "hidden": "the weighted doubling totals — each power of two counted with its weight",
        "boundary": (
            "The arithmetico-geometric summation identity is classical textbook material;"
            " nothing new is claimed about it."
        ),
    },
}

_KIND_WORDS = {
    "closed_form": "closed form",
    "linear_recurrence": "linear recurrence",
    "ratio_law": "term-to-term ratio law",
}


def _world_number(world_id: str) -> int:
    match = re.search(r"(\d+)$", world_id)
    if match is None:
        raise SiteGenerationError(f"world id outside the receipt grammar: {world_id!r}")
    return int(match.group(1))


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _data_value(key: str, value: Any, shown: str | None = None) -> str:
    display = shown if shown is not None else _fmt_int(value) if isinstance(value, int) else value
    return (
        f'<span data-key="{_esc(key)}" data-value="{_esc(value)}">' f"{_esc(display)}</span>"
    )


def _families_explainer(
    ladder_count: int | None, billion: Artifact, heading_tag: str
) -> str:
    """The families-not-formulas explainer, shared by every paper and the index."""

    if ladder_count is not None:
        heading = (
            f"Why {_data_value('ladder_families', ladder_count)} families and not a billion"
            " formulas"
        )
    else:
        heading = "Why declared families and not a billion formulas"
    if billion.present:
        processed = billion.data["counts"]["processed"]
        contrast = (
            "The contrast is the gravity screen elsewhere in this logbook, which enumerates"
            f" all {_data_value('gravity_contrast_processed', processed)} of its candidate"
            " laws one by one precisely because those laws are nonlinear in their unknowns"
            " and admit no per-family solve: solve what you can, enumerate what you cannot."
        )
    else:
        contrast = (
            "The contrast is the gravity screen elsewhere in this logbook, which enumerates"
            " its whole declared family member by member precisely because those laws are"
            " nonlinear in their unknowns and admit no per-family solve: solve what you can,"
            " enumerate what you cannot."
        )
    count_phrase = (
        _data_value("ladder_families_body", ladder_count) + " declared families"
        if ladder_count is not None
        else "its declared families"
    )
    return (
        f"<{heading_tag}>{heading}</{heading_tag}>"
        "<p>Every family on the declared ladder is linear in its unknown coefficients, so no"
        " family is searched by trying formulas one at a time: a single exact linear solve"
        " either pins down the one member that fits the rows or proves that no member of that"
        " family can. That is why the funnel counts"
        f" {count_phrase} rather than individual formulas — each family is"
        " an infinite space of formulas, searched perfectly in one step. The ladder itself is"
        " a hand-declared, frozen list ordered by simplicity, which makes &ldquo;simplest"
        " surviving family&rdquo; a checkable claim relative to the declaration rather than a"
        " matter of taste. Anything outside the ladder is invisible to this search, and when"
        " the rows demand such a shape the system emits a typed blocker instead of an answer"
        " — the abstention worlds in this campaign are exactly that, receipted."
        f" {contrast}</p>"
    )


# ---------------------------------------------------------------------------
# Rediscovery papers: page assembly
# ---------------------------------------------------------------------------


def _paper_status_banner() -> str:
    return _status_banner(
        "REDISCOVERED",
        (
            "Nothing here is unsolved: this question was settled long ago, and the write-up"
            " says so in its title. The machine's contribution is provenance — it reached the"
            " classical answer unaided, from anonymized rows, under a sealed blind protocol —"
            " and every number below is read from the campaign's receipts at build time."
        ),
    )


def _paper_rows_section(data: dict[str, Any], alias: str) -> str:
    rows = data.get("public_rows")
    parts: list[str] = []
    parts.append(
        "<p>Strip the story away and this table is the entire Phase A input for this world:"
        " an opaque world id, values under the neutral alias"
        f" <code>{_esc(alias)}</code>, and nothing else — no name, no source, no hint.</p>"
    )
    if not isinstance(rows, list):
        parts.append(
            f'<div class="missing"><strong>{_esc(MISSING_NOTE)}.</strong> The anonymized rows'
            " are rendered from the world receipt, which does not record them.</div>"
        )
        return "\n".join(parts)
    body_rows = "".join(
        "<tr>"
        f'<td class="num">{_esc(row.get("point"))}</td>'
        f'<td class="num">{_esc(row.get("value"))}</td>'
        "</tr>"
        for row in rows
    )
    parts.append(
        '<div class="scroll"><table><thead><tr><th class="num">point</th>'
        f'<th class="num">{_esc(alias)}(point)</th></tr></thead>'
        f"<tbody>{body_rows}</tbody></table></div>"
    )
    parts.append(f'<p class="small sub">{_esc(ROWS_CAPTION)}</p>')
    transformations = data.get("declared_transformations") or []
    for transformation in transformations:
        parts.append(
            "<p>One derived-row construction was declared in the public config before the"
            " run and logged in the receipt, verbatim:"
            f" <code>{_esc(transformation.get('statement', ''))}</code> (transformation id"
            f" <code>{_esc(transformation.get('transformation_id', ''))}</code>). The blind"
            " stages saw those derived rows under the same seal discipline.</p>"
        )
    return "\n".join(parts)


def _paper_methods_section(
    campaign: dict[str, Any], data: dict[str, Any], billion: Artifact
) -> str:
    chronology = campaign.get("chronology", {})
    probe = chronology.get("denied_probe", {})
    claims = campaign.get("claims", {})
    counts = campaign.get("counts", {})
    attempted = probe.get("attempted_target_reads")
    denied = probe.get("denied_target_reads")
    exposed = probe.get("denied_content_bytes_exposed")
    before_freeze = claims.get("target_records_read_before_candidate_freeze")
    batches = chronology.get("unseal_batches")
    post_unseal = counts.get("post_unseal_generation_events")
    root = data.get("phase_a_root") or chronology.get("phase_a_root") or ""
    parts: list[str] = []
    sentences: list[str] = []
    sentences.append(
        "Before anything was searched, the anonymized rows, the frozen stage ladder, and the"
        " full Phase A record for every world were committed under a single root hash,"
        f" <code>{_esc(str(root)[:16])}&hellip;</code>, while the classical answers sat in a"
        " separate sealed fixture."
    )
    if isinstance(attempted, int) and isinstance(denied, int) and isinstance(exposed, int):
        sentences.append(
            "During the blind phase the campaign deliberately attempted"
            f" {_data_value('chron_attempted_reads', attempted)} in-process"
            f" {_plural(attempted, 'read', 'reads')} of that fixture, and"
            f" {_data_value('chron_denied_reads', denied)}"
            f" {_plural(denied, 'was', 'were')} denied with"
            f" {_data_value('chron_denied_bytes', exposed)} bytes of content exposed."
        )
        sentences.append(
            "The receipt states the enforcement's own boundary: it covers this process's"
            " Python file-read surfaces, and is not an operating-system sandbox."
        )
    else:
        sentences.append(
            f"[{_esc(MISSING_NOTE)}: the campaign receipt does not record the denied-probe"
            " counts.]"
        )
    if isinstance(before_freeze, int):
        sentences.append(
            f"Exactly {_data_value('chron_reads_before_freeze', before_freeze)} target"
            f" {_plural(before_freeze, 'record was', 'records were')} read before the"
            " candidate freeze."
        )
    if isinstance(batches, int):
        sentences.append(
            f"The seal opened {_data_value('chron_unseal_batches', batches)}"
            f" {_plural(batches, 'time', 'times')}, in one atomic batch, after which the"
            " discovered and classical statements were compared by exact symbolic"
            " equivalence."
        )
    if isinstance(post_unseal, int):
        sentences.append(
            f"And {_data_value('chron_post_unseal', post_unseal)} post-unseal candidate-"
            f"generation {_plural(post_unseal, 'event is', 'events are')} recorded: nothing"
            " was generated, tuned, or repaired after the answers became visible."
        )
    parts.append("<p>" + " ".join(sentences) + "</p>")
    b1_counts = {}
    for stage in data.get("phase_a", {}).get("stages", []):
        if stage.get("stage_id") == "b1_basis_synthesis":
            b1_counts = (stage.get("receipt") or {}).get("counts", {})
            break
    ladder = b1_counts.get("ladder_entries")
    parts.append(
        _families_explainer(ladder if isinstance(ladder, int) else None, billion, "h3")
    )
    return "\n".join(parts)


def _paper_abstention_box(data: dict[str, Any], prose: dict[str, str]) -> str:
    """The honest-abstention sidebar, rendered only when the trail records it."""

    phase = data.get("phase_a", {})
    stages = {stage.get("stage_id"): stage for stage in phase.get("stages", [])}
    b1 = stages.get("b1_basis_synthesis")
    b7 = stages.get("b7_structural_repair")
    b3 = stages.get("b3_conjecture_generation")
    if b1 is None or b7 is None or b3 is None:
        return ""
    if b1.get("decision") != "BLOCK" or b7.get("decision") != "BLOCK":
        return ""
    if any(stage.get("decision") == "PASS" for stage in phase.get("stages", [])):
        return ""  # some later declared route found an exact law — that is not abstention
    closed_form = next(
        (
            conjecture
            for conjecture in (b3.get("receipt") or {}).get("conjectures", [])
            if conjecture.get("kind") == "closed_form"
        ),
        None,
    )
    if closed_form is None or closed_form.get("status") != "NOT_PROPOSED":
        return ""
    b1_counts = (b1.get("receipt") or {}).get("counts", {})
    strategies = (b7.get("receipt") or {}).get("rejected_strategies", [])
    strategy_items = "".join(
        f"<li><code>{_esc(strategy.get('strategy'))}</code> &mdash;"
        f" <code>{_esc(strategy.get('status'))}</code>, reason"
        f" <code>{_esc(strategy.get('reason'))}</code></li>"
        for strategy in strategies
    )
    abstain_sentence = prose.get("abstain", "")
    return (
        '<div class="aside-box"><p class="hd">An honest abstention</p>'
        "<p>This world's trail is a refusal, quoted from the receipts: every one of the"
        f" {_data_value('abstain_families', b1_counts.get('ladder_entries'))} declared basis"
        " families was rejected (stage <code>b1_basis_synthesis</code>, decision"
        f" <code>{_esc(b1.get('decision'))}</code>, first blocker"
        f" <code>{_esc((b1.get('receipt') or {}).get('first_blocker'))}</code>); every"
        " declared repair was rejected too (stage <code>b7_structural_repair</code>,"
        f" decision <code>{_esc(b7.get('decision'))}</code>):</p>"
        f"<ul>{strategy_items}</ul>"
        "<p>And the conjecture stage declined to guess: kind <code>closed_form</code> is"
        f" recorded as <code>{_esc(closed_form.get('status'))}</code>, reason"
        f" <code>{_esc(closed_form.get('reason'))}</code>."
        f" {_esc(abstain_sentence)}</p></div>"
    )


def _paper_killed_by_row_box(data: dict[str, Any]) -> str:
    """The killed-by-one-row sidebar — only if the trail records a refuted recurrence."""

    for stage in data.get("phase_a", {}).get("stages", []):
        if stage.get("stage_id") != "b3_conjecture_generation":
            continue
        for conjecture in (stage.get("receipt") or {}).get("conjectures", []):
            if (
                conjecture.get("kind") == "linear_recurrence"
                and conjecture.get("status") == "REFUTED"
                and conjecture.get("refutation_witness") is not None
            ):
                witness = conjecture.get("refutation_witness")
                return (
                    '<div class="aside-box"><p class="hd">Killed by one row</p>'
                    "<p>The trail records a recurrence that fit the visible prefix and died"
                    " on held-out data, verbatim:"
                    f" <code>{_esc(conjecture.get('statement'))}</code>, status"
                    f" <code>{_esc(conjecture.get('status'))}</code>, refutation witness"
                    f" <code>{_esc(json.dumps(witness, sort_keys=True))}</code>. One row was"
                    " enough; no partial credit is awarded anywhere in the pipeline.</p>"
                    "</div>"
                )
    return ""


def _b3_survivors(data: dict[str, Any]) -> list[dict[str, Any]]:
    for stage in data.get("phase_a", {}).get("stages", []):
        if stage.get("stage_id") == "b3_conjecture_generation":
            return [
                conjecture
                for conjecture in (stage.get("receipt") or {}).get("conjectures", [])
                if conjecture.get("status") == "SURVIVED"
            ]
    return []


def _paper_survivors_section(data: dict[str, Any]) -> str:
    candidate = data.get("phase_a", {}).get("candidate") or {}
    principal_statement = candidate.get("statement")
    survivors = _b3_survivors(data)
    alternates = [
        conjecture
        for conjecture in survivors
        if conjecture.get("statement") != principal_statement
    ]
    parts: list[str] = []
    if not alternates:
        parts.append(
            "<p>Exactly one statement survived the holdout in this world — the principal"
            " result above. Nothing else was proposed and confirmed, so there is no"
            " alternates ledger to show.</p>"
        )
        return "\n".join(parts)
    parts.append(
        "<p>The conjecture stage proposed from a prefix of the rows and attacked every"
        " proposal on held-out rows. Besides the principal result, these statements"
        " survived, listed with their holdout support:</p>"
    )
    items = []
    for conjecture in alternates:
        support = conjecture.get("support")
        support_html = (
            _data_value(f"alt_support_{conjecture.get('kind')}", support)
            if isinstance(support, int)
            else "&mdash;"
        )
        items.append(
            "<li>"
            + _statement_html(str(conjecture.get("statement")))
            + f'<span class="small sub"> &mdash; kind <code>{_esc(conjecture.get("kind"))}'
            f"</code>, {support_html} holdout confirmations</span></li>"
        )
    parts.append("<ul>" + "".join(items) + "</ul>")
    principal_twin = next(
        (
            conjecture
            for conjecture in survivors
            if conjecture.get("statement") == principal_statement
        ),
        None,
    )
    specificity_note = ""
    if principal_twin is not None:
        specificity = principal_twin.get("specificity", {})
        specificity_note = (
            " the receipt records its specificity as"
            f" <code>{_esc(specificity.get('numerator'))}/"
            f"{_esc(specificity.get('denominator'))}</code> with"
            f" <code>{_esc(principal_twin.get('parameters'))}</code> confirmed parameters,"
            " the top of this world's list;"
        )
    parts.append(
        "<p>The principal result is principal because it is the most specific survivor —"
        + specificity_note
        + " an exact defining law pins down every value, while sign, monotonicity, and"
        " congruence facts each admit many sequences. The selection rule was fixed before"
        " the run, verbatim:"
        f" <code>{_esc(data.get('phase_a', {}).get('candidate_selection_rule'))}</code>.</p>"
    )
    return "\n".join(parts)


def _paper_result_section(data: dict[str, Any], summary: dict[str, Any]) -> str:
    candidate = data.get("phase_a", {}).get("candidate") or {}
    unseal = data.get("unseal", {})
    comparison = unseal.get("comparison", {})
    discovered = str(summary.get("discovered_statement"))
    target = str(summary.get("target_statement"))
    parts: list[str] = []
    kind_word = _KIND_WORDS.get(str(candidate.get("kind")), "statement")
    parts.append(
        f"<p>Side by side: the {_esc(kind_word)} the machine wrote down blind, and the"
        " classical form that sat in the sealed fixture.</p>"
    )
    parts.append(
        '<div class="duo">'
        '<div class="half"><div class="hd">Discovered, blind</div>'
        + _statement_html(discovered)
        + '</div><div class="half"><div class="hd">The classical form</div>'
        + _statement_html(target)
        + "</div></div>"
    )
    detail = comparison.get("detail", {})
    residual = detail.get("residual")
    residual_html = (
        " with residual " + _data_value("result_residual", residual)
        if isinstance(residual, str)
        else ""
    )
    parts.append(
        "<p>The receipt's comparison method, verbatim:"
        f" <code>{_esc(comparison.get('method'))}</code> &mdash; equivalent:"
        f" <code>{_esc(comparison.get('equivalent'))}</code>{residual_html}."
        f" Verdict, verbatim: <code>{_esc(summary.get('verdict'))}</code>.</p>"
    )
    parts.append(
        f"<p><strong>Attribution.</strong> {_esc(summary.get('attribution'))}</p>"
    )
    return "\n".join(parts)


def _paper_verification_section(
    data: dict[str, Any], summary: dict[str, Any]
) -> str:
    parts: list[str] = []
    holdout = summary.get("holdout_confirmations")
    if isinstance(holdout, int):
        parts.append(
            "<p>The principal statement was confirmed on"
            f" {_data_value('verify_holdout', holdout)} held-out"
            f" {_plural(holdout, 'row', 'rows')} that no proposing stage ever saw. Survival"
            " is never proof; the holdout is a falsifier that happened to find nothing.</p>"
        )
    routes = [
        route
        for route in data.get("phase_a", {}).get("prover_routes", [])
        if isinstance(route.get("receipt"), dict) and route.get("lean_source_emitted")
    ]
    if not routes:
        parts.append(
            "<p>No proof-kernel source was emitted for this world: the receipt's prover"
            " trail records no applicable route, and the campaign receipt counts this world"
            " among the exact rediscoveries without proof. That boundary is stated rather"
            " than papered over.</p>"
        )
        return "\n".join(parts)
    parts.append(
        "<p>The prover emitted the Lean source below, rendered verbatim from the receipt."
        " Each file was checked exactly, locally, in rational arithmetic; independent"
        " kernel verification pending CI, and <code>kernel_verified</code> is"
        " <code>false</code> in every receipt on this page.</p>"
    )
    requirement = routes[0]["receipt"].get("kernel_verification_requirement")
    if requirement:
        parts.append(f"<blockquote>{_esc(requirement)}</blockquote>")
    for route in routes:
        receipt = route["receipt"]
        counts = receipt.get("counts", {})
        obligations = counts.get("obligations")
        failing = counts.get("obligations_failing_exact_local_check")
        obligation_note = ""
        if isinstance(obligations, int) and isinstance(failing, int):
            discharged = obligations - failing
            obligation_note = (
                " &mdash; "
                + _data_value("lean_obligations", f"{discharged}/{obligations}")
                + " obligations discharged in the exact local check"
            )
        parts.append(
            f"<h3>Route <code>{_esc(route.get('route'))}</code>, decision"
            f" <code>{_esc(route.get('decision'))}</code>{obligation_note}</h3>"
        )
        survivor = route.get("survivor_statement")
        if survivor:
            parts.append(
                f"<p class=\"small sub\">Proves the survivor statement"
                f" <code>{_esc(survivor)}</code>.</p>"
            )
        note = route.get("note")
        if note:
            parts.append(f'<p class="small sub">Receipt note, verbatim: {_esc(note)}.</p>')
        parts.append(f"<pre><code>{_esc(receipt.get('lean_source', ''))}</code></pre>")
    return "\n".join(parts)


def _paper_not_shown_section(
    summary: dict[str, Any], prose: dict[str, str], campaign: dict[str, Any]
) -> str:
    claims = campaign.get("claims", {})
    items = [
        f"<li>{_esc(prose.get('boundary', ''))}</li>",
        (
            "<li>No novelty of any kind is claimed. The campaign receipt records"
            f" <code>novelty_claimed: {_esc(claims.get('novelty_claimed'))}</code> and"
            " <code>rediscovery_of_classical_results:"
            f" {_esc(claims.get('rediscovery_of_classical_results'))}</code>, and the house"
            " rules forbid treating corpus absence as novelty.</li>"
        ),
        (
            "<li>Survival on held-out rows is never proof, and the exact local check of any"
            " emitted Lean is never kernel verification; the campaign receipt records"
            f" <code>kernel_verified_lean: {_esc(claims.get('kernel_verified_lean'))}</code>"
            " pending the CI kernel run.</li>"
        ),
        (
            "<li>The blind enforcement guards this process's own file-read surfaces; the"
            " receipt names its scope verbatim and does not claim an operating-system"
            " sandbox.</li>"
        ),
    ]
    note = summary.get("note")
    if note:
        items.append(
            f"<li>Receipt note, verbatim: <code>{_esc(note)}</code>.</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _paper_references_section(
    artifact: Artifact, dozen: Artifact, summary: dict[str, Any], campaign: dict[str, Any]
) -> str:
    parts: list[str] = ["<h3>Receipts rendered on this page</h3>"]
    items: list[str] = []
    if artifact.present:
        items.append(
            f"<li>{_github(artifact.path)} &mdash; seal {_sha_abbrev(artifact.sha256)}</li>"
        )
    else:
        items.append(f"<li><code>{_esc(artifact.path)}</code> &mdash; {_esc(MISSING_NOTE)}</li>")
    if dozen.present:
        items.append(
            f"<li>{_github(dozen.path)} &mdash; campaign receipt, seal"
            f" {_sha_abbrev(dozen.sha256)}</li>"
        )
    fixture = campaign.get("source_bindings", {}).get("target_fixture", {})
    if fixture.get("path"):
        items.append(
            f"<li>{_github(str(fixture['path']))} &mdash; the sealed target fixture the"
            f" blind phase was denied from reading, file hash"
            f" {_sha_abbrev(fixture.get('file_sha256'))}&dagger;</li>"
        )
    parts.append("<ul>" + "".join(items) + "</ul>")
    parts.append("<h3>Classical citation</h3>")
    parts.append(f"<p>{_esc(summary.get('attribution'))}</p>")
    parts.append(
        '<p class="small sub">&dagger; file-byte hash (the fixture is not a sealed JSON'
        " receipt).</p>"
    )
    return "\n".join(parts)


def _paper_world_page(
    summary: dict[str, Any],
    artifact: Artifact,
    dozen: Artifact,
    artifacts: dict[str, Artifact],
    commit: str,
) -> bytes:
    campaign = dozen.data if dozen.present else {}
    classical_id = str(summary.get("classical_id"))
    prose = WORLD_PROSE.get(classical_id, {"name": classical_id, "hidden": classical_id})
    number = _world_number(str(summary.get("world_id", "")))
    alias = str(campaign.get("sequence_alias", "a"))
    body: list[str] = []
    body.append('<p class="kicker">Blinded rediscovery papers</p>')
    body.append(f"<h1>Blind rediscovery {number}: {_esc(prose['name'])}</h1>")
    body.append('<p class="byline">The Invariant Project</p>')
    body.append(
        f'<p class="sub small">content as of <code>{_esc(commit)}</code> &mdash; dated by'
        " content, not by clock; every number on this page is read from the campaign's"
        " sealed receipts at build time.</p>"
    )
    body.append(_paper_status_banner())
    body.append(_status_key())
    holdout = summary.get("holdout_confirmations")
    data = artifact.data if artifact.present else {}
    rows = data.get("public_rows") if artifact.present else None
    row_count_html = (
        _data_value("paper_rows", len(rows)) if isinstance(rows, list) else "a table of"
    )
    body.append("<h2>Abstract</h2>")
    abstract: list[str] = []
    abstract.append(
        f"Hidden behind this world's opaque label was {_esc(prose['hidden'])}."
    )
    abstract.append(
        f"The machine saw {row_count_html} anonymized value pairs under the neutral alias"
        f" <code>{_esc(alias)}</code> — no name, no story, no hint of a source."
    )
    kind_word = _KIND_WORDS.get(
        str((data.get("phase_a", {}).get("candidate") or {}).get("kind")), "exact statement"
    )
    if isinstance(holdout, int):
        abstract.append(
            f"Working blind, it recovered the exact {_esc(kind_word)} and confirmed it on"
            f" {_data_value('paper_holdout', holdout)} held-out"
            f" {_plural(holdout, 'row', 'rows')} it had never used."
        )
    else:
        abstract.append(f"Working blind, it recovered the exact {_esc(kind_word)}.")
    abstract.append(
        "After the single unseal the discovered statement matched the classical answer"
        f" exactly: {_esc(summary.get('attribution'))}"
    )
    if summary.get("lean_emitted"):
        abstract.append(
            "A proof-kernel source was emitted for the result; its independent kernel check"
            " is pending CI."
        )
    elif "abstain" in prose:
        abstract.append(
            "It also declined, correctly, to claim a closed form that would have required"
            " irrational numbers outside its declared grammar."
        )
    abstract.append(_esc(REDISCOVERY_SENTENCE))
    body.append("<p>" + " ".join(abstract) + "</p>")
    body.append("<h2>The question</h2>")
    if artifact.present:
        body.append(_paper_rows_section(data, alias))
    else:
        body.append(_missing_block(artifact, "anonymized rows"))
    body.append("<h2>Methods</h2>")
    if dozen.present and artifact.present:
        body.append(_paper_methods_section(campaign, data, artifacts["billion"]))
    elif dozen.present:
        body.append(_paper_methods_section(campaign, {}, artifacts["billion"]))
    else:
        body.append(_missing_block(dozen, "sealed-blind chronology"))
    body.append("<h2>The elimination funnel</h2>")
    if artifact.present:
        model = _funnel_model(data)
        if model["bars"]:
            body.append(
                "<p>The path below is this world's actual narrowing, stage by stage,"
                " derived only from the receipt trail — no stage is drawn that the receipt"
                " does not record.</p>"
            )
            body.append(_funnel_figure(model))
        else:
            body.append(
                f'<p class="sub">[{_esc(MISSING_NOTE)}: the receipt trail records no stage'
                " counts, so no funnel is drawn]</p>"
            )
        body.append(_paper_abstention_box(data, prose))
        body.append(_paper_killed_by_row_box(data))
    else:
        body.append(_missing_block(artifact, "elimination funnel"))
    body.append("<h2>What else survived</h2>")
    if artifact.present:
        body.append(_paper_survivors_section(data))
    else:
        body.append(_missing_block(artifact, "surviving statements"))
    body.append("<h2>Result</h2>")
    if artifact.present:
        body.append(_paper_result_section(data, summary))
    else:
        body.append(_missing_block(artifact, "result comparison"))
    body.append("<h2>Verification</h2>")
    if artifact.present:
        body.append(_paper_verification_section(data, summary))
    else:
        body.append(_missing_block(artifact, "verification"))
    body.append("<h2>What this does not show</h2>")
    body.append(_paper_not_shown_section(summary, prose, campaign))
    body.append("<h2>References</h2>")
    body.append(_paper_references_section(artifact, dozen, summary, campaign))
    body.append('<p class="small"><a href="/papers">&larr; back to all twelve papers</a></p>')
    title = f"Blind rediscovery {number}: {prose['name']} · {SITE_TITLE}"
    return _page("papers", title, "\n".join(body), commit)


def _papers_progress_block(unsolved: Artifact) -> str:
    parts: list[str] = ["<h2>Progress reports (unsolved dozen)</h2>"]
    if not unsolved.present:
        parts.append(
            f'<div class="missing"><strong>{_esc(MISSING_NOTE)}.</strong> A companion'
            " campaign over questions without known answers is planned at"
            f" <code>{_esc(UNSOLVED_CAMPAIGN_PATH)}</code>; that campaign not yet published"
            " in this repository. When its receipts land, progress reports — including"
            " honest failures — will be listed here with the same ceremony as the papers"
            " above.</div>"
        )
        return "\n".join(parts)
    parts.append(
        "<p>The companion unsolved-dozen campaign has published receipts. Full progress-"
        "report pages are not built yet; the evidence is linked directly:</p>"
    )
    worlds = unsolved.data.get("world_results")
    items: list[str] = []
    if isinstance(worlds, list) and worlds:
        for world in worlds:
            label = str(
                world.get("classical_id") or world.get("world_id") or "unnamed world"
            )
            receipt_path = world.get("world_receipt_path")
            if isinstance(receipt_path, str) and receipt_path:
                items.append(f"<li><code>{_esc(label)}</code>: {_github(receipt_path)}</li>")
            else:
                items.append(f"<li><code>{_esc(label)}</code></li>")
    items.append(
        f"<li>Campaign receipt: {_github(unsolved.path)} &mdash; seal"
        f" {_sha_abbrev(unsolved.sha256)}</li>"
    )
    parts.append("<ul>" + "".join(items) + "</ul>")
    return "\n".join(parts)


def _papers_index_page(artifacts: dict[str, Artifact], commit: str) -> bytes:
    dozen = artifacts["dozen"]
    body: list[str] = []
    body.append('<p class="kicker">Blinded rediscovery papers</p>')
    body.append("<h1>Twelve old truths, found blind</h1>")
    body.append(
        "<p>Each page below is a short publication-style write-up of one blinded"
        " rediscovery: a classical, long-settled result was hidden behind an opaque label,"
        " the machine saw only anonymized rows, and the sealed protocol opened exactly once"
        " at the end to compare answers. The centerpiece of every paper is its elimination"
        " funnel — the world's actual narrowing path, drawn from the receipt trail and"
        " nothing else. Nothing here is new mathematics, every page says so, and the"
        " abstentions are written up with the same ceremony as the finds.</p>"
    )
    if not dozen.present:
        body.append(_missing_block(dozen, "rediscovery campaign"))
        body.append(_papers_progress_block(artifacts["unsolved"]))
        return _page("papers", f"Papers · {SITE_TITLE}", "\n".join(body), commit)
    campaign = dozen.data
    worlds = campaign.get("world_results", [])
    ladder = None
    for world in worlds:
        artifact = artifacts.get(DOZEN_WORLD_PREFIX + str(world.get("classical_id")))
        if artifact is not None and artifact.present:
            for stage in artifact.data.get("phase_a", {}).get("stages", []):
                if stage.get("stage_id") == "b1_basis_synthesis":
                    candidate = (stage.get("receipt") or {}).get("counts", {}).get(
                        "ladder_entries"
                    )
                    if isinstance(candidate, int):
                        ladder = candidate
                    break
        if ladder is not None:
            break
    body.append(_families_explainer(ladder, artifacts["billion"], "h2"))
    counts = campaign.get("counts", {})
    tiles = [
        _tile(
            _esc(_fmt_int(counts.get("worlds", 0))),
            "sealed worlds in the campaign",
            dozen.path,
            data_key="papers_worlds",
            data_value=str(counts.get("worlds", 0)),
        ),
        _tile(
            _esc(_fmt_int(counts.get("rediscovered_total", 0))),
            "classical results rediscovered under the blind",
            dozen.path,
            data_key="papers_rediscovered",
            data_value=str(counts.get("rediscovered_total", 0)),
        ),
        _tile(
            _esc(_fmt_int(counts.get("rediscovered_with_proof", 0))),
            "with an emitted proof-kernel source (kernel check pending CI)",
            dozen.path,
            data_key="papers_with_proof",
            data_value=str(counts.get("rediscovered_with_proof", 0)),
        ),
        _tile(
            _esc(_fmt_int(counts.get("holdout_confirmations_total", 0))),
            "holdout confirmations across all twelve principal results",
            dozen.path,
            data_key="papers_holdout_total",
            data_value=str(counts.get("holdout_confirmations_total", 0)),
        ),
        _tile(
            _esc(_fmt_int(counts.get("missed", 0))),
            "worlds missed — the count the protocol exists to keep honest",
            dozen.path,
            kind="neg",
            data_key="papers_missed",
            data_value=str(counts.get("missed", 0)),
        ),
    ]
    body.append('<div class="tiles">' + "".join(tiles) + "</div>")
    body.append(
        f'<p class="small sub">Campaign decision, verbatim:'
        f" <code>{_esc(campaign.get('decision'))}</code> &middot; content seal"
        f" {_sha_abbrev(dozen.sha256)} &middot; "
        + _github(dozen.path, "view the campaign receipt on GitHub")
        + "</p>"
    )
    body.append("<h2>The ledger</h2>")
    rows: list[str] = []
    cards: list[str] = []
    for world in worlds:
        classical_id = str(world.get("classical_id"))
        prose = WORLD_PROSE.get(classical_id, {"name": classical_id})
        artifact = artifacts.get(DOZEN_WORLD_PREFIX + classical_id)
        one_liner = "&mdash;"
        if artifact is not None and artifact.present:
            model = _funnel_model(artifact.data)
            if model["bars"]:
                one_liner = _esc(
                    "→".join(entry["display"] for entry in model["bars"])
                )
        number = _world_number(str(world.get("world_id", "")))
        href = f"/papers/{classical_id}"
        rows.append(
            "<tr>"
            f'<td class="num">{number}</td>'
            f'<td class="mono"><a href="{_esc(href)}"><code>{_esc(classical_id)}</code></a></td>'
            f"<td>{_esc(prose['name'])}</td>"
            f'<td class="mono">{_esc(world.get("verdict"))}</td>'
            f'<td class="mono">{"yes" if world.get("lean_emitted") else "no"}</td>'
            f'<td class="mono">{one_liner}</td>'
            "</tr>"
        )
        cards.append(
            f'<a class="card-link" href="{_esc(href)}">'
            f'<div class="t">Blind rediscovery {number}: {_esc(prose["name"])}</div>'
            f'<div class="d">{_esc(world.get("verdict"))}</div>'
            f'<div class="f">{one_liner}</div></a>'
        )
    body.append(
        '<div class="scroll"><table><thead><tr><th class="num">#</th><th>world</th>'
        "<th>classical name</th><th>verdict</th><th>proof</th><th>funnel</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    body.append(
        '<p class="small sub">The funnel column is each paper\'s elimination funnel in one'
        " line — every number in it is read from that world's receipt trail; the papers"
        " draw the same numbers as labeled bars.</p>"
    )
    body.append("<h2>The papers</h2>")
    body.append('<div class="cards">' + "".join(cards) + "</div>")
    body.append("<h2>The claims block, verbatim</h2>")
    claims = campaign.get("claims", {})
    claim_rows = "".join(
        "<tr>"
        f'<td class="mono">{_esc(key)}</td>'
        f'<td class="mono">{_esc(claims[key])}</td>'
        "</tr>"
        for key in sorted(claims)
    )
    body.append(
        "<p>Copied key for key from the campaign receipt's <code>claims</code> block —"
        " the boundary travels with the evidence:</p>"
        '<div class="scroll"><table><thead><tr><th>claim key</th><th>value</th></tr>'
        f"</thead><tbody>{claim_rows}</tbody></table></div>"
    )
    body.append(_papers_progress_block(artifacts["unsolved"]))
    return _page("papers", f"Papers · {SITE_TITLE}", "\n".join(body), commit)


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
        "paper.html": _paper_page(artifacts, facts, commit),
        "problems.html": _problems_index_page(artifacts["queue"], artifacts, commit),
        "papers.html": _papers_index_page(artifacts, commit),
        "gravity.html": _gravity_page(artifacts, commit),
        "collatz.html": _collatz_page(artifacts, facts, commit),
        "evidence.html": _evidence_page(artifacts, commit),
        "submit.html": _submit_page(commit),
        "method.html": _method_page(facts, commit),
    }
    queue = artifacts["queue"]
    if queue.present:
        for entry in queue.data["entries"]:
            pages[f"problems/{entry['id']}.html"] = _problem_detail_page(
                entry, queue, artifacts, commit
            )
    dozen = artifacts["dozen"]
    if dozen.present:
        for world in dozen.data.get("world_results", []):
            classical_id = str(world.get("classical_id"))
            world_artifact = artifacts[DOZEN_WORLD_PREFIX + classical_id]
            pages[f"papers/{classical_id}.html"] = _paper_world_page(
                world, world_artifact, dozen, artifacts, commit
            )
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
