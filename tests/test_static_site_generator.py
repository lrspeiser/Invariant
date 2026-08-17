"""Public-site generator gates.

The site is only as honest as the checks that pin it to the repository's evidence.
The load-bearing tests here are the negative ones: the generator source must not
contain the headline numerals (they are read from receipts at build time), the
emitted HTML must contain no scalar-score vocabulary, failures must get equal
billing with their real numbers, and a missing optional receipt must produce an
explicit not-yet-published block rather than a crash or a silent omission.

The redesign adds equal-or-stronger gates: every problem page and case study must
carry exactly one status banner from the five-word vocabulary (with the vocabulary
defined on the same page), the Collatz page must open with the mandated OPEN text,
no page may contain a standalone "solved" claim, receipt formulas must round-trip
through the ASCII->LaTeX translator into build-time MathML, and the site must stay
fully self-contained (no external resources beyond documented link targets).

The rediscovery papers add their own gates: a paper exists for exactly the worlds
in the sealed campaign receipt, every elimination-funnel number equals the value
re-derived here from that world's receipt trail (via the data-key mechanism, with
one SVG bar per recorded stage), the abstention and killed-by-one-row sidebars
appear exactly where the trail supports them, the alternates section lists exactly
the trail's survivors, and both the discovered and classical statements render as
build-time MathML on every paper.
"""

from __future__ import annotations

import html as html_module
import json
import re
import shutil
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sigma_theory_compiler.problem_queue import ENTRY_KEYS, MACHINE_FORM_KINDS
from sigma_theory_compiler.static_site_generator import (
    ARTIFACT_PATHS,
    COLLATZ_STATUS_TEXT,
    DECLARED_FAMILIES_PHRASE,
    FOOTER_CREED,
    FUNNEL_FAMILY_NOTE,
    GARDNER_OPENING,
    HEAD_TO_HEAD_CAPTION,
    LATEX_CASE_VIEW_GRAMMAR,
    LATEX_RAR_LAW,
    LATEX_SIGMA_HALVING,
    LATEX_SIGMA_POW2,
    MISSING_NOTE,
    REDISCOVERY_SENTENCE,
    ROWS_CAPTION,
    STATUS_DEFINITIONS,
    SUBMIT_NOTICE,
    UNSOLVED_CAMPAIGN_PATH,
    SiteGenerationError,
    build_site,
    formula_ascii_to_latex,
    latex_to_mathml,
    main,
    render_site,
    statement_ascii_to_latex,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE = ROOT / "src" / "sigma_theory_compiler" / "static_site_generator.py"
TEST_COMMIT = "deadbeef" * 5

BANNED_SCORE_TERMS = ("truth score", "probability", "confidence", "% true", "likelihood")

CASE_STUDY_PAGES = (
    "collatz.html",
    "gravity.html",
    "case-studies/balmer-bohr.html",
)

#: The sealed rediscovery directory name contains the letters "solved"; the path is
#: a filename, not a claim, so the standalone-"solved" gate strips exactly this
#: substring (and nothing else) before scanning.
DOZEN_DIR = "runs/math/solved-dozen"

_TAGS = re.compile(r"<[^>]+>")


@pytest.fixture(scope="module")
def pages() -> dict[str, bytes]:
    return render_site(ROOT, TEST_COMMIT)


@pytest.fixture(scope="module")
def receipts() -> dict[str, dict]:
    loaded = {}
    for key in ("queue", "billion", "lensing", "sweep", "dozen", "case_study"):
        loaded[key] = json.loads((ROOT / ARTIFACT_PATHS[key]).read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def world_receipts(receipts) -> dict[str, dict]:
    loaded = {}
    for world in receipts["dozen"]["world_results"]:
        path = ROOT / world["world_receipt_path"]
        loaded[world["classical_id"]] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _text(pages: dict[str, bytes], name: str) -> str:
    return pages[name].decode("utf-8")


def _flat(page_text: str) -> str:
    """The rendered text content: tags stripped, entities resolved."""

    return html_module.unescape(_TAGS.sub("", page_text))


def _tile_values(page_text: str) -> dict[str, str]:
    return dict(re.findall(r'data-key="([^"]+)" data-value="([^"]+)"', page_text))


def _stamped_pages(pages: dict[str, bytes]) -> list[str]:
    """Every page that must carry exactly one status banner."""

    return sorted(
        name
        for name in pages
        if name.startswith(("problems/", "papers/")) or name in CASE_STUDY_PAGES
    )


# ---------------------------------------------------------------------------
# Determinism and page set
# ---------------------------------------------------------------------------


def test_two_builds_are_byte_identical(tmp_path):
    first = build_site(ROOT, tmp_path / "a", TEST_COMMIT)
    second = build_site(ROOT, tmp_path / "b", TEST_COMMIT)
    assert set(first) == set(second)
    for name in sorted(first):
        assert first[name] == second[name], name
        assert (tmp_path / "a" / name).read_bytes() == first[name], name


def test_page_set_covers_every_queue_entry_and_campaign_world(pages, receipts):
    expected = {
        "index.html",
        "paper.html",
        "problems.html",
        "papers.html",
        "gravity.html",
        "collatz.html",
        "evidence.html",
        "submit.html",
        "method.html",
        "case-studies.html",
        "case-studies/balmer-bohr.html",
    }
    expected |= {f"problems/{entry['id']}.html" for entry in receipts["queue"]["entries"]}
    expected |= {
        f"papers/{world['classical_id']}.html" for world in receipts["dozen"]["world_results"]
    }
    assert set(pages) == expected


# ---------------------------------------------------------------------------
# Hard rule 1: every quantitative headline is read from an artifact
# ---------------------------------------------------------------------------


def test_headline_numbers_are_read_from_receipts(pages, receipts):
    tiles = _tile_values(_text(pages, "index.html"))
    billion_counts = receipts["billion"]["counts"]
    lensing_counts = receipts["lensing"]["counts"]
    assert tiles["candidates_processed"] == str(billion_counts["processed"])
    assert tiles["lensing_pass"] == str(lensing_counts["lensing_pass"])
    assert tiles["cluster_pass"] == str(lensing_counts["cluster_pass"])
    assert tiles["sweep_hi"] == str(receipts["sweep"]["range"]["hi"])
    assert tiles["problem_count"] == str(len(receipts["queue"]["entries"]))
    index_text = _text(pages, "index.html")
    assert f"{billion_counts['processed']:,}" in index_text
    assert f"{lensing_counts['lensing_pass']:,}" in index_text


def test_sweep_range_and_decision_rendered(pages, receipts):
    collatz_text = _text(pages, "collatz.html")
    sweep = receipts["sweep"]
    tiles = _tile_values(collatz_text)
    assert tiles["sweep_hi"] == str(sweep["range"]["hi"])
    assert tiles["sweep_decision"] == sweep["decision"]
    assert tiles["sweep_checked"] == str(sweep["counts"]["checked"])
    assert tiles["sweep_undecided"] == str(sweep["undecided"]["count"])
    assert "NO_COUNTEREXAMPLE" in sweep["decision"]
    exponent = len(str(sweep["range"]["hi"])) - 1
    assert str(sweep["range"]["hi"]) == "1" + "0" * exponent
    assert f"10^{exponent}" in collatz_text
    assert html_module.escape(sweep["statement"]["text"], quote=True) in collatz_text


def test_generator_source_contains_no_headline_numerals(receipts, world_receipts):
    source = GENERATOR_SOURCE.read_text(encoding="utf-8")
    closest = receipts["lensing"]["cluster_negative"]["closest_cluster_approach"]
    banned = {
        str(receipts["billion"]["counts"]["processed"]),
        str(receipts["billion"]["counts"]["fp64_survivors"]),
        str(receipts["billion"]["throughput_candidates_per_second"]),
        receipts["billion"]["elapsed_seconds"],
        str(receipts["lensing"]["counts"]["lensing_pass"]),
        str(receipts["lensing"]["counts"]["fp32_union_survivors"]),
        str(receipts["lensing"]["throughput_candidates_per_second"]),
        str(receipts["sweep"]["range"]["hi"]),
        str(receipts["sweep"]["counts"]["checked"]),
        str(receipts["sweep"]["throughput_per_second"]),
        closest["max_deviation"],
        receipts["billion"]["content_sha256"],
        receipts["lensing"]["content_sha256"],
        receipts["sweep"]["content_sha256"],
        receipts["queue"]["content_sha256"],
        receipts["dozen"]["content_sha256"],
        receipts["dozen"]["chronology"]["phase_a_root"],
        receipts["case_study"]["content_sha256"],
        receipts["case_study"]["chronology"]["phase_a_root"],
        str(receipts["case_study"]["counts"]["total_declared_views"]),
        str(receipts["case_study"]["counts"]["views_rejected"]),
        receipts["case_study"]["blind_race"]["candidate"]["constant_decimal"],
        receipts["case_study"]["blind_race"]["candidate"]["relative_spread"],
        receipts["case_study"]["blind_race"]["unseal"][
            "holdout_max_relative_residual"
        ],
        receipts["case_study"]["derivation"]["rydberg_numerics"][
            "derived_rydberg_per_m"
        ],
        receipts["case_study"]["derivation"]["rydberg_numerics"][
            "relative_error_vs_measured"
        ],
        "0/21",
        "21/21",
        "70/70",
        "118/118",
        "200/200",
    }
    for world in receipts["dozen"]["world_results"]:
        banned.add(world["discovered_statement"])
        banned.add(world["target_statement"])
        banned.add(world["attribution"])
        banned.add(world["world_receipt_sha256"])
    for world_receipt in world_receipts.values():
        banned.add(world_receipt["content_sha256"])
    for numeral in sorted(banned):
        assert numeral not in source, f"generator source hard-codes {numeral!r}"


# ---------------------------------------------------------------------------
# Hard rule 2: no scalar-score vocabulary anywhere
# ---------------------------------------------------------------------------


def test_no_scalar_score_vocabulary(pages):
    for name in sorted(pages):
        lowered = _text(pages, name).lower()
        for term in BANNED_SCORE_TERMS:
            assert term not in lowered, (name, term)


# ---------------------------------------------------------------------------
# Hard rule 3: failures get equal billing, with their real numbers
# ---------------------------------------------------------------------------


def test_sealed_negative_is_the_headline_on_gravity(pages, receipts):
    gravity_text = _text(pages, "gravity.html")
    lensing = receipts["lensing"]
    assert lensing["decision"] in gravity_text
    closest = lensing["cluster_negative"]["closest_cluster_approach"]
    tolerance = lensing["config"]["cluster"]["fp64_thresholds"]["consistency"]
    assert closest["max_deviation"] in gravity_text
    assert closest["formula"] in gravity_text
    assert "sealed negative" in gravity_text.lower()
    tiles = _tile_values(gravity_text)
    assert tiles["cluster_pass"] == str(lensing["counts"]["cluster_pass"])
    assert tiles["closest_cluster_max_deviation"] == closest["max_deviation"]
    assert tiles["cluster_tolerance"] == tolerance
    quantum = Decimal(1).scaleb(-4)
    assert format(Decimal(closest["max_deviation"]).quantize(quantum), "f") in gravity_text
    assert format(Decimal(tolerance), "f") in gravity_text


def test_documented_failures_have_equal_billing(pages):
    index_text = _text(pages, "index.html")
    method_text = _text(pages, "method.html")
    for text in (index_text, method_text):
        assert "0/21" in text
        assert "70/70" in text
        assert "documented in repository docs" in text
        assert "INDEPENDENT_DISCOVERY_TRIAL.md" in text
    assert "21/21" in method_text
    assert "200/200" in method_text
    assert "118/118" in method_text
    assert "GOALS_AND_MEASURED_OUTCOMES.md" in method_text


def test_doc_numbers_match_the_docs(pages):
    method_text = _text(pages, "method.html")
    idt = (ROOT / ARTIFACT_PATHS["idt_doc"]).read_text(encoding="utf-8")
    goals = (ROOT / ARTIFACT_PATHS["goals_doc"]).read_text(encoding="utf-8")
    blind = re.search(r"Blind semantic formula guessing \| (\d+) PASS / (\d+) REJECT", idt)
    assert blind is not None
    passes, rejects = int(blind.group(1)), int(blind.group(2))
    assert f"{passes}/{passes + rejects}" in method_text
    rejections = re.search(r"(\d+)/(\d+) formal rejections", idt)
    assert rejections is not None
    assert f"{rejections.group(1)}/{rejections.group(2)}" in method_text
    conditioned = re.search(r"(\d+)/(\d+) candidates passed with exact certificates", goals)
    assert conditioned is not None
    assert f"{conditioned.group(1)}/{conditioned.group(2)}" in method_text
    holdout = re.search(r"(\d+) holdout confirmations", idt)
    assert holdout is not None
    assert f"{int(holdout.group(1)):,} holdout confirmations" in _text(pages, "collatz.html")


# ---------------------------------------------------------------------------
# Hard rule 4: fail-soft on missing optional artifacts
# ---------------------------------------------------------------------------


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ARTIFACT_PATHS.values():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return root


def test_missing_optional_receipts_fail_soft(tmp_path):
    root = _fixture_root(tmp_path)
    (root / ARTIFACT_PATHS["lensing"]).unlink()
    (root / ARTIFACT_PATHS["lean"]).unlink()
    (root / ARTIFACT_PATHS["idt_doc"]).unlink()
    pages = render_site(root, TEST_COMMIT)
    gravity_text = _text(pages, "gravity.html")
    assert MISSING_NOTE in gravity_text
    assert ARTIFACT_PATHS["lensing"] in gravity_text
    assert "Lensing and cluster campaign" in gravity_text
    collatz_text = _text(pages, "collatz.html")
    assert MISSING_NOTE in collatz_text
    assert ARTIFACT_PATHS["lean"] in collatz_text
    assert MISSING_NOTE in _text(pages, "method.html")
    assert MISSING_NOTE in _text(pages, "evidence.html")
    assert MISSING_NOTE in _text(pages, "paper.html")
    assert "index.html" in pages


def test_missing_queue_fails_soft_without_detail_pages(tmp_path):
    root = _fixture_root(tmp_path)
    (root / ARTIFACT_PATHS["queue"]).unlink()
    pages = render_site(root, TEST_COMMIT)
    assert MISSING_NOTE in _text(pages, "problems.html")
    assert not any(name.startswith("problems/") for name in pages)


# ---------------------------------------------------------------------------
# HTML integrity: parseable, self-contained, links resolve
# ---------------------------------------------------------------------------

_VOID_TAGS = {"meta", "br", "hr", "img", "input", "link", "wbr", "col", "source"}


class _AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.start_count = 0
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.link_tags = 0
        self.script_tags = 0

    def handle_starttag(self, tag, attrs):
        self.start_count += 1
        attributes = dict(attrs)
        if tag == "link":
            self.link_tags += 1
        if tag == "script":
            self.script_tags += 1
        if "href" in attributes and attributes["href"] is not None:
            self.hrefs.append(attributes["href"])
        if "src" in attributes and attributes["src"] is not None:
            self.srcs.append(attributes["src"])
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> near {self.stack[-3:]}")
            return
        self.stack.pop()


def _audit(page_text: str) -> _AuditParser:
    parser = _AuditParser()
    parser.feed(page_text)
    parser.close()
    return parser


def test_every_page_parses_as_html(pages):
    for name in sorted(pages):
        parser = _audit(_text(pages, name))
        assert parser.errors == [], (name, parser.errors)
        assert parser.stack == [], (name, parser.stack)
        assert parser.start_count > 20, name


def test_internal_links_resolve_and_pages_are_self_contained(pages):
    names = set(pages)
    for name in sorted(pages):
        parser = _audit(_text(pages, name))
        assert parser.link_tags == 0, name
        assert parser.srcs == [], (name, parser.srcs)
        if name == "submit.html":
            assert parser.script_tags >= 1
        else:
            assert parser.script_tags == 0, name
        for href in parser.hrefs:
            if href.startswith("https://github.com/lrspeiser/Invariant"):
                continue
            assert not href.startswith(("http:", "https:", "//", "mailto:")), (name, href)
            path = href.split("#", 1)[0]
            if not path:
                continue
            assert path.startswith("/"), (name, href)
            target = "index.html" if path == "/" else path.lstrip("/") + ".html"
            assert target in names, (name, href)


def test_no_external_resources_beyond_documented_targets(pages):
    """Belt and braces on self-containment: no url()/@import in CSS, and every

    absolute URL on every page is either a GitHub link target or the (non-fetched)
    MathML namespace identifier."""

    allowed = (
        "https://github.com/lrspeiser/Invariant",
        "http://www.w3.org/1998/Math/MathML",
    )
    url_pattern = re.compile(r"https?://[^\s\"'<>]+")
    for name in sorted(pages):
        page_text = _text(pages, name)
        assert "@import" not in page_text, name
        assert "url(" not in page_text.lower(), name
        for url in url_pattern.findall(page_text):
            assert url.startswith(allowed), (name, url)


# ---------------------------------------------------------------------------
# Footer, vercel.json, committed output
# ---------------------------------------------------------------------------


def test_footer_present_on_every_page(pages):
    for name in sorted(pages):
        page_text = _text(pages, name)
        assert FOOTER_CREED in page_text, name
        assert f"Site content as of <code>{TEST_COMMIT}</code>" in page_text, name
        assert "src/sigma_theory_compiler/static_site_generator.py" in page_text, name


def test_vercel_json_exact():
    raw = (ROOT / "vercel.json").read_text(encoding="utf-8")
    assert json.loads(raw) == {
        "buildCommand": "",
        "outputDirectory": "public",
        "cleanUrls": True,
        "trailingSlash": False,
    }
    assert raw == (
        '{"buildCommand": "", "outputDirectory": "public", '
        '"cleanUrls": true, "trailingSlash": false}\n'
    )


def test_committed_public_matches_generator_output():
    public = ROOT / "public"
    index = public / "index.html"
    assert index.is_file(), "public/ must be committed (run the generator)"
    match = re.search(
        r"Site content as of <code>([0-9a-f]{40})</code>", index.read_text(encoding="utf-8")
    )
    assert match is not None
    pages = render_site(ROOT, match.group(1))
    on_disk = {
        path.relative_to(public).as_posix(): path.read_bytes()
        for path in sorted(public.rglob("*.html"))
    }
    assert set(on_disk) == set(pages)
    for name in sorted(pages):
        assert on_disk[name] == pages[name], name


def test_cli_build_validate_and_reject_paths(tmp_path):
    output = tmp_path / "site"
    assert main(["--root", str(ROOT), "--output", str(output), "--commit", TEST_COMMIT]) == 0
    assert (
        main(["--root", str(ROOT), "--output", str(output), "--commit", TEST_COMMIT, "--validate"])
        == 0
    )
    (output / "index.html").write_bytes(b"tampered")
    assert (
        main(["--root", str(ROOT), "--output", str(output), "--commit", TEST_COMMIT, "--validate"])
        == 1
    )
    assert main(["--root", str(ROOT), "--output", str(output), "--commit", "not-a-sha"]) == 2


# ---------------------------------------------------------------------------
# Content: Pareto verbatim, Lean in full, problems honest, evidence complete
# ---------------------------------------------------------------------------


def test_pareto_front_rendered_verbatim_with_margin_columns(pages, receipts):
    gravity_text = _text(pages, "gravity.html")
    front = receipts["billion"]["pareto_front"]
    assert front
    for entry in front:
        assert entry["formula"] in gravity_text
        assert entry["newton_error"] in gravity_text
        assert entry["flatness"] in gravity_text
    assert f"The {len(front)}-entry Pareto front" in gravity_text


def test_crosscheck_and_scope_are_receipt_backed(pages, receipts):
    gravity_text = _text(pages, "gravity.html")
    base_crosscheck = receipts["billion"]["crosscheck"]
    campaign_crosscheck = receipts["lensing"]["crosscheck"]
    assert f"{base_crosscheck['disagreements']:,} disagreements" in gravity_text
    assert f"{campaign_crosscheck['lensing_disagreements']:,} lensing disagreements" in gravity_text
    assert f"{campaign_crosscheck['cluster_disagreements']:,} cluster disagreements" in gravity_text
    for key in ("billion", "lensing"):
        assert html_module.escape(receipts[key]["scope"], quote=True) in gravity_text


def test_named_lensing_control_is_present(pages, receipts):
    gravity_text = _text(pages, "gravity.html")
    assert "flattens curves but fails lensing" in gravity_text
    linear_u = receipts["lensing"]["controls"]["linear_u"]
    assert linear_u["formula"] in gravity_text
    assert linear_u["lensing"]["worst_consistency"] in gravity_text


def test_lean_source_rendered_in_full_with_claim_boundary(pages):
    collatz_text = _text(pages, "collatz.html")
    lean_source = (ROOT / ARTIFACT_PATHS["lean"]).read_text(encoding="utf-8")
    assert html_module.escape(lean_source, quote=True) in collatz_text
    assert "Claim boundary" in collatz_text
    assert "termination" in collatz_text


def test_problem_pages_are_honest_about_flags_and_link_evidence(pages, receipts):
    for entry in receipts["queue"]["entries"]:
        page_text = _text(pages, f"problems/{entry['id']}.html")
        assert html_module.escape(entry["statement"], quote=True) in page_text
        assert html_module.escape(entry["source_citation"], quote=True) in page_text
        assert html_module.escape(entry["believed_open_because"], quote=True) in page_text
        assert html_module.escape(entry["progress_definition"], quote=True) in page_text
        assert entry["machine_form"]["kind"] in page_text
        if entry["control_rediscovery"]:
            assert "REDISCOVERY CONTROL" in page_text
        if entry["synthetic"]:
            assert "SYNTHETIC" in page_text
    assert "/collatz" in _text(pages, "problems/collatz_stopping_time.html")
    for problem_id in (
        "baryonic_rotation_law",
        "lensing_dynamics_consistency",
        "cluster_missing_mass",
    ):
        assert "/gravity" in _text(pages, f"problems/{problem_id}.html")


def test_evidence_lists_every_consumed_artifact(pages, receipts):
    evidence_text = _text(pages, "evidence.html")
    for rel in ARTIFACT_PATHS.values():
        assert rel in evidence_text
        assert f"https://github.com/lrspeiser/Invariant/blob/main/{rel}" in evidence_text
    for key in ("queue", "billion", "lensing", "sweep"):
        seal = receipts[key]["content_sha256"]
        assert seal[:16] in evidence_text
        assert seal in evidence_text


def test_submit_page_mirrors_the_sealed_schema(pages):
    submit_text = _text(pages, "submit.html")
    assert SUBMIT_NOTICE in submit_text
    for key in sorted(ENTRY_KEYS):
        assert f"<code>{key}</code>" in submit_text
    for kind in sorted(MACHINE_FORM_KINDS):
        assert kind in submit_text
        for field in MACHINE_FORM_KINDS[kind]:
            assert f"f-mf-{kind}-{field}" in submit_text
    assert "issues/new" in submit_text
    assert "floats are forbidden" in submit_text.lower()


# ---------------------------------------------------------------------------
# The status system: five words, one banner per problem page and case study
# ---------------------------------------------------------------------------


def test_status_banners_exactly_one_per_problem_and_case_study(pages):
    stamped = set(_stamped_pages(pages))
    assert stamped, "expected stamped pages"
    for name in sorted(pages):
        count = _text(pages, name).count('class="status-banner')
        expected = 1 if name in stamped else 0
        assert count == expected, (name, count)


def test_status_vocabulary_is_defined_wherever_stamped(pages):
    for name in _stamped_pages(pages):
        page_text = _text(pages, name)
        for word, _css, definition in STATUS_DEFINITIONS:
            assert word in page_text, (name, word)
            assert html_module.escape(definition, quote=True) in page_text, (name, word)
        stamps = re.findall(r'<span class="stamp">STATUS: ([^<]+)\.</span>', page_text)
        assert len(stamps) == 1, (name, stamps)
        assert stamps[0] in {word for word, _css, _definition in STATUS_DEFINITIONS}, name


def test_collatz_banner_is_the_mandated_open_text(pages):
    collatz_text = _text(pages, "collatz.html")
    assert "unsolved" in collatz_text
    assert "well known to mathematicians" in collatz_text
    assert ("STATUS: OPEN. " + COLLATZ_STATUS_TEXT) in _flat(collatz_text)


def test_gravity_banner_is_the_sealed_negative(pages):
    gravity_text = _text(pages, "gravity.html")
    assert 'class="status-banner status-negative"' in gravity_text
    assert "STATUS: SEALED NEGATIVE." in _flat(gravity_text)


def test_no_standalone_solved_claim_anywhere(pages):
    """The word "solved" may appear only inside "unsolved"/"resolved" forms.

    This is deliberately stronger than "no OPEN problem is called solved": nothing
    on the site is solved, so no page gets to use the bare word at all.  The one
    carve-out is the sealed rediscovery directory name, which contains the letters
    "solved" as a path component; the exact path substring is stripped before the
    scan because a filename is a citation, not a claim."""

    for name in sorted(pages):
        flat = _flat(_text(pages, name)).lower().replace(DOZEN_DIR, "")
        assert re.search(r"\bsolved\b", flat) is None, name


def test_problem_status_banners_are_receipt_aware(tmp_path):
    root = _fixture_root(tmp_path)
    with_receipts = render_site(root, TEST_COMMIT)
    cluster_flat = _flat(_text(with_receipts, "problems/cluster_missing_mass.html"))
    assert "STATUS: SEALED NEGATIVE." in cluster_flat
    (root / ARTIFACT_PATHS["lensing"]).unlink()
    without = render_site(root, TEST_COMMIT)
    cluster_flat = _flat(_text(without, "problems/cluster_missing_mass.html"))
    assert "STATUS: OPEN." in cluster_flat
    assert "not yet published" in cluster_flat
    assert "STATUS: OPEN." in _flat(_text(without, "gravity.html"))


# ---------------------------------------------------------------------------
# Mathematics: ASCII -> LaTeX -> MathML, at build time, or fail loudly
# ---------------------------------------------------------------------------


def test_formula_translator_roundtrips_every_receipt_formula(pages, receipts):
    formulas = {entry["formula"] for entry in receipts["billion"]["pareto_front"]}
    assert len(formulas) >= 60
    for entry in receipts["lensing"]["exact_verification"]:
        formulas.add(entry["formula"])
    for control in receipts["lensing"]["controls"].values():
        formulas.add(control["formula"])
    formulas.add(
        receipts["lensing"]["cluster_negative"]["closest_cluster_approach"]["formula"]
    )
    for formula in sorted(formulas):
        latex = formula_ascii_to_latex(formula)
        assert "\\frac" in latex, formula
        assert latex.endswith("u = y^{-1/2}"), formula
        assert "<math" in latex_to_mathml(latex), formula
    example = "nu(y) = [(1 + 3u^5) / (1 + u^4)]^1,  u = y^(-1/2)"
    assert formula_ascii_to_latex(example) == (
        r"\nu(y) = \left[\frac{1 + 3u^{5}}{1 + u^{4}}\right]^{1}, \quad u = y^{-1/2}"
    )
    with pytest.raises(SiteGenerationError):
        formula_ascii_to_latex("nu(y) = exp(u)")
    gravity_text = _text(pages, "gravity.html")
    assert gravity_text.count("<math") >= len(receipts["billion"]["pareto_front"])


def test_mathml_present_for_both_sigma_identities_on_collatz(pages):
    collatz_text = _text(pages, "collatz.html")
    for latex in (LATEX_SIGMA_HALVING, LATEX_SIGMA_POW2):
        assert latex_to_mathml(latex) in collatz_text, latex
    assert collatz_text.count("<math") >= 2
    assert latex_to_mathml(LATEX_RAR_LAW, display="block") in _text(pages, "gravity.html")


# ---------------------------------------------------------------------------
# Voice and aesthetic: the Gardner opening, paper sections, grid-paper CSS
# ---------------------------------------------------------------------------


def test_index_opens_with_the_gardner_passage(pages):
    index_text = _text(pages, "index.html")
    assert GARDNER_OPENING in index_text
    assert index_text.index(GARDNER_OPENING) < index_text.index("<h2>")


def test_case_study_pages_use_the_paper_sections_in_order(pages):
    expected = [
        "Abstract",
        "The question",
        "What we did",
        "What we found",
        "What this does not show",
        "Methods",
        "References",
    ]
    for name in CASE_STUDY_PAGES:
        headings = re.findall(r"<h2>([^<]+)</h2>", _text(pages, name))
        assert headings == expected, (name, headings)
        assert '<details class="methods">' in _text(pages, name)


def test_paper_page_structure_numbers_and_frankness(pages, receipts):
    paper_text = _text(pages, "paper.html")
    headings = re.findall(r"<h2>([^<]+)</h2>", paper_text)
    assert headings == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Limitations",
        "References",
    ]
    assert "The Invariant Project" in paper_text
    assert "Why this is not a mathematics-journal submission" in paper_text
    assert TEST_COMMIT in paper_text
    assert re.search(r"\d{4}-\d{2}-\d{2}", _flat(paper_text)) is None, "wall-clock date leaked"
    assert f"{receipts['billion']['counts']['processed']:,}" in paper_text
    assert f"{receipts['lensing']['counts']['lensing_pass']:,}" in paper_text
    closest = receipts["lensing"]["cluster_negative"]["closest_cluster_approach"]
    assert closest["max_deviation"] in paper_text
    assert "0/21" in paper_text
    assert "21/21" in paper_text
    goals = (ROOT / ARTIFACT_PATHS["goals_doc"]).read_text(encoding="utf-8")
    prior_art = re.search(r"Prior-art audit: (\d+) records", goals)
    assert prior_art is not None
    assert f"{int(prior_art.group(1)):,}-record" in paper_text
    assert 'href="/paper"' in _text(pages, "index.html")


def test_grid_paper_and_typewriter_styling_on_every_page(pages):
    for name in sorted(pages):
        page_text = _text(pages, name)
        assert page_text.count("repeating-linear-gradient") >= 4, name
        assert '"Courier New"' in page_text, name
        assert "prefers-color-scheme: dark" in page_text, name
        assert "rotate(-0.5deg)" in page_text, name
        assert "font-size: 17px" in page_text, name


# ---------------------------------------------------------------------------
# Rediscovery papers: receipt-derived funnels, sidebars, alternates, MathML
# ---------------------------------------------------------------------------

_FUNNEL_RECT = re.compile(
    r'<rect class="bar(?: hit)?"[^>]*data-key="funnel_([^"]+)" data-value="([^"]+)"'
)


def _paper_text(pages: dict[str, bytes], classical_id: str) -> str:
    return _text(pages, f"papers/{classical_id}.html")


def _section(page_text: str, heading: str, next_heading: str) -> str:
    start = page_text.index(f"<h2>{heading}</h2>")
    end = page_text.index(f"<h2>{next_heading}</h2>")
    assert start < end
    return page_text[start:end]


def _expected_funnel(world_receipt: dict) -> list[tuple[str, str]]:
    """Independently re-derive the funnel bars from a world receipt's trail."""

    phase = world_receipt["phase_a"]
    bars: list[tuple[str, str]] = []
    b1 = next(s for s in phase["stages"] if s["stage_id"] == "b1_basis_synthesis")
    if b1["decision"] == "PASS":
        counts = b1["receipt"]["counts"]
        examined = counts["entries_examined"]
        rejected = counts["entries_rejected_before_acceptance"]
        bars += [
            ("declared_families", str(counts["ladder_entries"])),
            ("families_examined", str(examined)),
            ("families_rejected", str(rejected)),
            ("family_accepted", str(examined - rejected)),
            ("holdout_confirmations", str(b1["receipt"]["result"]["confirmations"])),
        ]
        for route in phase["prover_routes"]:
            if route.get("route") == "b5_lemma_decomposition" and route.get("receipt"):
                route_counts = route["receipt"]["counts"]
                discharged = (
                    route_counts["obligations"]
                    - route_counts["obligations_failing_exact_local_check"]
                )
                bars.append(
                    ("proof_obligations", f"{discharged}/{route_counts['obligations']}")
                )
                break
        return bars
    for stage in phase["stages"]:
        stage_id = stage["stage_id"]
        receipt = stage.get("receipt") or {}
        counts = receipt.get("counts", {})
        if stage_id == "b1_basis_synthesis":
            bars.append(("declared_families", str(counts["ladder_entries"])))
        elif stage_id == "b7_structural_repair":
            bars.append(("repair_strategies", str(counts["strategies_attempted"])))
        elif stage_id == "b3_conjecture_generation":
            bars += [
                ("statement_kinds", str(counts["declared_statement_kinds"])),
                ("proposed", str(counts["proposed"])),
                ("survived", str(counts["survived"])),
            ]
        elif stage_id == "b2_nonlinear_coefficient_search" and stage["decision"] == "PASS":
            bars += [
                ("ratio_models", str(counts["declared_models"])),
                ("ratio_models_rejected", str(counts["models_rejected_before_acceptance"])),
                ("ratio_model_accepted", "1"),
                ("ratio_confirmations", str(receipt["result"]["confirmations"])),
            ]
    bars.append(("principal_result", "1"))
    return bars


def _b3_conjectures(world_receipt: dict) -> list[dict]:
    for stage in world_receipt["phase_a"]["stages"]:
        if stage["stage_id"] == "b3_conjecture_generation":
            return stage["receipt"]["conjectures"]
    return []


def _abstention_expected(world_receipt: dict) -> bool:
    stages = {s["stage_id"]: s for s in world_receipt["phase_a"]["stages"]}
    if "b1_basis_synthesis" not in stages or "b7_structural_repair" not in stages:
        return False
    if stages["b1_basis_synthesis"]["decision"] != "BLOCK":
        return False
    if stages["b7_structural_repair"]["decision"] != "BLOCK":
        return False
    if any(s["decision"] == "PASS" for s in world_receipt["phase_a"]["stages"]):
        return False
    closed_form = next(
        (c for c in _b3_conjectures(world_receipt) if c["kind"] == "closed_form"), None
    )
    return closed_form is not None and closed_form["status"] == "NOT_PROPOSED"


def _killed_by_row_expected(world_receipt: dict) -> bool:
    return any(
        conjecture["kind"] == "linear_recurrence"
        and conjecture["status"] == "REFUTED"
        and conjecture.get("refutation_witness") is not None
        for conjecture in _b3_conjectures(world_receipt)
    )


def test_papers_exist_for_exactly_the_campaign_worlds(pages, receipts):
    worlds = [world["classical_id"] for world in receipts["dozen"]["world_results"]]
    assert sorted(name for name in pages if name.startswith("papers/")) == sorted(
        f"papers/{world}.html" for world in worlds
    )
    assert "papers.html" in pages
    for name in sorted(pages):
        assert 'href="/papers"' in _text(pages, name), name


def test_paper_headings_and_masthead(pages, receipts):
    expected = [
        "Abstract",
        "The question",
        "Methods",
        "The elimination funnel",
        "What else survived",
        "Result",
        "Verification",
        "What this does not show",
        "References",
    ]
    for world in receipts["dozen"]["world_results"]:
        page_text = _paper_text(pages, world["classical_id"])
        headings = re.findall(r"<h2>([^<]+)</h2>", page_text)
        assert headings == expected, (world["classical_id"], headings)
        number = int(world["world_id"].rsplit("_", 1)[1])
        assert f"<h1>Blind rediscovery {number}: " in page_text
        assert "The Invariant Project" in page_text
        assert f"content as of <code>{TEST_COMMIT}</code>" in page_text
        assert REDISCOVERY_SENTENCE in _flat(page_text)
        assert html_module.escape(world["attribution"], quote=True) in page_text


def test_paper_rows_table_is_the_receipt_rows(pages, receipts, world_receipts):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        rows = world_receipts[classical_id]["public_rows"]
        section = _section(page_text, "The question", "Methods")
        assert ROWS_CAPTION in section
        assert section.count("<tr>") == len(rows) + 1  # header row + one per receipt row
        for row in rows:
            assert f'<td class="num">{row["point"]}</td>' in section
        tiles = _tile_values(page_text)
        assert tiles["paper_rows"] == str(len(rows))


def test_funnel_numbers_equal_receipt_derived_values(pages, receipts, world_receipts):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        expected = _expected_funnel(world_receipts[classical_id])
        rendered = _FUNNEL_RECT.findall(page_text)
        assert rendered == expected, classical_id
        assert page_text.count('<rect class="bar') == len(expected), classical_id
        assert page_text.count("<svg") == 1, classical_id


def test_funnel_first_bar_counts_declared_families_never_candidates(
    pages, receipts, world_receipts
):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        flat = _flat(page_text)
        b1 = next(
            s
            for s in world_receipts[classical_id]["phase_a"]["stages"]
            if s["stage_id"] == "b1_basis_synthesis"
        )
        ladder = b1["receipt"]["counts"]["ladder_entries"]
        assert f"{ladder} {DECLARED_FAMILIES_PHRASE}" in flat, classical_id
        assert f"{ladder} candidates" not in flat, classical_id
        caption = re.search(r"<figcaption>(.*?)</figcaption>", page_text, re.DOTALL)
        assert caption is not None, classical_id
        assert "declared families" in caption.group(1), classical_id
        assert html_module.escape(FUNNEL_FAMILY_NOTE, quote=True) in caption.group(1)


def test_funnel_rejection_reasons_match_certificates(pages, receipts, world_receipts):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        tiles = _tile_values(_paper_text(pages, classical_id))
        phase = world_receipts[classical_id]["phase_a"]
        for stage in phase["stages"]:
            certificate_key, field = {
                "b1_basis_synthesis": ("b1", "strictly_simpler_entries_rejected"),
                "b2_nonlinear_coefficient_search": ("b2", "strictly_simpler_models_rejected"),
            }.get(stage["stage_id"], (None, None))
            if certificate_key is None or stage["decision"] != "PASS":
                continue
            rejected = stage["receipt"]["minimality_certificate"][field]
            reasons: dict[str, int] = {}
            for entry in rejected:
                reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
            for reason, count in reasons.items():
                assert tiles[f"funnel_{certificate_key}_reason_{reason}"] == str(count), (
                    classical_id,
                    reason,
                )


def test_abstention_box_present_exactly_where_the_trail_supports_it(
    pages, receipts, world_receipts
):
    supported = {
        world["classical_id"]
        for world in receipts["dozen"]["world_results"]
        if _abstention_expected(world_receipts[world["classical_id"]])
    }
    assert supported == {"fibonacci", "lucas", "pell"}
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        has_box = "An honest abstention" in page_text
        assert has_box == (classical_id in supported), classical_id
        if has_box:
            assert "NOT_PROPOSED" in page_text
            assert "square root of" in _flat(page_text)
            b1 = next(
                s
                for s in world_receipts[classical_id]["phase_a"]["stages"]
                if s["stage_id"] == "b1_basis_synthesis"
            )
            tiles = _tile_values(page_text)
            assert tiles["abstain_families"] == str(b1["receipt"]["counts"]["ladder_entries"])
            for strategy in next(
                s
                for s in world_receipts[classical_id]["phase_a"]["stages"]
                if s["stage_id"] == "b7_structural_repair"
            )["receipt"]["rejected_strategies"]:
                assert strategy["strategy"] in page_text
                assert strategy["reason"] in page_text


def test_catalan_killed_by_row_box_matches_receipt_truth(pages, receipts, world_receipts):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        expected = _killed_by_row_expected(world_receipts[classical_id])
        present = "Killed by one row" in _paper_text(pages, classical_id)
        assert present == expected, classical_id
    assert "catalan_ratio" in world_receipts  # the box's subject world is in the campaign


def test_alternates_section_lists_exactly_the_trail_survivors(
    pages, receipts, world_receipts
):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        section = _section(page_text, "What else survived", "Result")
        principal = world_receipts[classical_id]["phase_a"]["candidate"]["statement"]
        survivors = [
            conjecture
            for conjecture in _b3_conjectures(world_receipts[classical_id])
            if conjecture["status"] == "SURVIVED"
        ]
        alternates = [c for c in survivors if c["statement"] != principal]
        if not alternates:
            assert "Exactly one statement survived" in section, classical_id
            assert 'class="receipt-form"' not in section, classical_id
            continue
        assert section.count('class="receipt-form"') == len(alternates), classical_id
        tiles = _tile_values(section)
        for conjecture in alternates:
            escaped = html_module.escape(conjecture["statement"], quote=True)
            assert escaped in section, (classical_id, conjecture["statement"])
            assert tiles[f"alt_support_{conjecture['kind']}"] == str(conjecture["support"])


def test_mathml_for_both_result_statements_on_every_paper(pages, receipts):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        section = _section(page_text, "Result", "Verification")
        for statement in (world["discovered_statement"], world["target_statement"]):
            mathml = latex_to_mathml(statement_ascii_to_latex(statement))
            assert mathml in section, (classical_id, statement)
            assert html_module.escape(statement, quote=True) in section
        assert section.count("<math") >= 2, classical_id


def test_paper_verification_renders_lean_in_full_or_says_none(
    pages, receipts, world_receipts
):
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        page_text = _paper_text(pages, classical_id)
        section = _section(page_text, "Verification", "What this does not show")
        routes = [
            route
            for route in world_receipts[classical_id]["phase_a"]["prover_routes"]
            if route.get("lean_source_emitted") and route.get("receipt")
        ]
        assert world["lean_emitted"] == bool(routes), classical_id
        if routes:
            assert "independent kernel verification pending CI" in section, classical_id
            for route in routes:
                escaped = html_module.escape(route["receipt"]["lean_source"], quote=True)
                assert escaped in section, (classical_id, route["route"])
        else:
            assert "No proof-kernel source was emitted" in section, classical_id
            assert "<pre>" not in section, classical_id
        tiles = _tile_values(page_text)
        assert tiles["verify_holdout"] == str(world["holdout_confirmations"]), classical_id


def test_paper_chronology_numbers_are_receipt_backed(pages, receipts):
    chronology = receipts["dozen"]["chronology"]
    probe = chronology["denied_probe"]
    claims = receipts["dozen"]["claims"]
    counts = receipts["dozen"]["counts"]
    for world in receipts["dozen"]["world_results"]:
        tiles = _tile_values(_paper_text(pages, world["classical_id"]))
        assert tiles["chron_attempted_reads"] == str(probe["attempted_target_reads"])
        assert tiles["chron_denied_reads"] == str(probe["denied_target_reads"])
        assert tiles["chron_denied_bytes"] == str(probe["denied_content_bytes_exposed"])
        assert tiles["chron_reads_before_freeze"] == str(
            claims["target_records_read_before_candidate_freeze"]
        )
        assert tiles["chron_unseal_batches"] == str(chronology["unseal_batches"])
        assert tiles["chron_post_unseal"] == str(counts["post_unseal_generation_events"])


def test_families_explainer_on_every_paper_and_the_index(pages, receipts, world_receipts):
    ladder_values = set()
    for world_receipt in world_receipts.values():
        for stage in world_receipt["phase_a"]["stages"]:
            if stage["stage_id"] == "b1_basis_synthesis":
                ladder_values.add(stage["receipt"]["counts"]["ladder_entries"])
    assert len(ladder_values) == 1
    ladder = ladder_values.pop()
    heading = f"Why {ladder} families and not a billion formulas"
    processed = receipts["billion"]["counts"]["processed"]
    for world in receipts["dozen"]["world_results"]:
        page_text = _paper_text(pages, world["classical_id"])
        assert heading in _flat(page_text), world["classical_id"]
        tiles = _tile_values(page_text)
        assert tiles["ladder_families"] == str(ladder)
        assert tiles["gravity_contrast_processed"] == str(processed)
    index_text = _text(pages, "papers.html")
    assert heading in _flat(index_text)
    assert index_text.index("billion formulas") < index_text.index("<h2>The ledger</h2>")


def test_papers_index_ledger_matches_campaign_and_funnels(
    pages, receipts, world_receipts
):
    index_text = _text(pages, "papers.html")
    tiles = _tile_values(index_text)
    counts = receipts["dozen"]["counts"]
    assert tiles["papers_worlds"] == str(counts["worlds"])
    assert tiles["papers_rediscovered"] == str(counts["rediscovered_total"])
    assert tiles["papers_with_proof"] == str(counts["rediscovered_with_proof"])
    assert tiles["papers_holdout_total"] == str(counts["holdout_confirmations_total"])
    assert tiles["papers_missed"] == str(counts["missed"])
    for world in receipts["dozen"]["world_results"]:
        classical_id = world["classical_id"]
        assert f'href="/papers/{classical_id}"' in index_text
        assert world["verdict"] in index_text
        one_liner = "→".join(
            display for _key, display in _expected_funnel(world_receipts[classical_id])
        )
        assert one_liner in _flat(index_text), classical_id
    flat = _flat(index_text)
    assert "21 candidates" not in flat


def test_papers_progress_report_block_tracks_unsolved_campaign(pages):
    index_text = _text(pages, "papers.html")
    assert "Progress reports (unsolved dozen)" in index_text
    if (ROOT / UNSOLVED_CAMPAIGN_PATH).is_file():
        assert "campaign not yet published" not in index_text
        assert UNSOLVED_CAMPAIGN_PATH in index_text
    else:
        assert "campaign not yet published" in index_text
        assert UNSOLVED_CAMPAIGN_PATH in index_text


def test_unsolved_campaign_listing_when_receipt_lands(tmp_path):
    root = _fixture_root(tmp_path)
    target = root / UNSOLVED_CAMPAIGN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "content_sha256": "ab" * 32,
                "world_results": [
                    {
                        "classical_id": "sample_open_world",
                        "world_receipt_path": "runs/math/unsolved-dozen/sample.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pages = render_site(root, TEST_COMMIT)
    index_text = _text(pages, "papers.html")
    assert "campaign not yet published" not in index_text
    assert "sample_open_world" in index_text
    assert "runs/math/unsolved-dozen/sample.json" in index_text


def test_papers_fail_soft_without_world_receipts(tmp_path):
    root = _fixture_root(tmp_path)  # copies the campaign but not the world receipts
    pages = render_site(root, TEST_COMMIT)
    campaign = json.loads((ROOT / ARTIFACT_PATHS["dozen"]).read_text(encoding="utf-8"))
    for world in campaign["world_results"]:
        page_text = _text(pages, f"papers/{world['classical_id']}.html")
        assert MISSING_NOTE in page_text, world["classical_id"]
        assert '<rect class="bar' not in page_text, world["classical_id"]
        assert REDISCOVERY_SENTENCE in _flat(page_text)
    index_text = _text(pages, "papers.html")
    assert "papers.html" in pages
    assert "The ledger" in index_text


def test_papers_fail_soft_without_campaign(tmp_path):
    root = _fixture_root(tmp_path)
    (root / ARTIFACT_PATHS["dozen"]).unlink()
    pages = render_site(root, TEST_COMMIT)
    assert not any(name.startswith("papers/") for name in pages)
    assert MISSING_NOTE in _text(pages, "papers.html")


def test_world_receipt_seals_match_the_campaign_ledger(receipts, world_receipts):
    for world in receipts["dozen"]["world_results"]:
        world_receipt = world_receipts[world["classical_id"]]
        assert world_receipt["content_sha256"] == world["world_receipt_sha256"]
        assert world_receipt["phase_a_root"] == receipts["dozen"]["chronology"]["phase_a_root"]


def test_statement_translator_rejects_out_of_grammar_text():
    for bad in ("a(n) = exp(n)", "a(n) := n", "a(n) = n!", "a(n) = 1 & 2"):
        with pytest.raises(SiteGenerationError):
            statement_ascii_to_latex(bad)


# ---------------------------------------------------------------------------
# Case studies: the Balmer/Bohr head-to-head
# ---------------------------------------------------------------------------

CASE_STUDY_PAGE = "case-studies/balmer-bohr.html"


@pytest.fixture(scope="module")
def case_runtime() -> dict:
    return json.loads(
        (ROOT / ARTIFACT_PATHS["case_study_runtime"]).read_text(encoding="utf-8")
    )


def test_case_study_pages_exist_and_are_reachable_from_the_nav(pages):
    assert "case-studies.html" in pages
    assert CASE_STUDY_PAGE in pages
    for name in sorted(pages):
        assert 'href="/case-studies"' in _text(pages, name), name
    index_text = _text(pages, "case-studies.html")
    assert 'href="/case-studies/balmer-bohr"' in index_text
    assert 'href="/case-studies"' in _text(pages, CASE_STUDY_PAGE)


def test_case_study_index_carries_no_banner_and_lists_the_study(pages, receipts):
    index_text = _text(pages, "case-studies.html")
    assert index_text.count('class="status-banner') == 0
    assert "Balmer 1885 and Bohr 1913" in index_text
    tiles = _tile_values(index_text)
    space = receipts["case_study"]["blind_race"]["search_space"]
    assert tiles["csi_views"] == str(space["total_declared_views"])
    assert tiles["csi_admitted"] == str(space["views_admitted"])
    assert receipts["case_study"]["verdict"] in index_text
    assert receipts["case_study"]["content_sha256"][:16] in index_text


def test_case_study_numbers_are_read_from_the_receipt(pages, receipts):
    page_text = _text(pages, CASE_STUDY_PAGE)
    values = _tile_values(page_text)
    receipt = receipts["case_study"]
    blind = receipt["blind_race"]
    space = blind["search_space"]
    candidate = blind["candidate"]
    unseal = blind["unseal"]
    numerics = receipt["derivation"]["rydberg_numerics"]

    assert values["cs_views_declared"] == str(space["total_declared_views"])
    assert values["cs_views_evaluated"] == str(space["views_evaluated"])
    assert values["cs_views_rejected"] == str(space["views_rejected"])
    assert values["cs_views_admitted"] == str(space["views_admitted"])
    assert values["cs_total_views"] == str(space["total_declared_views"])
    assert values["cs_abs_views"] == str(space["total_declared_views"])
    assert values["cs_rejected_earlier"] == str(candidate["rejected_earlier_views"])
    assert values["cs_found_constant"] == candidate["constant_decimal"]
    assert values["cs_spread"] == candidate["relative_spread"]
    assert values["cs_offset"] == str(candidate["recovered_index_offset"])
    assert values["cs_sealed_constant"] == unseal["sealed_constant_decimal"]
    assert values["cs_rounded_constant"] == unseal["constant_rounded_to_published_places"]
    assert values["cs_holdout_worst"] == unseal["holdout_max_relative_residual"]
    assert values["cs_holdout_tolerance"] == unseal["holdout_relative_tolerance"]
    assert values["cs_rydberg_derived"] == numerics["derived_rydberg_per_m"]
    assert values["cs_rydberg_measured"] == numerics["measured_rydberg_per_m"]
    assert values["cs_rydberg_error"] == numerics["relative_error_vs_measured"]
    assert values["cs_abs_error"] == numerics["relative_error_vs_measured"]

    family = blind["declared_view_family"]
    for key in (
        "shift_range",
        "index_exponent_range",
        "quadratic_exponent_range",
        "offset_range",
    ):
        assert values[f"cs_bound_{key}_low"] == str(family[key][0])
        assert values[f"cs_bound_{key}_high"] == str(family[key][1])

    for row in blind["public_rows"]:
        assert values[f"cs_row_m_{row['m']}"] == str(row["m"])
        assert values[f"cs_row_v_{row['m']}"] == row["v_decimal"]

    probe = receipt["chronology"]["denied_probe"]
    assert values["cs_probe_attempted"] == str(probe["attempted_target_reads"])
    assert values["cs_probe_denied"] == str(probe["denied_target_reads"])
    assert values["cs_probe_bytes"] == str(probe["denied_content_bytes_exposed"])
    assert values["cs_probe_post"] == str(receipt["counts"]["post_unseal_generation_events"])


def test_case_study_pages_resolve_every_receipt_value(pages):
    """A missing receipt key must fail the build, never render as an empty slot."""

    for name in ("case-studies.html", CASE_STUDY_PAGE):
        page_text = _text(pages, name)
        assert 'data-value="None"' not in page_text, name
        assert 'data-value=""' not in page_text, name
        assert ">None<" not in page_text, name
        assert "&mdash;</td>" not in page_text or name == "case-studies.html"


def test_case_study_holdout_table_matches_the_receipt(pages, receipts):
    values = _tile_values(_text(pages, CASE_STUDY_PAGE))
    holdout = receipts["case_study"]["blind_race"]["unseal"]["holdout"]
    assert len(holdout) == 3
    for row in holdout:
        label = row["m"]
        assert values[f"cs_hold_pred_{label}"] == row["predicted_decimal"]
        assert values[f"cs_hold_meas_{label}"] == row["measured_decimal"]
        assert values[f"cs_hold_res_{label}"] == row["residual_decimal"]
        assert values[f"cs_hold_rel_{label}"] == row["relative_residual"]
        assert row["within_declared_tolerance"] is True
    flat = _flat(_text(pages, CASE_STUDY_PAGE))
    assert "before the numbers they are scored against could be read" in flat


def test_case_study_tolerance_ladder_is_rendered_in_full(pages, receipts):
    values = _tile_values(_text(pages, CASE_STUDY_PAGE))
    ladder = receipts["case_study"]["blind_race"]["tolerance_robustness"]
    assert len(ladder) >= 5
    for rung in ladder:
        assert values[f"cs_ladder_{rung['relative_tolerance']}"] == str(rung["views_admitted"])
    page_text = _text(pages, CASE_STUDY_PAGE)
    declared = [rung for rung in ladder if rung["is_the_declared_tolerance"]]
    assert len(declared) == 1
    assert declared[0]["relative_tolerance"] in page_text


def test_case_study_renders_receipt_latex_as_build_time_mathml(pages, receipts):
    page_text = _text(pages, CASE_STUDY_PAGE)
    receipt = receipts["case_study"]
    blind = receipt["blind_race"]
    expected = [
        blind["candidate"]["latex"],
        blind["candidate"]["invariant_latex"],
        blind["unseal"]["classical_latex"],
        receipt["derivation"]["loop_closure"]["constant_identity_latex"],
    ]
    expected += [step["latex"] for step in receipt["derivation"]["steps"]]
    for latex in expected:
        assert latex_to_mathml(latex, display="block") in page_text, latex
    assert latex_to_mathml(LATEX_CASE_VIEW_GRAMMAR, display="block") in page_text
    assert page_text.count("<math") >= len(expected) + 1
    assert blind["candidate"]["statement"] in _flat(page_text)
    assert blind["unseal"]["target_statement"] in _flat(page_text)


def test_case_study_head_to_head_cites_intervals_and_never_estimates_effort(
    pages, receipts, case_runtime
):
    page_text = _text(pages, CASE_STUDY_PAGE)
    flat = _flat(page_text)
    values = _tile_values(page_text)
    head = receipts["case_study"]["head_to_head"]
    assert HEAD_TO_HEAD_CAPTION in flat
    for key in ("balmer_1885", "bohr_1913"):
        human = head[key]["human_timescale"]
        assert values[f"cs_years_{key}"] == str(human["documented_interval_years"])
        assert human["personal_effort_duration"] == "not precisely documented"
    assert flat.count("not precisely documented") >= 2
    assert values["cs_head_space"] == str(head["engine_empirical"]["search_space_size"])
    measured = case_runtime["measured_seconds"]
    assert values["cs_wall_engine_empirical"] == measured["blind_race"]
    assert values["cs_wall_engine_derivation"] == measured["derivation"]
    for note in head["comparison_notes"]:
        assert html_module.escape(note, quote=True) in page_text


def test_case_study_states_the_exact_arithmetic_refusal(pages, receipts):
    flat = _flat(_text(pages, CASE_STUDY_PAGE))
    blind = receipts["case_study"]["blind_race"]
    exact = blind["b1_on_the_winning_column"]
    assert exact["note"] in flat
    assert exact["decision"] == "BLOCK"
    assert exact["first_blocker"] in flat
    assert blind["candidate_selection_rule"] in flat
    assert blind["unseal"]["verdict_rule"] in flat
    assert receipts["case_study"]["scope"] in flat


def test_case_study_pages_fail_soft_without_the_receipt(tmp_path):
    root = _fixture_root(tmp_path)
    (root / ARTIFACT_PATHS["case_study"]).unlink()
    (root / ARTIFACT_PATHS["case_study_runtime"]).unlink()
    pages = render_site(root, TEST_COMMIT)
    detail = _text(pages, CASE_STUDY_PAGE)
    assert MISSING_NOTE in detail
    assert ARTIFACT_PATHS["case_study"] in detail
    assert detail.count('class="status-banner') == 1
    assert MISSING_NOTE in _text(pages, "case-studies.html")
    headings = re.findall(r"<h2>([^<]+)</h2>", detail)
    assert headings == [
        "Abstract",
        "The question",
        "What we did",
        "What we found",
        "What this does not show",
        "Methods",
        "References",
    ]


def test_case_study_reproducibility_round_trip(tmp_path, receipts):
    """Build, write, re-render from disk, and byte-compare the case-study pages."""

    output = tmp_path / "site"
    first = build_site(ROOT, output, TEST_COMMIT)
    on_disk = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(output.rglob("*.html"))
    }
    assert on_disk[CASE_STUDY_PAGE] == first[CASE_STUDY_PAGE]
    assert on_disk["case-studies.html"] == first["case-studies.html"]
    second = render_site(ROOT, TEST_COMMIT)
    assert second[CASE_STUDY_PAGE] == first[CASE_STUDY_PAGE]
    assert second["case-studies.html"] == first["case-studies.html"]
    assert receipts["case_study"]["content_sha256"] not in GENERATOR_SOURCE.read_text(
        encoding="utf-8"
    )
