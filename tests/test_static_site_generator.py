"""Public-site generator gates.

The site is only as honest as the checks that pin it to the repository's evidence.
The load-bearing tests here are the negative ones: the generator source must not
contain the headline numerals (they are read from receipts at build time), the
emitted HTML must contain no scalar-score vocabulary, failures must get equal
billing with their real numbers, and a missing optional receipt must produce an
explicit not-yet-published block rather than a crash or a silent omission.
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
    FOOTER_CREED,
    MISSING_NOTE,
    SUBMIT_NOTICE,
    build_site,
    main,
    render_site,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE = ROOT / "src" / "sigma_theory_compiler" / "static_site_generator.py"
TEST_COMMIT = "deadbeef" * 5

BANNED_SCORE_TERMS = ("truth score", "probability", "confidence", "% true", "likelihood")


@pytest.fixture(scope="module")
def pages() -> dict[str, bytes]:
    return render_site(ROOT, TEST_COMMIT)


@pytest.fixture(scope="module")
def receipts() -> dict[str, dict]:
    loaded = {}
    for key in ("queue", "billion", "lensing", "sweep"):
        loaded[key] = json.loads((ROOT / ARTIFACT_PATHS[key]).read_text(encoding="utf-8"))
    return loaded


def _text(pages: dict[str, bytes], name: str) -> str:
    return pages[name].decode("utf-8")


def _tile_values(page_text: str) -> dict[str, str]:
    return dict(re.findall(r'data-key="([^"]+)" data-value="([^"]+)"', page_text))


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


def test_page_set_covers_every_queue_entry(pages, receipts):
    expected = {
        "index.html",
        "problems.html",
        "gravity.html",
        "collatz.html",
        "evidence.html",
        "submit.html",
        "method.html",
    }
    expected |= {f"problems/{entry['id']}.html" for entry in receipts["queue"]["entries"]}
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


def test_generator_source_contains_no_headline_numerals(receipts):
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
        "0/21",
        "21/21",
        "70/70",
        "118/118",
        "200/200",
    }
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


# ---------------------------------------------------------------------------
# Footer, vercel.json, committed output
# ---------------------------------------------------------------------------


def test_footer_present_on_every_page(pages):
    for name in sorted(pages):
        page_text = _text(pages, name)
        assert FOOTER_CREED in page_text, name
        assert f"Site content as of <code>{TEST_COMMIT}</code>" in page_text, name


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
    for problem_id in ("baryonic_rotation_law", "lensing_dynamics_consistency", "cluster_missing_mass"):
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
