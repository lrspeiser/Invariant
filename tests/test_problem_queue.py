"""A2 problem-queue gates.

The queue is only as honest as its validator.  The load-bearing tests are the
negative ones: every schema, flag, citation, float, and seal violation must fail
closed, and the shipped artifact is pinned by its content hash so no edit can land
without resealing here too.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.problem_queue import (
    CLAIMS,
    ENTRY_KEYS,
    MACHINE_FORM_KINDS,
    QUEUE_SCHEMA,
    ProblemQueueError,
    iter_math_targets,
    iter_physics_targets,
    load_queue,
    main,
    seal_queue,
    summarize_queue,
    validate_queue,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

QUEUE_PATH = Path(__file__).resolve().parents[1] / "configs" / "problem_queue_v1.json"
PINNED_CONTENT_SHA256 = "d1d5a724ff698640d235d17cbcba994b17e30ee7832f7bf9a2234d6325690eb5"


@pytest.fixture()
def queue() -> dict[str, object]:
    return load_queue(QUEUE_PATH)


def _entry(queue: dict[str, object], entry_id: str) -> dict[str, object]:
    matches = [item for item in queue["entries"] if item["id"] == entry_id]
    assert len(matches) == 1, entry_id
    return matches[0]


def _reseal(body_and_seal: dict[str, object]) -> dict[str, object]:
    """Recompute the seal so a tamper test hits its target check, not the seal."""

    body = {key: value for key, value in body_and_seal.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


def _write(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "queue.json"
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return path


# ---------------------------------------------------------------------------
# The shipped artifact
# ---------------------------------------------------------------------------


def test_shipped_queue_loads_with_ten_entries(queue):
    assert queue["schema_version"] == QUEUE_SCHEMA
    assert len(queue["entries"]) == 10
    validate_queue(queue)


def test_shipped_queue_is_pinned_by_content_hash(queue):
    """Any edit to the shipped queue must be resealed here, deliberately."""

    assert queue["content_sha256"] == PINNED_CONTENT_SHA256


def test_shipped_file_is_canonical_and_byte_stable(queue):
    assert QUEUE_PATH.read_bytes() == canonical_json_bytes(queue) + b"\n"


def test_resealing_the_same_entries_is_deterministic(queue):
    assert seal_queue(queue["entries"]) == queue


def test_every_entry_cites_a_source_and_documents_openness(queue):
    for entry in queue["entries"]:
        assert set(entry) == ENTRY_KEYS
        assert len(entry["source_citation"]) > 20, entry["id"]
        citation = entry["source_citation"]
        assert any(char.isdigit() for char in citation) or "/" in citation, entry["id"]
        assert len(entry["believed_open_because"]) > 20, entry["id"]
        assert entry["progress_definition"], entry["id"]


def test_non_open_entries_say_so_instead_of_implying_openness(queue):
    """'Believed open' is documented, never inferred — and never faked."""

    for entry_id in (
        "continued_fraction_e_pattern",
        "catalan_like_recurrence_holdout",
        "quantified_inequality_families",
    ):
        assert _entry(queue, entry_id)["believed_open_because"].startswith("Not")


def test_domain_counts_split_seven_math_three_physics(queue):
    summary = summarize_queue(queue)
    assert summary["counts"] == {
        "control_rediscovery": 1,
        "entries": 10,
        "math": 7,
        "physics": 3,
        "synthetic": 1,
    }
    assert summary["content_sha256"] == PINNED_CONTENT_SHA256
    assert summary["claims"] == CLAIMS
    assert _entry(queue, "collatz_stopping_time")["domain"] == "math/dynamics"
    assert _entry(queue, "erdos_straus")["domain"] == "math/number_theory"
    assert _entry(queue, "aliquot_276")["domain"] == "math/number_theory"


def test_iterators_partition_the_queue_by_domain(queue):
    math_ids = [entry["id"] for entry in iter_math_targets(queue)]
    physics_ids = [entry["id"] for entry in iter_physics_targets(queue)]
    assert physics_ids == [
        "baryonic_rotation_law",
        "lensing_dynamics_consistency",
        "cluster_missing_mass",
    ]
    assert len(math_ids) == 7
    assert set(math_ids) | set(physics_ids) == {entry["id"] for entry in queue["entries"]}
    assert not set(math_ids) & set(physics_ids)


def test_control_and_synthetic_flags_are_honest(queue):
    controls = {entry["id"] for entry in queue["entries"] if entry["control_rediscovery"]}
    synthetics = {entry["id"] for entry in queue["entries"] if entry["synthetic"]}
    assert controls == {"continued_fraction_e_pattern"}
    assert synthetics == {"catalan_like_recurrence_holdout"}
    for entry in queue["entries"]:
        assert isinstance(entry["control_rediscovery"], bool)
        assert isinstance(entry["synthetic"], bool)


def test_collatz_entry_records_generator_and_prior_probes(queue):
    entry = _entry(queue, "collatz_stopping_time")
    assert entry["machine_form"]["kind"] == "sequence_rows"
    assert entry["machine_form"]["generator"] == "collatz_total_stopping_time"
    assert "sigma(2^k) = k" in entry["progress_definition"]
    assert "sigma(2n) = sigma(n) + 1" in entry["progress_definition"]


def test_machine_forms_use_only_declared_kinds(queue):
    for entry in queue["entries"]:
        kind = entry["machine_form"]["kind"]
        assert kind in MACHINE_FORM_KINDS
        assert set(entry["machine_form"]) == {"kind"} | set(MACHINE_FORM_KINDS[kind])


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_wrong_schema_version_fails_closed(queue):
    tampered = copy.deepcopy(queue)
    tampered["schema_version"] = "invariant-problem-queue-2.0"
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(tampered))


def test_extra_and_missing_top_level_keys_fail_closed(queue):
    extra = copy.deepcopy(queue)
    extra["notes"] = "smuggled"
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(extra))
    missing = {key: value for key, value in copy.deepcopy(queue).items() if key != "entries"}
    with pytest.raises(ProblemQueueError):
        validate_queue(missing)


def test_extra_and_missing_entry_keys_fail_closed(queue):
    extra = copy.deepcopy(queue)
    extra["entries"][0]["confidence"] = "high"
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(extra))
    missing = copy.deepcopy(queue)
    del missing["entries"][0]["source_citation"]
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(missing))


def test_machine_form_violations_fail_closed(queue):
    unknown_kind = copy.deepcopy(queue)
    unknown_kind["entries"][0]["machine_form"] = {"kind": "oracle_lookup", "url": "x"}
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(unknown_kind))
    extra_field = copy.deepcopy(queue)
    extra_field["entries"][0]["machine_form"]["hint"] = "smuggled"
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(extra_field))
    wrong_type = copy.deepcopy(queue)
    wrong_type["entries"][0]["machine_form"]["generator"] = 7
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(wrong_type))


def test_duplicate_ids_fail_closed(queue):
    tampered = copy.deepcopy(queue)
    tampered["entries"][1]["id"] = tampered["entries"][0]["id"]
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(tampered))


def test_empty_or_padded_citation_fails_closed(queue):
    for bad in ("", "   ", " padded citation "):
        tampered = copy.deepcopy(queue)
        tampered["entries"][0]["source_citation"] = bad
        with pytest.raises(ProblemQueueError):
            validate_queue(_reseal(tampered))


def test_float_smuggling_fails_closed_everywhere(queue):
    """Floats must be rejected wherever they hide, before any seal arithmetic."""

    in_machine_form = copy.deepcopy(queue)
    in_machine_form["entries"][0]["machine_form"]["max_point"] = 64.0
    with pytest.raises(ProblemQueueError):
        validate_queue(in_machine_form)
    in_entry = copy.deepcopy(queue)
    in_entry["entries"][2]["statement"] = 3.14
    with pytest.raises(ProblemQueueError):
        validate_queue(in_entry)
    at_top = copy.deepcopy(queue)
    at_top["schema_version"] = 1.0
    with pytest.raises(ProblemQueueError):
        validate_queue(at_top)


def test_flags_must_be_real_booleans(queue):
    as_int = copy.deepcopy(queue)
    as_int["entries"][0]["control_rediscovery"] = 1
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(as_int))
    as_string = copy.deepcopy(queue)
    as_string["entries"][0]["synthetic"] = "false"
    with pytest.raises(ProblemQueueError):
        validate_queue(_reseal(as_string))


def test_tampered_body_or_seal_fails_closed(queue, tmp_path):
    edited_body = copy.deepcopy(queue)
    edited_body["entries"][0]["believed_open_because"] = "Because the corpus lacks it."
    with pytest.raises(ProblemQueueError):
        validate_queue(edited_body)
    corrupt_seal = copy.deepcopy(queue)
    seal = corrupt_seal["content_sha256"]
    corrupt_seal["content_sha256"] = ("0" if seal[0] != "0" else "1") + seal[1:]
    with pytest.raises(ProblemQueueError):
        validate_queue(corrupt_seal)
    with pytest.raises(ProblemQueueError):
        load_queue(_write(tmp_path, corrupt_seal))


def test_load_rejects_noncanonical_encoding(queue, tmp_path):
    """Same content, same seal, different bytes: still refused."""

    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProblemQueueError):
        load_queue(path)


def test_iterators_refuse_an_unvalidated_queue(queue):
    tampered = copy.deepcopy(queue)
    tampered["entries"][0]["domain"] = "astrology"
    with pytest.raises(ProblemQueueError):
        iter_math_targets(_reseal(tampered))
    with pytest.raises(ProblemQueueError):
        iter_physics_targets({"entries": []})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exits_zero_on_the_shipped_queue(capsys):
    assert main(("--queue", str(QUEUE_PATH), "--validate")) == 0
    assert PINNED_CONTENT_SHA256 in capsys.readouterr().out
    assert main(("--queue", str(QUEUE_PATH))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["entries"] == 10


def test_cli_exits_nonzero_on_invalid_or_missing_queue(queue, tmp_path, capsys):
    tampered = copy.deepcopy(queue)
    tampered["entries"][0]["source_citation"] = ""
    path = _write(tmp_path, _reseal(tampered))
    assert main(("--queue", str(path), "--validate")) == 1
    assert "INVALID" in capsys.readouterr().out
    assert main(("--queue", str(tmp_path / "absent.json"), "--validate")) == 1
    with pytest.raises(SystemExit):
        main(())


# ---------------------------------------------------------------------------
# Queue v2: the unsolved dozen
# ---------------------------------------------------------------------------

QUEUE_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "problem_queue_v2.json"
PINNED_V2_CONTENT_SHA256 = "6109e436a97b97b798257adee590ce3ed7200d8eeaf21451d971ce67a19414db"

DOZEN_IDS = (
    "lychrel_196",
    "brocard_problem",
    "erdos_moser",
    "lehmer_totient",
    "giuga_conjecture",
    "odd_perfect_number",
    "odd_untouchable",
    "twin_prime_infinitude",
    "gilbreath_conjecture",
    "ulam_sequence_structure",
    "recaman_coverage",
    "singmaster_conjecture",
)


class TestProblemQueueV2:
    """The v2 queue: all ten v1 entries unchanged plus the twelve-problem dozen."""

    @pytest.fixture()
    def queue_v2(self) -> dict[str, object]:
        return load_queue(QUEUE_V2_PATH)

    def test_loads_seals_and_is_pinned_with_22_entries(self, queue_v2):
        assert queue_v2["schema_version"] == QUEUE_SCHEMA
        assert len(queue_v2["entries"]) == 22
        validate_queue(queue_v2)
        assert queue_v2["content_sha256"] == PINNED_V2_CONTENT_SHA256
        assert QUEUE_V2_PATH.read_bytes() == canonical_json_bytes(queue_v2) + b"\n"
        assert seal_queue(queue_v2["entries"]) == queue_v2

    def test_v1_entries_are_carried_over_unchanged(self, queue, queue_v2):
        assert queue_v2["entries"][:10] == queue["entries"]

    def test_domain_counts_are_nineteen_math_three_physics(self, queue_v2):
        summary = summarize_queue(queue_v2)
        assert summary["counts"] == {
            "control_rediscovery": 1,
            "entries": 22,
            "math": 19,
            "physics": 3,
            "synthetic": 1,
        }
        assert summary["entry_ids"][10:] == list(DOZEN_IDS)

    def test_every_new_entry_cites_real_literature(self, queue_v2):
        for entry_id in DOZEN_IDS:
            entry = _entry(queue_v2, entry_id)
            assert len(entry["source_citation"]) > 40, entry_id
            assert any(char.isdigit() for char in entry["source_citation"]), entry_id
            assert len(entry["believed_open_because"]) > 40, entry_id
            assert "2026" in entry["believed_open_because"], entry_id
            assert entry["progress_definition"], entry_id

    def test_named_citations_are_present_verbatim(self, queue_v2):
        expectations = {
            "lychrel_196": ("A023108", "VanLandingham"),
            "brocard_problem": ("Brocard", "Ramanujan", "Berndt", "Galway"),
            "erdos_moser": ("Moser", "Gallot", "Moree", "Zudilin", "2011"),
            "lehmer_totient": ("Lehmer", "1932", "B37"),
            "giuga_conjecture": ("Giuga", "1950", "Borwein", "1996"),
            "odd_perfect_number": ("Ochem", "Rao", "2012"),
            "odd_untouchable": ("B10", "Erdos", "1973"),
            "twin_prime_infinitude": ("A8", "Zhang", "Polymath", "A007508"),
            "gilbreath_conjecture": ("Gilbreath", "Odlyzko", "1993"),
            "ulam_sequence_structure": ("Ulam", "1964", "Steinerberger", "A002858"),
            "recaman_coverage": ("A005132",),
            "singmaster_conjecture": ("Singmaster", "1971", "Kane", "2007"),
        }
        for entry_id, needles in expectations.items():
            citation = _entry(queue_v2, entry_id)["source_citation"]
            for needle in needles:
                assert needle in citation, (entry_id, needle)

    def test_all_twelve_flags_are_false(self, queue_v2):
        for entry_id in DOZEN_IDS:
            entry = _entry(queue_v2, entry_id)
            assert entry["control_rediscovery"] is False, entry_id
            assert entry["synthetic"] is False, entry_id

    def test_machine_forms_reuse_only_registered_kinds(self, queue_v2):
        kinds = {}
        for entry_id in DOZEN_IDS:
            form = _entry(queue_v2, entry_id)["machine_form"]
            assert form["kind"] in MACHINE_FORM_KINDS, entry_id
            assert set(form) == {"kind"} | set(MACHINE_FORM_KINDS[form["kind"]]), entry_id
            kinds.setdefault(form["kind"], []).append(entry_id)
        assert sorted(kinds) == [
            "diophantine_family",
            "integer_trajectory",
            "sequence_rows",
        ]
        assert len(kinds["diophantine_family"]) == 6
        assert len(kinds["integer_trajectory"]) == 2
        assert len(kinds["sequence_rows"]) == 4


# ---------------------------------------------------------------------------
# Queue v3: the FLT-adjacent exponent-Diophantine targets
# ---------------------------------------------------------------------------

QUEUE_V3_PATH = Path(__file__).resolve().parents[1] / "configs" / "problem_queue_v3.json"
PINNED_V3_CONTENT_SHA256 = "76f3d3e9ec639c63da86de995b938439414c79d93eda1374770dff9212ccc299"

FLT_ADJACENT_IDS = ("beal_conjecture", "fermat_catalan", "erdos_straus_sweeper_target")


class TestProblemQueueV3:
    """The v3 queue: all 22 v2 entries unchanged plus the three sweeper targets."""

    @pytest.fixture()
    def queue_v2(self) -> dict[str, object]:
        return load_queue(QUEUE_V2_PATH)

    @pytest.fixture()
    def queue_v3(self) -> dict[str, object]:
        return load_queue(QUEUE_V3_PATH)

    def test_loads_seals_and_is_pinned_with_25_entries(self, queue_v3):
        assert queue_v3["schema_version"] == QUEUE_SCHEMA
        assert len(queue_v3["entries"]) == 25
        validate_queue(queue_v3)
        assert queue_v3["content_sha256"] == PINNED_V3_CONTENT_SHA256
        assert QUEUE_V3_PATH.read_bytes() == canonical_json_bytes(queue_v3) + b"\n"
        assert seal_queue(queue_v3["entries"]) == queue_v3

    def test_v2_entries_are_carried_over_byte_unchanged(self, queue_v2, queue_v3):
        assert queue_v3["entries"][:22] == queue_v2["entries"]
        assert canonical_json_bytes(queue_v2["entries"])[1:-1] in canonical_json_bytes(
            queue_v3["entries"]
        )

    def test_domain_counts_are_twenty_two_math_three_physics(self, queue_v3):
        summary = summarize_queue(queue_v3)
        assert summary["counts"] == {
            "control_rediscovery": 1,
            "entries": 25,
            "math": 22,
            "physics": 3,
            "synthetic": 1,
        }
        assert summary["entry_ids"][22:] == list(FLT_ADJACENT_IDS)

    def test_every_new_entry_cites_real_literature(self, queue_v3):
        for entry_id in FLT_ADJACENT_IDS:
            entry = _entry(queue_v3, entry_id)
            assert entry["domain"] == "math/number_theory", entry_id
            assert len(entry["source_citation"]) > 40, entry_id
            assert any(char.isdigit() for char in entry["source_citation"]), entry_id
            assert len(entry["believed_open_because"]) > 40, entry_id
            assert "2026" in entry["believed_open_because"], entry_id
            assert entry["progress_definition"], entry_id
            assert entry["control_rediscovery"] is False, entry_id
            assert entry["synthetic"] is False, entry_id

    def test_named_citations_are_present_verbatim(self, queue_v3):
        expectations = {
            "beal_conjecture": ("Mauldin", "1997", "Notices", "1,000,000", "Norvig"),
            "fermat_catalan": ("Darmon", "Granville", "1995", "Poonen", "2007"),
            "erdos_straus_sweeper_target": (
                "Elsholtz",
                "Tao",
                "1107.1010",
                "Salez",
                "1406.6307",
            ),
        }
        for entry_id, needles in expectations.items():
            citation = _entry(queue_v3, entry_id)["source_citation"]
            for needle in needles:
                assert needle in citation, (entry_id, needle)

    def test_all_three_are_diophantine_family_machine_forms(self, queue_v3):
        for entry_id in FLT_ADJACENT_IDS:
            form = _entry(queue_v3, entry_id)["machine_form"]
            assert form["kind"] == "diophantine_family", entry_id
            assert set(form) == {"kind"} | set(MACHINE_FORM_KINDS["diophantine_family"])
        assert _entry(queue_v3, "beal_conjecture")["machine_form"]["parameter"] == "base_bound"
        assert _entry(queue_v3, "fermat_catalan")["machine_form"]["parameter_min"] == 1

    def test_sweeper_reentry_cross_references_the_v1_entry(self, queue_v3):
        """The Erdos-Straus re-entry must point at the original entry, keep its
        parameterization compatible, and claim no separate mathematical target."""

        original = _entry(queue_v3, "erdos_straus")
        reentry = _entry(queue_v3, "erdos_straus_sweeper_target")
        assert reentry["id"] != original["id"]
        assert "erdos_straus" in reentry["statement"]
        assert "erdos_straus" in reentry["progress_definition"]
        assert reentry["machine_form"]["parameter"] == original["machine_form"]["parameter"]
        assert (
            reentry["machine_form"]["parameter_min"]
            == original["machine_form"]["parameter_min"]
        )
        assert "4/n = 1/x + 1/y + 1/z" in reentry["machine_form"]["equation"]
        assert "10^17" in reentry["believed_open_because"]

    def test_beal_landscape_is_phrased_as_documented_report(self, queue_v3):
        """The verification landscape is Norvig's documented report, not our claim."""

        entry = _entry(queue_v3, "beal_conjecture")
        assert "Norvig" in entry["believed_open_because"]
        assert "report" in entry["believed_open_because"]
        assert "250,000" in entry["believed_open_because"]

    def test_ten_known_fermat_catalan_solutions_are_stated(self, queue_v3):
        statement = _entry(queue_v3, "fermat_catalan")["statement"]
        assert "ten solutions" in statement
        for known in ("2^5 + 7^2 = 3^4", "7^3 + 13^2 = 2^9", "2^7 + 17^3 = 71^2"):
            assert known in statement
