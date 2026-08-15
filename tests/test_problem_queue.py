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
