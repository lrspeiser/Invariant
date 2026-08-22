"""Gates for the creativity measure wired into the loop's own receipt.

The measure was already tested as a measure.  What is tested here is the wiring, and the wiring
has exactly one claim to defend: *the number in the receipt is a measurement of the programs in
the receipt*.  A creativity block is not a note the campaign left about itself -- it is a
statement that can be recomputed from the sealed population sitting next to it, and this file is
the set of ways of making that statement false:

* edit a number and leave the seal -- caught by the seal;
* edit a number and re-seal the block honestly -- caught only by recomputation, which is why
  recomputation exists;
* declare a tolerance that merges every behaviour into one, then compute honestly under it --
  caught by pinning the declared parameters, because a receipt may not choose the measurement
  that flatters it;
* quietly drop a program that produced no behaviour, so the measure would not have noticed --
  caught by the population digest;
* seal no creativity block at all -- caught as absence.

Each must fail replay, and the good receipt must pass it.  The last test in the file is the
control on the control: the honest receipt replays clean, so a red result here means tampering
was detected and not that the detector fires at everything.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import pytest

from sigma_theory_compiler import creativity_measure as cm
from sigma_theory_compiler import creativity_receipt as cr
from sigma_theory_compiler import funsearch_loop as fl
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

FAST = fl.LoopConfig(generations=6, islands=2, proposals_per_call=3, seed=20260819)


def _run(tmp_path: Path, key: str = "blinded_sequence_rule", **overrides) -> dict:
    """One real search, sealed exactly as a campaign seals it."""

    problem = fl.declared_problems()[key]
    config = dataclasses.replace(FAST, **overrides)
    governor = fl.SpendGovernor(tmp_path / f"ledger-{key}-{config.seed}.json", 500, 5000, 1)
    block = fl.run_problem(
        problem,
        config,
        fl.MockMutationProposer(config.seed, problem.mutation_bank),
        governor,
        **({"seeded_programs": fl.RESPONSE_PROBE_PROGRAMS} if "response" in key else {}),
    )
    block["run_label"] = key
    return block


@pytest.fixture(scope="module")
def block(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("creativity"))


@pytest.fixture(scope="module")
def receipt(block: dict) -> dict:
    return {"problems": [copy.deepcopy(block)]}


# ---------------------------------------------------------------------------
# The loop measures itself
# ---------------------------------------------------------------------------


def test_a_run_seals_its_own_behavioural_diversity(block: dict) -> None:
    """The point of the task: the number arrives with the run, not from a later script."""

    sealed = block["creativity"]
    assert sealed["schema_version"] == cr.SCHEMA
    assert sealed["measured_origin"] == "proposed"
    assert sealed["measure"]["schema_version"] == cm.SCHEMA
    assert int(sealed["measure"]["population"]["distinct_behaviours"]) > 1
    assert float(sealed["measure"]["effective_novel_behaviours"]) > 1.0


def test_the_block_is_measured_over_the_list_that_is_sealed(block: dict) -> None:
    """Not a separately-assembled population: the same records, or it is not checkable."""

    proposed = [
        record for record in block["sealed_programs"] if record["origin"] == "proposed"
    ]
    sealed = block["creativity"]
    assert sealed["programs_offered"] == len(proposed)
    assert sealed["measured_population_sha256"] == canonical_sha256(
        sorted(record["program_sha256"] for record in proposed)
    )
    without = sum(1 for record in proposed if not record["outputs"])
    assert sealed["programs_without_a_behaviour"] == without
    assert int(sealed["measure"]["population"]["programs"]) == len(proposed) - without


def test_seeds_and_planted_probes_are_not_counted_as_the_proposers_work(tmp_path: Path) -> None:
    block = _run(tmp_path, "blinded_response_law")
    origins = {record["origin"] for record in block["sealed_programs"]}
    assert origins - {"proposed"}, "this run must contain non-proposed programs to be a control"
    assert block["creativity"]["programs_offered"] == sum(
        1 for record in block["sealed_programs"] if record["origin"] == "proposed"
    )


def test_nothing_on_the_certificate_path_is_a_float(block: dict) -> None:
    """``canonical_sha256`` rejects floats outright, so sealing the block is the proof."""

    sealed = block["creativity"]
    body = {key: value for key, value in sealed.items() if key != "content_sha256"}
    assert canonical_sha256(body) == sealed["content_sha256"]
    with pytest.raises(SchemaViolation):
        canonical_sha256({"leaked": 1.5})


def test_measuring_the_same_population_twice_is_byte_identical(block: dict) -> None:
    first = cr.creativity_block(block["sealed_programs"])
    second = cr.creativity_block(block["sealed_programs"])
    assert first == second
    assert first["content_sha256"] == block["creativity"]["content_sha256"]


# ---------------------------------------------------------------------------
# The control: a block that disagrees with its own programs must fail replay
# ---------------------------------------------------------------------------


def _faults(receipt: dict) -> set[str]:
    report = cr.replay_creativity(receipt)
    assert report["identical"] is False
    return {fault["fault"] for fault in report["mismatches"]}


def _shifted(value: str) -> str:
    """A different decimal of the same shape, so the tamper cannot land on the true value."""

    return format(Decimal(value) + Decimal("1.5"), ".6f")


def _reseal(sealed: dict) -> None:
    """Re-hash a block so only recomputation, not the seal, can catch the edit."""

    sealed["measure"]["content_sha256"] = canonical_sha256(
        {key: value for key, value in sealed["measure"].items() if key != "content_sha256"}
    )
    sealed["content_sha256"] = canonical_sha256(
        {key: value for key, value in sealed.items() if key != "content_sha256"}
    )


def test_an_edited_number_fails_replay(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["problems"][0]["creativity"]["measure"]["effective_novel_behaviours"] = "99.000000"
    assert _faults(tampered) == {"block_seal_broken", "recomputation_disagrees"}


def test_an_edited_and_resealed_number_still_fails_replay(receipt: dict) -> None:
    """The test that makes the block worth reading: a tidy forgery is still a forgery."""

    tampered = copy.deepcopy(receipt)
    sealed = tampered["problems"][0]["creativity"]
    sealed["measure"]["effective_novel_behaviours"] = "99.000000"
    _reseal(sealed)
    report = cr.replay_creativity(tampered)
    assert report["identical"] is False
    assert {fault["fault"] for fault in report["mismatches"]} == {"recomputation_disagrees"}
    assert "measure.effective_novel_behaviours" in {
        fault.get("field") for fault in report["mismatches"]
    }


def test_a_widened_tolerance_cannot_be_declared_into_the_receipt(receipt: dict) -> None:
    """Merging every behaviour into one would otherwise make a collapsed run look focused."""

    tampered = copy.deepcopy(receipt)
    sealed = tampered["problems"][0]["creativity"]
    honest = int(sealed["measure"]["population"]["distinct_behaviours"])
    sealed["measure"] = cm.measure_creativity(
        tampered["problems"][0]["sealed_programs"], tolerance=10.0, origin="proposed"
    )
    merged = int(sealed["measure"]["population"]["distinct_behaviours"])
    assert merged < honest, "a tolerance of 10 must actually merge behaviours to be a tamper"
    _reseal(sealed)
    assert _faults(tampered) == {
        "declared_measurement_parameters_changed",
        "recomputation_disagrees",
    }


def test_dropping_a_program_the_measure_ignored_still_fails_replay(receipt: dict) -> None:
    """A proposal that never executed has no behaviour, so only the digest can catch its loss."""

    tampered = copy.deepcopy(receipt)
    programs = tampered["problems"][0]["sealed_programs"]
    dead = [
        index
        for index, record in enumerate(programs)
        if record["origin"] == "proposed" and not record["outputs"]
    ]
    if not dead:
        pytest.skip("this run produced no unexecutable proposals to drop")
    programs.pop(dead[0])
    report = cr.replay_creativity(tampered)
    assert report["identical"] is False
    fields = {fault.get("field") for fault in report["mismatches"]}
    assert "measured_population_sha256" in fields
    assert report["mismatches"][0]["run_label"] == "blinded_sequence_rule"


def test_a_run_that_measured_nothing_fails_replay(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["problems"][0].pop("creativity")
    assert _faults(tampered) == {"creativity_block_absent"}


def test_the_honest_receipt_replays_clean(receipt: dict) -> None:
    """The control on the control: the detector is not simply always red."""

    report = cr.replay_creativity(receipt)
    assert report["identical"] is True
    assert report["mismatches"] == []
    assert report["blocks_checked"] == 1
    assert len(report["checks"]) == 3


def test_replay_from_receipt_fails_when_the_creativity_block_is_forged(receipt: dict) -> None:
    """The whole-receipt replay, not just the creativity one, must go red."""

    clean = fl.replay_from_receipt(receipt)
    assert clean["identical"] is True
    assert clean["creativity"]["identical"] is True

    tampered = copy.deepcopy(receipt)
    sealed = tampered["problems"][0]["creativity"]
    sealed["measure"]["wasted_variation_ratio"] = _shifted(
        sealed["measure"]["wasted_variation_ratio"]
    )
    _reseal(sealed)
    forged = fl.replay_from_receipt(tampered)
    assert forged["mismatches"] == [], "no program score was touched; only the claim about them"
    assert forged["creativity"]["identical"] is False
    assert forged["identical"] is False


def test_validate_receipt_refuses_a_forged_creativity_block(receipt: dict) -> None:
    """Validation is run-aborting, so a self-contradicting receipt is never written."""

    tampered = copy.deepcopy(receipt)
    sealed = tampered["problems"][0]["creativity"]
    sealed["measure"]["population"]["distinct_behaviours"] = 999
    _reseal(sealed)
    with pytest.raises(fl.FunSearchError) as caught:
        fl._validate_creativity(tampered)
    assert "disagrees with the programs" in str(caught.value)
    fl._validate_creativity(receipt)


# ---------------------------------------------------------------------------
# The A/B
# ---------------------------------------------------------------------------


def test_a_receipt_compared_with_itself_moves_nothing(receipt: dict) -> None:
    report = cr.compare_receipts(receipt, receipt)
    assert report["verdict"] == "unchanged"
    assert report["tally_on_the_headline"]["better"] == 0
    assert report["tally_on_the_headline"]["worse"] == 0
    for row in report["rows"].values():
        assert {item["delta"] for item in row["comparison"]["rows"]} == {"0.000000"}
    assert report["content_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )


def test_the_ab_names_the_run_that_tried_more_different_things(
    receipt: dict, tmp_path: Path
) -> None:
    """A longer search on the same problem explores more, and the report has to say so."""

    longer = {"problems": [_run(tmp_path, generations=18, islands=3, seed=99)]}
    before = float(
        receipt["problems"][0]["creativity"]["measure"]["effective_novel_behaviours"]
    )
    after = float(longer["problems"][0]["creativity"]["measure"]["effective_novel_behaviours"])
    assert after > before, "the longer run must actually be more diverse for this to be a test"

    report = cr.compare_receipts(receipt, longer)
    assert report["verdict"] == "better"
    row = report["rows"]["blinded_sequence_rule"]["comparison"]
    headline = next(
        item for item in row["rows"] if item["metric"] == "effective_novel_behaviours"
    )
    assert headline["verdict"] == "better"
    assert float(headline["delta"]) == pytest.approx(after - before, abs=1e-6)

    backwards = cr.compare_receipts(longer, receipt)
    assert backwards["verdict"] == "worse"


def test_rewriting_a_behaviour_cannot_win_the_ab(receipt: dict) -> None:
    """The anti-gaming control at receipt level: padding with spellings must not read better."""

    padded = copy.deepcopy(receipt)
    programs = padded["problems"][0]["sealed_programs"]
    twin = next(
        record
        for record in programs
        if record["origin"] == "proposed" and record["outputs"]
    )
    for index in range(25):
        clone = copy.deepcopy(twin)
        clone["source"] = f"{twin['source']}\n# spelling {index}\n"
        clone["program_sha256"] = canonical_sha256(clone["source"])
        programs.append(clone)
    padded["problems"][0]["creativity"] = cr.creativity_block(programs)

    report = cr.compare_receipts(receipt, padded)
    row = report["rows"]["blinded_sequence_rule"]["comparison"]
    headline = next(
        item for item in row["rows"] if item["metric"] == "effective_novel_behaviours"
    )
    waste = next(item for item in row["rows"] if item["metric"] == "wasted_variation_ratio")
    assert float(headline["delta"]) <= 0.0, "26 spellings of one behaviour is not more creative"
    assert waste["verdict"] == "worse"
    assert report["verdict"] in {"worse", "unchanged"}


def test_a_receipt_without_blocks_is_measured_and_says_so(receipt: dict) -> None:
    """An older receipt is comparable, but the report never calls a recomputation a promise."""

    older = copy.deepcopy(receipt)
    older["problems"][0].pop("creativity")
    read = cr.receipt_creativity(older)
    label = read["run_labels"]["blinded_sequence_rule"]
    assert label["provenance"] == "recomputed_from_sealed_programs"
    assert label["headline"] == cr.headline_numbers(receipt["problems"][0]["creativity"])
    assert cr.receipt_creativity(receipt)["run_labels"]["blinded_sequence_rule"][
        "provenance"
    ] == "sealed_in_the_receipt"


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _write(receipt: dict, path: Path) -> Path:
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_the_cli_replays_a_receipt_and_fails_on_a_forgery(
    receipt: dict, tmp_path: Path, capsys
) -> None:
    good = _write(receipt, tmp_path / "good.json")
    assert cr.main(["replay", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True

    tampered = copy.deepcopy(receipt)
    sealed = tampered["problems"][0]["creativity"]
    sealed["measure"]["known_collapse_fraction"] = _shifted(
        sealed["measure"]["known_collapse_fraction"]
    )
    _reseal(sealed)
    bad = _write(tampered, tmp_path / "bad.json")
    assert cr.main(["replay", str(bad)]) == 1
    assert json.loads(capsys.readouterr().out)["identical"] is False


def test_the_cli_compares_two_receipts_and_writes_a_sealed_report(
    receipt: dict, tmp_path: Path, capsys
) -> None:
    longer = {"problems": [_run(tmp_path, generations=18, islands=3, seed=99)]}
    before = _write(receipt, tmp_path / "before.json")
    after = _write(longer, tmp_path / "after.json")
    output = tmp_path / "ab.json"
    assert cr.main(["compare", "--before", str(before), "--after", str(after),
                    "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["verdict"] == "better"
    assert written["content_sha256"] == canonical_sha256(
        {key: value for key, value in written.items() if key != "content_sha256"}
    )


def test_the_cli_reports_one_receipt(receipt: dict, tmp_path: Path, capsys) -> None:
    path = _write(receipt, tmp_path / "one.json")
    assert cr.main(["show", str(path)]) == 0
    read = json.loads(capsys.readouterr().out)
    assert read["totals"]["run_labels"] == 1
    assert read["totals"]["distinct_behaviours"] == int(
        receipt["problems"][0]["creativity"]["measure"]["population"]["distinct_behaviours"]
    )


def test_the_cli_refuses_a_file_that_is_not_a_receipt(tmp_path: Path, capsys) -> None:
    path = tmp_path / "not-a-receipt.json"
    path.write_text(json.dumps({"nothing": "here"}), encoding="utf-8")
    assert cr.main(["show", str(path)]) == 2
    assert "problems" in capsys.readouterr().err


def test_two_receipts_with_no_shared_run_label_are_not_silently_equal(receipt: dict) -> None:
    other = copy.deepcopy(receipt)
    other["problems"][0]["run_label"] = "blinded_response_law"
    report = cr.compare_receipts(receipt, other)
    assert report["verdict"] == "no_shared_run_labels"
    assert report["only_in_before"] == ["blinded_sequence_rule"]
    assert report["only_in_after"] == ["blinded_response_law"]
