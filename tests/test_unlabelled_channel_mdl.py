"""Tests for unlabelled channels and their description-length price.

The module makes four kinds of claim, and each is tested by trying to break it:

* **the codelengths are lengths of messages** -- so the tests decode the messages, and a
  reconstruction that missed by more than half a resolution step would fail;
* **widening the slot is free for a law that ignores the width** -- so the tests price the
  same law on a three-channel table and on its one-channel collapse and demand equality, with
  two controls that must *not* come out equal;
* **the obstruction floor is a lower bound over all functions ignoring a channel** -- so the
  tests brute-force thousands of such functions and demand that not one of them codes the data
  below the floor;
* **using a channel is paid for** -- so the tests take a channel with real but tiny signal and
  demand it is refused at the declared price and admitted only when the price is set to zero.

Nothing here compares floats.  Every quantity on the certificate path is an ``int`` or a
``Fraction``, and one test walks the receipt asserting that no float survived into it.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import unlabelled_channel_mdl as ucm

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "runs" / "math" / "unlabelled-channel-mdl" / "receipt.json"


@pytest.fixture(scope="module")
def tables() -> dict[str, ucm.ChannelTable]:
    return ucm.declared_tables()


@pytest.fixture(scope="module")
def laws() -> dict[str, ucm.Expr]:
    return ucm.declared_law_space()


@pytest.fixture(scope="module")
def built() -> dict:
    return ucm.build_receipt()


# ---------------------------------------------------------------------------
# The codes are codes: every length is the length of something that decodes
# ---------------------------------------------------------------------------


def test_gamma_code_round_trips_and_its_length_is_the_declared_one() -> None:
    for magnitude in list(range(1, 200)) + [1023, 1024, 1025, 2**20]:
        bits = ucm.encode_gamma(magnitude)
        assert len(bits) == ucm.gamma_bits(magnitude)
        assert ucm.decode_gamma(bits, 0) == (magnitude, len(bits))


def test_gamma_code_is_prefix_free_on_a_bounded_alphabet() -> None:
    words = [ucm.encode_gamma(magnitude) for magnitude in range(1, 400)]
    for index, word in enumerate(words):
        for other in words[index + 1 :]:
            assert not other.startswith(word)
            assert not word.startswith(other)


def test_signed_integer_code_round_trips_and_is_monotone_in_magnitude() -> None:
    previous = 0
    for value in range(300):
        bits = ucm.encode_integer(value)
        assert len(bits) == ucm.integer_code_bits(value)
        assert ucm.decode_integer(bits, 0) == (value, len(bits))
        assert ucm.integer_code_bits(-value) == ucm.integer_code_bits(value)
        assert ucm.decode_integer(ucm.encode_integer(-value), 0)[0] == -value
        assert ucm.integer_code_bits(value) >= previous
        previous = ucm.integer_code_bits(value)
    assert ucm.integer_code_bits(0) == 1


def test_a_concatenated_residual_message_is_unambiguous() -> None:
    codes = [0, 1, -1, 7, -128, 0, 512]
    message = ucm.encode_residuals(codes)
    assert len(message) == ucm.data_bits(codes)
    position = 0
    recovered = []
    for _ in codes:
        value, position = ucm.decode_integer(message, position)
        recovered.append(value)
    assert recovered == codes
    assert position == len(message)


def test_every_declared_law_encodes_and_decodes_back_to_itself(laws) -> None:
    for law_id, expr in laws.items():
        bits = ucm.encode_expression(expr)
        recovered, position = ucm.decode_expression(bits)
        assert position == len(bits), law_id
        assert recovered == expr, law_id
        assert ucm.model_bits(expr) == len(bits), law_id


def test_a_codelength_is_hand_checkable(tables, laws) -> None:
    """Pin the arithmetic to numbers a reader can recompute with a pencil."""

    assert ucm.NODE_TAG_BITS == 3
    assert ucm.CHANNEL_INDEX_BITS == 4
    # mul(3, u): mul tag 3, const(3) = tag 3 + sign 1 + gamma(4) 5 + gamma(1) 1, chan 3 + 4.
    assert ucm.model_bits(laws["three_u"]) == 3 + (3 + 1 + 5 + 1) + (3 + 4) == 20
    # The exact law leaves every residual at zero, and a zero costs the one-bit floor.
    table = tables["main"]
    predicted = ucm.predictions(table, laws["u_times_two_plus_w"])
    assert predicted is not None
    assert set(ucm.residual_codes(table, predicted)) == {0}
    assert ucm.data_bits(ucm.residual_codes(table, predicted)) == len(table.rows) == 160


def test_the_residual_message_reconstructs_every_observation(tables, laws) -> None:
    table = tables["main"]
    for law_id, expr in laws.items():
        predicted = ucm.predictions(table, expr)
        assert predicted is not None, law_id
        codes = ucm.residual_codes(table, predicted)
        message = ucm.encode_residuals(codes)
        assert len(message) == ucm.data_bits(codes)
        recovered = ucm.decode_observations(message, table, predicted)
        for observed, value, resolution in zip(
            table.observations, recovered, table.resolutions, strict=True
        ):
            assert abs(observed - value) <= resolution / 2


def test_the_quantiser_is_exact_integer_arithmetic() -> None:
    assert ucm._round_half_up(Fraction(1, 2)) == 1
    assert ucm._round_half_up(Fraction(-1, 2)) == 0
    assert ucm._round_half_up(Fraction(-3, 2)) == -1
    assert ucm._round_half_up(Fraction(499, 1000)) == 0
    assert ucm._ceil_fraction(Fraction(-1, 2)) == 0
    assert ucm._ceil_fraction(Fraction(7, 3)) == 3


# ---------------------------------------------------------------------------
# Widening the slot is free for a law that does not use the width
# ---------------------------------------------------------------------------


def test_a_law_ignoring_the_extra_channels_costs_what_it_costs_on_one_channel(
    tables, laws
) -> None:
    wide = tables["main"]
    narrow = ucm.collapsed_table(wide, (0,))
    compared = 0
    for law_id, expr in laws.items():
        referenced = ucm.referenced_channels(expr)
        if referenced and max(referenced) > 0:
            continue
        wide_length = ucm.codelength(wide, law_id, expr)
        narrow_length = ucm.codelength(narrow, law_id, expr)
        assert wide_length is not None and narrow_length is not None
        assert wide_length.model_bits == narrow_length.model_bits, law_id
        assert wide_length.data_bits == narrow_length.data_bits, law_id
        assert wide_length.admission_bits == narrow_length.admission_bits, law_id
        assert wide_length.total_bits == narrow_length.total_bits, law_id
        compared += 1
    assert compared >= 5


def test_the_admission_fee_is_charged_per_used_channel_not_per_declared_channel(
    tables, laws
) -> None:
    """The fee must scale with what the law uses, or widening the slot would tax everyone."""

    table = tables["main"]
    assert table.arity == 3
    one = ucm.codelength(table, "three_u", laws["three_u"])
    two = ucm.codelength(table, "u_times_two_plus_w", laws["u_times_two_plus_w"])
    three = ucm.codelength(
        table, "all_three", laws["u_times_two_plus_w_plus_x_over_768"]
    )
    assert one is not None and two is not None and three is not None
    fee = ucm.DECLARED_POLICY.channel_admission_bits
    assert (one.admission_bits, two.admission_bits, three.admission_bits) == (
        fee,
        2 * fee,
        3 * fee,
    )


def test_control_a_channel_using_law_is_not_expressible_on_one_channel(tables, laws) -> None:
    narrow = ucm.collapsed_table(tables["main"], (0,))
    with pytest.raises(ucm.ChannelMDLError):
        ucm.codelength(narrow, "reads_channel_one", laws["u_times_two_plus_w"])


def test_control_a_channel_using_law_does_not_cost_the_same_as_a_one_channel_law(
    tables, laws
) -> None:
    wide = tables["main"]
    ignoring = ucm.codelength(wide, "three_u", laws["three_u"])
    using = ucm.codelength(wide, "u_times_two_plus_w", laws["u_times_two_plus_w"])
    assert ignoring is not None and using is not None
    assert ignoring.total_bits != using.total_bits
    assert using.data_bits < ignoring.data_bits
    assert using.admission_bits > ignoring.admission_bits


def test_an_inert_channel_term_is_ignored_behaviourally_but_still_costs_model_bits(
    tables, laws
) -> None:
    table = tables["main"]
    plain = ucm.codelength(table, "plain", laws["u_times_two_plus_w"])
    inert = ucm.codelength(table, "inert", laws["u_times_two_plus_w_plus_inert_x"])
    assert plain is not None and inert is not None
    assert 2 in ucm.referenced_channels(laws["u_times_two_plus_w_plus_inert_x"])
    assert inert.used_channels == plain.used_channels == (0, 1)
    assert inert.admission_bits == plain.admission_bits
    assert inert.data_bits == plain.data_bits
    assert inert.model_bits > plain.model_bits


# ---------------------------------------------------------------------------
# Dependence is measured, not read
# ---------------------------------------------------------------------------


def test_dependence_is_measured_from_behaviour_not_from_the_written_channels(tables) -> None:
    table = tables["main"]
    written = ucm.add(ucm.mul(ucm.const(3), ucm.chan(0)), ucm.mul(ucm.const(0), ucm.chan(2)))
    unwritten = ucm.mul(ucm.const(3), ucm.chan(0))
    assert 2 in ucm.referenced_channels(written)
    assert 2 not in ucm.referenced_channels(unwritten)
    assert ucm.channel_dependence(table, written, 2).verdict == ucm.INDEPENDENT
    assert ucm.channel_dependence(table, unwritten, 2).verdict == ucm.INDEPENDENT
    assert ucm.channel_dependence(table, written, 0).verdict == ucm.DEPENDS


def test_an_incomplete_design_returns_undecided_rather_than_a_guess(tables, laws) -> None:
    incomplete = tables["incomplete"]
    assert incomplete.witness_pairs(1) == ()
    verdict = ucm.channel_dependence(incomplete, laws["u_times_two_plus_w"], 1)
    assert verdict.verdict == ucm.UNDECIDED
    assert verdict.witness_pairs == 0
    # And an undecidable channel is charged for rather than waved through.
    length = ucm.codelength(incomplete, "law", laws["three_u"])
    assert length is not None
    assert 1 in length.used_channels


def test_two_identical_rows_are_not_treated_as_evidence() -> None:
    table = ucm.ChannelTable(
        table_id="duplicated",
        arity=2,
        rows=((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))),
        observations=(Fraction(3), Fraction(3)),
        resolutions=(Fraction(1, 8), Fraction(1, 8)),
    )
    assert table.witness_pairs(1) == ()
    assert ucm.channel_dependence(table, ucm.chan(1), 1).verdict == ucm.UNDECIDED


# ---------------------------------------------------------------------------
# The obstruction floor: a lower bound over *all* functions ignoring a channel
# ---------------------------------------------------------------------------


def _small_table() -> ucm.ChannelTable:
    levels = (
        (Fraction(1), Fraction(2)),
        (Fraction(0), Fraction(1), Fraction(2)),
    )
    return ucm.factorial_table(
        "small", levels, lambda row: row[0] * (2 + row[1]), Fraction(1, 8)
    )


def _data_bits_of_ignoring_function(
    table: ucm.ChannelTable, dropped: tuple[int, ...], choose
) -> int:
    """Data bits of a function constant on every group that differs only in ``dropped``."""

    total = 0
    for key, members in table.contexts(dropped).items():
        prediction = choose(key, members)
        for position in members:
            residual = (
                table.observations[position] - prediction
            ) / table.resolutions[position]
            total += ucm.integer_code_bits(ucm._round_half_up(residual))
    return total


def test_no_exhaustively_swept_ignoring_function_beats_the_floor() -> None:
    table = _small_table()
    floor = ucm.ignoring_codelength_floor(table, (1,))["floor_data_bits"]
    step = Fraction(1, 32)
    best_seen = None
    groups = table.contexts((1,))
    # Minimise each group independently over a fine rational sweep; the group minima are
    # independent, so their sum is the exact minimum of the sweep over the whole table.
    total_minimum = 0
    for members in groups.values():
        low = min(table.observations[position] for position in members) - 1
        high = max(table.observations[position] for position in members) + 1
        candidate = low
        group_best = None
        while candidate <= high:
            bits = sum(
                ucm.integer_code_bits(
                    ucm._round_half_up(
                        (table.observations[position] - candidate)
                        / table.resolutions[position]
                    )
                )
                for position in members
            )
            group_best = bits if group_best is None else min(group_best, bits)
            candidate += step
        total_minimum += group_best
    best_seen = total_minimum
    assert best_seen >= floor
    assert floor > len(table.rows) * ucm.integer_code_bits(0)


def test_no_randomly_drawn_ignoring_function_beats_the_floor(tables) -> None:
    table = tables["main"]
    floor = ucm.ignoring_codelength_floor(table, (1,))["floor_data_bits"]
    rng = random.Random(20260819)
    for _ in range(200):
        picks: dict[tuple[Fraction, ...], Fraction] = {}

        def choose(key, members, picks=picks, rng=rng):
            if key not in picks:
                sample = table.observations[rng.choice(members)]
                picks[key] = sample + Fraction(rng.randint(-40, 40), 512)
            return picks[key]

        assert _data_bits_of_ignoring_function(table, (1,), choose) >= floor


def test_the_best_declared_law_ignoring_the_channel_respects_the_floor(tables, laws) -> None:
    table = tables["main"]
    floor = ucm.ignoring_codelength_floor(table, (1,))["floor_data_bits"]
    for law_id, expr in laws.items():
        if ucm.channel_dependence(table, expr, 1).verdict != ucm.INDEPENDENT:
            continue
        length = ucm.codelength(table, law_id, expr)
        assert length is not None
        assert length.data_bits >= floor, law_id


def test_a_table_with_no_extra_dependence_yields_only_the_trivial_floor(tables) -> None:
    table = tables["flat"]
    for channel in (1, 2):
        report = ucm.ignoring_codelength_floor(table, (channel,))
        assert report["floor_data_bits"] == report["trivial_floor_data_bits"]
        assert report["groups_forcing_bits"] == 0


def test_the_obstruction_does_not_move_when_the_price_moves(tables, laws) -> None:
    table = tables["main"]
    expensive = ucm.adjudicate_channel(table, laws, 1, ucm.ChannelPolicy(10**6, 1, 16))
    free = ucm.adjudicate_channel(table, laws, 1, ucm.ChannelPolicy(0, 0, 16))
    assert (
        expensive["obstruction"]["floor_data_bits"] == free["obstruction"]["floor_data_bits"]
    )
    assert expensive["obstruction"]["verdict"] == free["obstruction"]["verdict"] == ucm.CERTIFIED
    assert expensive["adoption"]["verdict"] == ucm.NOT_ADOPTED
    assert free["adoption"]["verdict"] == ucm.ADOPTED


# ---------------------------------------------------------------------------
# The positive, and the controls that have to fail
# ---------------------------------------------------------------------------


def test_the_load_bearing_channel_is_certified_and_adopted(tables, laws) -> None:
    verdict = ucm.adjudicate_channel(tables["main"], laws, 1)
    assert verdict["obstruction"]["verdict"] == ucm.CERTIFIED
    assert verdict["adoption"]["verdict"] == ucm.ADOPTED
    assert verdict["adoption"]["net_bits"] > ucm.DECLARED_POLICY.channel_admission_bits
    assert verdict["obstruction"]["gap_bits"] > 0
    assert verdict["obstruction_channel_zero_alone"]["verdict"] == ucm.CERTIFIED
    assert "no function of u alone" in verdict["headline"]
    assert "channel w is load bearing" in verdict["headline"]


def test_control_the_decoy_channel_is_not_adopted(tables, laws) -> None:
    verdict = ucm.adjudicate_channel(tables["main"], laws, 2)
    assert verdict["obstruction"]["verdict"] == ucm.NOT_CERTIFIED
    assert verdict["adoption"]["verdict"] == ucm.NOT_ADOPTED
    assert verdict["adoption"]["net_bits"] < 0
    assert verdict["obstruction"]["floor_data_bits"] == (
        verdict["obstruction"]["trivial_floor_data_bits"]
    )


def test_control_the_load_bearing_index_is_not_hard_wired(tables, laws) -> None:
    swapped = tables["swapped"]
    assert ucm.adjudicate_channel(swapped, laws, 1)["adoption"]["verdict"] == ucm.NOT_ADOPTED
    assert ucm.adjudicate_channel(swapped, laws, 1)["obstruction"]["verdict"] == (
        ucm.NOT_CERTIFIED
    )
    moved = ucm.adjudicate_channel(swapped, laws, 2)
    assert moved["adoption"]["verdict"] == ucm.ADOPTED
    assert moved["obstruction"]["verdict"] == ucm.CERTIFIED


def test_control_a_channel_with_real_but_cheap_signal_is_refused_by_the_price(
    tables, laws
) -> None:
    weak = tables["weak"]
    priced = ucm.adjudicate_channel(weak, laws, 2, ucm.DECLARED_POLICY)
    assert priced["adoption"]["verdict"] == ucm.NOT_ADOPTED
    # The signal is real: ignoring the channel costs strictly more data bits.
    assert priced["best_ignoring"]["data_bits"] > priced["best_using"]["data_bits"]
    assert 0 < priced["adoption"]["net_bits"] < ucm.DECLARED_POLICY.channel_admission_bits
    # And the price is what refuses it: set the fee to zero and the same data adopts.
    free = ucm.adjudicate_channel(weak, laws, 2, ucm.ChannelPolicy(0, 0, 16))
    assert free["adoption"]["verdict"] == ucm.ADOPTED
    assert priced["adoption"]["largest_fee_that_still_adopts"] == (
        priced["adoption"]["net_bits"] - ucm.DECLARED_POLICY.adoption_margin_bits
    )


def test_an_undecidable_design_produces_no_verdict_at_all(tables, laws) -> None:
    verdict = ucm.adjudicate_channel(tables["incomplete"], laws, 1)
    assert verdict["adoption"]["net_bits"] is None
    assert verdict["adoption"]["verdict"] == ucm.NOT_ADOPTED
    assert verdict["obstruction"]["verdict"] == ucm.NOT_CERTIFIED
    assert verdict["laws_undecided"]
    assert verdict["laws_using_the_channel"] == 0
    assert verdict["laws_ignoring_the_channel"] == 0


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------


def _walk(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_no_float_survives_onto_the_certificate_path(built) -> None:
    for leaf in _walk(built):
        assert isinstance(leaf, (int, str, type(None))), repr(leaf)
        assert not isinstance(leaf, float), repr(leaf)


def test_every_intermediate_quantity_is_exact(tables, laws) -> None:
    table = tables["main"]
    predicted = ucm.predictions(table, laws["u_times_two_plus_w"])
    assert predicted is not None
    assert all(isinstance(item, Fraction) for item in predicted)
    codes = ucm.residual_codes(table, predicted)
    assert all(isinstance(item, int) and not isinstance(item, bool) for item in codes)
    assert all(isinstance(item, Fraction) for row in table.rows for item in row)
    assert all(isinstance(item, Fraction) for item in table.observations)


def test_the_policy_binds_a_declared_price_not_a_measured_one() -> None:
    with pytest.raises(ucm.ChannelMDLError):
        ucm.ChannelPolicy(channel_admission_bits=-1)
    with pytest.raises(ucm.ChannelMDLError):
        ucm.ChannelPolicy(adoption_margin_bits=-1)


# ---------------------------------------------------------------------------
# Malformed declarations
# ---------------------------------------------------------------------------


def test_a_table_refuses_a_nonpositive_resolution() -> None:
    with pytest.raises(ucm.ChannelMDLError):
        ucm.ChannelTable("bad", 1, ((Fraction(1),),), (Fraction(1),), (Fraction(0),))


def test_a_table_refuses_a_ragged_row() -> None:
    with pytest.raises(ucm.ChannelMDLError):
        ucm.ChannelTable(
            "bad",
            2,
            ((Fraction(1), Fraction(2)), (Fraction(1),)),
            (Fraction(1), Fraction(2)),
            (Fraction(1, 8), Fraction(1, 8)),
        )


def test_a_law_reading_a_channel_the_table_lacks_is_refused() -> None:
    table = ucm.ChannelTable(
        "one", 1, ((Fraction(1),),), (Fraction(1),), (Fraction(1, 8),)
    )
    with pytest.raises(ucm.ChannelMDLError):
        ucm.predictions(table, ucm.chan(1))


def test_an_expression_outside_the_declared_channel_width_is_refused() -> None:
    with pytest.raises(ucm.ChannelMDLError):
        ucm.chan(ucm.MAX_CHANNELS)


def test_a_division_by_zero_makes_a_law_inadmissible_rather_than_crashing(tables) -> None:
    table = tables["main"]
    broken = ucm.div(ucm.chan(0), ucm.const(0))
    assert ucm.evaluate(broken, table.rows[0]) is None
    assert ucm.predictions(table, broken) is None
    assert ucm.codelength(table, "broken", broken) is None


# ---------------------------------------------------------------------------
# The bridge from executed programs
# ---------------------------------------------------------------------------


def test_sandbox_decimal_output_becomes_an_exact_rational() -> None:
    values = ucm.exact_predictions_from_outputs(["0.25", "1.5", "2", "-0.0625", "1e-3"])
    assert values == (
        Fraction(1, 4),
        Fraction(3, 2),
        Fraction(2),
        Fraction(-1, 16),
        Fraction(1, 1000),
    )
    with pytest.raises(ucm.ChannelMDLError):
        ucm.exact_predictions_from_outputs(["nan"])


def test_a_population_of_programs_is_adjudicated_against_the_same_floor(tables) -> None:
    table = tables["main"]
    exact = tuple(row[0] * (2 + row[1]) for row in table.rows)
    ignoring = tuple(3 * row[0] for row in table.rows)
    executed = {
        "reads_the_channel": ("def rule(u, w, x):\n    return u * (2.0 + w)", exact),
        "ignores_the_channel": ("def rule(u, w, x):\n    return 3.0 * u", ignoring),
    }
    verdict = ucm.adjudicate_programs(table, executed, 1)
    assert verdict["obstruction"]["floor_data_bits"] == (
        ucm.ignoring_codelength_floor(table, (1,))["floor_data_bits"]
    )
    assert verdict["obstruction"]["verdict"] == ucm.CERTIFIED
    assert verdict["adoption"]["verdict"] == ucm.ADOPTED
    assert verdict["laws_using_the_channel"] == 1
    assert verdict["laws_ignoring_the_channel"] == 1


def test_a_program_population_with_no_channel_user_returns_no_adoption(tables) -> None:
    table = tables["main"]
    ignoring = tuple(3 * row[0] for row in table.rows)
    executed = {"ignores": ("def rule(u, w, x):\n    return 3.0 * u", ignoring)}
    verdict = ucm.adjudicate_programs(table, executed, 1)
    assert verdict["adoption"]["net_bits"] is None
    assert verdict["adoption"]["verdict"] == ucm.NOT_ADOPTED


def test_a_program_is_priced_on_its_own_bytes_and_whitespace_does_not_change_it() -> None:
    left = "def rule(u, w, x):\n    return 3.0 * u"
    right = "def rule(u, w, x):   \n    return 3.0 * u   \n\n"
    assert ucm.program_model_bits(left) == ucm.program_model_bits(right)
    assert ucm.program_model_bits(left + "\n    # a longer program") > (
        ucm.program_model_bits(left)
    )


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def test_the_built_receipt_validates(built) -> None:
    ucm.validate_receipt(built)


def test_every_negative_control_is_rejected(built) -> None:
    controls = built["negative_controls"]
    assert controls
    for name, control in controls.items():
        assert control["rejected"] is True, name


def test_the_receipt_seal_detects_a_tampered_verdict(built) -> None:
    tampered = json.loads(json.dumps(built))
    tampered["verdicts"][0]["adoption"]["verdict"] = ucm.ADOPTED
    with pytest.raises(ucm.ChannelMDLError):
        ucm.validate_receipt(tampered)


def test_the_receipt_refuses_an_adoption_that_did_not_repay_the_price(built) -> None:
    tampered = json.loads(json.dumps(built))
    target = next(
        item for item in tampered["verdicts"] if item["adoption"]["verdict"] == ucm.ADOPTED
    )
    target["adoption"]["net_bits"] = target["adoption"]["admission_bits"] - 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = ucm.canonical_sha256(tampered)
    with pytest.raises(ucm.ChannelMDLError):
        ucm.validate_receipt(tampered)


def test_the_receipt_refuses_a_certificate_resting_on_the_trivial_floor(built) -> None:
    tampered = json.loads(json.dumps(built))
    target = next(
        item for item in tampered["verdicts"] if item["obstruction"]["verdict"] == ucm.CERTIFIED
    )
    target["obstruction"]["floor_data_bits"] = target["obstruction"][
        "trivial_floor_data_bits"
    ]
    tampered.pop("content_sha256")
    tampered["content_sha256"] = ucm.canonical_sha256(tampered)
    with pytest.raises(ucm.ChannelMDLError):
        ucm.validate_receipt(tampered)


def test_the_receipt_refuses_a_free_channel(built) -> None:
    tampered = json.loads(json.dumps(built))
    tampered["policy"]["channel_admission_bits"] = 0
    tampered["policy_sha256"] = ucm.canonical_sha256(tampered["policy"])
    tampered.pop("content_sha256")
    tampered["content_sha256"] = ucm.canonical_sha256(tampered)
    with pytest.raises(ucm.ChannelMDLError):
        ucm.validate_receipt(tampered)


def test_the_receipt_refuses_a_leaked_concept(built) -> None:
    tampered = json.loads(json.dumps(built))
    tampered["blindness"]["violations"] = ["radius"]
    tampered.pop("content_sha256")
    tampered["content_sha256"] = ucm.canonical_sha256(tampered)
    with pytest.raises(ucm.ChannelMDLError):
        ucm.validate_receipt(tampered)


def test_no_declared_surface_names_a_concept(built) -> None:
    assert built["blindness"]["violations"] == []
    rendered = json.dumps(built, sort_keys=True).lower()
    for term in ucm.FORBIDDEN_VOCABULARY:
        assert term not in rendered, term


def test_the_receipt_reports_exactly_one_load_bearing_channel_per_declared_table(
    built,
) -> None:
    certified = {
        (item["table_id"], item["channel_symbol"])
        for item in built["verdicts"]
        if item["obstruction"]["verdict"] == ucm.CERTIFIED
        and item["adoption"]["verdict"] == ucm.ADOPTED
    }
    assert ("main", "w") in certified
    assert ("swapped", "x") in certified
    assert ("main", "x") not in certified
    assert ("swapped", "w") not in certified
    assert ("weak", "x") not in certified
    assert not any(table == "flat" for table, _ in certified)


def test_the_receipt_on_disk_is_the_regenerated_one(built) -> None:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    stored = json.loads(RECEIPT.read_text(encoding="utf-8"))
    ucm.validate_receipt(stored)
    assert stored["content_sha256"] == built["content_sha256"]
