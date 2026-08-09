from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
    total_search_count,
)


def test_python_reference_count_matches_billion_contract() -> None:
    assert total_search_count(50, 6) == 1_088_651_720


def test_reference_ordinal_decoder_is_unique_on_small_complete_space() -> None:
    total = total_search_count(6, 3)
    encodings = set()
    for ordinal in range(total):
        decoded = decode_ordinal(6, 3, ordinal)
        encoding = (tuple(decoded["term_ids"]), tuple(decoded["signs"]))
        assert encoding not in encodings
        encodings.add(encoding)
    assert len(encodings) == total


def test_basis_and_candidate_identifiers_are_deterministic() -> None:
    basis = build_basis(50)
    assert len({term["expression"] for term in basis}) == 50
    decoded = decode_ordinal(50, 6, 1_000_000)
    assert correction_expression(decoded, basis) == correction_expression(decoded, build_basis(50))
    assert candidate_id("FROZEN", decoded) == candidate_id("FROZEN", decoded)
