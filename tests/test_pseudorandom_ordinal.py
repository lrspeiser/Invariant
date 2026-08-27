from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.pseudorandom_ordinal import (
    PseudorandomChunkSchedule,
    PseudorandomOrdinalError,
    PseudorandomOrdinalPermutation,
    build_prefix_receipt,
    validate_prefix_receipt,
)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 16, 17, 63, 64, 65, 257, 1000])
def test_full_traversal_is_a_permutation(size: int) -> None:
    permutation = PseudorandomOrdinalPermutation(size, "full-coverage-test")

    assert sorted(permutation.iter()) == list(range(size))


def test_seed_is_deterministic_and_changes_order() -> None:
    first = PseudorandomOrdinalPermutation(1000, "seed-a")
    replay = PseudorandomOrdinalPermutation(1000, "seed-a")
    other = PseudorandomOrdinalPermutation(1000, "seed-b")

    assert list(first.iter(stop_position=100)) == list(replay.iter(stop_position=100))
    assert list(first.iter(stop_position=100)) != list(other.iter(stop_position=100))


def test_random_access_resume_reproduces_the_same_suffix() -> None:
    permutation = PseudorandomOrdinalPermutation(10_000, "resume-test")
    complete = list(permutation.iter(stop_position=2000))

    assert list(permutation.iter(start_position=731, stop_position=2000)) == complete[731:]
    assert permutation.at(731) == complete[731]


def test_trillion_scale_prefix_is_unique_without_materializing_the_space() -> None:
    size = 2_127_732_389_840
    permutation = PseudorandomOrdinalPermutation(size, "trillion-prefix-test")
    prefix = list(permutation.iter(stop_position=10_000))

    assert permutation.domain_size < 4 * size
    assert len(prefix) == len(set(prefix)) == 10_000
    assert all(0 <= ordinal < size for ordinal in prefix)


def test_prefix_receipt_replays_and_tampering_fails_closed() -> None:
    receipt = build_prefix_receipt(size=1_000_003, seed="receipt-seed", sample_count=4096)
    validate_prefix_receipt(receipt, seed="receipt-seed")

    tampered = copy.deepcopy(receipt)
    tampered["sample"]["first_ordinals"][0] += 1
    with pytest.raises(PseudorandomOrdinalError, match="seal"):
        validate_prefix_receipt(tampered, seed="receipt-seed")


def test_invalid_bounds_fail_before_iteration() -> None:
    with pytest.raises(PseudorandomOrdinalError, match="size"):
        PseudorandomOrdinalPermutation(0, "seed")
    permutation = PseudorandomOrdinalPermutation(10, "seed")
    with pytest.raises(PseudorandomOrdinalError, match="position"):
        permutation.at(10)
    with pytest.raises(PseudorandomOrdinalError, match="interval"):
        list(permutation.iter(start_position=8, stop_position=7))


def test_chunk_schedule_covers_uneven_space_once_and_resumes() -> None:
    schedule = PseudorandomChunkSchedule(1003, 100, "chunk-test")
    chunks = list(schedule.iter())
    covered = []
    for chunk in chunks:
        covered.extend(range(chunk["start_ordinal"], chunk["stop_ordinal_exclusive"]))

    assert schedule.chunk_count == 11
    assert sorted(covered) == list(range(1003))
    assert sum(chunk["formula_count"] for chunk in chunks) == 1003
    assert list(schedule.iter(start_position=4)) == chunks[4:]


def test_trillion_chunk_schedule_is_small_and_gpu_shaped() -> None:
    schedule = PseudorandomChunkSchedule(2_127_732_389_840, 10_000_000, "gpu-scale")
    prefix = list(schedule.iter(stop_position=1000))

    assert schedule.chunk_count == 212_774
    assert len({chunk["chunk_id"] for chunk in prefix}) == 1000
    assert all(chunk["formula_count"] == 10_000_000 for chunk in prefix)
