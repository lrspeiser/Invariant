"""Stateless collision-free pseudorandom traversal of very large ordinal spaces."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .sigma_core import canonical_sha256

ALGORITHM = "cycle_walked_splitmix_feistel_8round_v1"
SCHEMA = "invariant-pseudorandom-ordinal-receipt-1.0"
MAXIMUM_SIZE = 1 << 64
_MASK64 = (1 << 64) - 1
_ROUND_COUNT = 8


class PseudorandomOrdinalError(ValueError):
    """A permutation request or replay receipt is invalid."""


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _round_keys(size: int, seed: str) -> tuple[int, ...]:
    material = hashlib.sha256(f"{ALGORITHM}\0{size}\0{seed}".encode()).digest()
    root = int.from_bytes(material[:8], "little")
    return tuple(_mix64(root ^ round_index) for round_index in range(_ROUND_COUNT))


@dataclass(frozen=True, slots=True)
class PseudorandomOrdinalPermutation:
    """A keyed bijection over ``range(size)`` with constant-memory random access.

    The Feistel network permutes the next even-bit power-of-two domain. Cycle walking restricts
    that bijection to the requested interval, preserving one-to-one complete coverage. The design
    is a reproducible search scheduler, not a cryptographic primitive.
    """

    size: int
    seed: str
    _bits: int = field(init=False, repr=False)
    _keys: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or not 1 <= self.size <= MAXIMUM_SIZE
        ):
            raise PseudorandomOrdinalError("size must be an integer between 1 and 2^64")
        if not isinstance(self.seed, str) or not self.seed:
            raise PseudorandomOrdinalError("seed must be nonempty text")
        bits = max(2, (self.size - 1).bit_length())
        if bits % 2:
            bits += 1
        object.__setattr__(self, "_bits", bits)
        object.__setattr__(self, "_keys", _round_keys(self.size, self.seed))

    @property
    def domain_size(self) -> int:
        return 1 << self._bits

    def _feistel(self, value: int) -> int:
        half_bits = self._bits // 2
        half_mask = (1 << half_bits) - 1
        left = value >> half_bits
        right = value & half_mask
        for key in self._keys:
            function = _mix64(right ^ key) & half_mask
            left, right = right, left ^ function
        return (left << half_bits) | right

    def at_with_cycle_steps(self, position: int) -> tuple[int, int]:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < self.size
        ):
            raise PseudorandomOrdinalError("position lies outside the permutation")
        if self.size == 1:
            return 0, 1
        value = position
        steps = 0
        while True:
            value = self._feistel(value)
            steps += 1
            if value < self.size:
                return value, steps

    def at(self, position: int) -> int:
        return self.at_with_cycle_steps(position)[0]

    def iter(self, *, start_position: int = 0, stop_position: int | None = None) -> Iterator[int]:
        stop = self.size if stop_position is None else stop_position
        if (
            isinstance(start_position, bool)
            or isinstance(stop, bool)
            or not isinstance(start_position, int)
            or not isinstance(stop, int)
            or not 0 <= start_position <= stop <= self.size
        ):
            raise PseudorandomOrdinalError("invalid permutation interval")
        for position in range(start_position, stop):
            yield self.at(position)

    def descriptor(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "size": self.size,
            "seed_sha256": hashlib.sha256(self.seed.encode("utf-8")).hexdigest(),
            "feistel_rounds": _ROUND_COUNT,
            "power_of_two_domain_size": self.domain_size,
            "properties": {
                "complete_coverage_if_fully_traversed": True,
                "constant_memory_random_access": True,
                "duplicates": 0,
                "cryptographic_security_claimed": False,
            },
        }


@dataclass(frozen=True, slots=True)
class PseudorandomChunkSchedule:
    """Pseudorandomize large GPU work blocks while keeping each kernel launch contiguous."""

    total_size: int
    chunk_size: int
    seed: str
    _permutation: PseudorandomOrdinalPermutation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_size, bool)
            or not isinstance(self.total_size, int)
            or not 1 <= self.total_size <= MAXIMUM_SIZE
        ):
            raise PseudorandomOrdinalError("total size must be an integer between 1 and 2^64")
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or not 1 <= self.chunk_size <= self.total_size
        ):
            raise PseudorandomOrdinalError("chunk size is outside the search space")
        chunk_count = (self.total_size + self.chunk_size - 1) // self.chunk_size
        object.__setattr__(
            self,
            "_permutation",
            PseudorandomOrdinalPermutation(chunk_count, f"{self.seed}\0chunks"),
        )

    @property
    def chunk_count(self) -> int:
        return self._permutation.size

    def at(self, position: int) -> dict[str, int]:
        chunk_id = self._permutation.at(position)
        start = chunk_id * self.chunk_size
        stop = min(self.total_size, start + self.chunk_size)
        return {
            "schedule_position": position,
            "chunk_id": chunk_id,
            "start_ordinal": start,
            "stop_ordinal_exclusive": stop,
            "formula_count": stop - start,
        }

    def iter(
        self, *, start_position: int = 0, stop_position: int | None = None
    ) -> Iterator[dict[str, int]]:
        stop = self.chunk_count if stop_position is None else stop_position
        if (
            isinstance(start_position, bool)
            or isinstance(stop, bool)
            or not isinstance(start_position, int)
            or not isinstance(stop, int)
            or not 0 <= start_position <= stop <= self.chunk_count
        ):
            raise PseudorandomOrdinalError("invalid chunk-schedule interval")
        for position in range(start_position, stop):
            yield self.at(position)

    def descriptor(self) -> dict[str, Any]:
        return {
            "algorithm": "pseudorandom_chunk_permutation_with_contiguous_inner_ordinals_v1",
            "total_size": self.total_size,
            "chunk_size": self.chunk_size,
            "chunk_count": self.chunk_count,
            "chunk_permutation": self._permutation.descriptor(),
            "properties": {
                "complete_nonoverlapping_coverage_if_fully_traversed": True,
                "constant_memory_random_access_resume": True,
                "gpu_contiguous_inner_loop": True,
            },
        }


def build_prefix_receipt(
    *, size: int, seed: str, sample_count: int
) -> dict[str, Any]:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= min(size, 1_000_000)
    ):
        raise PseudorandomOrdinalError("sample count is outside the supported prefix bound")
    permutation = PseudorandomOrdinalPermutation(size, seed)
    prefix = []
    cycle_steps = []
    for position in range(sample_count):
        ordinal, steps = permutation.at_with_cycle_steps(position)
        prefix.append(ordinal)
        cycle_steps.append(steps)
    if len(set(prefix)) != sample_count:
        raise PseudorandomOrdinalError("permutation prefix contains a collision")
    body = {
        "schema_version": SCHEMA,
        "permutation": permutation.descriptor(),
        "sample": {
            "count": sample_count,
            "first_ordinals": prefix[:16],
            "ordinal_sequence_sha256": canonical_sha256(prefix),
            "minimum_ordinal": min(prefix),
            "maximum_ordinal": max(prefix),
            "cycle_walk_total_steps": sum(cycle_steps),
            "cycle_walk_maximum_steps": max(cycle_steps),
            "all_in_range": all(0 <= value < size for value in prefix),
            "all_unique": True,
        },
        "claims": {
            "formula_candidates_evaluated": 0,
            "trillion_scale_iteration_completed": False,
            "historical_novelty_established": False,
        },
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_prefix_receipt(
    receipt: Mapping[str, Any], *, seed: str
) -> None:
    if receipt.get("schema_version") != SCHEMA:
        raise PseudorandomOrdinalError("permutation receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise PseudorandomOrdinalError("permutation receipt seal changed")
    permutation = receipt.get("permutation", {})
    sample = receipt.get("sample", {})
    if not isinstance(permutation, Mapping) or not isinstance(sample, Mapping):
        raise PseudorandomOrdinalError("permutation receipt structure changed")
    expected = build_prefix_receipt(
        size=int(permutation["size"]),
        seed=seed,
        sample_count=int(sample["count"]),
    )
    if dict(receipt) != expected:
        raise PseudorandomOrdinalError("permutation receipt replay changed")


__all__ = [
    "ALGORITHM",
    "MAXIMUM_SIZE",
    "PseudorandomChunkSchedule",
    "PseudorandomOrdinalError",
    "PseudorandomOrdinalPermutation",
    "build_prefix_receipt",
    "validate_prefix_receipt",
]
