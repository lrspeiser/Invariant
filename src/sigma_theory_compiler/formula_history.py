from __future__ import annotations

import bisect
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

_HEADER = struct.Struct("<8sHHQQQQ")
_RECORD = struct.Struct("<QBBH6H")
_MAGIC = b"SGSURV2\0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_combination(n: int, values: list[int]) -> int:
    """Rank a sorted combination in the same lexicographic order as Generator v2."""
    if values != sorted(set(values)) or any(value < 0 or value >= n for value in values):
        raise ValueError("term ids must be a sorted unique in-range combination")
    rank = 0
    start = 0
    for position, value in enumerate(values):
        remaining = len(values) - position - 1
        for skipped in range(start, value):
            rank += 1 if remaining == 0 else math.comb(n - skipped - 1, remaining)
        start = value + 1
    return rank


def encode_ordinal(basis_count: int, max_action_terms: int, term_ids: list[int], sign_mask: int) -> int:
    terms = len(term_ids)
    if not 1 <= terms <= max_action_terms:
        raise ValueError("term count outside formula space")
    if not 0 <= sign_mask < 2**terms:
        raise ValueError("sign mask outside formula space")
    offset = sum(math.comb(basis_count, k) * 2**k for k in range(1, terms))
    return offset + (rank_combination(basis_count, term_ids) << terms) + sign_mask


def _parse_signed_terms(expression: str) -> list[tuple[int, str]] | None:
    text = "".join(expression.split())
    if not text:
        return None
    if text[0] not in "+-":
        return [(1, text)]
    result: list[tuple[int, str]] = []
    index = 0
    while index < len(text):
        sign = 1 if text[index] == "+" else -1
        index += 1
        if index >= len(text) or text[index] != "(":
            return None
        start = index + 1
        depth = 1
        index += 1
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        if depth:
            return None
        result.append((sign, text[start : index - 1]))
        if index < len(text) and text[index] not in "+-":
            return None
    return result


class GeneratorFormulaHistory:
    """Exact, random-access view of a compact Generator v2 formula space and survivor ledger."""

    def __init__(
        self,
        manifest_path: str | Path,
        basis_path: str | Path,
        survivor_directory: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.basis_path = Path(basis_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.basis = json.loads(self.basis_path.read_text(encoding="utf-8"))
        if not isinstance(self.basis, list):
            raise TypeError("basis library must be a JSON array")
        if len(self.basis) != int(self.manifest["basis_count"]):
            raise ValueError("basis count disagrees with generator manifest")
        canonical_basis = json.dumps(
            self.basis, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if hashlib.sha256(canonical_basis).hexdigest() != self.manifest["basis_library_sha256"]:
            raise ValueError("basis library hash disagrees with generator manifest")
        raw_directory = Path(survivor_directory or self.manifest["survivor_export_directory"])
        if not raw_directory.is_absolute():
            candidates = [
                (self.manifest_path.parent / raw_directory).resolve(),
                (self.manifest_path.parents[2] / raw_directory).resolve()
                if len(self.manifest_path.parents) >= 3
                else raw_directory.resolve(),
                raw_directory.resolve(),
            ]
            raw_directory = next((path for path in candidates if path.is_dir()), candidates[-1])
        self.survivor_directory = raw_directory.resolve()
        self._basis_by_expression = {
            "".join(str(item["expression"]).split()): int(item["id"]) for item in self.basis
        }
        self._blocks = sorted(self.manifest.get("blocks", []), key=lambda item: item["start_ordinal"])
        self._block_starts = [int(item["start_ordinal"]) for item in self._blocks]

    def describe(self) -> dict[str, Any]:
        return {
            "protocol_version": self.manifest["protocol_version"],
            "generator_version": self.manifest["generator_version"],
            "basis_count": int(self.manifest["basis_count"]),
            "max_action_terms": int(self.manifest["max_action_terms"]),
            "total_declared_actions": int(self.manifest["total_declared_actions"]),
            "processed_actions": int(self.manifest["processed_actions"]),
            "complete_declared_space": bool(self.manifest["complete_declared_space"]),
            "survivor_count": int(self.manifest.get("survivor_count", 0)),
            "manifest_sha256": _sha256(self.manifest_path),
            "basis_sha256": _sha256(self.basis_path),
        }

    def _decompose(self, expression: str) -> tuple[list[int], int] | None:
        parsed = _parse_signed_terms(expression)
        if not parsed:
            return None
        term_signs: list[tuple[int, int]] = []
        for sign, term in parsed:
            term_id = self._basis_by_expression.get(term)
            if term_id is None:
                return None
            term_signs.append((term_id, sign))
        if len({term_id for term_id, _ in term_signs}) != len(term_signs):
            return None
        term_signs.sort()
        term_ids = [term_id for term_id, _ in term_signs]
        sign_mask = sum(
            1 << position for position, (_, sign) in enumerate(term_signs) if sign > 0
        )
        return term_ids, sign_mask

    def _candidate_id(self, term_ids: list[int], sign_mask: int) -> str:
        digest = hashlib.sha256()
        digest.update(b"SIGMA-GENERATOR-V2\0")
        digest.update(self.manifest["protocol_version"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes([len(term_ids)]))
        for position, term_id in enumerate(term_ids):
            digest.update(int(term_id).to_bytes(2, "little"))
            digest.update(bytes([1 if sign_mask & (1 << position) else 0]))
        return f"STC2-{digest.hexdigest()[:24]}"

    def _block_for(self, ordinal: int) -> dict[str, Any] | None:
        index = bisect.bisect_right(self._block_starts, ordinal) - 1
        if index < 0:
            return None
        block = self._blocks[index]
        return block if ordinal < int(block["end_ordinal_exclusive"]) else None

    def _is_exported_survivor(self, ordinal: int, term_ids: list[int], sign_mask: int) -> bool:
        block = self._block_for(ordinal)
        if not block or not block.get("survivor_export"):
            return False
        export = block["survivor_export"]
        path = self.survivor_directory / export["file"]
        with path.open("rb") as handle:
            header = handle.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError(f"truncated survivor header: {path}")
            magic, version, record_size, block_index, start, end, count = _HEADER.unpack(header)
            if (magic, version, record_size) != (_MAGIC, 1, _RECORD.size):
                raise ValueError(f"unsupported survivor ledger: {path}")
            expected = (
                int(block["block_index"]),
                int(block["start_ordinal"]),
                int(block["end_ordinal_exclusive"]),
                int(export["record_count"]),
            )
            if (block_index, start, end, count) != expected:
                raise ValueError(f"survivor header disagrees with manifest: {path}")
            low, high = 0, count
            while low < high:
                middle = (low + high) // 2
                handle.seek(_HEADER.size + middle * _RECORD.size)
                record = _RECORD.unpack(handle.read(_RECORD.size))
                if record[0] < ordinal:
                    low = middle + 1
                else:
                    high = middle
            if low >= count:
                return False
            handle.seek(_HEADER.size + low * _RECORD.size)
            record = _RECORD.unpack(handle.read(_RECORD.size))
        found_ordinal, term_count, found_mask, reserved, *found_ids = record
        if found_ordinal != ordinal:
            return False
        if reserved or found_ids[:term_count] != term_ids or found_mask != sign_mask:
            raise ValueError("survivor record disagrees with deterministic ordinal decoding")
        return True

    def query(self, expression: str) -> dict[str, Any]:
        decomposition = self._decompose(expression)
        base = {
            "formula_space": self.manifest["protocol_version"],
            "exact_project_history_match": False,
            "scientific_validity_claimed": False,
        }
        if decomposition is None:
            return {**base, "status": "outside_exact_generator_syntax"}
        term_ids, sign_mask = decomposition
        ordinal = encode_ordinal(
            int(self.manifest["basis_count"]),
            int(self.manifest["max_action_terms"]),
            term_ids,
            sign_mask,
        )
        tested = (
            int(self.manifest["start_ordinal"])
            <= ordinal
            < int(self.manifest["end_ordinal_exclusive"])
        )
        survived = self._is_exported_survivor(ordinal, term_ids, sign_mask) if tested else False
        if not tested:
            outcome = "outside_manifest_processed_range"
        elif survived:
            outcome = "survived_sampled_static_export"
        else:
            outcome = "rejected_before_sampled_static_survivor_export"
        return {
            **base,
            "status": "exact_generator_formula",
            "exact_project_history_match": tested,
            "candidate_id": self._candidate_id(term_ids, sign_mask),
            "ordinal": ordinal,
            "term_ids": term_ids,
            "sign_mask": sign_mask,
            "tested_in_manifest": tested,
            "recorded_outcome": outcome,
            "outcome_scope": "generator screening history, not empirical or formal validity",
        }
