from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rgbt_uav.data.manifest import SampleRecord


def _rank(pair_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()


def select_deterministic_records(
    records: Sequence[SampleRecord],
    *,
    count: int,
    seed: int,
    split: str,
) -> list[SampleRecord]:
    if not 0 < count <= len(records):
        raise ValueError("count must be positive and no larger than records")
    if any(record.split != split for record in records):
        raise ValueError(f"expected split {split}")
    if len({record.pair_id for record in records}) != len(records):
        raise ValueError("duplicate pair_id")
    return sorted(records, key=lambda item: (_rank(item.pair_id, seed=seed), item.pair_id))[:count]


def records_sha256(records: Sequence[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.pair_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
