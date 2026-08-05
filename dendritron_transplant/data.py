"""Deterministic synthetic language with phrase-addressable conditional rules."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticCorpus:
    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    vocabulary_size: int


def _sequence(rng: random.Random, *, chunks: int, vocabulary_size: int) -> list[int]:
    if vocabulary_size < 96:
        raise ValueError("synthetic grammar requires vocabulary_size >= 96")
    values = [1]  # BOS
    for _ in range(chunks):
        topic = rng.randrange(8)
        relation = rng.randrange(4)
        subject = 3 + topic
        relation_token = 12 + relation
        object_token = 20 + ((topic * 3 + relation) % 16)
        answer = 40 + topic * 4 + relation
        cadence = 76 + ((topic + relation) % 8)
        # The answer is a deterministic function of the preceding trigram.
        values.extend([subject, relation_token, object_token, answer, cadence])
    values.append(2)  # EOS
    return values


def build_synthetic_corpus(
    *,
    training_sequences: int = 512,
    validation_sequences: int = 128,
    chunks_per_sequence: int = 5,
    vocabulary_size: int = 96,
    seed: int = 7,
) -> SyntheticCorpus:
    if min(training_sequences, validation_sequences, chunks_per_sequence) < 1:
        raise ValueError("corpus sizes must be positive")
    rng = random.Random(seed)
    train = [
        _sequence(rng, chunks=chunks_per_sequence, vocabulary_size=vocabulary_size)
        for _ in range(training_sequences)
    ]
    validation = [
        _sequence(rng, chunks=chunks_per_sequence, vocabulary_size=vocabulary_size)
        for _ in range(validation_sequences)
    ]
    train_tensor = torch.tensor(train, dtype=torch.long)
    validation_tensor = torch.tensor(validation, dtype=torch.long)
    return SyntheticCorpus(
        train_inputs=train_tensor[:, :-1],
        train_targets=train_tensor[:, 1:],
        validation_inputs=validation_tensor[:, :-1],
        validation_targets=validation_tensor[:, 1:],
        vocabulary_size=vocabulary_size,
    )


def count_ngrams(
    sequences: Tensor | Sequence[Sequence[int]],
    *,
    orders: Iterable[int] = (2, 3),
) -> Counter[tuple[int, ...]]:
    if isinstance(sequences, Tensor):
        rows = sequences.detach().to("cpu", dtype=torch.long).tolist()
    else:
        rows = [[int(value) for value in row] for row in sequences]
    counts: Counter[tuple[int, ...]] = Counter()
    for row in rows:
        for order in orders:
            for start in range(len(row) - int(order) + 1):
                counts[tuple(row[start : start + int(order)])] += 1
    return counts


def top_phrase_inventory(
    sequences: Tensor | Sequence[Sequence[int]],
    *,
    per_order: int = 128,
    orders: tuple[int, ...] = (2, 3),
) -> tuple[list[tuple[int, ...]], Counter[tuple[int, ...]]]:
    counts = count_ngrams(sequences, orders=orders)
    selected: list[tuple[int, ...]] = []
    for order in orders:
        candidates = [
            (phrase, count)
            for phrase, count in counts.items()
            if len(phrase) == order
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected.extend(phrase for phrase, _ in candidates[:per_order])
    return selected, counts
