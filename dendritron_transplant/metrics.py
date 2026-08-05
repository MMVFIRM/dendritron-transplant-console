"""CPU benchmark helpers with explicit stored-versus-active reporting."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from time import perf_counter
from typing import Callable, Generic, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class LatencyReport(Generic[T]):
    median_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]
    last_output: T


def benchmark_callable(
    function: Callable[[], T],
    *,
    warmup: int = 5,
    iterations: int = 25,
) -> LatencyReport[T]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    output: T | None = None
    for _ in range(warmup):
        output = function()
    timings: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        output = function()
        timings.append((perf_counter() - start) * 1_000.0)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    assert output is not None
    return LatencyReport(
        median_ms=float(statistics.median(timings)),
        p95_ms=float(ordered[p95_index]),
        samples_ms=tuple(timings),
        last_output=output,
    )


def module_parameter_bytes(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
