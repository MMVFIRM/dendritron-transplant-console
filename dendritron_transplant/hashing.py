"""Deterministic, configuration-locked hash addressing for exact-memory misses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import HashMemoryConfig


@dataclass(frozen=True)
class HashAddressStats:
    valid_by_order: dict[int, int]
    maximum_by_order: dict[int, int]


def hash_ngram_ids(
    token_ids: tuple[int, ...] | list[int],
    *,
    order: int,
    head: int,
    rows: int,
    config: HashMemoryConfig,
) -> int:
    """Hash one exact suffix into the configured table without clamping."""

    if len(token_ids) != order:
        raise ValueError(f"expected {order} token IDs, found {len(token_ids)}")
    if order not in config.rows_by_order:
        raise ValueError(f"order {order} is not configured")
    if rows != config.rows_by_order[order]:
        raise ValueError("address rows must match the shared hash configuration")
    if not 0 <= head < config.heads:
        raise ValueError("hash head is outside the configured range")

    state = (1_000_033 + order * 10_007 + head * 1_000_037) % config.modulus
    for position, raw_token_id in enumerate(token_ids):
        token_id = int(raw_token_id)
        if token_id < 0:
            raise ValueError("token IDs must be nonnegative")
        state = (
            state * config.multiplier + token_id + (position + 1) * 97
        ) % config.modulus
    result = int(state % rows)
    if not 0 <= result < rows:  # defensive; modulo should make this impossible
        raise RuntimeError("hash function produced an out-of-range address")
    return result


class HashAddressor:
    """Build batched multi-head n-gram addresses from one shared config."""

    def __init__(self, config: HashMemoryConfig) -> None:
        self.config = config

    def build(self, input_ids: Tensor) -> dict[int, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.numel() and int(input_ids.min()) < 0:
            raise ValueError("input_ids must be nonnegative")

        batch, length = input_ids.shape
        result: dict[int, Tensor] = {}
        device = input_ids.device
        for order, rows in self.config.rows_by_order.items():
            addresses = torch.full(
                (batch, length, self.config.heads),
                -1,
                dtype=torch.long,
                device=device,
            )
            if length >= order:
                windows = input_ids.unfold(1, order, 1)
                for head in range(self.config.heads):
                    state = torch.full(
                        windows.shape[:2],
                        (1_000_033 + order * 10_007 + head * 1_000_037)
                        % self.config.modulus,
                        dtype=torch.long,
                        device=device,
                    )
                    for position in range(order):
                        state = (
                            state * self.config.multiplier
                            + windows[..., position]
                            + (position + 1) * 97
                        ) % self.config.modulus
                    addresses[:, order - 1 :, head] = state % rows
            self.validate_order(addresses, order)
            result[order] = addresses
        return result

    def validate_order(self, addresses: Tensor, order: int) -> None:
        if order not in self.config.rows_by_order:
            raise ValueError(f"unconfigured hash order {order}")
        rows = self.config.rows_by_order[order]
        valid = addresses >= 0
        if bool((addresses[valid] >= rows).any()):
            maximum = int(addresses[valid].max()) if bool(valid.any()) else -1
            raise ValueError(
                f"order-{order} hash address {maximum} exceeds table rows {rows}"
            )

    def stats(self, addresses: dict[int, Tensor]) -> HashAddressStats:
        valid_by_order: dict[int, int] = {}
        maximum_by_order: dict[int, int] = {}
        for order, tensor in addresses.items():
            self.validate_order(tensor, order)
            valid = tensor >= 0
            valid_by_order[order] = int(valid.sum())
            maximum_by_order[order] = (
                int(tensor[valid].max()) if bool(valid.any()) else -1
            )
        return HashAddressStats(valid_by_order, maximum_by_order)
