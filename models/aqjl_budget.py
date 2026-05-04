"""
A-QJL helpers: validate layer-group boundaries, surrogate budget B = sum_j Delta_j * m_j,
snap sketch widths to kernel tile multiples, and repair allocations after snapping.
"""
from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple


def snap_sketch_width(m: int, multiple: int = 64, m_min: int = 64, m_max: int = 2048) -> int:
    """Round m down to a multiple of `multiple`, clamped to [m_min, m_max]."""
    m = int(m)
    snapped = max(m_min, (m // multiple) * multiple)
    return min(m_max, snapped)


def group_layer_counts(boundaries: Sequence[int], num_layers: int) -> List[int]:
    """Return [Delta_0, ..., Delta_{G-1}] for half-open strata [r_j, r_{j+1})."""
    r = [0] + list(boundaries) + [num_layers]
    return [r[j + 1] - r[j] for j in range(len(r) - 1)]


def surrogate_budget(boundaries: Sequence[int], sketch_widths: Sequence[int], num_layers: int) -> int:
    """
    Linear surrogate B(m) = sum_j Delta_j * m_j (paper Eq. budget).
    boundaries has length G-1; sketch_widths has length G.
    """
    deltas = group_layer_counts(boundaries, num_layers)
    if len(deltas) != len(sketch_widths):
        raise ValueError(
            f"group count mismatch: {len(deltas)} layer strata vs {len(sketch_widths)} sketch widths"
        )
    return int(sum(d * m for d, m in zip(deltas, sketch_widths)))


def validate_layer_group_boundaries(boundaries: Sequence[int], num_layers: int) -> None:
    """
    Enforce 0 < r_1 < ... < r_{G-1} < L (zero-based layer indices; L = num_layers).
    """
    if not boundaries:
        raise ValueError("layer_group_boundaries must be non-empty for multi-group A-QJL")
    b = list(boundaries)
    if any(x != int(x) for x in b):
        raise ValueError("layer_group_boundaries must be integers")
    b = [int(x) for x in b]
    if b != sorted(b):
        raise ValueError(f"layer_group_boundaries must be strictly increasing sorted; got {b}")
    if b[0] <= 0:
        raise ValueError(f"first boundary must be > 0; got {b[0]}")
    if b[-1] >= num_layers:
        raise ValueError(
            f"last boundary must be < num_layers ({num_layers}); got {b[-1]}"
        )
    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            raise ValueError(f"boundaries must be strictly increasing; got {b}")


def validate_aqjl_sketch_widths(sketch_widths: Sequence[int], multiple: int = 64) -> None:
    for m in sketch_widths:
        if m < multiple:
            raise ValueError(f"sketch width {m} must be >= {multiple}")
        if m % multiple != 0:
            raise ValueError(
                f"sketch width {m} must be a multiple of {multiple} for kernel alignment"
            )


def validate_aqjl_config(
    num_layers: int,
    boundaries: Sequence[int],
    sketch_widths: Sequence[int],
) -> None:
    validate_layer_group_boundaries(boundaries, num_layers)
    if len(boundaries) + 1 != len(sketch_widths):
        raise ValueError(
            "need len(layer_group_boundaries) + 1 == len(key_quantization_bits_per_group): "
            f"{len(boundaries)} + 1 vs {len(sketch_widths)}"
        )
    validate_aqjl_sketch_widths(sketch_widths)


def snap_sketch_widths_inplace(
    widths: List[int], multiple: int = 64, m_min: int = 64, m_max: int = 2048
) -> List[int]:
    out = [snap_sketch_width(m, multiple, m_min, m_max) for m in widths]
    if out != widths:
        warnings.warn(
            "Adjusted sketch widths to multiples of "
            f"{multiple} in [{m_min}, {m_max}]: {widths} -> {out}",
            UserWarning,
            stacklevel=2,
        )
    return out


def repair_surrogate_budget(
    n_layers: List[int],
    sketch_widths: List[int],
    group_sensitivity: List[float],
    budget_target: int,
    multiple: int = 64,
    m_min: int = 64,
    m_max: int = 2048,
) -> List[int]:
    """
    After snapping, B = sum n_j m_j may differ from budget_target.
    - If B > budget_target: repeatedly decrease m in the group with lowest sensitivity
      (greedy shave) until within budget or no move left.
    - If B < budget_target: increase m in the highest-sensitivity group until at cap or budget met.
    """
    k = [snap_sketch_width(x, multiple, m_min, m_max) for x in sketch_widths]
    n_groups = len(k)

    def current_B() -> int:
        return sum(n * m for n, m in zip(n_layers, k))

    B = current_B()
    max_steps = sum(n_layers) * 32
    steps = 0

    while B > budget_target and steps < max_steps:
        steps += 1
        candidates = [
            i for i in range(n_groups) if k[i] - multiple >= m_min
        ]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: group_sensitivity[idx])
        k[i] -= multiple
        B = current_B()

    steps = 0
    while B < budget_target and steps < max_steps:
        steps += 1
        candidates = [i for i in range(n_groups) if k[i] + multiple <= m_max]
        if not candidates:
            break
        i = max(candidates, key=lambda idx: group_sensitivity[idx])
        k[i] += multiple
        B = current_B()

    # Growth moves in steps of n_i*multiple; trim if we overshot the cap.
    steps = 0
    while B > budget_target and steps < max_steps:
        steps += 1
        candidates = [i for i in range(n_groups) if k[i] - multiple >= m_min]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: group_sensitivity[idx])
        k[i] -= multiple
        B = current_B()

    return k


def percentile_boundaries(sensitivity: Sequence[float], num_groups: int, num_layers: int) -> List[int]:
    """
    Place boundaries so each group carries approximately equal *total* sensitivity mass.
    prefix[b] = sum_{i < b} s_i; choose smallest b in {prev+1, ..., L-1} with prefix[b] >= g * total / G.
    """
    if num_groups < 2:
        raise ValueError("num_groups must be >= 2")
    s = [float(x) for x in sensitivity]
    if len(s) != num_layers:
        raise ValueError("sensitivity length must equal num_layers")
    total = sum(s)
    if total <= 0:
        return [num_layers * (i + 1) // num_groups for i in range(num_groups - 1)]

    prefix = [0.0]
    for v in s:
        prefix.append(prefix[-1] + v)

    boundaries: List[int] = []
    prev = 0
    for g in range(1, num_groups):
        target = total * (g / num_groups)
        b = None
        for idx in range(prev + 1, num_layers):
            if prefix[idx] >= target:
                b = idx
                break
        if b is None:
            b = num_layers - 1
        b = max(prev + 1, min(b, num_layers - 1))
        boundaries.append(b)
        prev = b

    return boundaries
