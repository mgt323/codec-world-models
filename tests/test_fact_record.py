"""Tests for shared quantization and FactRecord construction."""

from __future__ import annotations

import pytest

from world.schema import (
    VALUE_BIN_EDGES,
    FactRecord,
    NoiseBin,
    Observation,
    SensorSource,
    facts_from_observation,
    quantize_value,
)


def test_quantize_value_exactly_on_bin_edge() -> None:
    """Value on an interior edge: right-open bins put the edge in the upper bin.

    Edges are 0.0, 0.1, ..., 1.0. For value < edges[i+1] return i.
    So 0.3 is not < 0.3 (fails bin starting at 0.2); it is < 0.4 → bin index 3.
    """
    assert VALUE_BIN_EDGES[3] == pytest.approx(0.3)
    assert VALUE_BIN_EDGES[4] == pytest.approx(0.4)
    assert quantize_value(0.3) == 3


def test_quantize_value_out_of_range_clips_to_end_bins() -> None:
    """Below first edge → bin 0; at/above last edge → last bin (len(edges)-2)."""
    last_bin = len(VALUE_BIN_EDGES) - 2
    assert quantize_value(-0.5) == 0
    assert quantize_value(-1e9) == 0
    assert quantize_value(1.0) == last_bin
    assert quantize_value(1.01) == last_bin
    assert quantize_value(2.5) == last_bin


def test_fact_record_ignores_stale_value_when_unobserved() -> None:
    """Unobserved a_value must not leak into FactRecord even if non-None."""
    shared = dict(
        b_observed=True,
        b_value=0.38,
        noise_bin=NoiseBin.MEDIUM,
        n_samples=3,
        source=SensorSource.SENSOR_B,
    )
    obs_clean = Observation(
        a_observed=False,
        a_value=None,
        **shared,
    )
    obs_stale = Observation(
        a_observed=False,
        a_value=0.99,  # stale; must be ignored
        **shared,
    )

    facts_clean = facts_from_observation(obs_clean)
    facts_stale = facts_from_observation(obs_stale)

    assert facts_clean == facts_stale
    assert isinstance(facts_clean, FactRecord)
    assert facts_clean.a_obs is False
    assert facts_clean.a_val_q is None
    assert facts_stale.a_val_q is None
    # Observed B still quantized normally.
    assert facts_clean.b_obs is True
    assert facts_clean.b_val_q == quantize_value(0.38)
