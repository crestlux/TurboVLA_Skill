"""Shared offline/live FT-history rate conversion; never reads future anchors."""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter


def resampling_contract(settings):
    if settings is None:
        return None
    expected = {"source_hz", "target_hz", "source_length", "num_taps", "cutoff_ratio", "rate_basis", "method"}
    if set(settings) - expected:
        raise ValueError("Unknown FT-history resampling setting")
    result = {"method": "causal_fir_decimate", "num_taps": 21, "cutoff_ratio": .4,
              "rate_basis": "nominal; per-sample timestamps unavailable", **settings}
    if result["method"] != "causal_fir_decimate":
        raise ValueError("FT history resampling method must be causal_fir_decimate")
    for key in ("source_hz", "target_hz"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"FT-history {key} must be finite and positive")
    ratio = result["source_hz"] / result["target_hz"]
    factor = round(ratio)
    if factor < 1 or not np.isclose(ratio, factor, rtol=0, atol=1e-8):
        raise ValueError("FT history supports integer downsampling factors only")
    length, taps = result.get("source_length"), result["num_taps"]
    if type(length) is not int or length < 1 or length % factor:
        raise ValueError("FT source length must be positive and divisible by the downsampling factor")
    if type(taps) is not int or taps < 3 or taps % 2 != 1:
        raise ValueError("FT FIR num_taps must be an odd integer >= 3")
    if not 0 < result["cutoff_ratio"] < .5:
        raise ValueError("FT FIR cutoff_ratio must be between zero and 0.5")
    if not isinstance(result["rate_basis"], str) or not result["rate_basis"].strip():
        raise ValueError("FT rate_basis must describe the source rate evidence")
    return result


def resample_ft_history(values, settings):
    """Input [...,source_length,6]; outputs end at the newest source sample.

    FIR is causal, restarted for each complete window with first-value left
    padding. Its group delay is (num_taps-1)/(2*source_hz). Identity conversion
    leaves every source value unchanged. Rate labels remain nominal if the
    source has no individual timestamps.
    """
    settings = resampling_contract(settings)
    values = np.asarray(values, dtype=np.float32)
    if settings is None:
        return values
    if values.ndim < 2 or values.shape[-2:] != (settings["source_length"], 6):
        raise ValueError("FT history does not match its source_length and six channels")
    factor = round(settings["source_hz"] / settings["target_hz"])
    if factor == 1:
        return values
    coefficients = firwin(settings["num_taps"], settings["target_hz"] * settings["cutoff_ratio"],
                           fs=settings["source_hz"], window="hamming")
    padding = [(0, 0)] * values.ndim
    padding[-2] = (len(coefficients) - 1, 0)
    extended = np.pad(values, padding, mode="edge")
    filtered = lfilter(coefficients, [1.], extended, axis=-2)[..., len(coefficients)-1:, :]
    return np.ascontiguousarray(filtered[..., factor-1::factor, :], dtype=np.float32)
