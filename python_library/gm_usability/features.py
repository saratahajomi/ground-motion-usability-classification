"""Python translation of data_preparation_09172026_full_features.R.

Feature spelling, R type-7 quantiles, trailing means, peak boundaries,
and nonfinite context handling are preserved.
"""
from pathlib import Path
import math
import re
import warnings
import numpy as np
import pandas as pd
from ._ranking import FULL_FEATURE_NAMES, RANKED_FEATURE_NAMES


def _validate(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Time and amplitude must be equal-length one-dimensional arrays")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 64 or np.std(y, ddof=1) == 0:
        raise ValueError("Waveform requires at least 64 finite samples and nonzero variance")
    if np.any(np.diff(x) <= 0):
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    dt = np.median(np.diff(x))
    if not np.isfinite(dt) or not 1e-6 <= dt <= 1:
        raise ValueError("Median sampling interval must be between 1e-6 and 1 second")
    return x, y, dt


def read_waveforms(path):
    """Return (record_id, time_seconds, amplitude) per TXT or MiniSEED trace.

    MiniSEED traces remain separate; no merging, resampling, unit conversion,
    or instrument-response removal is performed.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] < 2:
            raise ValueError("TXT requires time and amplitude columns")
        x = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()
        x, y, _ = _validate(x, y)
        return [(path.stem, x, y)]
    if path.suffix.lower() not in {".mseed", ".miniseed", ".ms", ".seed"}:
        raise ValueError("Supported inputs are TXT and MiniSEED")
    from obspy import read
    result = []
    for i, trace in enumerate(read(str(path), format="MSEED")):
        y = np.asarray(np.ma.filled(trace.data, np.nan), dtype=float)
        record_id = f"{path.stem}:{trace.id}:{trace.stats.starttime}:{i}"
        try:
            x, y, _ = _validate(trace.times(), y)
        except ValueError as error:
            warnings.warn(f"Skipping invalid MiniSEED trace {record_id}: {error}", UserWarning, stacklevel=2)
            continue
        result.append((record_id, x, y))
    if not result:
        raise ValueError("MiniSEED contains no valid traces (at least 64 finite samples, nonzero variance, and valid sampling interval required)")
    return result


def _quantile(v, p):
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p / 100, method="linear")) if len(v) else np.nan


def _mean(signal, n):
    result = np.full(len(signal), np.nan)
    # R stats::filter(..., sides=1): incomplete initial windows are NA.
    sums = np.concatenate(([0.0], np.cumsum(signal)))
    result[n - 1:] = (sums[n:] - sums[:-n]) / n
    return result


def extract_features(time, amplitude, mode="full"):
    """Generate 1,368 full features or only the embedded top 500.

    Reduced generation computes only the selected parameter combinations.
    The returned Series preserves exact source names, including double underscores.
    """
    if mode not in {"full", "reduced"}:
        raise ValueError("mode must be 'full' or 'reduced'")
    names = FULL_FEATURE_NAMES if mode == "full" else RANKED_FEATURE_NAMES[:500]
    x, y, dt = _validate(time, amplitude)
    # Centering improves conditioning while retaining lm(y ~ x)'s intercept.
    xc = x - np.mean(x)
    fit = np.linalg.lstsq(np.column_stack((np.ones(len(x)), xc)), y, rcond=None)[0]
    signal = np.abs(y - (fit[0] + fit[1] * xc))
    ratios, runs, envelopes = {}, {}, {}
    output = {}
    for name in names:
        if name.startswith("stalta__"):
            sta, lta, threshold, duration = map(float, re.fullmatch(
                r"stalta__sta([\d.]+)s__lta([\d.]+)s__thr([\d.]+)__min_exceedance_duration([\d.]+)s", name).groups())
            key = (sta, lta)
            if key not in ratios:
                ns = max(1, round(sta * (1 / dt)))
                nl = max(ns + 1, round(lta * (1 / dt)))
                ratios[key] = None if max(ns, nl) >= len(signal) else _mean(signal, ns) / (_mean(signal, nl) + 1e-12)
            ratio = ratios[key]
            if ratio is None or not np.any(np.isfinite(ratio)):
                output[name] = np.nan
                continue
            run_key = (sta, lta, threshold)
            if run_key not in runs:
                mask = np.isfinite(ratio) & (ratio >= threshold)
                edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                runs[run_key] = max(lengths, default=0)
            output[name] = float(runs[run_key] >= max(1, math.ceil(duration / dt - 1e-12)))
        else:
            mw, step, sp, width, percentile = map(int, re.fullmatch(
                r"peak_context__mw(\d+)s__step(\d+)s__spP(\d+)__w(\d+)s__ratioP(\d+)__median", name).groups())
            key = (mw, step, sp)
            if key not in envelopes:
                n = max(4, round(mw / dt))
                stride = max(1, round(step / dt))
                starts = np.arange(0, len(signal) - n + 1, stride)
                if n >= len(signal) or len(starts) < 4:
                    envelopes[key] = None
                else:
                    t = x[starts + (n - 1) // 2]
                    env = np.array([_quantile(signal[s:s+n], sp) for s in starts])
                    keep = np.isfinite(t) & np.isfinite(env)
                    envelopes[key] = (t[keep], env[keep]) if keep.sum() >= 4 else None
            envelope = envelopes[key]
            if envelope is None:
                output[name] = np.nan
                continue
            t, env = envelope
            peak = t[np.argmax(env)]
            lo, hi = peak - width / 2, peak + width / 2
            numerator = _quantile(env[(t >= lo) & (t <= hi)], percentile)
            sides = []
            for segment in (env[t < lo], env[t > hi]):
                denominator = _quantile(segment, percentile)
                sides.append(np.nan if not np.isfinite(numerator) else
                             np.inf if not len(segment) else
                             np.nan if not np.isfinite(denominator) or denominator <= 0 else
                             numerator / denominator)
            # R checks both for finiteness before median(..., na.rm=TRUE).
            output[name] = np.nan if not np.any(np.isfinite(sides)) else float(np.median(np.asarray(sides)[~np.isnan(sides)]))
    return pd.Series(output, index=names, dtype=float)
