"""Testable, deterministic adaptive learning for the Klondike scanner.

Never use prices after the snapshot date to build a signal. Outcomes are
calculated later, from *completed* daily bars only, and unavailable future
observations are never treated as losses/wins. The adjustments are descriptive,
not calibrated success probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LEARNING_MODEL_VERSION = "adaptive-r1"
TRAINING_COST_PCT = 0.10  # Modelled round-trip costs, percentages of notional.
MIN_RESOLVED = 30
MIN_BUCKET = 12
MAX_SCORE_ADJUSTMENT = 8.0


def number(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if isfinite(v) else default
    except (ValueError, TypeError, OverflowError):
        return default


def bucket_features(row: Mapping[str, Any]) -> dict[str, str]:
    """Categorical features known at decision time; no future data."""
    distance = number(row.get("support_distance_atr"))
    ratio = number(row.get("volume_ratio"))
    trend = number(row.get("trend_score"))
    rr = number(row.get("rr1"))
    return {
        "support_atr": "missing" if not isfinite(distance) else ("near" if distance <= 1 else "far"),
        "volume": "missing" if not isfinite(ratio) else ("elevated" if ratio >= 1.2 else "normal"),
        "trend": "missing" if not isfinite(trend) else ("strong" if trend >= 70 else "other"),
        "rr": "missing" if not isfinite(rr) else ("high" if rr >= 2 else "low"),
    }


def outcome_from_bars(row: Mapping[str, Any], historical: pd.DataFrame) -> dict[str, Any]:
    """Outcome at next session's open; reject ambiguous days and re-adjusted data.

    Uses the next completed daily open only if it falls inside the snapshot's
    admissible execution range and the original risk/reward constraint.
    """
    pending = {"status": "PENDING", "outcome": None,
               "realized_entry": None, "realized_return_pct": None, "realized_r": None,
               "resolved_at": None}
    if not bool(row.get("buy_allowed")):
        return {**pending, "status": "INELIGIBLE", "outcome": "INELIGIBLE"}
    if historical is None or historical.empty or not {"Open", "High", "Low", "Close"}.issubset(historical):
        return pending
    try:
        signal_date = pd.Timestamp(row["signal_date"]).date()
        times = pd.to_datetime(historical.index)
        valid_signal = np.flatnonzero(np.array([t.date() == signal_date for t in times]))
        if not len(valid_signal):
            return pending
        idx = int(valid_signal[-1])
        reference = number(row.get("snapshot_close"))
        old_close = number(historical["Close"].iloc[idx])
        if not (reference > 0 and old_close > 0) or abs(old_close / reference - 1) > .002:
            return {**pending, "status": "DATA_CHANGED", "outcome": "DATA_CHANGED"}
        if idx + 1 >= len(historical):
            return pending
        stop = number(row.get("stop_price"))
        target = number(row.get("target_price"))
        lower = number(row.get("min_execution_price"))
        upper = number(row.get("max_execution_price"))
        open_price = number(historical["Open"].iloc[idx + 1])
        if not (stop > 0 and target > stop and lower > stop and upper >= lower and open_price > 0):
            return {**pending, "status": "INVALID", "outcome": "INVALID"}
        if not (lower <= open_price <= upper and stop < open_price < target and
                (target - open_price) / (open_price - stop) >= 1.5):
            return {**pending, "status": "NOT_FILLED", "outcome": "NOT_FILLED",
                    "resolved_at": str(times[idx + 1].date())}
        horizon = int(row.get("horizon") or 5)
        if not 1 <= horizon <= 30:
            return {**pending, "status": "INVALID", "outcome": "INVALID"}
        path = historical.iloc[idx + 1:idx + 1 + horizon]
        outcome, sale = None, None
        sold_on = None
        for candle_date, bar in path.iterrows():
            opn, low, high = (number(bar[c]) for c in ("Open", "Low", "High"))
            if not all(isfinite(x) for x in (opn, low, high)):
                return pending
            if opn <= stop:
                outcome, sale = "SL", opn  # downward gap / adverse fill
            elif opn >= target:
                outcome, sale = "TP1", target  # conservative: do not claim better gap fill
            elif low <= stop and high >= target:
                return {**pending, "status": "AMBIGUOUS", "outcome": "AMBIGUOUS",
                        "resolved_at": str(pd.Timestamp(candle_date).date())}
            elif low <= stop:
                outcome, sale = "SL", stop
            elif high >= target:
                outcome, sale = "TP1", target
            if outcome:
                sold_on = str(pd.Timestamp(candle_date).date())
                break
        if outcome is None:
            if len(path) < horizon:
                return pending
            sale = number(path["Close"].iloc[-1])
            if not isfinite(sale):
                return pending
            outcome, sold_on = "TIME_EXIT", str(pd.Timestamp(path.index[-1]).date())
        ret = (sale / open_price - 1) * 100
        raw_r = (sale - open_price) / (open_price - stop)
        return {"status": "RESOLVED", "outcome": outcome,
                "realized_entry": round(open_price, 7),
                "realized_return_pct": round(ret, 6), "realized_r": round(raw_r, 6),
                "resolved_at": sold_on}
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
        return {**pending, "status": "INVALID", "outcome": "INVALID"}


def learned_adjustment(candidate: Mapping[str, Any], resolved: Sequence[Mapping[str, Any]],
                       cost_percent: float = TRAINING_COST_PCT) -> dict[str, Any]:
    """Empirical-Bayes score adjustment learned from *resolved* prior signals.

    Uses mean realized R compared with training baseline, with shrinkage,
    minimum sample guards, clipping and cost assumptions. Never modifies hard
    risk gates. No use of a label from the current scan.
    """
    dataset = []
    for row in resolved:
        if row.get("status") != "RESOLVED" or row.get("outcome") not in {"SL", "TP1", "TIME_EXIT"}:
            continue
        risk = number(row.get("realized_entry")) - number(row.get("stop_price"))
        entry = number(row.get("realized_entry"))
        r = number(row.get("realized_r"))
        if not (isfinite(r) and risk > 0 and entry > 0):
            continue
        net_r = r - entry * (cost_percent / 100) / risk
        dataset.append((bucket_features(row), float(np.clip(net_r, -4, 4))))
    n = len(dataset)
    if n < MIN_RESOLVED:
        return {"adjustment": 0., "samples": n, "status": "INSUFFICIENT_DATA",
                "explanation": f"Učení čeká na alespoň {MIN_RESOLVED} jednoznačně vyhodnocených vstupů."}
    baseline = float(np.mean([r for _, r in dataset]))
    candidate_features = bucket_features(candidate)
    components = []
    for name, category in candidate_features.items():
        group = [ret for feat, ret in dataset if feat[name] == category]
        if len(group) < MIN_BUCKET or category == "missing":
            continue
        # Shrink category's excess outcome against full observed population.
        # 30 pseudo-observations damp the small-sample effects.
        shrink = len(group) / (len(group) + 30)
        contribution = 5.0 * shrink * (float(np.mean(group)) - baseline)
        components.append((name, category, len(group), contribution))
    adjustment = float(np.clip(np.mean([x[3] for x in components]),
                               -MAX_SCORE_ADJUSTMENT, MAX_SCORE_ADJUSTMENT)) if components else 0.0
    return {"adjustment": round(adjustment, 2), "samples": n,
            "status": "ACTIVE" if components else "INSUFFICIENT_BUCKETS",
            "baseline_r": round(baseline, 3),
            "explanation": "; ".join(f"{name}={cat}: {count} vzorků" for name, cat, count, _ in components)
                            or "Pro daný profil zatím není dost srovnatelných výsledků."}
