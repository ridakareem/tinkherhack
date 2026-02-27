"""
services/forgetting_curve_service.py
=====================================
Core mathematical engine for the Cognitive Memory Analytics System.

THEORY:
-------
Based on Ebbinghaus's forgetting curve, memory retention decays exponentially:

    R(t) = R₀ · exp(-λ · (t - t₀))

Where:
    t₀  = time when memory was encoded (quiz attempt time, in simulated days)
    R₀  = initial retention at t₀ (derived from quiz score, 0.0–1.0)
    λ   = decay rate (higher = faster forgetting)
    t   = query time (simulated days)

PIECEWISE STRUCTURE:
--------------------
The full memory curve is the UNION of all decay segments over time.
Each quiz reattempt:
  1. Closes the active segment (sets t_end)
  2. Starts a new segment at the reattempt time with a new R₀ from the score

Segments are NEVER overwritten. This preserves the full memory history,
enabling visualization of:
  - Decay between sessions
  - Retention "jumps" after reattempts (spaced repetition effect)

RETENTION FROM SCORE:
---------------------
R₀ = score_pct (e.g., 0.8 → 80% initial retention)
A minimum floor of 0.2 is applied to prevent zero retention.

DECAY RATE (λ):
---------------
Default baseline: λ = 0.1 (forgetting ~10% per simulated day)
ML-learned λ overrides the default when available.
"""

import math
from typing import List, Dict, Optional


# --- Constants ---

BASELINE_LAMBDA = 0.1       # Default decay rate (used when no ML params available)
MIN_RETENTION = 0.2         # Floor: retention never drops below 20%
MAX_RETENTION = 1.0         # Cap: retention never exceeds 100%
R0_MIN_FLOOR = 0.2          # Minimum initial retention after a quiz


def score_to_r0(score_pct: float) -> float:
    """
    Convert a quiz score (0.0–1.0) to initial retention R₀.

    A perfect score yields R₀ = 1.0.
    The floor ensures even a 0% score yields some retention.

    Args:
        score_pct: Fraction of questions answered correctly (0.0–1.0)

    Returns:
        R₀ value clamped to [R0_MIN_FLOOR, MAX_RETENTION]
    """
    r0 = max(R0_MIN_FLOOR, min(MAX_RETENTION, score_pct))
    return round(r0, 4)


def retention_at(r0: float, lambda_val: float, t0: float, t: float) -> float:
    """
    Compute memory retention at time t for a single decay segment.

    Formula:
        R(t) = R₀ · exp(-λ · (t - t₀))

    Args:
        r0: Initial retention at segment start (0.0–1.0)
        lambda_val: Decay rate (>0)
        t0: Segment start time (simulated days)
        t: Query time (simulated days)

    Returns:
        Retention value clamped to [MIN_RETENTION, MAX_RETENTION]
        Returns 0.0 if t < t0 (segment hasn't started yet)
    """
    if t < t0:
        return 0.0
    raw = r0 * math.exp(-lambda_val * (t - t0))
    return round(max(MIN_RETENTION, min(MAX_RETENTION, raw)), 4)


def build_curve_points(
    segments: List[Dict],
    t_start: float,
    t_end: float,
    num_points: int = 200
) -> List[Dict]:
    """
    Generate a list of (time, retention) points for the full piecewise curve.

    For each time point, the ACTIVE segment is used.
    A segment is active if: t0 <= t < t_end (or t_end is None = still active).

    Args:
        segments: List of decay segment dicts with keys:
                  t0, r0, lambda_val, t_end (may be None)
        t_start: Start of the time range to plot
        t_end: End of the time range to plot
        num_points: Number of time points to generate

    Returns:
        List of dicts: [{"t": float, "retention": float}, ...]
    """
    if not segments or num_points < 2:
        return []

    step = (t_end - t_start) / (num_points - 1)
    curve = []

    for i in range(num_points):
        t = t_start + i * step
        r = _retention_for_time(segments, t)
        curve.append({"t": round(t, 3), "retention": r})

    return curve


def _retention_for_time(segments: List[Dict], t: float) -> float:
    """
    Find the active decay segment for time t and compute retention.

    Segments are matched by: t0 <= t < t_end (t_end=None means open-ended).
    If no segment matches, returns MIN_RETENTION.

    Args:
        segments: Ordered list of decay segment dicts
        t: Query time (simulated days)

    Returns:
        Retention value (float)
    """
    active_seg = None
    for seg in segments:
        seg_t0 = seg["t0"]
        seg_t_end = seg["t_end"]  # May be None (open)
        if t >= seg_t0:
            if seg_t_end is None or t < seg_t_end:
                active_seg = seg
                break  # Segments sorted newest-first; first match wins

    if active_seg is None:
        return MIN_RETENTION

    return retention_at(
        r0=active_seg["r0"],
        lambda_val=active_seg["lambda_val"],
        t0=active_seg["t0"],
        t=t
    )


def close_active_segment(db, user_id: int, topic: str, t_close: float):
    """
    Close the currently open decay segment for a user+topic.

    Called before creating a new segment (i.e., on reattempt).
    Sets t_end on the most recent open segment.

    Args:
        db: SQLite database connection
        user_id: Current user's ID
        topic: Topic being studied
        t_close: Simulated time at which segment ends
    """
    db.execute("""
        UPDATE decay_segments
        SET t_end = ?
        WHERE user_id = ? AND topic = ? AND t_end IS NULL
    """, (t_close, user_id, topic))
    db.commit()


def create_decay_segment(
    db,
    attempt_id: int,
    user_id: int,
    topic: str,
    t0: float,
    score_pct: float,
    lambda_val: float
) -> int:
    """
    Insert a new decay segment after a quiz attempt.

    Called after every quiz completion to start a new piecewise segment.

    Args:
        db: SQLite database connection
        attempt_id: ID of the quiz attempt that created this segment
        user_id: Current user's ID
        topic: Topic being studied
        t0: Simulated time of the attempt (segment start)
        score_pct: Quiz score (used to derive R₀)
        lambda_val: Decay rate (from ML params or baseline)

    Returns:
        ID of the newly created decay segment
    """
    r0 = score_to_r0(score_pct)

    cursor = db.execute("""
        INSERT INTO decay_segments (attempt_id, user_id, topic, t0, r0, lambda_val, t_end)
        VALUES (?, ?, ?, ?, ?, ?, NULL)
    """, (attempt_id, user_id, topic, t0, r0, lambda_val))
    db.commit()
    return cursor.lastrowid


def get_segments_for_topic(db, user_id: int, topic: str) -> List[Dict]:
    """
    Retrieve all decay segments for a user+topic, sorted by t0 descending.

    Newest segment first so _retention_for_time finds the active one quickly.

    Args:
        db: SQLite database connection
        user_id: Current user's ID
        topic: Topic name

    Returns:
        List of segment dicts with keys: id, t0, r0, lambda_val, t_end
    """
    rows = db.execute("""
        SELECT id, t0, r0, lambda_val, t_end
        FROM decay_segments
        WHERE user_id = ? AND topic = ?
        ORDER BY t0 DESC
    """, (user_id, topic)).fetchall()

    return [dict(row) for row in rows]
