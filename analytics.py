"""
Analytics Module - Pro-Vision v2.0
Tracks blink rate, eye closure duration, and drowsiness scoring over time.
"""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class AnalyticsSnapshot:
    """Point-in-time analytics summary."""
    blink_rate_per_minute: float
    avg_eye_closure_ms: float
    drowsiness_score: float
    session_duration_s: float
    total_blinks: int
    long_closure_count: int


class SessionAnalytics:
    """
    Tracks real-time metrics across a monitoring session.
    - Blink rate (per minute, rolling 60-second window)
    - Eye closure duration (ms per closure event)
    - Drowsiness score (0-100, passed in from classifier)
    """

    LONG_CLOSURE_THRESHOLD_MS = 400  # closures longer than this flag fatigue

    def __init__(self):
        self._session_start = time.time()

        # Ring buffer of (timestamp, blink=True) for rolling blink rate
        self._blink_timestamps: deque = deque()
        self._BLINK_WINDOW_S = 60.0

        # Closure tracking
        self._closure_start: float | None = None
        self._closure_durations: deque = deque(maxlen=50)
        self._long_closure_count = 0
        self._total_blinks = 0

        # Drowsiness score history for chart display
        self.score_history: deque = deque(maxlen=300)
        self.blink_rate_history: deque = deque(maxlen=300)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def record_frame(
        self,
        eyes_open: bool,
        is_blink: bool,
        drowsy_score: float,
    ) -> None:
        """
        Call once per processed frame.

        Parameters
        ----------
        eyes_open : bool
            True when at least one eye is detected open.
        is_blink : bool
            True on the single frame where a completed blink is registered.
        drowsy_score : float
            Current 0-100 score from DrowsinessClassifier.
        """
        now = time.time()

        # --- Closure duration tracking ---
        if not eyes_open:
            if self._closure_start is None:
                self._closure_start = now
        else:
            if self._closure_start is not None:
                duration_ms = (now - self._closure_start) * 1000.0
                self._closure_durations.append(duration_ms)
                if duration_ms > self.LONG_CLOSURE_THRESHOLD_MS:
                    self._long_closure_count += 1
                self._closure_start = None

        # --- Blink counting ---
        if is_blink:
            self._total_blinks += 1
            self._blink_timestamps.append(now)

        # Prune blink timestamps outside the rolling window
        cutoff = now - self._BLINK_WINDOW_S
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        # --- History buffers for chart rendering ---
        self.score_history.append(drowsy_score)
        self.blink_rate_history.append(self._blink_rate_per_minute())

    def snapshot(self) -> AnalyticsSnapshot:
        """Return a current analytics snapshot."""
        return AnalyticsSnapshot(
            blink_rate_per_minute=self._blink_rate_per_minute(),
            avg_eye_closure_ms=self._avg_closure_ms(),
            drowsiness_score=self.score_history[-1] if self.score_history else 0.0,
            session_duration_s=time.time() - self._session_start,
            total_blinks=self._total_blinks,
            long_closure_count=self._long_closure_count,
        )

    def reset(self) -> None:
        """Reset all counters for a new session."""
        self.__init__()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _blink_rate_per_minute(self) -> float:
        """Rolling blink count over the last 60 seconds, scaled to per-minute."""
        if not self._blink_timestamps:
            return 0.0
        elapsed = min(time.time() - self._session_start, self._BLINK_WINDOW_S)
        if elapsed < 1.0:
            return 0.0
        return (len(self._blink_timestamps) / elapsed) * 60.0

    def _avg_closure_ms(self) -> float:
        if not self._closure_durations:
            return 0.0
        return sum(self._closure_durations) / len(self._closure_durations)
