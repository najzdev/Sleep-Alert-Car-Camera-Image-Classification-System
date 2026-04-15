"""
Alert System - Pro-Vision v2.0
Multi-level alert management with cooldown, beep, and event logging.

Alert levels
------------
NORMAL   -> driver attentive
WARNING  -> eyes closing frequently (moderate risk)
DROWSY   -> sustained eye closure (high risk, beep triggers)
"""

import json
import time
import threading
import platform
from datetime import datetime
from collections import deque
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    NORMAL  = "NORMAL"
    WARNING = "WARNING"
    DROWSY  = "DROWSY"


# ---------------------------------------------------------------------------
# Event data model
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp:   str
    alert_level: str
    message:     str
    drowsy_score: float = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event logger
# ---------------------------------------------------------------------------

class EventLogger:
    """Appends events to an in-memory deque and optionally to disk."""

    def __init__(self, max_events: int = 200, log_file: str = "vision_log.json"):
        self.log_file = Path(log_file)
        self.events: deque = deque(maxlen=max_events)
        self._load()

    # ---- public ---

    def log(
        self,
        level: AlertLevel,
        message: str,
        score: float = None,
    ) -> Event:
        event = Event(
            timestamp=datetime.now().isoformat(),
            alert_level=level.value,
            message=message,
            drowsy_score=score,
        )
        self.events.append(event)
        self._save()
        return event

    def get_recent(self, count: int = 10) -> list:
        return list(self.events)[-count:]

    def get_by_level(self, level: AlertLevel, count: int = 10) -> list:
        return [e for e in self.events if e.alert_level == level.value][-count:]

    def clear(self) -> None:
        self.events.clear()
        self._save()

    # ---- private ---

    def _save(self) -> None:
        try:
            with open(self.log_file, "w") as fh:
                json.dump([e.to_dict() for e in self.events], fh, indent=2)
        except OSError as exc:
            print(f"[EventLogger] Could not save: {exc}")

    def _load(self) -> None:
        if not self.log_file.exists():
            return
        try:
            with open(self.log_file, "r") as fh:
                for item in json.load(fh):
                    self.events.append(Event(**item))
        except Exception as exc:
            print(f"[EventLogger] Could not load: {exc}")


# ---------------------------------------------------------------------------
# Beep utility (cross-platform, non-blocking)
# ---------------------------------------------------------------------------

def _beep_non_blocking(frequency: int = 1000, duration_ms: int = 200, repeats: int = 3, gap_ms: int = 120) -> None:
    """
    Fire multiple beeps in a daemon thread (non-blocking).
    repeats : number of beeps (default = 3)
    gap_ms  : delay between beeps
    """
    def _play():
        try:
            sys_name = platform.system()

            for _ in range(repeats):
                if sys_name == "Windows":
                    import winsound
                    winsound.Beep(frequency, duration_ms)

                elif sys_name == "Darwin":
                    import subprocess
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Ping.aiff"],
                        check=False, capture_output=True,
                    )

                else:
                    import subprocess
                    result = subprocess.run(
                        ["beep", f"-f{frequency}", f"-l{duration_ms}"],
                        capture_output=True,
                    )
                    if result.returncode != 0:
                        print("\a", end="", flush=True)

                time.sleep(gap_ms / 1000.0)

        except Exception:
            for _ in range(repeats):
                print("\a", end="", flush=True)
                time.sleep(gap_ms / 1000.0)

    threading.Thread(target=_play, daemon=True).start()

# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Evaluates the current drowsiness score and triggers alerts.

    Thresholds (from config["alerts"]):
        warning_threshold  : score above which WARNING fires
        critical_threshold : score above which DROWSY fires
        beep_cooldown      : seconds between audio alerts

    The manager tracks the previous status to log only on transitions,
    preventing log spam during sustained drowsy periods.
    """

    # BGR colors for border overlays
    _COLORS = {
        AlertLevel.NORMAL:  (0, 200, 0),    # green
        AlertLevel.WARNING: (0, 165, 255),   # orange
        AlertLevel.DROWSY:  (0, 0, 220),     # red
    }

    _MESSAGES = {
        AlertLevel.NORMAL:  "NORMAL: Driver is attentive",
        AlertLevel.WARNING: "WARNING: Eyes closing frequently",
        AlertLevel.DROWSY:  "DROWSY: WAKE UP - Eyes closed too long",
    }

    def __init__(self, config: dict, logger: EventLogger = None):
        self._cfg = config["alerts"]
        self.logger = logger or EventLogger()
        self._last_beep: float = 0.0
        self._last_level: AlertLevel | None = None

    # ---- public ---

    def evaluate(self, drowsy_score: float) -> dict:
        """
        Compute alert level and decide whether audio/visual should fire.

        Returns
        -------
        dict with keys:
            level         : AlertLevel
            message       : str
            border_color  : (B, G, R) tuple
            play_beep     : bool
            log_event     : bool  (True when level changed)
        """
        level = self._classify(drowsy_score)
        level_changed = level != self._last_level
        now = time.time()
        cooldown_elapsed = (now - self._last_beep) > self._cfg["beep_cooldown"]

        play_beep = (level == AlertLevel.DROWSY) and cooldown_elapsed

        if play_beep:
            self._last_beep = now
            _beep_non_blocking()

        if level_changed:
            self.logger.log(
                AlertLevel[level.value],
                self._MESSAGES[level],
                drowsy_score,
            )
            self._last_level = level

        return {
            "level":        level,
            "message":      self._MESSAGES[level],
            "border_color": self._COLORS[level],
            "play_beep":    play_beep,
            "log_event":    level_changed,
        }

    # ---- private ---

    def _classify(self, score: float) -> AlertLevel:
        if score >= self._cfg["critical_threshold"]:
            return AlertLevel.DROWSY
        if score >= self._cfg["warning_threshold"]:
            return AlertLevel.WARNING
        return AlertLevel.NORMAL
