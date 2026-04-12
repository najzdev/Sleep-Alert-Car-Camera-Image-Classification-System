"""
Alert System & Event Logger - Pro-Vision v2.0
Handles multi-level alerts and persistent event logging
"""

import json
import time
from datetime import datetime
from collections import deque
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Event:
    """Represents a system event"""
    timestamp: str
    alert_level: str
    message: str
    drowsy_score: float = None
    
    def to_dict(self):
        return asdict(self)


class EventLogger:
    """Persistent event logging system"""
    
    def __init__(self, max_events: int = 100, log_file: str = "vision_log.json"):
        self.events = deque(maxlen=max_events)
        self.log_file = Path(log_file)
        self._load_from_disk()
    
    def log(self, alert_level: AlertLevel, message: str, drowsy_score: float = None):
        """Record an event"""
        event = Event(
            timestamp=datetime.now().isoformat(),
            alert_level=alert_level.value,
            message=message,
            drowsy_score=drowsy_score
        )
        self.events.append(event)
        self._save_to_disk()
        return event
    
    def _save_to_disk(self):
        """Persist events to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([e.to_dict() for e in self.events], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save logs to disk: {e}")
    
    def _load_from_disk(self):
        """Load events from JSON file if exists"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.events.append(Event(**item))
            except Exception as e:
                print(f"Warning: Could not load logs from disk: {e}")
    
    def get_recent(self, count: int = 10) -> list:
        """Get N most recent events"""
        return list(self.events)[-count:]
    
    def get_by_level(self, level: AlertLevel, count: int = 10) -> list:
        """Get events by alert level"""
        filtered = [e for e in self.events if e.alert_level == level.value]
        return filtered[-count:]
    
    def clear(self):
        """Clear all events"""
        self.events.clear()
        self._save_to_disk()


class AlertManager:
    """Manages multi-level alerts with cooldown and deduplication"""
    
    def __init__(self, config: dict, logger: EventLogger = None):
        self.config = config["alerts"]
        self.logger = logger or EventLogger()
        self.last_beep_time = 0
        self.last_status = None
    
    def should_trigger_audio(self) -> bool:
        """Check if enough time has passed since last alert"""
        return (time.time() - self.last_beep_time) > self.config["beep_cooldown"]
    
    def record_beep(self):
        """Update last beep timestamp"""
        self.last_beep_time = time.time()
    
    def evaluate(self, drowsy_score: float, status_enum) -> dict:
        """
        Evaluate alert requirements and log status changes
        Returns: {should_alert, alert_level, alert_message}
        """
        status = status_enum.value
        
        alert_info = {
            "should_alert": False,
            "alert_level": AlertLevel.INFO,
            "alert_message": "",
            "border_color": (0, 255, 0)  # BGR format
        }
        
        # Log status changes
        if status != self.last_status:
            self.logger.log(AlertLevel.INFO, f"Status changed to {status}", drowsy_score)
            self.last_status = status
        
        # Determine alert level
        if drowsy_score > self.config["critical_threshold"]:
            alert_info["alert_level"] = AlertLevel.CRITICAL
            alert_info["should_alert"] = self.should_trigger_audio()
            alert_info["alert_message"] = "🚨 CRITICAL: WAKE UP IMMEDIATELY!"
            alert_info["border_color"] = (0, 0, 255)  # Red
            self.logger.log(AlertLevel.CRITICAL, 
                          "Driver sleeping detected", drowsy_score)
        
        elif drowsy_score > self.config["warning_threshold"]:
            alert_info["alert_level"] = AlertLevel.WARNING
            alert_info["alert_message"] = "⚠️ WARNING: Eyes closing frequently"
            alert_info["border_color"] = (0, 165, 255)  # Orange
            self.logger.log(AlertLevel.WARNING, 
                          "Drowsiness detected", drowsy_score)
        
        else:
            alert_info["alert_level"] = AlertLevel.INFO
            alert_info["alert_message"] = "✅ AWAKE: Driver is attentive"
            alert_info["border_color"] = (0, 255, 0)  # Green
        
        if alert_info["should_alert"]:
            self.record_beep()
        
        return alert_info
