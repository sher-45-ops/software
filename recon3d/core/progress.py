"""Progress events, stage taxonomy, logging and cancellation (spec #30, #31).

Every pipeline stage reports through :class:`ProgressReporter`, which fans out
to in-process subscribers (the REST/WebSocket layer, the studio UI, the CLI
renderer) and to a per-project ``events.jsonl`` file so a human can replay what
happened after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .store import append_jsonl, utc_now

# --------------------------------------------------------------------------
# Stage taxonomy - these strings appear in CLI, API, WebSocket and MCP output
# --------------------------------------------------------------------------
STAGES: List[str] = [
    "queued",
    "ingestion",
    "validation",
    "segmentation",
    "camera_estimation",
    "depth_estimation",
    "point_cloud",
    "mesh_reconstruction",
    "mesh_cleanup",
    "uv",
    "texture",
    "materials",
    "optimization",
    "rigging",
    "lod",
    "preview",
    "comparison",
    "refinement",
    "export",
    "done",
]

#: Rough share of total runtime per stage - used for honest overall progress.
STAGE_WEIGHTS: Dict[str, float] = {
    "queued": 0.0,
    "ingestion": 0.02,
    "validation": 0.03,
    "segmentation": 0.05,
    "camera_estimation": 0.10,
    "depth_estimation": 0.18,
    "point_cloud": 0.14,
    "mesh_reconstruction": 0.16,
    "mesh_cleanup": 0.06,
    "uv": 0.04,
    "texture": 0.10,
    "materials": 0.02,
    "optimization": 0.03,
    "rigging": 0.03,
    "lod": 0.02,
    "preview": 0.01,
    "comparison": 0.01,
    "refinement": 0.0,
    "export": 0.0,
    "done": 0.0,
}


def overall_progress(stage: str, stage_progress: float) -> float:
    """Map (stage, stage_progress) to a 0-100 overall figure."""
    done = 0.0
    for s in STAGES:
        if s == stage:
            done += STAGE_WEIGHTS.get(s, 0.0) * max(0.0, min(1.0, stage_progress))
            break
        done += STAGE_WEIGHTS.get(s, 0.0)
    return round(min(99.9, done * 100.0), 1)


@dataclass
class ProgressEvent:
    """A single progress notification."""

    stage: str
    progress: float
    message: str = ""
    project: str = ""
    job: str = ""
    level: str = "info"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "progress": round(float(self.progress), 2),
            "message": self.message,
            "project": self.project,
            "job": self.job,
            "level": self.level,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class CancellationToken:
    """Cooperative cancellation shared between threads, processes and the API.

    Inside one process the token is an event.  A *watch file* extends it across
    processes: ``recon3d cancel <job-id>`` touches ``<intermediate>/jobs/<id>.cancel``
    and a running reconstruction - in the CLI or in the API server - stops at the
    next checkpoint boundary and keeps everything it already finished.
    """

    def __init__(self, watch: Optional[Path] = None) -> None:
        self._event = threading.Event()
        self.reason: str = ""
        self.watch: Optional[Path] = Path(watch) if watch else None

    def cancel(self, reason: str = "cancelled by operator") -> None:
        self.reason = reason
        self._event.set()

    def watch_file(self, path: Path) -> None:
        self.watch = Path(path)

    def request_path(self) -> Optional[Path]:
        return self.watch

    @property
    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self.watch is not None:
            try:
                if self.watch.exists():
                    self.reason = self.reason or "cancel requested by operator"
                    return True
            except OSError:  # pragma: no cover - unreadable volume
                return False
        return False

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            from ..errors import CancelledError

            raise CancelledError(self.reason or "cancelled")


class ProgressReporter:
    """Fan-out progress sink shared by all pipeline stages."""

    def __init__(
        self,
        *,
        project: str = "",
        job: str = "",
        events_path: Optional[Path] = None,
        token: Optional[CancellationToken] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.project = project
        self.job = job
        self.events_path = Path(events_path) if events_path else None
        self.token = token or CancellationToken()
        self.logger = logger or logging.getLogger("recon3d")
        self._subscribers: List[Callable[[ProgressEvent], None]] = []
        self._history: List[ProgressEvent] = []
        self._lock = threading.RLock()
        self.stage = "queued"
        self.started = time.time()
        self._stage_started = time.time()
        self._stage_eta: Dict[str, float] = {}

    # -- subscription ---------------------------------------------------
    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    @property
    def history(self) -> List[ProgressEvent]:
        with self._lock:
            return list(self._history)

    # -- emitting -------------------------------------------------------
    def emit(self, event: ProgressEvent) -> ProgressEvent:
        if not event.project:
            event.project = self.project
        if not event.job:
            event.job = self.job
        with self._lock:
            self._history.append(event)
            if len(self._history) > 5000:  # bound memory for very long jobs
                del self._history[:1000]
            subscribers = list(self._subscribers)
        if self.events_path is not None:
            try:
                append_jsonl(self.events_path, event.to_dict())
            except OSError:  # pragma: no cover - logging must never break a job
                pass
        log_line = f"[{event.stage}] {event.progress:5.1f}% {event.message}"
        if event.level in {"error", "critical"}:
            self.logger.error(log_line)
        elif event.level == "warning":
            self.logger.warning(log_line)
        else:
            self.logger.info(log_line)
        for cb in subscribers:
            try:
                cb(event)
            except Exception:  # pragma: no cover - a bad subscriber must not kill the job
                self.logger.debug("progress subscriber raised", exc_info=True)
        return event

    def stage_start(self, stage: str, message: str = "", **data: Any) -> ProgressEvent:
        self.stage = stage
        self._stage_started = time.time()
        return self.emit(
            ProgressEvent(stage=stage, progress=0.0, message=message or f"stage {stage} started",
                          data=dict(data))
        )

    def update(self, progress: float, message: str = "", stage: Optional[str] = None,
               **data: Any) -> ProgressEvent:
        stage_name = stage or self.stage
        payload = dict(data)
        payload["overall"] = overall_progress(stage_name, progress / 100.0)
        elapsed = time.time() - self.started
        payload["elapsed_s"] = round(elapsed, 2)
        if progress >= 1.0:
            payload["eta_s"] = 0
        else:
            done = max(1e-6, overall_progress(stage_name, progress / 100.0) / 100.0)
            if done > 0.02:
                payload["eta_s"] = round(elapsed / done - elapsed, 1)
        return self.emit(ProgressEvent(stage=stage_name, progress=progress, message=message,
                                       data=payload))

    def info(self, message: str, **data: Any) -> ProgressEvent:
        return self.emit(ProgressEvent(stage=self.stage, progress=self._last_progress(),
                                       message=message, level="info", data=dict(data)))

    def warning(self, message: str, **data: Any) -> ProgressEvent:
        return self.emit(ProgressEvent(stage=self.stage, progress=self._last_progress(),
                                       message=message, level="warning", data=dict(data)))

    def error(self, message: str, **data: Any) -> ProgressEvent:
        return self.emit(ProgressEvent(stage=self.stage, progress=self._last_progress(),
                                       message=message, level="error", data=dict(data)))

    def _last_progress(self) -> float:
        with self._lock:
            for ev in reversed(self._history):
                if ev.stage == self.stage:
                    return ev.progress
        return 0.0

    def check_cancelled(self) -> None:
        self.token.raise_if_cancelled()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            last = self._history[-1].to_dict() if self._history else None
        return {
            "project": self.project,
            "job": self.job,
            "stage": self.stage,
            "last_event": last,
            "cancelled": self.token.cancelled,
            "elapsed_s": round(time.time() - self.started, 2),
        }


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------
class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        payload = {
            "ts": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_LOGGING_CONFIGURED = False
_LOG_LOCK = threading.Lock()


def setup_logging(*, level: str = "info", log_file: Optional[Path] = None,
                  json_logs: bool = False, quiet: bool = False,
                  stream=None) -> logging.Logger:
    """Configure the ``recon3d`` logger exactly once per process."""
    global _LOGGING_CONFIGURED
    logger = logging.getLogger("recon3d")
    with _LOG_LOCK:
        if _LOGGING_CONFIGURED:
            if log_file is not None:
                _attach_file_handler(logger, log_file, json_logs)
            return logger
        logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        logger.propagate = False
        if not quiet and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            if stream is None:
                stream = sys.stderr if level == "error" else sys.stdout
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                JsonLogFormatter() if json_logs
                else logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
            )
            logger.addHandler(handler)
        if log_file is not None:
            _attach_file_handler(logger, log_file, json_logs)
        _LOGGING_CONFIGURED = True
    return logger


def _attach_file_handler(logger: logging.Logger, log_file: Path, json_logs: bool) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_file.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            return
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(JsonLogFormatter() if json_logs
                    else logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)


def iter_stage_names() -> Iterable[str]:
    return tuple(STAGES)
