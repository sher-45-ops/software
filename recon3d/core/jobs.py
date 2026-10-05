"""Job manager: background execution, cancellation, checkpoint/resume (spec #31).

A *job* is one pipeline run.  Jobs execute in a worker thread inside the engine
process (which works identically under the CLI, the REST API and the MCP
server).  Every finished stage writes a checkpoint to
``intermediate/<stage>/_checkpoint.json`` keyed by a hash of its inputs, so a
crashed or cancelled run resumes without repeating completed work - and a
follow-up job that only changes texture resolution re-uses the mesh.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import CancelledError, NotFoundError, Recon3DError, StageError
from .progress import CancellationToken, ProgressEvent, ProgressReporter, STAGES, utc_now
from .store import read_json, write_json


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # finished with non-fatal stage failures

    @property
    def terminal(self) -> bool:
        return self in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.PARTIAL}


@dataclass
class Job:
    """A pipeline execution record."""

    id: str
    project: str
    kind: str = "reconstruct"
    params: Dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    message: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    version: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    duration_s: float = 0.0
    resumable: bool = True
    resumed_from: Optional[str] = None

    # runtime only (never serialised)
    token: Optional[CancellationToken] = field(default=None, repr=False, compare=False)
    reporter: Optional[ProgressReporter] = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_events: bool = False, event_limit: int = 50) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "project": self.project,
            "kind": self.kind,
            "params": self.params,
            "state": self.state.value,
            "progress": round(self.progress, 2),
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "warnings": self.warnings,
            "stages": self.stages,
            "version": self.version,
            "duration_s": round(self.duration_s, 2),
            "resumable": self.resumable,
            "resumed_from": self.resumed_from,
        }
        if include_events and self.reporter is not None:
            data["events"] = [e.to_dict() for e in self.reporter.history[-event_limit:]]
        return data

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    def cancel(self, reason: str = "cancelled by operator") -> bool:
        if self.terminal:
            return False
        self.state = JobState.CANCELLING
        self.message = reason
        if self.token is not None:
            self.token.cancel(reason)
        return True


class JobManager:
    """Owns jobs, worker threads and checkpoints."""

    def __init__(self, *, max_concurrent: int = 1, log_dir: Optional[Path] = None) -> None:
        self._jobs: Dict[str, Job] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self.max_concurrent = max(1, int(max_concurrent))
        self.log_dir = Path(log_dir) if log_dir else None
        self._subscribers: List[Callable[[str, ProgressEvent], None]] = []

    # -- subscription ---------------------------------------------------
    def subscribe(self, callback: Callable[[str, ProgressEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _publish(self, job: Job, event: ProgressEvent) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(job.id, event)
            except Exception:  # pragma: no cover
                pass

    # -- lifecycle ------------------------------------------------------
    def create(self, project: str, *, kind: str = "reconstruct",
               params: Optional[Dict[str, Any]] = None, project_obj: Any = None,
               resumed_from: Optional[str] = None) -> Job:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, project=project, kind=kind, params=dict(params or {}),
                  resumed_from=resumed_from)
        job.token = CancellationToken()
        events_path = None
        if project_obj is not None:
            events_path = project_obj.events_path
            job.checkpoint_dir = str(project_obj.intermediate_dir)
        elif self.log_dir is not None:
            events_path = self.log_dir / f"{job_id}.jsonl"
        job.reporter = ProgressReporter(project=project, job=job_id, events_path=events_path,
                                        token=job.token)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("job not found", details={"job": job_id,
                                                          "known": list(self._jobs)})
        return job

    def list(self, project: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if project:
            jobs = [j for j in jobs if j.project == project]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def run_sync(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> Job:
        """Execute a job payload in the calling thread."""
        self._execute(job, fn)
        return job

    def submit(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> Job:
        """Execute a job payload on a background worker thread."""

        def _runner() -> None:
            self._execute(job, fn)

        thread = threading.Thread(target=_runner, name=f"recon3d-{job.id}", daemon=True)
        with self._lock:
            self._threads[job.id] = thread
        thread.start()
        return job

    def _execute(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> None:
        assert job.reporter is not None
        job.state = JobState.RUNNING
        job.started_at = utc_now()
        started = time.time()
        reporter = job.reporter

        def _on_event(event: ProgressEvent) -> None:
            job.stage = event.stage
            job.progress = float(event.data.get("overall", job.progress)) if event.data else job.progress
            if event.message:
                job.message = event.message
            if event.level == "warning" and event.message not in job.warnings:
                job.warnings.append(event.message)
            maybe_stage = job.stages.setdefault(event.stage, {"started_at": event.timestamp})
            maybe_stage["progress"] = event.progress
            maybe_stage["message"] = event.message
            self._publish(job, event)

        unsubscribe = reporter.subscribe(_on_event)
        try:
            result = fn(job) or {}
            job.result = result
            job.version = result.get("version") or job.version
            failed_stages = [s for s, info in job.stages.items()
                             if str(info.get("status", "")).lower() == "failed"]
            job.stages.setdefault("done", {})
            job.state = JobState.PARTIAL if failed_stages else JobState.COMPLETED
            if failed_stages:
                job.warnings.append(f"stages failed but were recovered/skipped: {failed_stages}")
            job.progress = 100.0
        except CancelledError as exc:
            job.state = JobState.CANCELLED
            job.error = exc.to_dict()
            reporter.warning(f"job cancelled: {exc.message}")
        except StageError as exc:
            job.state = JobState.FAILED
            job.error = exc.to_dict()
            job.resumable = exc.recoverable
            job.stages.setdefault(exc.stage, {})["status"] = "failed"
            reporter.error(f"stage '{exc.stage}' failed: {exc.message}", **exc.details)
        except Recon3DError as exc:
            job.state = JobState.FAILED
            job.error = exc.to_dict()
            reporter.error(exc.message, **exc.details)
        except Exception as exc:  # pragma: no cover - defensive
            job.state = JobState.FAILED
            job.error = {
                "error": True,
                "code": "internal_error",
                "message": str(exc),
                "details": {"traceback": traceback.format_exc()[-4000:]},
            }
            reporter.error(f"internal error: {exc}")
        finally:
            unsubscribe()
            job.duration_s = time.time() - started
            job.finished_at = utc_now()
            self._persist(job)

    def _persist(self, job: Job) -> None:
        if not job.checkpoint_dir:
            return
        try:
            path = Path(job.checkpoint_dir) / "jobs"
            path.mkdir(parents=True, exist_ok=True)
            write_json(path / f"{job.id}.json", job.to_dict())
        except OSError:  # pragma: no cover
            pass

    def cancel(self, job_id: str, reason: str = "cancelled by operator") -> Job:
        job = self.get(job_id)
        job.cancel(reason)
        return job

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Job:
        job = self.get(job_id)
        deadline = None if timeout is None else time.time() + timeout
        while not job.terminal:
            if deadline is not None and time.time() > deadline:
                break
            time.sleep(0.05)
        return job

    def active_count(self) -> int:
        return sum(1 for j in self.list() if j.state in {JobState.RUNNING, JobState.QUEUED})


# --------------------------------------------------------------------------
# Module-level default manager (used by CLI/API/MCP singletons)
# --------------------------------------------------------------------------
_DEFAULT: Optional[JobManager] = None
_DEFAULT_LOCK = threading.Lock()


def get_job_manager(log_dir: Optional[Path] = None, *, reset: bool = False) -> JobManager:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None or reset:
            _DEFAULT = JobManager(log_dir=log_dir)
        if log_dir is not None and _DEFAULT.log_dir is None:
            _DEFAULT.log_dir = Path(log_dir)
    return _DEFAULT


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------
def checkpoint_path(project: Any, stage: str) -> Path:
    return project.stage_dir(stage) / "_checkpoint.json"


def read_checkpoint(project: Any, stage: str) -> Optional[Dict[str, Any]]:
    data = read_json(checkpoint_path(project, stage))
    return data if isinstance(data, dict) else None


def write_checkpoint(project: Any, stage: str, *, inputs_hash: str,
                     outputs: Optional[Dict[str, Any]] = None,
                     statistics: Optional[Dict[str, Any]] = None,
                     status: str = "completed") -> Dict[str, Any]:
    payload = {
        "stage": stage,
        "status": status,
        "inputs_hash": inputs_hash,
        "outputs": outputs or {},
        "statistics": statistics or {},
        "finished_at": utc_now(),
        "project": getattr(project, "id", ""),
    }
    write_json(checkpoint_path(project, stage), payload)
    return payload


def checkpoint_valid(project: Any, stage: str, inputs_hash: str,
                     *, require_outputs: bool = True) -> bool:
    """True when a stage's cached outputs can be re-used for these inputs."""
    data = read_checkpoint(project, stage)
    if not data or data.get("inputs_hash") != inputs_hash or data.get("status") != "completed":
        return False
    if not require_outputs:
        return True
    outputs = data.get("outputs") or {}
    if not outputs:
        return False
    for rel in outputs.values():
        if not isinstance(rel, str):
            continue
        candidate = Path(rel)
        target = candidate if candidate.is_absolute() else Path(project.root) / candidate
        if not target.exists():
            return False  # corrupted-stage detection (spec #31)
    return True


def invalidate_stage(project: Any, stage: str) -> None:
    path = checkpoint_path(project, stage)
    if path.exists():
        path.unlink()
