"""Typed error hierarchy for the Recon3D engine.

Every error carries a machine-readable ``code`` plus an optional ``details``
mapping so that agents and API clients can react programmatically instead of
scraping English text.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class Recon3DError(Exception):
    """Base class for all engine errors."""

    code = "recon3d_error"
    http_status = 500

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class ConfigError(Recon3DError):
    code = "config_error"
    http_status = 400


class ValidationError(Recon3DError):
    """Bad input supplied by a user or agent."""

    code = "validation_error"
    http_status = 400


class SecurityError(Recon3DError):
    """A path or operation was rejected by the sandbox rules."""

    code = "security_error"
    http_status = 403


class NotFoundError(Recon3DError):
    code = "not_found"
    http_status = 404


class ConflictError(Recon3DError):
    code = "conflict"
    http_status = 409


class InsufficientDataError(Recon3DError):
    """The supplied references cannot support the requested reconstruction.

    This is deliberately *not* swallowed: spec requirement #5 forbids silently
    producing a bad model from bad input.
    """

    code = "insufficient_input"
    http_status = 422


class HardwareError(Recon3DError):
    """Requested operation exceeds the resources available on this machine."""

    code = "insufficient_hardware"
    http_status = 507


class StageError(Recon3DError):
    """A pipeline stage failed."""

    code = "stage_failed"
    http_status = 500

    def __init__(
        self,
        message: str,
        *,
        stage: str = "unknown",
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.stage = stage
        self.recoverable = recoverable
        self.details.setdefault("stage", stage)
        self.details.setdefault("recoverable", recoverable)


class CancelledError(Recon3DError):
    """The job was cancelled by the operator."""

    code = "cancelled"
    http_status = 409


class ModelError(Recon3DError):
    """An optional local model could not be downloaded, verified or loaded."""

    code = "model_error"
    http_status = 409


class BackendUnavailableError(Recon3DError):
    """An optional backend (COLMAP, Open3D, Blender, ONNX...) is missing."""

    code = "backend_unavailable"
    http_status = 501
