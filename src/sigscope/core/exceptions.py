"""Typed exceptions for sigscope. Every parser raises one of these, never a bare crash.

Covers: SR-01 (malformed/truncated/oversized input is rejected with a typed error).
"""

from __future__ import annotations


class SigscopeError(Exception):
    """Base class for all sigscope errors."""


class UnsupportedFormatError(SigscopeError):
    """Raised when a file's format is not recognised or not supported."""


class MalformedFileError(SigscopeError):
    """Raised when a file's header or structure is invalid or truncated."""


class FileTooLargeError(SigscopeError):
    """Raised when a file exceeds the configured maximum size before full allocation."""


class MissingParametersError(SigscopeError):
    """Raised when required format parameters (e.g. raw .IQ dtype) are not supplied
    and cannot be inferred from a SigMF sidecar. This is the documented
    'cannot proceed fully blind' case (FR-02 AC3): resolved via analyst input,
    never a silent guess.
    """
