"""
BatchResult
===========

Result dataclass for batch processing operations.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchResult:
    """Result of a batch processing operation.

    Attributes:
        total: Total number of files processed.
        successful: Number of successfully processed files.
        failed: Number of failed files.
        skipped: Number of skipped files.
        output_paths: List of output file paths.
        errors: Dictionary of path -> error message.
    """

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    output_paths: list[Path] = field(default_factory=list)
    errors: dict[Path, str] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.successful / self.total) * 100
