"""
CompressionResult
=================

Result dataclass from compression operations.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompressionResult:
    """Result from compression operation.

    Attributes:
        data: Compressed image bytes.
        size_kb: Final file size in kilobytes.
        quality: Final quality setting used.
        format: Output format.
        iterations: Number of optimization iterations.
    """

    data: bytes
    size_kb: float
    quality: int
    format: str
    iterations: int

    def save(self, path: str | Path) -> Path:
        """Save compressed data to file.

        Args:
            path: Output file path.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path
