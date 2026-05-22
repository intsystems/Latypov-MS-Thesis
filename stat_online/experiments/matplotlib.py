"""Matplotlib setup scoped to an experiment output directory."""

from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib(output_dir: str | Path):
    """Configure Matplotlib for headless file rendering.

    Experiment commands should not require callers to export ``MPLCONFIGDIR``.
    When plots are requested, keep Matplotlib's cache/config files next to the
    experiment artifacts instead of relying on a writable user home directory.
    """
    cache_dir = Path(output_dir) / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    return matplotlib


def import_pyplot(output_dir: str | Path):
    """Return ``matplotlib.pyplot`` after output-scoped configuration."""
    configure_matplotlib(output_dir)
    import matplotlib.pyplot as plt

    return plt
