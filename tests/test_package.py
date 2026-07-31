"""Tests for installed-package metadata."""

from importlib.metadata import version

import count_significance


def test_public_version_matches_distribution():
    """The public version is derived from the installed distribution."""
    assert count_significance.__version__ == version("count-significance")
