"""
Regression tests for https://github.com/asdf-format/asdf/issues/1334.

A view of a memmapped array (e.g. ``af["a"][:5]``) used after its file was
closed read unmapped memory and segfaulted.

https://github.com/asdf-format/asdf/pull/1668 avoided the segfault by no
longer closing the mmap, which left the file mapped (and open) for as long as
any view of it was referenced. The current fix closes the mmap again, and has
views check that their mapping is still open before touching it (see
``asdf.tags.core.MemmapArrayView``).
"""

import numpy as np
import psutil
import pytest

import asdf


def test_memmap_closes_with_context_manager(tmp_path):
    """
    Exiting ``with asdf.open(...)`` releases the file even while a view of one
    of its memmapped arrays is still referenced.

    This is what PR #1668 broke: holding a view kept the mmap -- which holds its
    own file descriptor -- open indefinitely outside the context manager until
    the view is garbage collected.
    """
    a = np.ones(10, dtype="uint8")
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": a}).write_to(fn)

    p = psutil.Process()
    orig_open = p.open_files()

    with asdf.open(fn, memmap=True) as af:
        # Create a view which persists after the close
        v = af["a"][:5]  # noqa: F841
        assert len(p.open_files()) > len(orig_open)

    assert len(p.open_files()) <= len(orig_open)


def test_memmap_view_access_after_close_raises(tmp_path):
    """
    A view of a memmapped array raises ``OSError`` when used after the file is
    closed, rather than segfaulting -- the original #1334 report.
    """
    a = np.ones(10, dtype="uint8")
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": a}).write_to(fn)

    with asdf.open(fn, memmap=True) as af:
        v = af["a"][:5]

    # Ufuncs (here ``==``) go through __array_ufunc__
    with pytest.raises(OSError, match="closed"):
        np.all(v == 1)

    # indexing goes through __getitem__
    with pytest.raises(OSError, match="closed"):
        v[0]
