import numpy as np
import pytest

import asdf


def test_memmap_closes_with_context_manager(tmp_path):
    """
    ``with asdf.open(...)`` must actually close the underlying file when the
    block exits, even if a memmapped array (or a view of one) is still
    referenced -- a context manager that leaves a file open is not managing
    the resource it was handed.

    https://github.com/asdf-format/asdf/issues/1334 was a real bug: a *view*
    of a memmapped array (e.g. ``af["a"][:5]``), accessed after the file
    closed, segfaulted instead of raising. The fix in gh-1668 addressed that
    by having close leave the mmap open (and thus the file handle open) for
    as long as any array/view derived from it was referenced -- silently,
    indefinitely. That broke the contract of ``with``: it stopped managing
    the resource at all. Closing is now unconditional, and the segfault is
    prevented by guarding the view instead (see below).
    """
    psutil = pytest.importorskip("psutil")

    a = np.ones(10, dtype="uint8")
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": a}).write_to(fn)

    p = psutil.Process()
    orig_open = p.open_files()

    with asdf.open(fn, memmap=True) as af:
        v = af["a"][:5]  # noqa: F841 (kept alive on purpose)
        assert len(p.open_files()) > len(orig_open)

    assert len(p.open_files()) <= len(orig_open)


def test_memmap_view_access_after_close_raises(tmp_path):
    """
    Regression test for the original issue #1334: accessing a view of a
    memmapped array after the file is closed used to segfault.

    Views of memmapped blocks are handed back as
    `asdf.tags.core.ndarray.MemmapArrayView`, which checks that the mapping
    is still open before touching it, so this raises instead.
    """
    a = np.ones(10, dtype="uint8")
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": a}).write_to(fn)

    with asdf.open(fn, memmap=True) as af:
        v = af["a"][:5]

    with pytest.raises(OSError, match="closed"):
        np.all(v == 1)

    with pytest.raises(OSError, match="closed"):
        v[0]


def test_memmap_array_raises_after_close(tmp_path):
    """
    The array/wrapper handed back by ``af["a"]`` re-validates itself against
    the file on every access (NDArrayType._make_array), so accessing it
    after the file is closed raises cleanly instead of touching freed
    memory.
    """
    a = np.ones(10, dtype="uint8")
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": a}).write_to(fn)

    with asdf.open(fn, memmap=True) as af:
        arr = af["a"]

    with pytest.raises(OSError, match="closed"):
        arr[:5]
