import numpy as np
import pytest

import asdf


def test_update_with_memmapped_data_can_make_view_data_invalid(tmp_path):
    """
    Calling update with memmapped data can create invalid data in memmap views

    https://github.com/asdf-format/asdf/issues/1530

    ``update()`` rewrites the file and remaps it, so a view taken before the
    call is left pointing at memory that is no longer valid. That used to
    return stale data (or segfault); the view now carries a guard for the
    mapping it came from, so touching it after the update raises instead.

    Access via the tree keeps working and reflects the updated file.
    """
    fn = tmp_path / "test.asdf"
    a = np.zeros(10, dtype="uint8")
    b = np.ones(10, dtype="uint8")
    ov = a[:3]

    af = asdf.AsdfFile({"a": a, "b": b})
    af.write_to(fn)

    with asdf.open(fn, mode="rw", memmap=True) as af:
        va = af["a"][:3]
        np.testing.assert_array_equal(a, af["a"])
        np.testing.assert_array_equal(b, af["b"])
        np.testing.assert_array_equal(va, ov)

        af["c"] = "a" * 10000
        af.update()

        # the view taken before the update is no longer valid, and says so
        with pytest.raises(OSError, match="closed"):
            np.all(va == ov)

        # ...while the tree still reflects the file
        np.testing.assert_array_equal(a, af["a"])
        np.testing.assert_array_equal(b, af["b"])
