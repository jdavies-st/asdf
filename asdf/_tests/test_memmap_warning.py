"""
Tests for the warning raised when a file is closed while memory mapped and
array data read from it is still referenced, keeping the file open.
"""

import io
import linecache
import warnings

import numpy as np
import pytest
from numpy import ma
from numpy.lib.stride_tricks import as_strided

import asdf
from asdf.exceptions import AsdfMemmapWarning, AsdfWarning


@pytest.fixture
def arrays_file(tmp_path):
    """A file holding a plain, a structured and a complex array."""
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile(
        {
            "a": np.arange(20, dtype="uint8"),
            "s": np.array([(1, 2.0), (3, 4.0)], dtype=[("i", "i4"), ("f", "f8")]),
            "c": np.arange(8, dtype="complex128"),
        }
    ).write_to(fn)
    return fn


# Every one of these shares memory with the file, so holding on to the result
# keeps the file open after the with block.
_SHARES_MEMORY = [
    pytest.param(lambda arr: arr[:5], id="slice"),
    pytest.param(lambda arr: arr[...], id="ellipsis"),
    pytest.param(lambda arr: arr[None], id="newaxis"),
    pytest.param(lambda arr: arr.view(), id="view"),
    pytest.param(lambda arr: arr.view("int8"), id="view_dtype"),
    pytest.param(lambda arr: arr.T, id="transpose_attribute"),
    pytest.param(lambda arr: arr.reshape(4, 5), id="reshape"),
    pytest.param(lambda arr: arr.ravel(), id="ravel"),
    pytest.param(lambda arr: arr.reshape(4, 5).swapaxes(0, 1), id="chained_swapaxes"),
    pytest.param(lambda arr: arr.reshape(4, 5)[:4, :4].diagonal(), id="chained_diagonal"),
    pytest.param(lambda arr: np.reshape(arr, (4, 5)), id="np_reshape"),
    pytest.param(lambda arr: np.transpose(arr), id="np_transpose"),
    pytest.param(lambda arr: np.expand_dims(arr, 0), id="np_expand_dims"),
    pytest.param(lambda arr: np.broadcast_to(arr, (3, 20)), id="np_broadcast_to"),
    pytest.param(lambda arr: np.split(arr, 4)[0], id="np_split"),
    pytest.param(lambda arr: np.asarray(arr), id="np_asarray"),
    pytest.param(lambda arr: arr.view(np.ndarray), id="view_ndarray"),
    pytest.param(lambda arr: arr.astype("uint8", copy=False), id="astype_without_copy"),
    pytest.param(lambda arr: as_strided(np.asarray(arr), (5,), (1,)), id="as_strided"),
    pytest.param(lambda arr: arr.flat, id="flat"),
    pytest.param(lambda arr: next(iter(arr.reshape(4, 5))), id="iterated_row"),
    pytest.param(lambda arr: ma.masked_array(np.asarray(arr)), id="masked_array"),
    pytest.param(lambda arr: arr.data, id="data_buffer"),
    pytest.param(lambda arr: np.frombuffer(np.asarray(arr), dtype="uint8"), id="frombuffer"),
    pytest.param(lambda arr: arr.ctypes, id="ctypes"),
]

# Each of these allocates its own memory, so the file is released on close.
_OWNS_MEMORY = [
    pytest.param(lambda arr: arr.copy(), id="copy"),
    pytest.param(lambda arr: arr[:5].copy(), id="slice_copy"),
    pytest.param(lambda arr: arr[[0, 1, 2]], id="fancy_index"),
    pytest.param(lambda arr: arr[arr > 5], id="boolean_index"),
    pytest.param(lambda arr: arr + 1, id="arithmetic"),
    pytest.param(lambda arr: arr.astype("int32"), id="astype"),
    pytest.param(lambda arr: arr.sum(), id="sum"),
    pytest.param(lambda arr: arr.tobytes(), id="tobytes"),
    pytest.param(lambda arr: arr.tolist(), id="tolist"),
    pytest.param(lambda arr: np.concatenate([arr, arr]), id="concatenate"),
    pytest.param(lambda arr: bytes(np.asarray(arr)), id="bytes"),
    pytest.param(lambda arr: ma.masked_greater(np.asarray(arr), 5), id="masked_greater"),
    pytest.param(lambda arr: arr[0], id="numeric_scalar"),
]


@pytest.mark.parametrize("op", _SHARES_MEMORY)
def test_warns_when_data_shares_memory_with_the_file(arrays_file, op):
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = op(af["a"])

    assert kept is not None


@pytest.mark.parametrize("op", _OWNS_MEMORY)
def test_no_warning_when_data_owns_its_memory(arrays_file, op):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True) as af:
            kept = op(af["a"])

    assert kept is not None


def test_warns_when_the_array_itself_is_kept(arrays_file):
    """The array returned by ``af[key]`` holds the mapping once it is read."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["a"]
            kept.sum()  # read the data, creating the memory map


def test_warns_when_a_view_is_stored_on_an_object(arrays_file):
    """Where the reference lives makes no difference to the check."""

    class Holder:
        def __init__(self, data):
            self.data = data

    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            holder = Holder(af["a"][:5])

    assert holder.data is not None


def test_warns_when_a_view_of_one_array_is_kept(arrays_file):
    """A file has a single mapping, so one view keeps all of it open."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            af["c"].sum()
            kept = af["a"][:5]

    assert kept is not None


def test_warns_when_a_structured_field_is_kept(arrays_file):
    """A field of a structured array is a view of it."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["s"]["i"]

    assert kept is not None


def test_warns_when_a_structured_row_is_kept(arrays_file):
    """A single record of a table is a view, which is easy to overlook."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["s"][0]

    assert kept is not None


def test_warns_when_the_real_part_is_kept(arrays_file):
    """.real of a complex array is a view of it."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["c"].real

    assert kept is not None


def test_warns_when_the_imaginary_part_is_kept(arrays_file):
    """.imag of a complex array is a view of it."""
    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["c"].imag

    assert kept is not None


def test_no_warning_when_nothing_is_kept(arrays_file):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True) as af:
            assert af["a"].sum() == 190


def test_no_warning_when_eagerly_loaded_and_nothing_is_kept(arrays_file):
    """Eager loading must not leave asdf itself holding the mapping."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True, lazy_load=False) as af:
            assert af["a"].sum() == 190


def test_warns_when_eagerly_loaded_and_a_view_is_kept(arrays_file):
    with (
        pytest.warns(AsdfMemmapWarning, match="still referenced"),
        asdf.open(arrays_file, memmap=True, lazy_load=False) as af,
    ):
        kept = af["a"][:5]

    assert kept is not None


def test_no_warning_when_the_data_is_never_read(arrays_file):
    """No data access means no memory map was ever created."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True) as af:
            assert af["a"].shape == (20,)


def test_no_warning_for_a_recursive_tree(tmp_path):
    """
    A tree that refers to itself is a reference cycle, so the mapping
    outlives the close until cycles are collected rather than because the
    caller kept array data.
    """
    fn = tmp_path / "recursive.asdf"
    tree: dict = {"a": np.arange(20, dtype="uint8")}
    tree["self"] = tree
    asdf.AsdfFile(tree).write_to(fn)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(fn, memmap=True) as af:
            assert af["a"].sum() == 190


def test_no_warning_when_the_view_is_deleted(arrays_file):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["a"][:5]
            assert kept[0] == 0
            del kept


def test_no_warning_without_memmap(arrays_file):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=False) as af:
            kept = af["a"][:5]

    assert kept[0] == 0


def test_no_warning_for_a_file_that_cannot_be_memory_mapped():
    """A BytesIO is read into memory, so there is no mapping to keep open."""
    buff = io.BytesIO()
    asdf.AsdfFile({"a": np.arange(20, dtype="uint8")}).write_to(buff)
    buff.seek(0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(buff, memmap=True) as af:
            kept = af["a"][:5]

    assert kept[0] == 0


def test_explicit_close_warns(arrays_file):
    af = asdf.open(arrays_file, memmap=True)
    kept = af["a"][:5]

    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        af.close()

    assert kept is not None


def test_closing_twice_warns_once(arrays_file):
    af = asdf.open(arrays_file, memmap=True)
    kept = af["a"][:5]

    with pytest.warns(AsdfMemmapWarning, match="still referenced"):
        af.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        af.close()

    assert kept is not None


def test_update_does_not_warn(tmp_path):
    """update() remaps the file, which is not a close."""
    fn = tmp_path / "test.asdf"
    asdf.AsdfFile({"a": np.zeros(10, dtype="uint8")}).write_to(fn)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(fn, mode="rw", memmap=True) as af:
            af["a"][:5] = 1
            af.update()


def test_warning_names_the_file(arrays_file):
    with pytest.warns(AsdfMemmapWarning, match=str(arrays_file)):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["a"][:5]

    assert kept is not None


def test_warning_recommends_copying(arrays_file):
    with pytest.warns(AsdfMemmapWarning, match=r"\.copy\(\)"):
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["a"][:5]

    assert kept is not None


def test_warning_inherits_from_asdf_warning():
    assert issubclass(AsdfMemmapWarning, AsdfWarning)


def test_warning_points_at_the_with_statement(arrays_file):
    """
    The warning's filename and lineno are the Python source location it is
    reported against, which should be the caller's with statement.
    """
    with pytest.warns(AsdfMemmapWarning, match="still referenced") as record:
        with asdf.open(arrays_file, memmap=True) as af:
            kept = af["a"][:5]

    warning = record.pop(AsdfMemmapWarning)
    source_line = linecache.getline(warning.filename, warning.lineno)
    assert source_line.strip() == "with asdf.open(arrays_file, memmap=True) as af:"
    assert kept is not None


def test_warning_points_at_the_close_call(arrays_file):
    af = asdf.open(arrays_file, memmap=True)
    kept = af["a"][:5]

    with pytest.warns(AsdfMemmapWarning, match="still referenced") as record:
        af.close()

    warning = record.pop(AsdfMemmapWarning)
    source_line = linecache.getline(warning.filename, warning.lineno)
    assert source_line.strip() == "af.close()"
    assert kept is not None


def test_bare_address_is_not_detected(arrays_file):
    """
    A raw address is an integer, so nothing connects it to the mapping and
    no warning is possible. This records the limit of the check; it is
    harmless because asdf leaves the mapping in place.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", AsdfMemmapWarning)
        with asdf.open(arrays_file, memmap=True) as af:
            address = af["a"].ctypes.data

    assert address != 0
