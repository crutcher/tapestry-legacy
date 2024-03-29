import hamcrest
import numpy as np

from tapestry.numpy_utils import (
    as_zarray,
    make_ndarray_immutable,
    ndarray_aggregate_equality,
    ndarray_hash,
    ndarray_le,
    ndarray_lt,
)
from tapestry.testlib import eggs, np_eggs


def test_ndarray_aggregate_equality():
    eggs.assert_true(ndarray_aggregate_equality([1, 2], np.array([1, 2])))
    eggs.assert_true(ndarray_aggregate_equality(np.array([1, 2]), [1, 2]))

    eggs.assert_false(ndarray_aggregate_equality([1, 3], np.array([1, 2])))
    eggs.assert_false(ndarray_aggregate_equality(np.array([1, 3]), [1, 2]))

    eggs.assert_false(ndarray_aggregate_equality([[2, 3], [2, 3]], np.array([1, 2])))


def test_ndarray_hash():
    eggs.assert_match(
        ndarray_hash(np.array([[2, 3], [4, 5]])),
        ndarray_hash(np.array([[2, 3], [4, 5]])),
    )

    eggs.assert_match(
        ndarray_hash(np.array([[2, 3], [4, 5]])),
        hamcrest.not_(ndarray_hash(np.array([[2, 3], [9, 8]]))),
    )


def test_make_ndarray_immutable():
    x = np.array([2, 3])
    x[0] = 7

    np_eggs.assert_ndarray_equals(
        x,
        [7, 3],
    )

    make_ndarray_immutable(x)

    eggs.assert_raises(
        lambda: x.__setitem__(0, 9),
        ValueError,
        "read-only",
    )

    np_eggs.assert_ndarray_equals(
        x,
        [7, 3],
    )


def test_as_zarray():
    np_eggs.assert_ndarray_equals(
        as_zarray([[2, 3], [4, 5]]),
        [[2, 3], [4, 5]],
    )
    eggs.assert_true(as_zarray([[2, 3]]).flags.writeable)
    eggs.assert_false(as_zarray([[2, 3]], immutable=True).flags.writeable)

    np_eggs.assert_ndarray_equals(
        as_zarray([[2, 3], [4, 5]], ndim=2),
        [[2, 3], [4, 5]],
    )

    eggs.assert_raises(
        lambda: as_zarray([[2, 3], [4, 5]], ndim=1),
        ValueError,
        r"of dim \(1\), found \(2\)",
    )

    eggs.assert_raises(
        lambda: as_zarray([3.5]),
        ValueError,
        "rounding error",
    )


def test_ndarray_lt():
    eggs.assert_true(
        ndarray_lt(
            np.array([1, 2, 1]),
            np.array([1, 3, 4]),
        )
    )

    eggs.assert_true(
        ndarray_lt(
            np.array([1, 2, 1]),
            np.array([1, 2, 1, 0]),
        )
    )

    eggs.assert_false(
        ndarray_lt(
            np.array([8, 0, 9, 7]),
            np.array([5, 1, 5, 7]),
        )
    )


def test_ndarray_le():
    eggs.assert_true(
        ndarray_le(
            np.array([1, 2, 1]),
            np.array([1, 3, 4]),
        )
    )

    eggs.assert_true(
        ndarray_le(
            np.array([1, 2, 1]),
            np.array([1, 2, 1]),
        )
    )

    eggs.assert_true(
        ndarray_le(
            np.array([1, 2, 1]),
            np.array([1, 2, 1, 0]),
        )
    )

    eggs.assert_true(
        ndarray_le(
            np.array([6, 8, 6, 0]),
            np.array([8, 5, 4, 4]),
        )
    )
