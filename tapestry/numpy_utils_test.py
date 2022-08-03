import hamcrest
import numpy as np

from tapestry.numpy_utils import as_zarray, np_hash, np_immutable_guard
from tapestry.testlib import eggs, np_eggs


def test_np_hash():
    eggs.assert_match(
        np_hash(np.array([[2, 3], [4, 5]])),
        np_hash(np.array([[2, 3], [4, 5]])),
    )

    eggs.assert_match(
        np_hash(np.array([[2, 3], [4, 5]])),
        hamcrest.not_(np_hash(np.array([[2, 3], [9, 8]]))),
    )


def test_np_immutable_guard():
    x = np.array([2, 3])
    x[0] = 7

    np_eggs.assert_ndarray_equals(
        x,
        [7, 3],
    )

    np_immutable_guard(x)

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
