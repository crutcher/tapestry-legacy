import unittest

import hamcrest
import numpy as np
from marshmallow_dataclass import dataclass

from tapestry.serialization.json_serializable import JsonLoadable
from tapestry.serialization.json_testlib import assert_json_serializable_roundtrip
from tapestry.testlib import eggs, np_eggs
from tapestry.zspace import ZAffineMap, ZArray, ZRange, ZRangeMap


def _assert_hash_eq(a, b):
    eggs.assert_match(
        hash(a),
        hash(a),
    )
    eggs.assert_match(
        hash(a),
        hamcrest.not_(hash(b)),
    )
    eggs.assert_match(
        a,
        a,
    )
    eggs.assert_match(
        a,
        hamcrest.not_(b),
    )


class ZArrayTest(unittest.TestCase):
    def test_zarray(self):
        @dataclass
        class Example(JsonLoadable):
            arr: ZArray

        actual = Example(np.array([3, 2]))

        assert_json_serializable_roundtrip(
            actual,
            {"arr": [3, 2]},
        )


class ZRangeTest(unittest.TestCase):
    def test_construction(self) -> None:
        eggs.assert_match(
            ZRange([2, 3]),
            hamcrest.has_properties(
                start=np_eggs.matches_ndarray([0, 0]),
                end=np_eggs.matches_ndarray([2, 3]),
            ),
        )

        eggs.assert_match(
            ZRange(start=[1, 1], end=[2, 3]),
            hamcrest.has_properties(
                start=np_eggs.matches_ndarray([1, 1]),
                end=np_eggs.matches_ndarray([2, 3]),
            ),
        )

        eggs.assert_raises(
            lambda: ZRange(start=[4, 0], end=[3, 3]),
            ValueError,
            "is not >=",
        )

        eggs.assert_raises(
            lambda: ZRange(start=3, end=[3, 3]),
            ValueError,
            r"of dim \(1\), found \(0\)",
        )

        eggs.assert_raises(
            lambda: ZRange(end=3),
            ValueError,
            r"of dim \(1\), found \(0\)",
        )

    def test_json(self):
        assert_json_serializable_roundtrip(
            ZRange(start=[1, 1], end=[2, 3]),
            {"start": [1, 1], "end": [2, 3]},
        )

    def test_empty(self) -> None:
        r = ZRange(start=[2, 3], end=[2, 3])
        eggs.assert_true(r.empty)
        eggs.assert_false(r.nonempty)

    def test_props(self) -> None:
        eggs.assert_match(
            ZRange(start=[2, 3], end=[3, 5]),
            hamcrest.has_properties(
                ndim=2,
                shape=np_eggs.matches_ndarray([1, 2]),
                size=2,
                empty=False,
                nonempty=True,
            ),
        )

    def test_hash_eq(self) -> None:
        start = np.array([0, 1])
        end = np.array([4, 5])

        # verify that hash and eq work for all embedded properties.
        _assert_hash_eq(
            ZRange(start=start, end=end),
            ZRange(start=start + 1, end=end),
        )
        _assert_hash_eq(
            ZRange(start=start, end=end),
            ZRange(start=start, end=end + 1),
        )

    def test_ordering(self) -> None:
        a = ZRange([2, 3])
        b = ZRange([2, 4])

        eggs.assert_match(
            a,
            hamcrest.equal_to(a),
        )
        eggs.assert_match(
            a,
            hamcrest.not_(hamcrest.equal_to(b)),
        )

        eggs.assert_true(a.__lt__(b))

        eggs.assert_match(
            a,
            hamcrest.less_than(b),
        )
        eggs.assert_match(
            a,
            hamcrest.less_than_or_equal_to(b),
        )
        eggs.assert_match(
            b,
            hamcrest.greater_than(a),
        )
        eggs.assert_match(
            b,
            hamcrest.greater_than_or_equal_to(a),
        )

    def test_inclusive_corners_empty(self) -> None:
        r = ZRange(start=[2, 3], end=[2, 3])
        eggs.assert_match(
            r.inclusive_corners,
            hamcrest.empty(),
        )

    def test_inclusive_corners(self) -> None:
        eggs.assert_match(
            ZRange([1, 2, 3]).inclusive_corners,
            hamcrest.contains_exactly(
                np_eggs.matches_ndarray([0, 0, 0]),
                np_eggs.matches_ndarray([0, 0, 2]),
                np_eggs.matches_ndarray([0, 1, 0]),
                np_eggs.matches_ndarray([0, 1, 2]),
            ),
        )


class ZAffineMapTest(unittest.TestCase):
    def test_construction(self):
        eggs.assert_match(
            ZAffineMap(
                projection=[[2, 0], [0, 1]],
                offset=[-5, 3],
            ),
            hamcrest.has_properties(
                projection=np_eggs.matches_ndarray([[2, 0], [0, 1]]),
                offset=np_eggs.matches_ndarray([-5, 3]),
            ),
        )

        eggs.assert_raises(
            lambda: ZAffineMap(
                projection=[[2, 0], [0, 1]],
                offset=[-5, 3, 7],
            ),
            ValueError,
            r"output shape \(2\) != offset shape \(3\)",
        )

        eggs.assert_raises(
            lambda: ZAffineMap(
                projection=3,
                offset=[-5, 3],
            ),
            ValueError,
            r"of dim \(2\), found \(0\)",
        )

        eggs.assert_raises(
            lambda: ZAffineMap(
                projection=[[2, 0], [0, 1]],
                offset=3,
            ),
            ValueError,
            r"of dim \(1\), found \(0\)",
        )

    def test_json(self):
        assert_json_serializable_roundtrip(
            ZAffineMap(
                projection=[[2, 0], [0, 1]],
                offset=[-5, 3],
            ),
            {"projection": [[2, 0], [0, 1]], "offset": [-5, 3]},
        )

    def test_hash_eq(self) -> None:
        projection = np.array([[2, 0], [0, 1]])
        offset = np.array([-5, 3])

        # verify that hash and eq work for all embedded properties.
        _assert_hash_eq(
            ZAffineMap(projection=projection, offset=offset),
            ZAffineMap(projection=projection + 1, offset=offset),
        )
        _assert_hash_eq(
            ZAffineMap(projection=projection, offset=offset),
            ZAffineMap(projection=projection, offset=offset + 1),
        )

    def test_props(self):
        eggs.assert_match(
            ZAffineMap(
                projection=[[2, 0, 1], [0, 1, 1]],
                offset=[-5, 3, 0],
            ),
            hamcrest.has_properties(
                in_dim=2,
                out_dim=3,
                constant=False,
            ),
        )

    def test_constant(self):
        eggs.assert_match(
            ZAffineMap(
                projection=[[0, 0], [0, 0]],
                offset=[-5, 0],
            ),
            hamcrest.has_properties(
                constant=True,
            ),
        )

    def test_marginal_strides(self):
        eggs.assert_match(
            ZAffineMap(
                projection=[[2, 0, 1], [0, 1, 1]],
                offset=[-5, 3, 1],
            ).marginal_strides(),
            np_eggs.matches_ndarray(
                [
                    [2, 0, 1],
                    [0, 1, 1],
                ],
            ),
        )


class ZRangeMapTest(unittest.TestCase):
    def test_construction(self):
        eggs.assert_match(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=[[2, 0], [0, 1]],
                    offset=[-5, 3],
                ),
                shape=[2, 2],
            ),
            hamcrest.has_properties(
                zaffine_map=ZAffineMap(
                    projection=[[2, 0], [0, 1]],
                    offset=[-5, 3],
                ),
                shape=np_eggs.matches_ndarray([2, 2]),
            ),
        )

        eggs.assert_raises(
            lambda: ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=[[2, 0], [0, 1]],
                    offset=[-5, 3],
                ),
                shape=[2, 2, 4],
            ),
            ValueError,
            r"out ndim \(2\) != shape ndim \(3\)",
        )

    def test_json(self):
        assert_json_serializable_roundtrip(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=[[2, 0], [0, 1]],
                    offset=[-5, 3],
                ),
                shape=[2, 2],
            ),
            {
                "zaffine_map": {"projection": [[2, 0], [0, 1]], "offset": [-5, 3]},
                "shape": [2, 2],
            },
        )

    def test_props(self):
        eggs.assert_match(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=[[2, 0, 1], [0, 1, 1]],
                    offset=[-5, 3, 0],
                ),
                shape=[2, 2, 1],
            ),
            hamcrest.has_properties(
                in_dim=2,
                out_dim=3,
                constant=False,
            ),
        )

    def test_hash_eq(self) -> None:
        projection = np.array([[2, 0], [0, 1]])
        offset = np.array([-5, 3])
        shape = np.array([2, 2])

        # verify that hash and eq work for all embedded properties.
        _assert_hash_eq(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection,
                    offset=offset,
                ),
                shape=shape,
            ),
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection + 1,
                    offset=offset,
                ),
                shape=shape,
            ),
        )
        _assert_hash_eq(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection,
                    offset=offset,
                ),
                shape=shape,
            ),
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection,
                    offset=offset + 1,
                ),
                shape=shape,
            ),
        )
        _assert_hash_eq(
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection,
                    offset=offset,
                ),
                shape=shape,
            ),
            ZRangeMap(
                zaffine_map=ZAffineMap(
                    projection=projection,
                    offset=offset,
                ),
                shape=shape + 1,
            ),
        )

    def test_point_to_range(self):
        rm = ZRangeMap(
            zaffine_map=ZAffineMap(
                projection=[[2, 0, 1], [0, 1, 1]],
                offset=[-1, -1, 0],
            ),
            shape=[2, 2, 1],
        )
        eggs.assert_match(
            rm.point_to_range([3, 2]),
            ZRange(
                start=[5, 1, 5],
                end=[7, 3, 6],
            ),
        )

    def test_call(self):
        rm = ZRangeMap(
            zaffine_map=ZAffineMap(
                projection=[[-2, 0, 1], [0, 1, 1]],
                offset=[-1, -1, 0],
            ),
            shape=[1, 1, 1],
        )
        eggs.assert_match(
            rm(
                ZRange(start=[1, 1], end=[3, 2]),
            ),
            ZRange(
                start=[-5, 0, 3],
                end=[-2, 1, 3],
            ),
        )

    def test_marginal(self):
        rm = ZRangeMap(
            zaffine_map=ZAffineMap(
                projection=[[1, 0, 1], [0, 2, 1]],
                offset=[0, 0, 0],
            ),
            shape=[1, 1, 4],
        )

        np_eggs.assert_ndarray_equals(
            rm.marginal_overlap(),
            [
                [0, 1, 3],
                [1, 0, 3],
            ],
        )

        np_eggs.assert_ndarray_equals(
            rm.marginal_waste(),
            [
                [0, 0, 0],
                [0, 1, 0],
            ],
        )
