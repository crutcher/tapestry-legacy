import unittest

import numpy as np
import torch

from tapestry import space
from tapestry.space_testlib import match_zpoint
from tapestry.testlib import eggs, np_eggs, torch_eggs


class ZPointTest(unittest.TestCase):
    def test_constructor(self):
        eggs.assert_match(
            space.ZPoint((2, 3)).as_tuple(),
            (2, 3),
        )

        eggs.assert_match(
            space.ZPoint([2, 3]).as_tuple(),
            (2, 3),
        )

        eggs.assert_match(
            space.ZPoint(torch.tensor([2, 3])).as_tuple(),
            (2, 3),
        )

        eggs.assert_match(
            space.ZPoint(np.array([2, 3])).as_tuple(),
            (2, 3),
        )

    def test_sequence(self):
        zp = space.ZPoint((2, 3))

        # iter
        eggs.assert_match(
            list(zp),
            [2, 3],
        )

        # __len__
        eggs.assert_match(len(zp), 2)

        # __getitem__
        eggs.assert_match(zp[1], 3)

    def test_eq(self):
        zp = space.ZPoint((2, 3))
        eggs.assert_true(zp == (2, 3))
        eggs.assert_true(zp != (1, 2, 3))
        eggs.assert_true(zp != (3,))

    def test_hash(self):
        zp = space.ZPoint((2, 3))
        eggs.assert_match(hash(zp), hash((2, 3)))

    def test_tensor(self):
        zp = space.ZPoint((2, 3))
        torch_eggs.assert_tensor_equals(
            zp.as_tensor(),
            (2, 3),
        )

    def test_numpy(self):
        zp = space.ZPoint((2, 3))
        np_eggs.assert_ndarray_equals(
            zp.as_numpy(),
            (2, 3),
        )

    def test_immutable(self):
        zp = space.ZPoint((2, 3))

        def op():
            del zp[0]

        eggs.assert_raises(
            op,
            TypeError,
            "ZPoint object does not support item deletion",
        )

        def op():
            zp[0] = 8

        eggs.assert_raises(
            op,
            TypeError,
            "ZPoint object does not support item assignment",
        )


class ZRangeTest(unittest.TestCase):
    def test_lifecycle(self):

        ib = space.ZRange(
            start=(2, 3),
            end=(2, 4),
        )

        eggs.assert_match(
            ib.start(),
            match_zpoint((2, 3)),
        )

        eggs.assert_match(
            ib.end(),
            match_zpoint((2, 4)),
        )

        torch_eggs.assert_tensor_equals(
            ib.shape(),
            (0, 1),
        )

        eggs.assert_match(ib.card(), 0)
