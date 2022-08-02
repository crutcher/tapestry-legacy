import unittest

import hamcrest

from tapestry.serialization.json_testlib import assert_json_serializable_roundtrip
from tapestry.testlib import eggs
from tapestry.zspace import ZRange


class ZRangeTest(unittest.TestCase):
    def test_basic(self) -> None:
        a = ZRange([2, 3])

        assert_json_serializable_roundtrip(
            a,
            {"start": [0, 0], "end": [2, 3]},
        )

        eggs.assert_match(
            hash(ZRange([2, 3])),
            hash(ZRange([2, 3])),
        )
        eggs.assert_match(
            hash(ZRange([2, 3])),
            hamcrest.not_(hash(ZRange([2, 4]))),
        )
