import unittest

from marshmallow_dataclass import dataclass

from tapestry.serialization.json_serializable import JsonLoadable
from tapestry.serialization.json_testlib import assert_json_serializable_roundtrip
from tapestry.testlib import eggs


@dataclass
class Example(JsonLoadable):
    x: int
    y: str


class JsonSerializableTest(unittest.TestCase):
    def test_dump_load(self) -> None:
        actual = Example(x=12, y="abc")
        expected = {
            "x": 12,
            "y": "abc",
        }

        assert_json_serializable_roundtrip(actual, expected)

        js = actual.dump_json_str()

        eggs.assert_match(
            type(actual).load_json_str(js).dump_json_data(),
            expected,
        )
