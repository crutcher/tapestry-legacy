import unittest
import uuid

import hamcrest

from tapestry import type_utils
from tapestry.testlib import eggs


class DictToParameterStrTest(unittest.TestCase):
    def test(self) -> None:
        a = uuid.uuid4()
        b = [1, 2]
        eggs.assert_match(
            type_utils.dict_to_parameter_str(
                dict(
                    a=a,
                    b=b,
                ),
            ),
            f"a={repr(a)}, b={repr(b)}",
        )


class EnsureUuidTest(unittest.TestCase):
    def test_coerce_uuid(self) -> None:
        a = uuid.uuid4()

        eggs.assert_match(
            type_utils.coerce_uuid(a),
            a,
        )

        eggs.assert_match(
            type_utils.coerce_uuid(str(a)),
            a,
        )

    def test_ensure_uuid(self) -> None:
        a = uuid.uuid4()

        eggs.assert_match(
            type_utils.ensure_uuid(a),
            a,
        )

        eggs.assert_match(
            type_utils.ensure_uuid(str(a)),
            a,
        )

        eggs.assert_match(
            type_utils.ensure_uuid(),
            hamcrest.all_of(
                hamcrest.instance_of(uuid.UUID),
                hamcrest.not_(a),
            ),
        )
