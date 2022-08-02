import hamcrest

import tapestry.serialization
from tapestry.testlib import eggs


def assert_json_serializable_roundtrip(actual, json_data) -> None:
    """
    Assert that a JsonSerializable class roundtrips to target json data.

    :param actual: the object.
    :param json_data: the expected json data.
    """
    eggs.assert_match(
        actual,
        hamcrest.instance_of(tapestry.serialization.json.JsonSerializable),
    )

    eggs.assert_match(
        actual.dump_json_data(),
        json_data,
    )

    eggs.assert_match(
        type(actual).load_json_data(json_data),
        actual,
    )
