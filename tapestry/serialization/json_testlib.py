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

    json_1st_gen = actual.dump_json_data()
    deserialized = type(actual).load_json_data(json_1st_gen)
    json_2nd_gen = deserialized.dump_json_data()

    eggs.assert_match(
        json_2nd_gen,
        json_data,
    )
