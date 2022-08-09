import hamcrest

from tapestry.serialization import json_serializable
from tapestry.testlib import eggs


def assert_json_serializable_roundtrip(actual, json_data) -> None:
    """
    Assert that a JsonSerializable class roundtrips to target json data.

    :param actual: the object.
    :param json_data: the expected json data.
    """
    eggs.assert_match(
        actual,
        hamcrest.instance_of(json_serializable.JsonSerializable),
    )

    json_1st_gen = actual.dump_json_data()
    deserialized = type(actual).load_json_data(json_1st_gen)

    hamcrest.assert_that(deserialized, hamcrest.instance_of(type(actual)))

    eggs.assert_match(
        deserialized.dump_json_data(),
        json_data,
    )
