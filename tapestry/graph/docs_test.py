import unittest
import uuid

import hamcrest

import tapestry.graph.docs
from tapestry.graph.docs import JsonSerializable
from tapestry.testlib import eggs


def assert_json_serializable_roundtrip(actual, json_data) -> None:
    """
    Assert that a JsonSerializable class roundtrips to target json data.

    :param actual: the object.
    :param json_data: the expected json data.
    """
    eggs.assert_match(
        actual,
        hamcrest.instance_of(JsonSerializable),
    )

    eggs.assert_match(
        actual.to_json_data(),
        json_data,
    )

    eggs.assert_match(
        type(actual).from_json_data(json_data),
        actual,
    )


class TestEnsureUuid(unittest.TestCase):
    def test_ensure_uuid(self):
        a = uuid.uuid4()

        eggs.assert_match(tapestry.graph.docs.ensure_uuid(a), a)
        eggs.assert_match(
            tapestry.graph.docs.ensure_uuid(),
            hamcrest.all_of(
                hamcrest.instance_of(uuid.UUID),
                hamcrest.is_not(a),
            ),
        )


class TestTapestryNodeDoc(unittest.TestCase):
    def test_lifecycle(self) -> None:
        node = tapestry.graph.docs.TapestryNodeDoc(type="foo")

        eggs.assert_match(
            node.type,
            "foo",
        )
        eggs.assert_match(
            node.id,
            hamcrest.instance_of(uuid.UUID),
        )

    def test_json(self) -> None:
        node = tapestry.graph.docs.TapestryNodeDoc(type="foo")

        assert_json_serializable_roundtrip(
            node,
            {
                "id": str(node.id),
                "type": "foo",
                "fields": {},
            },
        )


class TestTapestryGraphDoc(unittest.TestCase):
    def test_lifecycle(self) -> None:
        graph = tapestry.graph.docs.TapestryGraphDoc()

        node = tapestry.graph.docs.TapestryNodeDoc(type="foo")

        graph.add_node(node)

        eggs.assert_raises(
            lambda: graph.add_node(node),
            AssertionError,
            "already in graph",
        )

        eggs.assert_match(
            graph.nodes,
            hamcrest.has_entry(
                node.id,
                node,
            ),
        )

    def test_json(self) -> None:
        graph = tapestry.graph.docs.TapestryGraphDoc()
        node = tapestry.graph.docs.TapestryNodeDoc(type="foo")
        graph.add_node(node)

        assert_json_serializable_roundtrip(
            graph,
            {
                "id": str(graph.id),
                "nodes": {
                    str(node.id): node.to_json_data(),
                },
            },
        )
