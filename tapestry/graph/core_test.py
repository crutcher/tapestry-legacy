import unittest
import uuid

import hamcrest

from tapestry.graph import core
from tapestry.testlib import eggs


class TestEnsureUuid(unittest.TestCase):
    def test_ensure_uuid(self):
        a = uuid.uuid4()

        eggs.assert_match(core.ensure_uuid(a), a)
        eggs.assert_match(
            core.ensure_uuid(),
            hamcrest.all_of(
                hamcrest.instance_of(uuid.UUID),
                hamcrest.is_not(a),
            ),
        )


class TestTapestryType(unittest.TestCase):
    def test_basic(self) -> None:
        foo_type = core.TapestryType(type_name="foo")
        eggs.assert_match(
            foo_type.name(),
            "foo",
        )

    def test_bad_name(self) -> None:
        for name in [
            "",
            "foo..bar",
            "_foo",
            "9foo",
        ]:
            eggs.assert_raises(
                lambda: core.TapestryType(type_name=name),
                AssertionError,
                "Illegal type name",
            )


class TestTapestryNode(unittest.TestCase):
    def test_lifecycle(self) -> None:
        foo_type = core.TapestryType(type_name="foo")

        node = core.TapestryNode(node_type=foo_type)

        eggs.assert_match(
            node.node_type(),
            foo_type,
        )
        eggs.assert_match(
            node.node_id(),
            hamcrest.instance_of(uuid.UUID),
        )


class TestTapestryGraph(unittest.TestCase):
    def test_lifecycle(self) -> None:
        graph = core.TapestryGraph()

        foo_type = core.TapestryType(type_name="foo")
        node = core.TapestryNode(node_type=foo_type)

        graph.add_node(node)

        eggs.assert_raises(
            lambda: graph.add_node(node),
            AssertionError,
            "already in graph",
        )

        eggs.assert_match(
            graph.nodes_view(),
            hamcrest.contains_exactly(node),
        )
