import unittest
import uuid

import hamcrest

import tapestry.graph.docs as docs
from tapestry.serialization.json_testlib import assert_json_serializable_roundtrip
from tapestry.testlib import eggs


class TestEnsureUuid(unittest.TestCase):
    def test_ensure_uuid(self):
        a = uuid.uuid4()

        eggs.assert_match(docs.ensure_uuid(a), a)
        eggs.assert_match(
            docs.ensure_uuid(),
            hamcrest.all_of(
                hamcrest.instance_of(uuid.UUID),
                hamcrest.is_not(a),
            ),
        )


class TestTapestryNodeDoc(unittest.TestCase):
    def test_lifecycle(self) -> None:
        node = docs.TapestryNodeDoc(type="foo")

        eggs.assert_match(
            node.type,
            "foo",
        )
        eggs.assert_match(
            node.id,
            hamcrest.instance_of(uuid.UUID),
        )

    def test_json(self) -> None:
        node = docs.TapestryNodeDoc(type="foo")

        assert_json_serializable_roundtrip(
            node,
            {
                "id": str(node.id),
                "type": "foo",
                "fields": {},
            },
        )


class TestTapestryEdgeDoc(unittest.TestCase):
    def test_json(self) -> None:
        a = uuid.uuid4()
        b = uuid.uuid4()
        edge = docs.TapestryEdgeDoc(
            type="foo",
            source=a,
            target=b,
        )

        assert_json_serializable_roundtrip(
            edge,
            {
                "type": "foo",
                "source": str(a),
                "target": str(b),
            },
        )


class TestTapestryGraphDoc(unittest.TestCase):
    def test_nodes(self) -> None:
        graph = docs.TapestryGraphDoc()

        node = docs.TapestryNodeDoc(type="foo")

        graph.add_node(node)

        eggs.assert_raises(
            lambda: graph.add_node(node),
            ValueError,
            "already in graph",
        )

        eggs.assert_match(
            graph.nodes,
            hamcrest.contains_exactly(
                node,
            ),
        )

    def test_edges(self) -> None:
        graph = docs.TapestryGraphDoc()
        nodeA = docs.TapestryNodeDoc(type="foo")
        nodeB = docs.TapestryNodeDoc(type="bar")

        graph.add_node(nodeA)
        graph.add_node(nodeB)

        edge = docs.TapestryEdgeDoc(
            type="link",
            source=nodeA.id,
            target=nodeB.id,
        )
        graph.add_edge(
            type="link",
            source=nodeA.id,
            target=nodeB.id,
        )

        eggs.assert_raises(
            lambda: graph.add_edge(edge),
            ValueError,
            "already in graph",
        )

        eggs.assert_match(
            graph.find_edges(),
            hamcrest.contains_inanyorder(
                edge,
            ),
        )
        eggs.assert_match(
            graph.find_edges(types="unknown"),
            hamcrest.empty(),
        )
        eggs.assert_match(
            graph.find_edges(types="link", sources=uuid.uuid4()),
            hamcrest.empty(),
        )
        eggs.assert_match(
            graph.find_edges(types="link", targets=uuid.uuid4()),
            hamcrest.empty(),
        )
        eggs.assert_match(
            graph.find_edges(types="link"),
            hamcrest.contains_inanyorder(
                edge,
            ),
        )
        eggs.assert_match(
            graph.find_edges(sources=nodeA),
            hamcrest.contains_inanyorder(
                edge,
            ),
        )
        eggs.assert_match(
            graph.find_edges(targets=nodeA),
            hamcrest.empty(),
        )
        eggs.assert_match(
            graph.find_edges(targets=nodeB),
            hamcrest.contains_inanyorder(
                edge,
            ),
        )
        eggs.assert_match(
            graph.find_edges(sources=nodeB),
            hamcrest.empty(),
        )

        eggs.assert_match(
            graph.edges,
            hamcrest.contains_exactly(
                edge,
            ),
        )

        graph.remove_edge(edge)
        eggs.assert_raises(
            lambda: graph.remove_edge(edge),
            ValueError,
            "not in graph",
        )

        eggs.assert_match(
            graph.edges,
            hamcrest.empty(),
        )

    def test_json(self) -> None:
        graph = docs.TapestryGraphDoc()
        node = docs.TapestryNodeDoc(type="foo")
        graph.add_node(node)

        edge = graph.add_edge(
            type="foo",
            source=node.id,
            target=node.id,
        )

        assert_json_serializable_roundtrip(
            graph,
            {
                "id": str(graph.id),
                "nodes": [
                    node.dump_json_data(),
                ],
                "edges": [
                    edge.dump_json_data(),
                ],
            },
        )
