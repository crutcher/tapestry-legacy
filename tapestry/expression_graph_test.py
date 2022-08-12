import unittest
import uuid

import hamcrest
import numpy as np

from tapestry.expression_graph import (
    ExternalTensor,
    TapestryEdge,
    TapestryGraph,
    TapestryNode,
    TensorResult,
    TensorValue,
)
from tapestry.testlib import eggs


class DisjointAttributes(TapestryNode):
    """
    Test class which no other NodeAttributes is a subclass of.
    """


class TapestryNodeTest(unittest.TestCase):
    def test_clone(self) -> None:
        g = TapestryGraph()

        node = TapestryNode()

        eggs.assert_match(
            node.graph,
            hamcrest.none(),
        )

        node.graph = g

        eggs.assert_match(
            node.graph,
            g,
        )

        clone = node.clone()
        eggs.assert_match(
            clone.graph,
            hamcrest.none(),
        )


class TapestryGraphTest(unittest.TestCase):
    def test_clone(self) -> None:
        g = TapestryGraph()
        node = g.add_node(TapestryNode())

        clone = g.clone()

        eggs.assert_match(g, hamcrest.not_(hamcrest.same_instance(clone)))

        eggs.assert_match(
            clone.get_node(node.node_id),
            hamcrest.all_of(
                hamcrest.not_(hamcrest.same_instance(node)),
                hamcrest.has_property("node_id", node.node_id),
            ),
        )

    def test_list_nodes_of_type(self) -> None:
        g = TapestryGraph()

        adoc = ExternalTensor(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        g.add_node(adoc)

        bdoc = TensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype="torch.int64",
            name="B",
        )
        g.add_node(bdoc)

        eggs.assert_match(
            g.list_nodes(),
            hamcrest.contains_inanyorder(
                hamcrest.all_of(
                    hamcrest.instance_of(ExternalTensor),
                    hamcrest.has_property("name", "A"),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(TensorValue),
                    hamcrest.instance_of(TapestryNode),
                    hamcrest.has_property("name", "B"),
                ),
            ),
        )

        eggs.assert_match(
            g.list_nodes(ExternalTensor),
            hamcrest.only_contains(
                hamcrest.all_of(
                    hamcrest.instance_of(ExternalTensor),
                    hamcrest.has_property("name", "A"),
                ),
            ),
        )

    def test_get_node(self) -> None:
        g = TapestryGraph()

        adoc = ExternalTensor(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        g.add_node(adoc)

        bdoc = TensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype="torch.int64",
            name="B",
        )
        g.add_node(bdoc)

        # uuid lookup
        eggs.assert_match(
            g.get_node(adoc.node_id),
            hamcrest.all_of(
                hamcrest.instance_of(ExternalTensor),
                hamcrest.has_property("name", "A"),
            ),
        )

        # string lookup
        eggs.assert_match(
            g.get_node(str(adoc.node_id)),
            hamcrest.all_of(
                hamcrest.instance_of(ExternalTensor),
                hamcrest.has_property("name", "A"),
            ),
        )

        eggs.assert_match(
            g.get_node(adoc.node_id, ExternalTensor),
            hamcrest.all_of(
                hamcrest.instance_of(ExternalTensor),
                hamcrest.has_property("name", "A"),
            ),
        )

        eggs.assert_raises(
            lambda: g.get_node(adoc.node_id, DisjointAttributes),
            ValueError,
        )
        eggs.assert_raises(
            lambda: g.get_node(uuid.uuid4()),
            KeyError,
        )

    def test_schema(self) -> None:
        g = TapestryGraph()
        a = ExternalTensor(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        g.add_node(a)
        eggs.assert_raises(
            lambda: g.add_node(a),
            ValueError,
            "already in graph",
        )

        b = TensorValue(
            node_id=uuid.uuid4(),
            name="B",
            shape=np.array([2, 3]),
            dtype="torch.int64",
        )
        g.add_node(b)

        edge_node = TapestryEdge(
            node_id=uuid.uuid4(),
            name="child",
            source_node_id=a.node_id,
            target_node_id=b.node_id,
        )
        g.add_node(edge_node)

        s = TapestryGraph.build_load_schema(
            [
                TapestryEdge,
                TensorValue,
                ExternalTensor,
            ]
        )

        expected_json = {
            "nodes": {
                str(a.node_id): {
                    "__type__": "ExternalTensor",
                    **a.dump_json_data(),
                    "__edges__": [
                        {
                            **edge_node.dump_json_data(),
                            "__type__": "TapestryEdge",
                        }
                    ],
                },
                str(b.node_id): {
                    "__type__": "TensorValue",
                    **b.dump_json_data(),
                },
            },
        }

        eggs.assert_match(
            s.dump(g),
            expected_json,
        )
        eggs.assert_match(
            g.dump_json_data(),
            expected_json,
        )

        eggs.assert_match(
            s.dump(s.load(s.dump(g))),
            expected_json,
        )

    def test_assert_node_types(self) -> None:
        g = TapestryGraph()
        a = ExternalTensor(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        g.add_node(a)

        b = TensorValue(
            node_id=uuid.uuid4(),
            name="B",
            shape=np.array([2, 3]),
            dtype="torch.int64",
        )
        g.add_node(b)

        g.assert_node_types(
            [
                ExternalTensor,
                TensorValue,
            ]
        )

        eggs.assert_raises(
            lambda: g.assert_node_types(
                [
                    TensorResult,
                ],
            ),
            ValueError,
            r"\[ExternalTensor, TensorValue\]",
        )

    def test_edges(self) -> None:
        g = TapestryGraph()

        foo_node = TapestryNode(
            node_id=uuid.uuid4(),
            name="foo",
        )
        g.add_node(foo_node)

        edge_node = TapestryEdge(
            node_id=uuid.uuid4(),
            name="bar",
            source_node_id=foo_node.node_id,
            target_node_id=foo_node.node_id,
        )
        g.add_node(edge_node)

        illegal_source = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_node_id=edge_node.node_id,
            target_node_id=foo_node.node_id,
        )
        missing_source = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_node_id=uuid.uuid4(),
            target_node_id=foo_node.node_id,
        )
        illegal_target = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_node_id=foo_node.node_id,
            target_node_id=edge_node.node_id,
        )
        missing_target = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_node_id=foo_node.node_id,
            target_node_id=uuid.uuid4(),
        )

        for bad in (illegal_source, missing_source, illegal_target, missing_target):
            eggs.assert_raises(
                lambda: g.add_node(bad),
                ValueError,
            )
