import unittest
import uuid

import hamcrest
import numpy as np
import torch

from tapestry.expression_graph import (
    AggregateTensor,
    TapestryEdge,
    TapestryGraph,
    TapestryNode,
    TapestryTag,
    TensorShard,
    TensorValue,
)
from tapestry.testlib import eggs
from tapestry.zspace import ZRange


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

    def test_list_nodes_and_tags_and_edges(self) -> None:
        g = TapestryGraph()

        a_value = g.add_node(
            TensorShard(
                node_id=uuid.uuid4(),
                name="A",
                shape=np.array([2, 3]),
                dtype=torch.int64,
                slice=ZRange([2, 3]),
            ),
        )

        b_value = g.add_node(
            TensorValue(
                node_id=uuid.uuid4(),
                shape=np.array([2, 3]),
                dtype=torch.int64,
                name="B",
            ),
        )

        tag = g.add_node(
            TapestryTag(
                source_id=a_value.node_id,
            ),
        )

        eggs.assert_match(
            g.list_tags(),
            hamcrest.contains_exactly(
                tag,
            ),
        )

        edge = g.add_node(
            TapestryEdge(
                source_id=a_value.node_id,
                target_id=b_value.node_id,
            ),
        )

        eggs.assert_match(
            g.list_edges(),
            hamcrest.contains_exactly(
                edge,
            ),
        )

        # filtering out TapestryEdge:
        eggs.assert_match(
            g.list_nodes(),
            hamcrest.contains_exactly(
                hamcrest.all_of(
                    hamcrest.instance_of(TensorShard),
                    hamcrest.has_property("name", "A"),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(TensorValue),
                    hamcrest.instance_of(TapestryNode),
                    hamcrest.has_property("name", "B"),
                ),
            ),
        )

        # NOT filtering out:
        eggs.assert_match(
            g.list_nodes(exclude=None),
            hamcrest.contains_exactly(
                hamcrest.all_of(
                    hamcrest.instance_of(TensorShard),
                    hamcrest.has_property("name", "A"),
                    hamcrest.has_property("node_id", a_value.node_id),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(TensorValue),
                    hamcrest.instance_of(TapestryNode),
                    hamcrest.has_property("name", "B"),
                    hamcrest.has_property("node_id", b_value.node_id),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(TapestryTag),
                    hamcrest.has_property("node_id", tag.node_id),
                    hamcrest.has_property("source_id", a_value.node_id),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(TapestryEdge),
                    hamcrest.has_property("node_id", edge.node_id),
                    hamcrest.has_property("source_id", a_value.node_id),
                    hamcrest.has_property("target_id", b_value.node_id),
                ),
            ),
        )

        eggs.assert_match(
            g.list_nodes(TensorShard),
            hamcrest.only_contains(
                hamcrest.all_of(
                    hamcrest.instance_of(TensorShard),
                    hamcrest.has_property("name", "A"),
                ),
            ),
        )

    def test_get_node(self) -> None:
        g = TapestryGraph()

        adoc = TensorShard(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype=torch.int64,
            slice=ZRange([2, 3]),
        )
        g.add_node(adoc)

        bdoc = TensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype=torch.int64,
            name="B",
        )
        g.add_node(bdoc)

        # uuid lookup
        eggs.assert_match(
            g.get_node(adoc.node_id),
            hamcrest.all_of(
                hamcrest.instance_of(TensorShard),
                hamcrest.has_property("name", "A"),
            ),
        )

        # string lookup
        eggs.assert_match(
            g.get_node(str(adoc.node_id)),
            hamcrest.all_of(
                hamcrest.instance_of(TensorShard),
                hamcrest.has_property("name", "A"),
            ),
        )

        eggs.assert_match(
            g.get_node(adoc.node_id, TensorShard),
            hamcrest.all_of(
                hamcrest.instance_of(TensorShard),
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
        a = TensorShard(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype=torch.int64,
            slice=ZRange([2, 3]),
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
            dtype=torch.int64,
        )
        g.add_node(b)

        edge_node = TapestryEdge(
            node_id=uuid.uuid4(),
            name="child",
            source_id=a.node_id,
            target_id=b.node_id,
        )
        g.add_node(edge_node)

        s = TapestryGraph.build_load_schema(
            [
                TapestryEdge,
                TensorValue,
                TensorShard,
            ]
        )

        expected_json = {
            "nodes": {
                str(a.node_id): {
                    "__type__": "TensorShard",
                    **a.dump_json_data(),
                },
                str(b.node_id): {
                    "__type__": "TensorValue",
                    **b.dump_json_data(),
                },
                str(edge_node.node_id): {
                    **edge_node.dump_json_data(),
                    "__type__": "TapestryEdge",
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
        a = TensorShard(
            node_id=uuid.uuid4(),
            name="A",
            shape=np.array([2, 3]),
            dtype=torch.int64,
            slice=ZRange([2, 3]),
        )
        g.add_node(a)

        b = TensorValue(
            node_id=uuid.uuid4(),
            name="B",
            shape=np.array([2, 3]),
            dtype=torch.int64,
        )
        g.add_node(b)

        g.assert_node_types(
            [
                TensorShard,
                TensorValue,
            ]
        )

        eggs.assert_raises(
            lambda: g.assert_node_types(
                [
                    AggregateTensor,
                ],
            ),
            ValueError,
            r"\[TensorShard, TensorValue\]",
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
            source_id=foo_node.node_id,
            target_id=foo_node.node_id,
        )
        g.add_node(edge_node)

        missing_source = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_id=uuid.uuid4(),
            target_id=foo_node.node_id,
        )
        missing_target = TapestryEdge(
            node_id=uuid.uuid4(),
            name="illegal",
            source_id=foo_node.node_id,
            target_id=uuid.uuid4(),
        )

        for bad in (missing_source, missing_target):
            eggs.assert_raises(
                lambda: g.add_node(bad),
                ValueError,
            )
