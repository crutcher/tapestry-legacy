import unittest
import uuid

import hamcrest
import numpy as np
from overrides import overrides

from tapestry.expression_graph import (
    EdgeAttributes,
    ExternalTensorValue,
    GraphDoc,
    GraphHandle,
    NodeAttributes,
    NodeHandle,
    TensorSource,
    TensorValue,
)
from tapestry.testlib import eggs


class DisjointAttributes(NodeAttributes):
    """
    Test class which no other NodeAttributes is a subclass of.
    """


class DisjointNodeHandle(NodeHandle):
    """
    Test class which no other NodeHandle is a subclass of.
    """

    attrs: DisjointAttributes


class CommonNodeWrapperTestBase(unittest.TestCase):
    HANDLE_CLASS = NodeHandle
    """
    Wrapper class being tested.
    """

    def example_doc(self) -> NodeAttributes:
        """
        Subclasses can override.
        """
        return NodeAttributes(
            node_id=uuid.uuid4(),
            display_name="foo",
        )

    def test_common_construction(self) -> None:
        g = GraphHandle()

        attrs = self.example_doc()
        attrs_class = type(attrs)

        node = self.HANDLE_CLASS(
            graph=g,
            attrs=attrs,
        )

        eggs.assert_true(node.wraps_doc_type(attrs_class))
        node.assert_wraps_doc_type(attrs_class)

        eggs.assert_match(
            node.try_as_type(NodeHandle),
            hamcrest.instance_of(NodeHandle),
        )
        eggs.assert_match(
            node.force_as_type(NodeHandle),
            hamcrest.instance_of(NodeHandle),
        )

        eggs.assert_true(node.wraps_doc_type(attrs_class))
        node.assert_wraps_doc_type(attrs_class)

        eggs.assert_match(
            node.try_as_type(self.HANDLE_CLASS),
            hamcrest.instance_of(self.HANDLE_CLASS),
        )
        eggs.assert_match(
            node.force_as_type(self.HANDLE_CLASS),
            hamcrest.instance_of(self.HANDLE_CLASS),
        )

        if attrs_class != NodeAttributes:
            eggs.assert_false(node.wraps_doc_type(DisjointAttributes))
            eggs.assert_raises(
                lambda: node.assert_wraps_doc_type(DisjointAttributes),
                ValueError,
            )

        eggs.assert_match(
            node.try_as_type(DisjointNodeHandle),
            hamcrest.none(),
        )
        eggs.assert_raises(
            lambda: node.force_as_type(DisjointNodeHandle),
            ValueError,
        )


class NodeWrapperTest(CommonNodeWrapperTestBase):
    def test_eq_hash(self) -> None:
        g = GraphHandle()

        foo = NodeAttributes(
            node_id=uuid.uuid4(),
            display_name="foo",
        )
        bar = NodeAttributes(
            node_id=uuid.uuid4(),
            display_name="bar",
        )

        foo_node_1 = NodeHandle(
            graph=g,
            attrs=foo,
        )
        foo_node_2 = NodeHandle(
            graph=g,
            attrs=foo,
        )
        bar_node = NodeHandle(
            graph=g,
            attrs=bar,
        )

        eggs.assert_match(
            foo_node_1,
            foo_node_2,
        )

        eggs.assert_match(
            hash(foo_node_1),
            hash(foo_node_2),
        )

        eggs.assert_match(
            foo_node_1,
            hamcrest.not_(bar_node),
        )

        eggs.assert_match(
            hash(foo_node_1),
            hamcrest.not_(hash(bar_node)),
        )


class TensorSourceTest(CommonNodeWrapperTestBase):
    HANDLE_CLASS = TensorSource.Handle

    @overrides
    def example_doc(self) -> NodeAttributes:
        return TensorSource(
            node_id=uuid.uuid4(),
            shape=[2, 3],  # type: ignore
            dtype="torch.int64",
        )


class TensorValueTest(CommonNodeWrapperTestBase):
    HANDLE_CLASS = TensorValue.Handle

    @overrides
    def example_doc(self) -> NodeAttributes:
        return TensorValue(
            node_id=uuid.uuid4(),
            shape=[2, 3],  # type: ignore
            dtype="torch.int64",
        )


class ExternalTensorValueTest(CommonNodeWrapperTestBase):
    HANDLE_CLASS = ExternalTensorValue.Handle

    @overrides
    def example_doc(self) -> NodeAttributes:
        return ExternalTensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="abc",
        )


class ExpressionGraphTest(unittest.TestCase):
    def test_list_nodes_of_type(self) -> None:
        gdoc = GraphDoc()

        adoc = ExternalTensorValue(
            node_id=uuid.uuid4(),
            display_name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        gdoc.add_node(adoc)

        bdoc = TensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype="torch.int64",
            display_name="B",
        )
        gdoc.add_node(bdoc)

        g = GraphHandle(gdoc)

        eggs.assert_match(
            g.all_handles_of_type(NodeHandle),
            hamcrest.contains_inanyorder(
                hamcrest.all_of(
                    hamcrest.instance_of(NodeHandle),
                    hamcrest.has_property("attrs", adoc),
                ),
                hamcrest.all_of(
                    hamcrest.instance_of(NodeHandle),
                    hamcrest.has_property("attrs", bdoc),
                ),
            ),
        )

        eggs.assert_match(
            g.all_handles_of_type(ExternalTensorValue.Handle),
            hamcrest.contains_inanyorder(
                hamcrest.all_of(
                    hamcrest.instance_of(ExternalTensorValue.Handle),
                    hamcrest.has_property("attrs", adoc),
                ),
            ),
        )

    def test_get_node(self) -> None:
        gdoc = GraphDoc()

        adoc = ExternalTensorValue(
            node_id=uuid.uuid4(),
            display_name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        gdoc.add_node(adoc)

        bdoc = TensorValue(
            node_id=uuid.uuid4(),
            shape=np.array([2, 3]),
            dtype="torch.int64",
            display_name="B",
        )
        gdoc.add_node(bdoc)

        g = GraphHandle(gdoc)

        # uuid lookup
        eggs.assert_match(
            g.get_node(adoc.node_id, NodeHandle),
            hamcrest.all_of(
                hamcrest.instance_of(NodeHandle),
                hamcrest.has_property("attrs", adoc),
            ),
        )

        # string lookup
        eggs.assert_match(
            g.get_node(str(adoc.node_id), NodeHandle),
            hamcrest.all_of(
                hamcrest.instance_of(NodeHandle),
                hamcrest.has_property("attrs", adoc),
            ),
        )

        eggs.assert_match(
            g.get_node(adoc.node_id, ExternalTensorValue.Handle),
            hamcrest.all_of(
                hamcrest.instance_of(ExternalTensorValue.Handle),
                hamcrest.has_property("attrs", adoc),
            ),
        )

        eggs.assert_raises(
            lambda: g.get_node(adoc.node_id, DisjointNodeHandle),
            ValueError,
        )
        eggs.assert_raises(
            lambda: g.get_node(uuid.uuid4(), NodeHandle),
            KeyError,
        )


class GraphDocTest(unittest.TestCase):
    def test_schema(self) -> None:
        g = GraphDoc()
        a = ExternalTensorValue(
            node_id=uuid.uuid4(),
            display_name="A",
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
            display_name="B",
            shape=np.array([2, 3]),
            dtype="torch.int64",
        )
        g.add_node(b)

        edge_node = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="child",
            source_node_id=a.node_id,
            target_node_id=b.node_id,
        )
        g.add_node(edge_node)

        s = GraphDoc.build_load_schema(
            [
                EdgeAttributes,
                TensorSource,
                TensorValue,
                ExternalTensorValue,
            ]
        )

        expected_json = {
            "nodes": {
                str(a.node_id): {
                    "__type__": "ExternalTensorValue",
                    **a.dump_json_data(),
                    "__edges__": [
                        {
                            **edge_node.dump_json_data(),
                            "__type__": "EdgeAttributes",
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
        g = GraphDoc()
        a = ExternalTensorValue(
            node_id=uuid.uuid4(),
            display_name="A",
            shape=np.array([2, 3]),
            dtype="torch.int64",
            storage="pre:A",
        )
        g.add_node(a)

        b = TensorValue(
            node_id=uuid.uuid4(),
            display_name="B",
            shape=np.array([2, 3]),
            dtype="torch.int64",
        )
        g.add_node(b)

        g.assert_node_types(
            [
                ExternalTensorValue,
                TensorValue,
            ]
        )

        eggs.assert_raises(
            lambda: g.assert_node_types(
                [
                    TensorValue,
                ],
            ),
            ValueError,
            r"\[ExternalTensorValue\]",
        )

    def test_edges(self) -> None:
        g = GraphDoc()

        foo_node = NodeAttributes(
            node_id=uuid.uuid4(),
            display_name="foo",
        )
        g.add_node(foo_node)

        edge_node = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="bar",
            source_node_id=foo_node.node_id,
            target_node_id=foo_node.node_id,
        )
        g.add_node(edge_node)

        illegal_source = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=edge_node.node_id,
            target_node_id=foo_node.node_id,
        )
        missing_source = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=uuid.uuid4(),
            target_node_id=foo_node.node_id,
        )
        illegal_target = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=foo_node.node_id,
            target_node_id=edge_node.node_id,
        )
        missing_target = EdgeAttributes(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=foo_node.node_id,
            target_node_id=uuid.uuid4(),
        )

        for bad in (illegal_source, missing_source, illegal_target, missing_target):
            eggs.assert_raises(
                lambda: g.add_node(bad),
                ValueError,
            )
