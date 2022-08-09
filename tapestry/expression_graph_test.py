import unittest
import uuid

import hamcrest

from tapestry import attrs_docs, expression_graph
from tapestry.testlib import eggs


class DisjointAttrsDoc(attrs_docs.NodeAttrsDoc):
    """
    Test class which no other NodeAttrsDoc is a subclass of.
    """


class DisjointNodeWrapper(expression_graph.NodeWrapper):
    """
    Test class which no other NodeWrapper is a subclass of.
    """

    attrs: DisjointAttrsDoc


class CommonNodeWrapperTestBase(unittest.TestCase):
    WRAPPER_CLASS = expression_graph.NodeWrapper

    def example_doc(self) -> attrs_docs.NodeAttrsDoc:
        return attrs_docs.NodeAttrsDoc(name="foo")

    def test_common_construction(self) -> None:
        g = expression_graph.ExpressionGraph()

        attrs = self.example_doc()
        attrs_class = type(attrs)

        node = self.WRAPPER_CLASS(
            graph=g,
            attrs=attrs,
        )

        eggs.assert_true(node.wraps_doc_type(attrs_class))
        node.assert_wraps_doc_type(attrs_class)

        eggs.assert_match(
            node.try_as_type(expression_graph.NodeWrapper),
            hamcrest.instance_of(expression_graph.NodeWrapper),
        )
        eggs.assert_match(
            node.force_as_type(expression_graph.NodeWrapper),
            hamcrest.instance_of(expression_graph.NodeWrapper),
        )

        eggs.assert_true(node.wraps_doc_type(attrs_class))
        node.assert_wraps_doc_type(attrs_class)

        eggs.assert_match(
            node.try_as_type(self.WRAPPER_CLASS),
            hamcrest.instance_of(self.WRAPPER_CLASS),
        )
        eggs.assert_match(
            node.force_as_type(self.WRAPPER_CLASS),
            hamcrest.instance_of(self.WRAPPER_CLASS),
        )

        if attrs_class != attrs_docs.NodeAttrsDoc:
            eggs.assert_false(node.wraps_doc_type(DisjointAttrsDoc))
            eggs.assert_raises(
                lambda: node.assert_wraps_doc_type(DisjointAttrsDoc),
                ValueError,
            )

        eggs.assert_match(
            node.try_as_type(DisjointNodeWrapper),
            hamcrest.none(),
        )
        eggs.assert_raises(
            lambda: node.force_as_type(DisjointNodeWrapper),
            ValueError,
        )


class NodeWrapperTest(CommonNodeWrapperTestBase):
    def test_eq_hash(self) -> None:
        g = expression_graph.ExpressionGraph()

        foo = attrs_docs.NodeAttrsDoc(
            node_id=uuid.uuid4(),
            name="foo",
        )
        bar = attrs_docs.NodeAttrsDoc(
            node_id=uuid.uuid4(),
            name="bar",
        )

        foo_node_1 = expression_graph.NodeWrapper(
            graph=g,
            attrs=foo,
        )
        foo_node_2 = expression_graph.NodeWrapper(
            graph=g,
            attrs=foo,
        )
        bar_node = expression_graph.NodeWrapper(
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
    WRAPPER_CLASS = expression_graph.TensorSource

    def example_doc(self) -> attrs_docs.NodeAttrsDoc:
        return attrs_docs.TensorSourceAttrs(
            name="foo",
        )


class TensorValueTest(CommonNodeWrapperTestBase):
    WRAPPER_CLASS = expression_graph.TensorValue

    def example_doc(self) -> attrs_docs.NodeAttrsDoc:
        return attrs_docs.TensorValueAttrs(
            name="foo",
        )


class ExternalTensorValueTest(CommonNodeWrapperTestBase):
    WRAPPER_CLASS = expression_graph.ExternalTensorValue

    def example_doc(self) -> attrs_docs.NodeAttrsDoc:
        return attrs_docs.ExternalTensorValueAttrs(
            name="foo",
            storage="abc",
        )
