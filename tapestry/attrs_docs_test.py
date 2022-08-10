import unittest
import uuid
from typing import Any, Dict, Type

from tapestry import attrs_docs
from tapestry.serialization import json_testlib
from tapestry.testlib import eggs


class NodeAttrsDocTest(unittest.TestCase):
    DOC_CLASS: Type[attrs_docs.NodeAttrs] = attrs_docs.NodeAttrs
    """Overridable by subclasses."""

    def expected_json(self, node_id: uuid.UUID) -> Dict[str, Any]:
        return {
            "node_id": str(node_id),
            "display_name": "foo",
        }

    def test_lifecycle(self) -> None:
        node_id = uuid.uuid4()
        json = self.expected_json(node_id)
        node = self.DOC_CLASS.load_json_data(json)

        json_testlib.assert_json_serializable_roundtrip(
            node,
            json,
        )


class TensorSourceAttrsTest(NodeAttrsDocTest):
    DOC_CLASS = attrs_docs.TensorSourceAttrs


class TensorValueAttrsTest(NodeAttrsDocTest):
    DOC_CLASS = attrs_docs.TensorValueAttrs


class ExternalTensorSourceAttrsTest(NodeAttrsDocTest):
    DOC_CLASS = attrs_docs.ExternalTensorValueAttrs

    def expected_json(self, node_id: uuid.UUID) -> Dict[str, Any]:
        return {
            "node_id": str(node_id),
            "display_name": "foo",
            "storage": "abc",
        }


class OpGraphDocTest(unittest.TestCase):
    def test_schema(self) -> None:
        g = attrs_docs.GraphDoc()
        a = attrs_docs.ExternalTensorValueAttrs(
            node_id=uuid.uuid4(),
            display_name="A",
            storage="pre:A",
        )
        g.add_node(a)
        eggs.assert_raises(
            lambda: g.add_node(a),
            ValueError,
            "already in graph",
        )

        b = attrs_docs.TensorValueAttrs(
            node_id=uuid.uuid4(),
            display_name="B",
        )
        g.add_node(b)

        edge_node = attrs_docs.EdgeAttrs(
            node_id=uuid.uuid4(),
            display_name="child",
            source_node_id=a.node_id,
            target_node_id=b.node_id,
        )
        g.add_node(edge_node)

        s = attrs_docs.GraphDoc.build_load_schema(
            [
                attrs_docs.EdgeAttrs,
                attrs_docs.TensorSourceAttrs,
                attrs_docs.TensorValueAttrs,
                attrs_docs.ExternalTensorValueAttrs,
            ]
        )

        expected_json = {
            "nodes": {
                str(a.node_id): {
                    "__type__": "ExternalTensorValueAttrs",
                    **a.dump_json_data(),
                    "__edges__": [
                        {
                            **edge_node.dump_json_data(),
                            "__type__": "EdgeAttrs",
                        }
                    ],
                },
                str(b.node_id): {
                    "__type__": "TensorValueAttrs",
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
        g = attrs_docs.GraphDoc()
        a = attrs_docs.ExternalTensorValueAttrs(
            node_id=uuid.uuid4(),
            display_name="A",
            storage="pre:A",
        )
        g.add_node(a)

        b = attrs_docs.TensorValueAttrs(
            node_id=uuid.uuid4(),
            display_name="B",
        )
        g.add_node(b)

        g.assert_node_types(
            [
                attrs_docs.ExternalTensorValueAttrs,
                attrs_docs.TensorValueAttrs,
            ]
        )

        eggs.assert_raises(
            lambda: g.assert_node_types(
                [
                    attrs_docs.TensorValueAttrs,
                ],
            ),
            ValueError,
            r"\[ExternalTensorValueAttrs\]",
        )

    def test_edges(self) -> None:
        g = attrs_docs.GraphDoc()

        foo_node = attrs_docs.NodeAttrs(
            node_id=uuid.uuid4(),
            display_name="foo",
        )
        g.add_node(foo_node)

        edge_node = attrs_docs.EdgeAttrs(
            node_id=uuid.uuid4(),
            display_name="bar",
            source_node_id=foo_node.node_id,
            target_node_id=foo_node.node_id,
        )
        g.add_node(edge_node)

        illegal_source = attrs_docs.EdgeAttrs(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=edge_node.node_id,
            target_node_id=foo_node.node_id,
        )
        missing_source = attrs_docs.EdgeAttrs(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=uuid.uuid4(),
            target_node_id=foo_node.node_id,
        )
        illegal_target = attrs_docs.EdgeAttrs(
            node_id=uuid.uuid4(),
            display_name="illegal",
            source_node_id=foo_node.node_id,
            target_node_id=edge_node.node_id,
        )
        missing_target = attrs_docs.EdgeAttrs(
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
