import unittest
import uuid
from typing import Any, Dict, Type

from tapestry import attrs_docs
from tapestry.serialization import json_testlib
from tapestry.testlib import eggs


class NodeAttrsDocTest(unittest.TestCase):
    DOC_CLASS: Type[attrs_docs.NodeAttrsDoc] = attrs_docs.NodeAttrsDoc
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

        b = attrs_docs.TensorValueAttrs(
            node_id=uuid.uuid4(),
            display_name="B",
        )
        g.add_node(b)

        s = attrs_docs.GraphDoc.build_load_schema(
            [
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
