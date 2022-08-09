import unittest
import uuid
from typing import Any, Dict

from tapestry import attrs_docs
from tapestry.serialization import json_testlib
from tapestry.testlib import eggs


class NodeAttrsDocTest(unittest.TestCase):
    DOC_CLASS = attrs_docs.NodeAttrsDoc

    def expected_json(self, node_id: uuid.UUID) -> Dict[str, Any]:
        return {
            "node_id": str(node_id),
            "name": "foo",
        }

    def test_lifecycle(self) -> None:
        node = self.DOC_CLASS(name="foo")

        json_testlib.assert_json_serializable_roundtrip(
            node,
            self.expected_json(node.node_id),
        )

        node_id = uuid.uuid4()
        node = self.DOC_CLASS(
            node_id=node_id,
            name="foo",
        )
        json_testlib.assert_json_serializable_roundtrip(
            node,
            self.expected_json(node_id),
        )


class TensorSourceAttrsTest(unittest.TestCase):
    DOC_CLASS = attrs_docs.TensorSourceAttrs


class TensorValueAttrsTest(unittest.TestCase):
    DOC_CLASS = attrs_docs.TensorValueAttrs


class ExternalTensorSourceAttrsTest(unittest.TestCase):
    DOC_CLASS = attrs_docs.ExternalTensorValueAttrs

    def expected_json(self, node_id: uuid.UUID) -> Dict[str, Any]:
        return {
            "node_id": str(node_id),
            "name": "foo",
            "storage": "abc",
        }


class OpGraphDocTest(unittest.TestCase):
    def test_schema(self) -> None:
        g = attrs_docs.OpGraphDoc()
        a = attrs_docs.ExternalTensorValueAttrs(
            name="A",
            storage="pre:A",
        )
        g.add_node(a)

        b = attrs_docs.TensorValueAttrs(
            name="B",
        )
        g.add_node(b)

        s = attrs_docs.OpGraphDoc.build_load_schema(
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
        g = attrs_docs.OpGraphDoc()
        a = attrs_docs.ExternalTensorValueAttrs(
            name="A",
            storage="pre:A",
        )
        g.add_node(a)

        b = attrs_docs.TensorValueAttrs(
            name="B",
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
