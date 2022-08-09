import unittest
import uuid
from typing import Any, Dict

from tapestry import attrs_docs
from tapestry.serialization import json_testlib


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
    def test_json(self) -> None:
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

        json_testlib.assert_json_serializable_roundtrip(
            g,
            {
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
            },
        )
