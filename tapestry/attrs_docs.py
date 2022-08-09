import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Type, cast

import marshmallow
import marshmallow_dataclass
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema
from overrides import overrides

from tapestry.serialization.json_serializable import JsonDumpable, JsonLoadable


@marshmallow_dataclass.dataclass
class NodeAttrsDoc(JsonLoadable):
    node_id: uuid.UUID
    """Unique in a document."""

    display_name: str
    """May be repeated in a document."""


@marshmallow_dataclass.dataclass
class TensorSourceAttrs(NodeAttrsDoc):
    pass


@marshmallow_dataclass.dataclass
class TensorValueAttrs(TensorSourceAttrs):
    pass


@marshmallow_dataclass.dataclass
class ExternalTensorValueAttrs(TensorValueAttrs):
    storage: str


@dataclass
class GraphDoc(JsonDumpable):
    """
    Serializable NodeAttrsDoc graph.
    """

    nodes: Dict[uuid.UUID, NodeAttrsDoc]

    @classmethod
    def build_load_schema(
        cls,
        node_types: Iterable[Type[NodeAttrsDoc]],
    ) -> marshmallow.Schema:
        """
        Builds a load schema for a collection of node types.

        :param node_types: the node types to load.
        :return: a Schema.
        """

        class N(OneOfSchema):
            """
            Polymorphic type-dispatch wrapper for NodeAttrDoc subclasses.
            """

            type_field = "__type__"

            type_schemas = {
                cls.__name__: cast(Type[NodeAttrsDoc], cls).get_load_schema()
                for cls in node_types
            }

        class G(marshmallow.Schema):
            nodes = fields.Dict(
                fields.UUID,
                fields.Nested(N),
            )

            @marshmallow.post_load
            def post_load(self, data, **kwargs):
                return GraphDoc(**data)

        return G()

    @overrides
    def get_dump_schema(self) -> marshmallow.Schema:
        return self.build_load_schema(
            {type(n) for n in self.nodes.values()},
        )

    def __init__(
        self,
        *,
        nodes: Optional[Dict[uuid.UUID, NodeAttrsDoc]] = None,
    ):
        self.nodes = {}
        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

    def add_node(self, node: NodeAttrsDoc) -> None:
        """
        Add a node to the document.

        :param node: the node.
        """
        self.nodes[node.node_id] = node

    def assert_node_types(
        self,
        node_types: Iterable[Type[NodeAttrsDoc]],
    ) -> None:
        """
        Assert that the GraphDoc contains only the listed types.

        :param node_types: the node types.
        :raises ValueError: if there are type violations.
        """
        node_types = set(node_types)
        violations = {
            type(node) for node in self.nodes.values() if type(node) not in node_types
        }
        if violations:
            names = ", ".join(sorted(cls.__name__ for cls in violations))
            raise ValueError(f"Illegal node types found: [{names}]")
