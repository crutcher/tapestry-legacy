import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Type, cast

import marshmallow
import marshmallow_dataclass
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema

from tapestry.serialization.json_serializable import JsonDumpable, JsonSerializable
from tapestry.type_utils import ensure_uuid


@marshmallow_dataclass.dataclass
class NodeAttrsDoc(JsonSerializable):
    node_id: uuid.UUID
    name: str

    def __init__(
        self,
        *,
        name: str,
        node_id: Optional[uuid.UUID] = None,
    ):
        self.node_id = ensure_uuid(node_id)
        self.name = name


@marshmallow_dataclass.dataclass
class TensorSourceAttrs(NodeAttrsDoc):
    def __init__(
        self,
        *,
        name: str,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )


@marshmallow_dataclass.dataclass
class TensorValueAttrs(TensorSourceAttrs):
    def __init__(
        self,
        *,
        name: str,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )


@marshmallow_dataclass.dataclass
class ExternalTensorValueAttrs(TensorValueAttrs):
    storage: str

    def __init__(
        self,
        *,
        name: str,
        storage: str,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )
        self.storage = storage


@dataclass
class OpGraphDoc(JsonDumpable):
    # This class has special handling.
    #
    # Note that this is a normal python dataclass,
    # and it defines its schema directly.
    #
    # This is to trigger dump/load polymorphic dispatch for GraphNode;
    # which uses OneOfSchema; and to build a schema which calls that correctly.

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

        class S(OneOfSchema):
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
                fields.Nested(S),
            )

            @marshmallow.post_load
            def post_load(self, data, **kwargs):
                return OpGraphDoc(**data)

        return G()

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
        self.nodes[node.node_id] = node
