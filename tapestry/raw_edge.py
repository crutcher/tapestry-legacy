import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union, cast

import marshmallow
import marshmallow_dataclass
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema

from tapestry.serialization.json import JsonSerializable


def ensure_uuid(val: Optional[Union[str, uuid.UUID]] = None) -> uuid.UUID:
    if val is None:
        return uuid.uuid4()
    if isinstance(val, str):
        return uuid.UUID(val)
    if isinstance(val, uuid.UUID):
        return val
    raise ValueError(
        f"Unknown UUID coercion type: {type(val)}:: {val}",
    )


@marshmallow_dataclass.dataclass
class GraphNode(JsonSerializable):
    node_id: uuid.UUID
    name: str

    def __init__(
        self,
        name: str,
        *,
        node_id: Optional[uuid.UUID] = None,
    ):
        self.node_id = ensure_uuid(node_id)
        self.name = name


@marshmallow_dataclass.dataclass
class TensorSource(GraphNode):
    def __init__(
        self,
        name: str,
        *,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )


@marshmallow_dataclass.dataclass
class TensorValue(TensorSource):
    def __init__(
        self,
        name: str,
        *,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )


@marshmallow_dataclass.dataclass
class ExternalTensor(TensorValue):
    storage: str

    def __init__(
        self,
        name: str,
        storage: str,
        *,
        node_id: Optional[uuid.UUID] = None,
    ):
        super().__init__(
            name=name,
            node_id=node_id,
        )
        self.storage = storage


class GraphNodeSchema(OneOfSchema):
    type_schemas = {
        cls.__name__: cast(Type[GraphNode], cls).Schema
        for cls in [
            TensorValue,
            ExternalTensor,
        ]
    }

    type_field = "_node_type_"


@dataclass
class OpGraph(JsonSerializable):
    # This class has special handling.
    #
    # Note that this is a normal python dataclass,
    # and it defines its schema directly.
    #
    # This is to trigger dump/load polymorphic dispatch for GraphNode;
    # which uses OneOfSchema; and to build a schema which calls that correctly.

    nodes: Dict[uuid.UUID, GraphNode]

    class Schema(marshmallow.Schema):
        nodes = fields.Dict(
            fields.UUID,
            fields.Nested(GraphNodeSchema),
        )

        @marshmallow.post_load
        def post_load(self, data, **kwargs):
            return OpGraph(**data)

    def __init__(
        self,
        *,
        nodes: Optional[Dict[uuid.UUID, GraphNode]] = None,
    ):
        self.nodes = {}
        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node


def raw():
    a = ExternalTensor(
        name="A",
        storage="pre:A",
    )
    b = TensorValue(
        name="B",
    )
    print("a", a.pretty())
    print("b", b.pretty())

    g = OpGraph()
    g.add_node(a)
    g.add_node(b)
    print(
        "load(dump(g))",
        OpGraph.load_json_data(g.dump_json_data()).pretty(),
    )


if __name__ == "__main__":
    raw()
