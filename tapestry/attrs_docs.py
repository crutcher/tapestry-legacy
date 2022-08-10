import copy
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Type, cast

import marshmallow
import marshmallow_dataclass
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema
from overrides import overrides

from tapestry.serialization.json_serializable import JsonDumpable, JsonLoadable

NODE_TYPE_FIELD = "__type__"
EDGES_FIELD = "__edges__"


@marshmallow_dataclass.dataclass
class NodeAttrs(JsonLoadable):
    node_id: uuid.UUID
    """Unique in a document."""

    display_name: str
    """May be repeated in a document."""


@marshmallow_dataclass.dataclass
class EdgeAttrs(NodeAttrs):
    source_node_id: uuid.UUID
    target_node_id: uuid.UUID


@marshmallow_dataclass.dataclass
class TensorSourceAttrs(NodeAttrs):
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
    Serializable NodeAttrs graph.
    """

    nodes: Dict[uuid.UUID, NodeAttrs]

    @classmethod
    def build_load_schema(
        cls,
        node_types: Iterable[Type[NodeAttrs]],
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

            type_field = NODE_TYPE_FIELD

            type_schemas = {
                cls.__name__: cast(Type[NodeAttrs], cls).get_load_schema()
                for cls in node_types
            }

        class G(marshmallow.Schema):
            nodes = fields.Dict(
                fields.UUID,
                fields.Nested(N),
            )

            @marshmallow.post_dump
            def post_dump(self, data, **kwargs):
                nodes = data["nodes"]
                edges = [node for node in nodes.values() if "source_node_id" in node]
                for edge in edges:
                    del nodes[edge["node_id"]]

                    source_node = nodes[edge["source_node_id"]]

                    if EDGES_FIELD not in source_node:
                        source_node[EDGES_FIELD] = []

                    source_node[EDGES_FIELD].append(edge)

                return data

            @marshmallow.pre_load
            def pre_load(self, data, **kwargs):
                # don't mess with whatever the input source was.
                data = copy.deepcopy(data)

                nodes = data["nodes"]
                edges = []
                for node in nodes.values():
                    if EDGES_FIELD in node:
                        edges.extend(node[EDGES_FIELD])
                        del node[EDGES_FIELD]

                for edge in edges:
                    nodes[edge["node_id"]] = edge

                return data

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
        nodes: Optional[Dict[uuid.UUID, NodeAttrs]] = None,
    ):
        self.nodes = {}
        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

    def add_node(self, node: NodeAttrs) -> None:
        """
        Add a node to the document.

        EdgeAttrs nodes are validated that their `.source_node_id` and `.target_node_id`
        appear in the graph, and are not edges.

        :param node: the node.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already in graph.")

        if isinstance(node, EdgeAttrs):
            for port in ("source_node_id", "target_node_id"):
                port_id = getattr(node, port)
                if port_id not in self.nodes:
                    raise ValueError(
                        f"Edge {port}({port_id}) not in graph:\n\n{repr(node)}",
                    )

                port_node = self.nodes[port_id]
                if isinstance(port_node, EdgeAttrs):
                    raise ValueError(
                        f"Edge {port}({port_id}) is an edge:\n\n{repr(node)}"
                        f"\n\n ==[{port}]==>\n\n{repr(port_node)}",
                    )

        self.nodes[node.node_id] = node

    def assert_node_types(
        self,
        node_types: Iterable[Type[NodeAttrs]],
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
