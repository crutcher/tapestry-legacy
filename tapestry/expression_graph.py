import copy
from dataclasses import dataclass, field
import html
import typing
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
import uuid
import weakref

import marshmallow
from marshmallow import fields
import marshmallow_dataclass
from marshmallow_oneofschema import OneOfSchema
import numpy as np
from overrides import overrides
import pydot

from tapestry import zspace
from tapestry.serialization.json_serializable import JsonDumpable, JsonLoadable
from tapestry.type_utils import UUIDConvertable, coerce_optional_uuid, coerce_uuid

NODE_TYPE_FIELD = "__type__"
EDGES_FIELD = "__edges__"


_TapestryNodeT = TypeVar("_TapestryNodeT", bound="TapestryNode")
_TapestryEdgeT = TypeVar("_TapestryEdgeT", bound="TapestryEdge")

NodeIdCoercible = Union[UUIDConvertable, "TapestryNode"]


def coerce_node_id(val: NodeIdCoercible) -> uuid.UUID:
    if isinstance(val, TapestryNode):
        return val.node_id
    return coerce_uuid(val)


def coerce_optional_node_id(
    val: Optional[NodeIdCoercible],
) -> Optional[NodeIdCoercible]:
    if val is None:
        return None
    return coerce_node_id(val)


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TapestryNode(JsonLoadable):
    class Meta:
        exclude = ("_graph",)

    _graph: Any = None

    node_id: uuid.UUID = field(default_factory=uuid.uuid4)
    """Unique in a document."""

    name: Optional[str] = None

    def validate(self) -> None:
        self.assert_graph()

    def clone(self: _TapestryNodeT) -> _TapestryNodeT:
        val = copy.deepcopy(self)
        del val.graph
        return val

    @classmethod
    def node_type(cls) -> str:
        return cls.__qualname__

    def assert_graph(self) -> "TapestryGraph":
        g = self.graph
        if g is None:
            raise ValueError("No attached graph")
        return g

    @property
    def graph(self) -> Optional["TapestryGraph"]:
        """
        :return: the GraphDoc.
        :raises ValueError: if the object is not attached to a GraphDoc.
        :raises ReferenceError: if the attached GraphDoc is out of scope.
        """
        if self._graph is None:
            return None
        try:
            return cast(TapestryGraph, self._graph())
        except ReferenceError:
            return None

    @graph.setter
    def graph(self, graph: "TapestryGraph") -> None:
        self._graph = weakref.ref(graph)

    @graph.deleter
    def graph(self) -> None:
        self._graph = None


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TapestryEdge(TapestryNode):
    class Constraint:
        SOURCE_TYPE: TapestryNode
        TARGET_TYPE: TapestryNode

    source_id: uuid.UUID
    target_id: uuid.UUID

    @overload
    def source(self) -> TapestryNode:
        ...

    @overload
    def source(
        self,
        node_type: Type[_TapestryNodeT],
    ) -> _TapestryNodeT:
        ...

    def source(
        self,
        node_type: Type[TapestryNode | _TapestryNodeT] = TapestryNode,
    ) -> Union[TapestryNode, _TapestryNodeT]:
        return self.assert_graph().get_node(self.source_id, node_type)

    @overload
    def target(self) -> TapestryNode:
        ...

    @overload
    def target(
        self,
        node_type: Type[_TapestryNodeT],
    ) -> _TapestryNodeT:
        ...

    def target(
        self,
        node_type: Type[TapestryNode | _TapestryNodeT] = TapestryNode,
    ) -> Union[TapestryNode, _TapestryNodeT]:
        return self.assert_graph().get_node(self.target_id, node_type)

    @overrides
    def validate(self) -> None:
        super(TapestryEdge, self).validate()
        hints = typing.get_type_hints(self.Constraint)
        assert isinstance(self.source(), hints["SOURCE_TYPE"]), hints
        assert isinstance(self.target(), hints["TARGET_TYPE"])


@dataclass
class TapestryGraph(JsonDumpable):
    """
    Serializable NodeAttributes graph.
    """

    nodes: Dict[uuid.UUID, TapestryNode]

    @classmethod
    def build_load_schema(
        cls,
        node_types: Iterable[Type[TapestryNode]],
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
                cls.__qualname__: cast(Type[TapestryNode], cls).get_load_schema()
                for cls in node_types
            }

            def get_obj_type(self, obj):
                return type(obj).__qualname__

        class G(marshmallow.Schema):
            nodes = fields.Dict(
                fields.UUID,
                fields.Nested(N),
            )

            @marshmallow.post_dump
            def post_dump(self, data, **kwargs):
                nodes = data["nodes"]
                edges = [node for node in nodes.values() if "source_id" in node]
                for edge in edges:
                    del nodes[edge["node_id"]]

                    source_node = nodes[edge["source_id"]]

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
                graph = TapestryGraph(**data)
                for node in graph.nodes.values():
                    node.graph = graph
                return graph

        return G()

    @overrides
    def get_dump_schema(self) -> marshmallow.Schema:
        return self.build_load_schema(
            {type(n) for n in self.nodes.values()},
        )

    def __init__(
        self,
        *,
        nodes: Optional[Dict[uuid.UUID, TapestryNode]] = None,
    ):
        self.nodes = {}
        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

    def validate(self) -> None:
        for node in self.nodes.values():
            node.validate()

    def clone(self) -> "TapestryGraph":
        """
        Deep clone of a graph.

        :return: a new graph.
        """
        g = TapestryGraph()
        for node in self.nodes.values():
            g.add_node(node.clone())
        return g

    @overload
    def get_node(
        self,
        node_id: UUIDConvertable,
    ) -> TapestryNode:
        ...

    @overload
    def get_node(
        self,
        node_id: UUIDConvertable,
        node_type: Type[_TapestryNodeT],
    ) -> _TapestryNodeT:
        ...

    def get_node(
        self,
        node_id: UUIDConvertable,
        node_type: Type[TapestryNode] = TapestryNode,
    ) -> _TapestryNodeT:
        """
        Find a NodeAttributes by node_id, and wrap it in the given node type.

        :param node_id: the node_id, may be str or UUID.
        :param node_type: the expected type.
        :return: the wrapped node.
        """
        node_id = coerce_uuid(node_id)

        if not issubclass(node_type, TapestryNode):
            raise AssertionError(
                f"{node_type} is not a subclass of {TapestryNode}",
            )

        node = self.nodes[node_id]
        if not isinstance(node, node_type):
            raise ValueError(
                f"Node {node.node_id} is not of the expected type {node_type}:\n{node.pretty()}"
            )
        return cast(_TapestryNodeT, node)

    def remove_node(self, node: UUIDConvertable | TapestryNode) -> None:
        """
        Remove a node from the graph.

        :param node: the node to remove.
        :raises KeyError: if the node is not in the graph.
        """
        if not isinstance(node, TapestryNode):
            node = self.get_node(coerce_uuid(node))

        if node.node_id not in self.nodes:
            raise KeyError(f"Node {node.node_id} not in graph:\n{node.pretty()}")

        del node.graph
        del self.nodes[node.node_id]

    @overload
    def list_nodes(self) -> List[TapestryNode]:
        ...

    @overload
    def list_nodes(self, node_type: Type[_TapestryNodeT]) -> List[_TapestryNodeT]:
        ...

    def list_nodes(
        self,
        node_type: Type[TapestryNode | _TapestryNodeT] = TapestryNode,
    ) -> Union[List[TapestryNode], List[_TapestryNodeT]]:
        """
        List all nodes which are subclasses of the given type.

        :param node_type: the node wrapper type.
        :return: a list of nodes.
        """
        if not issubclass(node_type, TapestryNode):
            raise AssertionError(
                f"Class {node_type} is not a subclass of {TapestryNode}"
            )
        return [
            cast(_TapestryNodeT, node)
            for node in self.nodes.values()
            if isinstance(node, node_type)
        ]

    @overload
    def list_edges(
        self,
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
    ) -> List[TapestryEdge]:
        ...

    @overload
    def list_edges(
        self,
        edge_type: Type[_TapestryEdgeT],
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
    ) -> List[_TapestryEdgeT]:
        ...

    def list_edges(
        self,
        edge_type: Type[TapestryEdge | _TapestryEdgeT] = TapestryEdge,
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
    ) -> Union[List[TapestryEdge], List[_TapestryEdgeT]]:
        if not issubclass(edge_type, TapestryEdge):
            raise AssertionError(
                f"Class {edge_type} is not a subclass of {TapestryEdge}"
            )

        source_id = coerce_optional_uuid(source_id)
        target_id = coerce_optional_uuid(target_id)
        return [
            node
            for node in self.list_nodes(edge_type)
            if source_id is None or node.source_id == source_id
            if target_id is None or node.target_id == target_id
        ]

    def add_node(self, node: _TapestryNodeT) -> _TapestryNodeT:
        """
        Add a node to the document.

        EdgeAttributes nodes are validated that their `.source_id` and `.target_id`
        appear in the graph, and are not edges.

        :param node: the node.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already in graph.")

        if isinstance(node, TapestryEdge):
            for port in ("source_id", "target_id"):
                port_id = getattr(node, port)
                if port_id not in self.nodes:
                    raise ValueError(
                        f"Edge {port}({port_id}) not in graph:\n\n{repr(node)}",
                    )

                port_node = self.nodes[port_id]
                if isinstance(port_node, TapestryEdge):
                    raise ValueError(
                        f"Edge {port}({port_id}) is an edge:\n\n{repr(node)}"
                        f"\n\n ==[{port}]==>\n\n{repr(port_node)}",
                    )

        self.nodes[node.node_id] = node
        node.graph = self

        return node

    def assert_node_types(
        self,
        node_types: Iterable[Type[TapestryNode]],
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

    def to_dot(self, *, omit_ids: bool = True) -> pydot.Dot:
        def format_data_table_row(data: dict) -> str:
            return "".join(
                (
                    f'<tr><td align="right"><b>{k}:</b></td>'
                    f'<td align="left">{format_data(v)}</td></tr>'
                )
                for k, v in data.items()
            )

        def format_data(data):
            if isinstance(data, dict):
                return f"""<table border="0" cellborder="1" cellspacing="0">{format_data_table_row(data)}</table>"""

            else:
                return html.escape(str(data))

        dot = pydot.Dot("G", graph_type="digraph", bgcolor="red")
        dot.set_graph_defaults(bgcolor="white")
        dot.set_graph_defaults(rankdir="RL")
        for node in self.nodes.values():
            node_type = type(node).node_type()

            data = node.dump_json_data()
            is_edge = isinstance(node, TapestryEdge)

            if omit_ids:
                del data["node_id"]

                if is_edge:
                    del data["source_id"]
                    del data["target_id"]

            null_keys = [k for k, v in data.items() if v is None]
            for k in null_keys:
                del data[k]

            title = node_type
            if is_edge:
                title = f"Edge: {title}"

            label = f"""
                <table border="0" cellborder="1" cellspacing="0">
                  <tr><td colspan="2">{title}</td></tr>
                  {format_data_table_row(data)}
                  </table>
            """
            shape = "plain"
            if is_edge:
                shape = "rectangle"

            dot.add_node(
                pydot.Node(
                    str(node.node_id),
                    label=f"<{label}>",
                    shape=shape,
                ),
            )

        for node in self.nodes.values():
            if isinstance(node, TapestryEdge):
                dot.add_edge(
                    pydot.Edge(
                        str(node.source_id),
                        str(node.node_id),
                        arrowhead="none",
                    ),
                )
                dot.add_edge(
                    pydot.Edge(
                        str(node.node_id),
                        str(node.target_id),
                    ),
                )

        return dot


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorValue(TapestryNode):
    shape: zspace.ZArray
    dtype: str

    def __post_init__(self):
        self.shape = zspace.as_zarray(self.shape)


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorResult(TensorValue):
    pass


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class ExternalTensor(TensorValue):
    storage: str


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class BlockOperation(TapestryNode):
    index_space: zspace.ZRange

    @dataclass(kw_only=True)
    class TensorBinding(TapestryEdge):
        # force name to be required
        name: str
        selector: zspace.ZRangeMap

        def __post_init__(self):
            assert self.name

        def _validate(self, op: "BlockOperation", tensor: TensorValue):
            actual = np.array((op.index_space.ndim, tensor.shape.ndim))
            if (actual == self.selector.shape).all():
                raise AssertionError(
                    f"Selector Dimension Miss-match:\n"
                    f"{self.pretty()}\n\n"
                    f"Op {op.pretty()}\n\n"
                    f"Tensor {tensor.pretty()}"
                )

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Input(TensorBinding):
        class Constraint(TapestryEdge.Constraint):
            SOURCE_TYPE: "BlockOperation"
            TARGET_TYPE: TensorValue

        def validate(self) -> None:
            op = self.source(BlockOperation)
            tensor = self.target(TensorValue)
            self._validate(op, tensor)

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Result(TensorBinding):
        class Constraint(TapestryEdge.Constraint):
            SOURCE_TYPE: TensorValue
            TARGET_TYPE: "BlockOperation"

        def validate(self) -> None:
            tensor = self.source(TensorResult)
            op = self.target(BlockOperation)
            self._validate(op, tensor)

    def inputs(self) -> List[Input]:
        return self.assert_graph().list_edges(
            edge_type=BlockOperation.Input,
            source_id=self.node_id,
        )

    def results(self) -> List[Result]:
        return self.assert_graph().list_edges(
            edge_type=BlockOperation.Result,
            target_id=self.node_id,
        )
