import contextlib
import copy
from dataclasses import dataclass, field
import html
import inspect
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
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
from marshmallow_dataclass import NewType
from marshmallow_oneofschema import OneOfSchema
from overrides import overrides
import pydot
import torch

from tapestry import zspace
from tapestry.numpy_utils import as_zarray
from tapestry.serialization.json_serializable import JsonDumpable, JsonLoadable
from tapestry.type_utils import UUIDConvertable, coerce_optional_uuid, coerce_uuid
from tapestry.zspace import ZRange, ZRangeMap

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


def find_dtype(name: str) -> torch.dtype:
    if not name.startswith("torch."):
        raise AssertionError(f"Not a pytorch dtype: {name}")

    sname = name.removeprefix("torch.")
    dtype = getattr(torch, sname)
    assert isinstance(dtype, torch.dtype), f"{name} exists but is {type(dtype)}"
    return dtype


class DTypeField(fields.Field):
    """
    Marshmallow Field type for torch.dtype.

    Depends upon the following `setup.cfg` for mpyp:

    >>> [mypy]
    >>> plugins = marshmallow_dataclass.mypy
    """

    def __init__(self, *args, **kwargs):
        super(DTypeField, self).__init__(*args, **kwargs)

    def _serialize(self, value: torch.dtype, *args, **kwargs):
        return str(value)

    def _deserialize(self, value, *args, **kwargs):
        if value is None:
            return None

        return find_dtype(value)


DType = NewType("torch.dtype", torch.dtype, field=DTypeField)
"""
Marshmallow NewType for DTypeField.

Usage:

>>> @marshmallow_dataclass.dataclass
... class Example:
...     coords: DType
"""


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TapestryNode(JsonLoadable):
    class Meta:
        exclude = ("_graph",)

    class NodeControl:
        BG_COLOR: Optional[str] = None

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
class TapestryTag(TapestryNode):
    class EdgeControl:
        SOURCE_TYPE: TapestryNode

    source_id: uuid.UUID

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

    @overrides
    def validate(self) -> None:
        super(TapestryTag, self).validate()
        hints = typing.get_type_hints(self.EdgeControl)
        assert isinstance(self.source(), hints["SOURCE_TYPE"]), hints


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TapestryEdge(TapestryTag):
    class EdgeControl(TapestryTag.EdgeControl):
        TARGET_TYPE: TapestryNode

        INVERT_DEPENDENCY_FLOW: bool = False
        DISPLAY_ATTRIBUTES: bool = True

        RELAX_EDGE: bool = False

    target_id: uuid.UUID

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
        hints = typing.get_type_hints(self.EdgeControl)
        assert isinstance(self.target(), hints["TARGET_TYPE"]), hints


@dataclass
class TapestryGraph(JsonDumpable):
    """
    Serializable NodeAttributes graph.
    """

    class Meta:
        exclude = ("_validate_edits",)

    nodes: Dict[uuid.UUID, TapestryNode]
    observed: Set[uuid.UUID]

    _validate_edits: bool = True

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
        observed: Optional[Iterable[NodeIdCoercible]] = None,
    ):
        self.nodes = {}
        self.observed = set()

        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

        if observed:
            for node in observed:
                self.mark_observed(node)

    def validate(self) -> None:
        for node in self.nodes.values():
            node.validate()

    def validate_if_enabled(self) -> None:
        if self._validate_edits:
            self.validate()

    @contextlib.contextmanager
    def relax(self) -> typing.Iterator[None]:
        val = self._validate_edits
        self._validate_edits = False
        yield
        self._validate_edits = val
        self.validate_if_enabled()

    def clone(self) -> "TapestryGraph":
        """
        Deep clone of a graph.

        :return: a new graph.
        """

        # TODO: with graph edits, the node order might not be sufficient to load.
        # Because we enforce "exists" for edges, and run validate on add,
        # this will fail with some graph structures.
        #
        # doing something smarter should be fine.
        g = TapestryGraph()
        with g.relax():
            for node in self.nodes.values():
                g.add_node(node.clone())
            for node_id in self.observed:
                g.mark_observed(node_id)
        return g

    def add_node(self, node: _TapestryNodeT) -> _TapestryNodeT:
        """
        Add a node to the document.

        EdgeAttributes nodes are validated that their `.source_id` and `.target_id`
        appear in the graph, and are not edges.

        :param node: the node.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already in graph.")

        if self._validate_edits and isinstance(node, TapestryEdge):
            for port in ("source_id", "target_id"):
                port_id = getattr(node, port)
                if port_id not in self.nodes:
                    raise ValueError(
                        f"Edge {port}({port_id}) not in graph:\n\n{repr(node)}",
                    )

        self.nodes[node.node_id] = node
        node.graph = self

        self.validate_if_enabled()

        return node

    def mark_observed(self, node: NodeIdCoercible) -> None:
        node_id = coerce_node_id(node)
        node = self.get_node(node_id)

        if node_id not in self.nodes:
            raise ValueError(
                f"Can't observe a node not in the graph: {node}",
            )
        self.observed.add(node_id)

    def clear_observed(self, node: NodeIdCoercible) -> None:
        node_id = coerce_node_id(node)
        self.observed.remove(node_id)

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
                f"Node {node.node_id} is not of the expected type {node_type}:\n"
                f"{node.pretty()}"
            )
        return cast(_TapestryNodeT, node)

    def remove_node(
        self,
        node: NodeIdCoercible,
        *,
        remove_edges: bool = False,
    ) -> None:
        """
        Remove a node from the graph.

        :param node: the node to remove.
        :raises KeyError: if the node is not in the graph.
        """
        node_id = coerce_node_id(node)
        node = self.get_node(node_id)

        if node_id in self.observed:
            raise ValueError(f"Can't remove an observed node: {node_id}\n\n{node}")

        edges = {e.node_id: e for e in self.list_edges(source_id=node_id)}
        edges.update({e.node_id: e for e in self.list_edges(target_id=node_id)})
        if edges:
            if not remove_edges:
                raise AssertionError(
                    "Can't remove a node with live edges:\n"
                    + "\n".join(repr(e) for e in edges.values())
                )

            for e in edges:
                self.remove_node(e, remove_edges=True)

        del node.graph
        del self.nodes[node.node_id]

        if self._validate_edits:
            self.validate()

    @overload
    def list_nodes(
        self,
        *,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = TapestryEdge,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> List[TapestryNode]:
        ...

    @overload
    def list_nodes(
        self,
        node_type: Type[_TapestryNodeT],
        *,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = TapestryEdge,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> List[_TapestryNodeT]:
        ...

    def list_nodes(
        self,
        node_type: Type[TapestryNode | _TapestryNodeT] = TapestryNode,
        *,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = TapestryEdge,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> Union[List[TapestryNode], List[_TapestryNodeT]]:
        """
        List all nodes which are subclasses of the given type.

        By default, filters out all instances of TapestryEdge.

        :param node_type: the node wrapper type.
        :param exclude: Iterable of types (and sub-types) to exclude,
            defaults to `(TapestryEdge,)`.
        :return: a list of nodes.
        """
        if not issubclass(node_type, TapestryNode):
            raise AssertionError(
                f"Class {node_type} is not a subclass of {TapestryNode}"
            )

        def clean_class_list_arg(val) -> Tuple[Type[TapestryNode], ...]:
            if val is None:
                return tuple()

            if inspect.isclass(val):
                return (val,)

            else:
                return tuple(val)

        restrict = clean_class_list_arg(restrict)
        exclude = clean_class_list_arg(exclude)

        for t in restrict:
            if not issubclass(t, node_type):
                raise AssertionError(
                    f"include type ({t}) is not a subclass of node_type: {node_type}"
                )

        if issubclass(node_type, exclude):
            raise AssertionError(
                f"node_type: ({node_type.__qualname__}) is a"
                f" subclass of a filter_type: {exclude}"
            )

        return [
            cast(_TapestryNodeT, node)
            for node in self.nodes.values()
            if isinstance(node, node_type)
            if (not restrict or isinstance(node, restrict))
            if (not exclude or not isinstance(node, exclude))
            if (not filter or filter(node))
        ]

    @overload
    def list_edges(
        self,
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> List[TapestryEdge]:
        ...

    @overload
    def list_edges(
        self,
        edge_type: Type[_TapestryEdgeT],
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> List[_TapestryEdgeT]:
        ...

    def list_edges(
        self,
        edge_type: Type[TapestryEdge | _TapestryEdgeT] = TapestryEdge,
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
        restrict: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        exclude: Optional[
            Union[Type[TapestryNode], Iterable[Type[TapestryNode]]]
        ] = None,
        filter: Optional[Callable[[TapestryNode], bool]] = None,
    ) -> Union[List[TapestryEdge], List[_TapestryEdgeT]]:
        if not issubclass(edge_type, TapestryEdge):
            raise AssertionError(
                f"Class {edge_type} is not a subclass of {TapestryEdge}"
            )

        source_id = coerce_optional_uuid(source_id)
        target_id = coerce_optional_uuid(target_id)
        return [
            node
            for node in self.list_nodes(
                edge_type,
                restrict=restrict,
                exclude=exclude,
                filter=filter,
            )
            if source_id is None or node.source_id == source_id
            if target_id is None or node.target_id == target_id
        ]

    def get_singular_edge(
        self,
        edge_type: Type[_TapestryEdgeT],
        *,
        source_id: UUIDConvertable = None,
        target_id: UUIDConvertable = None,
    ) -> _TapestryEdgeT:
        edges = self.list_edges(
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
        )
        if not edges:
            raise ValueError(
                f"No matching edge {edge_type = }, {source_id =}, {target_id =}",
            )

        if len(edges) > 1:
            raise ValueError(
                f"Multiple matching edges {edge_type = }, {source_id =}, {target_id =}:\n",
                "\n".join((repr(e) for e in edges)),
            )

        return edges[0]

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

    def to_dot(
        self,
        *,
        omit_ids: bool = True,
    ) -> pydot.Dot:
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
                return (
                    '<table border="0" cellborder="1" cellspacing="0">'
                    f"{format_data_table_row(data)}</table>"
                )

            else:
                return html.escape(str(data))

        def gensym(idx: int) -> str:
            return hex(idx + 1)[2:].upper()

        node_syms = {}
        for node_idx, node in enumerate(self.list_nodes()):
            node_sym = f"N{gensym(node_idx)}"
            node_syms[node.node_id] = node_sym

            for edge_idx, edge in enumerate(self.list_edges(source_id=node.node_id)):
                edge_sym = f"{node_sym}.E{gensym(edge_idx)}"
                node_syms[edge.node_id] = edge_sym

        for edge in self.list_edges():
            if edge.node_id not in node_syms:
                node_idx += 1
                edge_sym = f"E{gensym(node_idx)}"
                node_syms[edge.node_id] = edge_sym

        dot = pydot.Dot("G", graph_type="digraph", bgcolor="red")
        dot.set_graph_defaults(nodesep=0.7)
        dot.set_graph_defaults(bgcolor="white")
        dot.set_graph_defaults(rankdir="BT")
        for node in self.nodes.values():
            node_type = type(node).node_type()

            if isinstance(node, TapestryEdge):
                if not node.EdgeControl.DISPLAY_ATTRIBUTES:
                    continue

            data = node.dump_json_data()

            is_tag = isinstance(node, TapestryTag)
            is_edge = isinstance(node, TapestryEdge)

            if omit_ids:
                del data["node_id"]

                if is_tag:
                    del data["source_id"]

                if is_edge:
                    del data["target_id"]

            null_keys = [k for k, v in data.items() if v is None]
            for k in null_keys:
                del data[k]

            title = f"{node_type}: {node_syms[node.node_id]}"

            if is_tag:
                title = f"Tag: {title}"

            if is_edge:
                title = f"Edge: {title}"

            table_attrs: Dict[str, Any] = dict(
                border=0,
                cellborder=1,
                cellspacing=0,
            )
            if node.NodeControl.BG_COLOR:
                table_attrs["bgcolor"] = node.NodeControl.BG_COLOR

            label = f"""
                <table {' '.join((f'{k}="{v}"' for k,v in table_attrs.items()))}>
                  <tr><td colspan="2">{title}</td></tr>
                  {format_data_table_row(data)}
                  </table>
            """
            node_attrs = dict(shape="plain")
            if is_tag:
                node_attrs["shape"] = "rectangle"

            dot.add_node(
                pydot.Node(
                    str(node.node_id),
                    label=f"<{label}>",
                    **node_attrs,
                ),
            )

        for node in self.nodes.values():
            if not isinstance(node, TapestryEdge):
                continue

            source = node.source_id
            target = node.target_id

            if node.EdgeControl.INVERT_DEPENDENCY_FLOW:
                source, target = target, source

            if not node.EdgeControl.DISPLAY_ATTRIBUTES:
                # there's only this edge.

                if node.EdgeControl.INVERT_DEPENDENCY_FLOW:
                    source_kwargs = dict(
                        dir="both",
                        arrowtail="normal",
                        arrowhead="oinv",
                    )
                else:
                    source_kwargs = dict(
                        dir="both",
                        arrowtail="oinv",
                        arrowhead="normal",
                    )

                if node.EdgeControl.RELAX_EDGE:
                    source_kwargs["constraint"] = "false"

                # distinguish no-attribute errors
                source_kwargs["style"] = "dashed"

                # title = f"{node.node_type()}: {node_syms[node.node_id]}"
                title = node.node_type()

                dot.add_edge(
                    pydot.Edge(
                        str(source),
                        str(target),
                        label=title,
                        **source_kwargs,
                    ),
                )

            else:
                # there's a dot-node for this edge.

                if node.EdgeControl.INVERT_DEPENDENCY_FLOW:
                    source_kwargs = dict(
                        dir="both",
                        arrowtail="normal",
                        arrowhead="odot",
                    )
                    target_kwargs = dict(
                        dir="both",
                        arrowtail="dot",
                        arrowhead="oinv",
                    )
                else:
                    source_kwargs = dict(
                        dir="both",
                        arrowtail="oinv",
                        arrowhead="dot",
                    )
                    target_kwargs = dict(
                        dir="both",
                        arrowtail="odot",
                    )

                if node.EdgeControl.RELAX_EDGE:
                    source_kwargs["constraint"] = "false"
                    target_kwargs["constraint"] = "false"

                dot.add_edge(
                    pydot.Edge(
                        str(source),
                        str(node.node_id),
                        **source_kwargs,
                    ),
                )
                dot.add_edge(
                    pydot.Edge(
                        str(node.node_id),
                        str(target),
                        **target_kwargs,
                    ),
                )

        if self.observed:
            dot.add_node(
                pydot.Node(
                    "Observer",
                    color="#E6B0AA",
                    style="filled",
                ),
            )
            for node_id in self.observed:
                dot.add_edge(
                    pydot.Edge(
                        "Observer",
                        str(node_id),
                    ),
                )

        return dot


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorValue(TapestryNode):
    class NodeControl(TapestryNode.NodeControl):
        BG_COLOR = "#F5EEf8"

    shape: zspace.ZArray
    dtype: DType

    def __post_init__(self):
        self.shape = zspace.as_zarray(self.shape)


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorShard(TensorValue):
    slice: zspace.ZRange

    def validate(self) -> None:
        super().validate()
        if self.slice not in ZRange(self.shape):
            raise ValueError(
                f"{self.node_type()} slice ({self.slice}) ∉ shape {self.shape}"
            )


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class AggregateTensor(TensorValue):
    @dataclass(kw_only=True)
    class Aggregates(TapestryEdge):
        class EdgeControl(TapestryEdge.EdgeControl):
            SOURCE_TYPE: "AggregateTensor"
            TARGET_TYPE: TensorShard
            DISPLAY_ATTRIBUTES = False

        class Meta(TapestryEdge.Meta):
            ordered = True

        def validate(self) -> None:
            super().validate()
            value = self.source(AggregateTensor)
            shard = self.target(TensorShard)
            if shard.slice not in ZRange(value.shape):
                raise ValueError(f"{self.node_type()} shard ({shard}) ∉ value {value}")

    def shards(self) -> List[TensorShard]:
        return [
            e.target(TensorShard)
            for e in self.assert_graph().list_edges(
                source_id=self.node_id,
                edge_type=AggregateTensor.Aggregates,
            )
        ]


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class PinnedTensor(TensorValue):
    class NodeControl(TapestryNode.NodeControl):
        BG_COLOR = "#C39BD3"

    storage: str


@dataclass(kw_only=True)
class BlockOpBindingEdgeBase(TapestryEdge):
    class EdgeControl(TapestryEdge.EdgeControl):
        SOURCE_TYPE: "BlockOperation"
        TARGET_TYPE: TensorValue

    class Meta(TapestryEdge.Meta):
        ordered = True

    # force name to be required
    name: str
    selector: zspace.ZRangeMap

    def __post_init__(self):
        assert self.name

    def validate(self) -> None:
        op = self.source(BlockOperation)
        tensor = self.target(TensorValue)

        try:
            result_range = self.selector(op.index_space)
        except ValueError as e:
            raise AssertionError(
                f"{op.name} :: {self.node_type()}[{self.name}] Selector incompatible with index_space:\n"
                f"  Index Space: {repr(op.index_space)}\n"
                f"{op.pretty(prefix='    > ')}\n"
                f"  Selector: {repr(self.selector.transform)}\n"
                f"{self.pretty(prefix='    > ')}"
            ) from e

        if len(result_range.start) != len(tensor.shape):
            raise AssertionError(
                f"{self.node_type()} Selector Dimension Miss-match:\n"
                f"  Index Space: {repr(op.index_space)}\n"
                f"  Selector: {repr(self.selector.transform)}\n"
                f"  Selected Space: {repr(result_range)}\n"
                f"  Tensor: {tensor.shape}\n\n"
                f"  {repr(self)}"
            )


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorIOBase(TapestryEdge):
    class EdgeControl(TapestryEdge.EdgeControl):
        TARGET_TYPE: TensorValue

    name: str
    slice: zspace.ZRange


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class ReadSlice(TensorIOBase):
    class NodeControl(TapestryNode.NodeControl):
        BG_COLOR = "#E8F8F5"


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class WriteSlice(TensorIOBase):
    class NodeControl(TapestryNode.NodeControl):
        BG_COLOR = "#F9EBEA"

    class EdgeControl(TensorIOBase.EdgeControl):
        INVERT_DEPENDENCY_FLOW = True


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class BlockOperation(TapestryNode):
    class NodeControl(TapestryNode.NodeControl):
        BG_COLOR = "#A9CCE3"

    class Meta(TapestryNode.Meta):
        ordered = True

    operation: str
    index_space: zspace.ZRange

    # costs must map to dim-1
    memory_cost: zspace.ZTransform
    compute_cost: zspace.ZTransform

    def validate(self) -> None:
        super().validate()
        if self.memory_cost.out_dim != 1:
            raise ValueError(
                f"Memory costs must map to a 1-dim space: {repr(self.memory_cost)}",
            )
        if self.compute_cost.out_dim != 1:
            raise ValueError(
                f"Compute costs must map to a 1-dim space: {repr(self.compute_cost)}",
            )

        if not (self.memory_cost.marginal_strides() >= 0).all():
            raise ValueError(
                f"Marginal memory costs must be positive: {repr(self.memory_cost)}",
            )
        if not (self.compute_cost.marginal_strides() >= 0).all():
            raise ValueError(
                f"Marginal compute costs must be positive: {repr(self.compute_cost)}",
            )

    # TODO: this kinda wants to be a "Tag"; which could be a base-class of TapestryEdge.
    # A tag could easily have only a target (or only a source?)
    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class SectionPlan(TapestryNode):
        class Meta(TapestryEdge.Meta):
            ordered = True

        class NodeControl(TapestryNode.NodeControl):
            BG_COLOR = "white"

        sections: zspace.ZArray

    @dataclass(kw_only=True)
    class Sections(TapestryEdge):
        class EdgeControl(TapestryEdge.EdgeControl):
            SOURCE_TYPE: "BlockOperation.SectionPlan"
            TARGET_TYPE: "BlockOperation"
            DISPLAY_ATTRIBUTES = False

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Shard(TapestryNode):
        class Meta(TapestryNode.Meta):
            ordered = True

        class NodeControl(TapestryNode.NodeControl):
            BG_COLOR = "#EAF2F8"

        index_slice: zspace.ZRange
        operation: str
        memory_cost: int
        compute_cost: int

        def inputs(self) -> List[ReadSlice]:
            return self.assert_graph().list_edges(
                edge_type=ReadSlice,
                source_id=self.node_id,
            )

        def results(self) -> List[WriteSlice]:
            return self.assert_graph().list_edges(
                edge_type=WriteSlice,
                source_id=self.node_id,
            )

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Selection(TapestryEdge):
        class EdgeControl(TapestryEdge.EdgeControl):
            # SOURCE_TYPE: "BlockOperation.Partition"
            # TARGET_TYPE: BlockOpBindingEdgeBase
            DISPLAY_ATTRIBUTES = False
            RELAX_EDGE = False

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Partition(TapestryEdge):
        class EdgeControl(TapestryEdge.EdgeControl):
            SOURCE_TYPE: "BlockOperation"
            TARGET_TYPE: "BlockOperation.Shard"
            INVERT_DEPENDENCY_FLOW = True
            DISPLAY_ATTRIBUTES = False
            RELAX_EDGE = False

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Input(BlockOpBindingEdgeBase):
        class NodeControl(TapestryNode.NodeControl):
            BG_COLOR = "#A3E4D7"

    @marshmallow_dataclass.add_schema
    @dataclass(kw_only=True)
    class Result(BlockOpBindingEdgeBase):
        class NodeControl(TapestryNode.NodeControl):
            BG_COLOR = "#E6B0AA"

        class EdgeControl(BlockOpBindingEdgeBase.EdgeControl):
            INVERT_DEPENDENCY_FLOW = True

    def inputs(self) -> List[Input]:
        return self.assert_graph().list_edges(
            edge_type=BlockOperation.Input,
            source_id=self.node_id,
        )

    def results(self) -> List[Result]:
        return self.assert_graph().list_edges(
            edge_type=BlockOperation.Result,
            source_id=self.node_id,
        )

    def bind_input(
        self,
        *,
        name: str,
        value: NodeIdCoercible,
        selector: ZRangeMap,
    ) -> Input:
        graph = self.assert_graph()
        value_id = coerce_node_id(value)

        # TODO: verify this is legal.

        return graph.add_node(
            BlockOperation.Input(
                name=name,
                source_id=self.node_id,
                target_id=value_id,
                selector=selector,
            )
        )

    def bind_result(
        self,
        *,
        name: str,
        selector: ZRangeMap,
        dtype: torch.dtype = torch.float16,
    ) -> AggregateTensor:
        graph = self.assert_graph()

        value = graph.add_node(
            AggregateTensor(
                name=name,
                shape=selector(self.index_space).end,
                dtype=dtype,
            )
        )

        graph.add_node(
            BlockOperation.Result(
                source_id=self.node_id,
                target_id=value.node_id,
                selector=selector,
                name=name,
            ),
        )

        return value

    def attach_section_plan(self, sections) -> SectionPlan:
        sections = as_zarray(sections)
        g = self.assert_graph()

        sp = g.add_node(
            BlockOperation.SectionPlan(
                sections=sections,
            )
        )

        g.add_node(
            BlockOperation.Sections(
                source_id=sp.node_id,
                target_id=self.node_id,
            )
        )

        return sp

    def add_shard(self, index_slice: ZRange) -> Shard:
        graph = self.assert_graph()

        if index_slice not in self.index_space:
            raise AssertionError(
                f"{index_slice = } does not fit in {self.index_space =}",
            )

        shard = graph.add_node(
            BlockOperation.Shard(
                index_slice=index_slice,
                name=self.name,
                operation=self.operation,
                compute_cost=self.compute_cost(index_slice.shape).item(),
                memory_cost=self.memory_cost(index_slice.shape).item(),
            )
        )

        graph.add_node(
            BlockOperation.Partition(
                source_id=self.node_id,
                target_id=shard.node_id,
            )
        )

        for input_edge in self.inputs():
            read_edge = graph.add_node(
                ReadSlice(
                    source_id=shard.node_id,
                    target_id=input_edge.target_id,
                    slice=input_edge.selector(index_slice),
                    name=input_edge.name,
                )
            )
            graph.add_node(
                BlockOperation.Selection(
                    source_id=read_edge.node_id,
                    target_id=input_edge.node_id,
                )
            )

        for result_edge in self.results():
            result = result_edge.target(AggregateTensor)
            slice = result_edge.selector(index_slice)
            tensor_shard = graph.add_node(
                TensorShard(
                    name=result_edge.name,
                    shape=result.shape,
                    dtype=result.dtype,
                    slice=slice,
                )
            )
            write_edge = graph.add_node(
                WriteSlice(
                    name=result_edge.name,
                    source_id=shard.node_id,
                    target_id=tensor_shard.node_id,
                    slice=slice,
                )
            )
            graph.add_node(
                BlockOperation.Selection(
                    source_id=write_edge.node_id,
                    target_id=result_edge.node_id,
                )
            )
            graph.add_node(
                AggregateTensor.Aggregates(
                    source_id=result.node_id,
                    target_id=tensor_shard.node_id,
                )
            )

        return shard

    def get_shards(self) -> List[Shard]:
        return [
            partition.target(BlockOperation.Shard)
            for partition in self.assert_graph().list_edges(
                edge_type=BlockOperation.Partition,
                source_id=self.node_id,
            )
        ]
