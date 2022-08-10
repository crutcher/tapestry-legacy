import copy
import typing
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Type, TypeVar, Union, cast

import marshmallow
import marshmallow_dataclass
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema
from overrides import overrides

from tapestry.serialization.json_serializable import JsonDumpable, JsonLoadable
from tapestry.type_utils import UUIDConvertable, coerce_uuid, dict_to_parameter_str

NODE_TYPE_FIELD = "__type__"
EDGES_FIELD = "__edges__"


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class NodeAttributes(JsonLoadable):
    node_id: uuid.UUID = field(default_factory=uuid.uuid4)
    """Unique in a document."""

    display_name: Optional[str] = None
    """May be repeated in a document."""


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class EdgeAttributes(NodeAttributes):
    source_node_id: uuid.UUID
    target_node_id: uuid.UUID


W = TypeVar("W", bound="NodeHandle")


# See forward references:
#  * TypeVar W
class NodeHandle:
    """
    Base class for node handles of NodeAttributes.

    Provides reference to the containing `.graph`, and the embedded `.attrs`.

    Equality and hashing is via node_id.
    """

    graph: "GraphHandle"
    """
    The attached GraphHandle.
    
    As wrapping nodes are transient, and the GraphHandle has no reference to them,
    only the underlying NodeAttrsDocs (which are not recursive); this does not
    need to be a weakref.
    """

    attrs: NodeAttributes
    """
    The wrapped NodeAttributes.
    
    The type of this annotation is used to determine if a subclass is permitted
    to wrap documents. Subclasses which wrap specialized NodeAttributes subclasses
    should declare a more restrictive type for `.attrs`.
    
    This means that typed handles subclasses have typed attrs.
    """

    def __init__(
        self,
        *,
        graph: "GraphHandle",
        attrs: NodeAttributes,
    ):
        self.graph = graph

        self.assert_wraps_doc_type(type(attrs))
        self.attrs = attrs

    def __eq__(self, other):
        if not isinstance(other, NodeHandle):
            return False

        return self.node_id == other.attrs.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    @classmethod
    def wraps_doc_type(
        cls,
        doc_type: Union[NodeAttributes, Type[NodeAttributes]],
    ) -> bool:
        """
        Will this NodeHandle subclass wrap the given NodeAttributes subclass?

        :param doc_type: the NodeAttributes type to test.
        :return: True if compatible.
        """
        if isinstance(doc_type, NodeAttributes):
            doc_type = type(doc_type)

        attrs_type = typing.get_type_hints(cls)["attrs"]

        return issubclass(doc_type, attrs_type)

    @classmethod
    def assert_wraps_doc_type(
        cls,
        doc_type: Union[NodeAttributes, Type[NodeAttributes]],
    ) -> None:
        """
        Assert that this NodeHandle subclass will wrap the given NodeAttributes subclass.

        :param doc_type: the NodeAttributes type to test.
        :raises ValueError: on type incompatibility.
        :raises AssertionError: if doc_type is not a NodeAttributes.
        """
        if isinstance(doc_type, NodeAttributes):
            doc_type = type(doc_type)

        if not issubclass(doc_type, NodeAttributes):
            raise AssertionError(
                f"{doc_type.__name__} is not a subclass of NodeAttributes",
            )

        if not cls.wraps_doc_type(doc_type):
            raise ValueError(f"{cls.__name__} cannot wrap {doc_type.__name__}")

    @property
    def node_id(self) -> uuid.UUID:
        """
        The node node_id.
        """
        return self.attrs.node_id

    @property
    def doc_type(self) -> Type[NodeAttributes]:
        """
        The type of the attrs.
        """
        return type(self.attrs)

    def __repr__(self):
        return (
            f"{type(self).__name__}[{self.doc_type.__name__}]"
            f"({dict_to_parameter_str(asdict(self.attrs))})"
        )

    def __str__(self):
        return repr(self)

    def force_as_type(self, handle_type: Type[W]) -> W:
        """
        Construct a new wrapper for `.attrs` of the given type, or throw.

        :param handle_type: the target wrapper type.
        :return: the new wrapper.
        :raises ValueError: if the type cannot wrap `.attrs`.
        """
        return handle_type(
            graph=self.graph,
            attrs=self.attrs,
        )

    def try_as_type(self, handle_type: Type[W]) -> Optional[W]:
        """
        Try to construct a new wrapper for `.attrs`, or return None.

        The assignment operator (:=) permits simple typed flow dispatch:

        >>> node: XyzzyWrapper
        >>> if foo := node.try_as_type(FooWrapper):
        >>>    do_foo_thing(foo.attrs.foo_attr)
        >>> elif bar := node.try_as_type(BarWrapper):
        >>>    do_bar_thing(bar.attrs.bar_attr)
        >>> else:
        >>>    do_xyzzy_thing(node)

        :param handle_type: the target wrapper type.
        :return: the new wrapper, or None.
        """
        if not handle_type.wraps_doc_type(self.attrs):
            return None

        return handle_type(
            graph=self.graph,
            attrs=self.attrs,
        )


@dataclass
class GraphDoc(JsonDumpable):
    """
    Serializable NodeAttributes graph.
    """

    nodes: Dict[uuid.UUID, NodeAttributes]

    @classmethod
    def build_load_schema(
        cls,
        node_types: Iterable[Type[NodeAttributes]],
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
                cls.__name__: cast(Type[NodeAttributes], cls).get_load_schema()
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
        nodes: Optional[Dict[uuid.UUID, NodeAttributes]] = None,
    ):
        self.nodes = {}
        if nodes is not None:
            for n in nodes.values():
                self.add_node(n)

    def add_node(self, node: NodeAttributes) -> None:
        """
        Add a node to the document.

        EdgeAttributes nodes are validated that their `.source_node_id` and `.target_node_id`
        appear in the graph, and are not edges.

        :param node: the node.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already in graph.")

        if isinstance(node, EdgeAttributes):
            for port in ("source_node_id", "target_node_id"):
                port_id = getattr(node, port)
                if port_id not in self.nodes:
                    raise ValueError(
                        f"Edge {port}({port_id}) not in graph:\n\n{repr(node)}",
                    )

                port_node = self.nodes[port_id]
                if isinstance(port_node, EdgeAttributes):
                    raise ValueError(
                        f"Edge {port}({port_id}) is an edge:\n\n{repr(node)}"
                        f"\n\n ==[{port}]==>\n\n{repr(port_node)}",
                    )

        self.nodes[node.node_id] = node

    def assert_node_types(
        self,
        node_types: Iterable[Type[NodeAttributes]],
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


class GraphHandle:
    doc: GraphDoc

    def __init__(
        self,
        doc: Optional[GraphDoc] = None,
    ):
        if doc is None:
            doc = GraphDoc()
        self.doc = doc

    def __repr__(self):
        return f"{type(self).__name__}(doc={repr(self.doc)})"

    def __str__(self):
        return str(self.doc.pretty())

    def all_handles_of_type(self, handle_type: Type[W]) -> List[W]:
        """
        List all nodes wrappable by the given wrapper type.

        :param handle_type: the node wrapper type.
        :return: a list of nodes.
        """
        if not issubclass(handle_type, NodeHandle):
            raise AssertionError(
                f"Class {handle_type} is not a subclass of {NodeHandle}"
            )
        return [
            handle_type(
                graph=self,
                attrs=node_doc,
            )
            for node_doc in self.doc.nodes.values()
            if handle_type.wraps_doc_type(type(node_doc))
        ]

    def get_node(
        self,
        node_id: UUIDConvertable,
        handle_type: Type[W],
    ) -> W:
        """
        Find a NodeAttributes by node_id, and wrap it in the given node type.

        :param node_id: the node_id, may be str or UUID.
        :param handle_type: the wrapper type.
        :return: the wrapped node.
        """
        node_id = coerce_uuid(node_id)

        if not issubclass(handle_type, NodeHandle):
            raise AssertionError(
                f"{handle_type} is not a subclass of {NodeHandle}",
            )

        return handle_type(
            graph=self,
            attrs=self.doc.nodes[node_id],
        )


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorSource(NodeAttributes):
    class Handle(NodeHandle):
        attrs: "TensorSource"


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class TensorValue(TensorSource):
    class Handle(TensorSource.Handle):
        attrs: "TensorValue"


@marshmallow_dataclass.add_schema
@dataclass(kw_only=True)
class ExternalTensorValue(TensorValue):
    class Handle(TensorValue.Handle):
        attrs: "ExternalTensorValue"

    storage: str
