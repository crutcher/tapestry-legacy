import typing
import uuid
from dataclasses import asdict
from typing import List, Optional, Type, TypeVar, Union

from tapestry.attrs_docs import (
    ExternalTensorValueAttrs,
    NodeAttrsDoc,
    OpGraphDoc,
    TensorSourceAttrs,
    TensorValueAttrs,
)
from tapestry.type_utils import UUIDConvertable, coerce_uuid, dict_to_parameter_str

W = TypeVar("W", bound="NodeWrapper")


# See forward references:
#  * TypeVar W
class NodeWrapper:
    """
    Base class for node wrappers of NodeAttrsDoc.

    Provides reference to the containing `.graph`, and the embedded `.attrs`.

    Equality and hashing is via node_id.
    """

    graph: "ExpressionGraph"
    """
    The attached ExpressionGraph.
    
    As wrapping nodes are transient, and the ExpressionGraph has no reference to them,
    only the underlying NodeAttrsDocs (which are not recursive); this does not
    need to be a weakref.
    """

    attrs: NodeAttrsDoc
    """
    The wrapped NodeAttrsDoc.
    
    The type of this annotation is used to determine if a subclass is permitted
    to wrap documents. Subclasses which wrap specialized NodeAttrsDoc subclasses
    should declare a more restrictive type for `.attrs`.
    
    This means that typed wrapper subclasses have typed attrs.
    """

    def __init__(
        self,
        *,
        graph: "ExpressionGraph",
        attrs: NodeAttrsDoc,
    ):
        self.graph = graph

        self.assert_wraps_doc_type(type(attrs))
        self.attrs = attrs

    def __eq__(self, other):
        if not isinstance(other, NodeWrapper):
            return False

        return self.node_id == other.attrs.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    @classmethod
    def wraps_doc_type(
        cls,
        doc_type: Union[NodeAttrsDoc, Type[NodeAttrsDoc]],
    ) -> bool:
        """
        Will this NodeWrapper subclass wrap the given NodeAttrsDoc subclass?

        :param doc_type: the NodeAttrsDoc type to test.
        :return: True if compatible.
        """
        if isinstance(doc_type, NodeAttrsDoc):
            doc_type = type(doc_type)

        attrs_type = typing.get_type_hints(cls)["attrs"]

        return issubclass(doc_type, attrs_type)

    @classmethod
    def assert_wraps_doc_type(
        cls,
        doc_type: Union[NodeAttrsDoc, Type[NodeAttrsDoc]],
    ) -> None:
        """
        Assert that this NodeWrapper subclass will wrap the given NodeAttrsDoc subclass.

        :param doc_type: the NodeAttrsDoc type to test.
        :raises ValueError: on type incompatibility.
        :raises AssertionError: if doc_type is not a NodeAttrsDoc.
        """
        if isinstance(doc_type, NodeAttrsDoc):
            doc_type = type(doc_type)

        if not issubclass(doc_type, NodeAttrsDoc):
            raise AssertionError(
                f"{doc_type.__name__} is not a subclass of NodeAttrsDoc",
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
    def doc_type(self) -> Type[NodeAttrsDoc]:
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

    def force_as_type(self, wrapper_type: Type[W]) -> W:
        """
        Construct a new wrapper for `.attrs` of the given type, or throw.

        :param wrapper_type: the target wrapper type.
        :return: the new wrapper.
        :raises ValueError: if the type cannot wrap `.attrs`.
        """
        return wrapper_type(
            graph=self.graph,
            attrs=self.attrs,
        )

    def try_as_type(self, wrapper_type: Type[W]) -> Optional[W]:
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

        :param wrapper_type: the target wrapper type.
        :return: the new wrapper, or None.
        """
        if not wrapper_type.wraps_doc_type(self.attrs):
            return None

        return wrapper_type(
            graph=self.graph,
            attrs=self.attrs,
        )


class TensorSource(NodeWrapper):
    attrs: TensorSourceAttrs


class TensorValue(TensorSource):
    attrs: TensorValueAttrs


class ExternalTensorValue(TensorValue):
    attrs: ExternalTensorValueAttrs


# See forward references:
#  * NodeWrapper
class ExpressionGraph:
    doc: OpGraphDoc

    def __init__(
        self,
        doc: Optional[OpGraphDoc] = None,
    ):
        if doc is None:
            doc = OpGraphDoc()
        self.doc = doc

    def __repr__(self):
        return f"ExpressionGraph(doc={repr(self.doc)})"

    def __str__(self):
        return str(self.doc.pretty())

    def list_nodes_of_type(self, wrapper_type: Type[W]) -> List[W]:
        """
        List all nodes wrappable by the given wrapper type.

        :param wrapper_type: the node wrapper type.
        :return: a list of nodes.
        """
        return [
            wrapper_type(
                graph=self,
                attrs=node_doc,
            )
            for node_doc in self.doc.nodes.values()
            if wrapper_type.wraps_doc_type(type(node_doc))
        ]

    def get_node(
        self,
        node_id: UUIDConvertable,
        wrapper_type: Type[W],
    ) -> W:
        """
        Find a NodeAttrsDoc by node_id, and wrap it in the given node type.

        :param node_id: the node_id, may be str or UUID.
        :param wrapper_type: the wrapper type.
        :return: the wrapped node.
        """
        node_id = coerce_uuid(node_id)

        if not issubclass(wrapper_type, NodeWrapper):
            raise AssertionError(
                f"{wrapper_type} is not a subclass of {NodeWrapper}",
            )

        return wrapper_type(
            graph=self,
            attrs=self.doc.nodes[node_id],
        )
