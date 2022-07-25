import re
import uuid
from typing import Dict, Iterable, Optional
from uuid import UUID


def ensure_uuid(val: Optional[UUID] = None) -> UUID:
    if val is None:
        return uuid.uuid4()
    return val


TYPE_NAME_PATTERN = "^[a-zA-Z][_a-zA-Z0-9]*(\.[a-zA-Z][_a-zA-Z0-9]*)*$"
TYPE_NAME_REGEX = re.compile(TYPE_NAME_PATTERN)


class TapestryType:
    _type_name: str

    def __init__(
        self,
        *,
        type_name: str,
    ):
        if not re.fullmatch(TYPE_NAME_REGEX, type_name):
            raise AssertionError(
                f'Illegal type name, found "{type_name}", expected: {TYPE_NAME_PATTERN}'
            )
        self._type_name = type_name

    def name(self) -> str:
        return self._type_name


class TapestryNode:
    _node_id: UUID
    _node_type: TapestryType

    def __init__(
        self,
        *,
        node_id: Optional[UUID] = None,
        node_type: TapestryType,
    ):
        self._node_id = ensure_uuid(node_id)
        self._node_type = node_type

    def node_id(self) -> UUID:
        return self._node_id

    def node_type(self) -> TapestryType:
        return self._node_type


class TapestryGraph:
    _graph_id: UUID

    # graph meta?
    # _graph_type: TapestryType

    _nodes: Dict[UUID, TapestryNode]

    def __init__(
        self,
        *,
        graph_id: Optional[UUID] = None,
    ):
        self._graph_id = ensure_uuid(graph_id)
        self._nodes = {}

    def add_node(self, node: TapestryNode) -> None:
        node_id = node.node_id()

        if node_id in self._nodes:
            raise AssertionError(f"Node {node_id} already in graph.")

        self._nodes[node_id] = node

    def nodes_view(self) -> Iterable[TapestryNode]:
        return self._nodes.values()
