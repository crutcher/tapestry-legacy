import copy
import re
import typing
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID

import marshmallow

if typing.TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from marshmallow_dataclass import dataclass

K = TypeVar("K")
V = TypeVar("V")


def ensure_dict(val: Optional[Mapping[K, V]]) -> Dict[K, V]:
    if val is None:
        val = {}
    return dict(val)


def ensure_list(val: Optional[Sequence[V]]) -> List[V]:
    if val is None:
        val = []
    return list(val)


def ensure_uuid(val: Optional[UUID] = None) -> UUID:
    if val is None:
        return uuid.uuid4()
    return val


C = TypeVar("C", bound="JsonSerializable")


class JsonSerializable:
    class Meta:
        ordered = True

    @classmethod
    def get_schema(cls) -> marshmallow.Schema:
        return getattr(cls, "Schema")()

    def dump_json_data(self) -> Any:
        return self.get_schema().dump(self)

    def dump_json_str(
        self,
        *,
        indent: Optional[int] = None,
    ) -> Any:
        return self.get_schema().dumps(self, indent=indent)

    @classmethod
    def load_json_data(cls: Type[C], data: Any) -> C:
        return cls.get_schema().load(data)

    @classmethod
    def load_json_str(cls: Type[C], data: str) -> C:
        return cls.get_schema().loads(data)

    def __str__(self):
        return self.dump_json_str()

    def pretty(self):
        return self.dump_json_str(indent=2)


TYPE_NAME_PATTERN = r"^[a-zA-Z][_a-zA-Z0-9]*(\.[a-zA-Z][_a-zA-Z0-9]*)*$"
TYPE_NAME_REGEX = re.compile(TYPE_NAME_PATTERN)


def assert_valid_typename(type_name: str) -> str:
    if not TYPE_NAME_REGEX.fullmatch(type_name):
        raise ValueError(
            f'Invalid tapestry typename, expected "{TYPE_NAME_PATTERN}", found: {type_name}'
        )

    return type_name


@dataclass(frozen=True)
class TapestryTargetDoc(JsonSerializable):
    _target_: UUID


@dataclass
class TapestryNodeDoc(JsonSerializable):
    """
    Serialization format for Tapestry Nodes.
    """

    id: UUID
    type: str
    fields: Dict[str, Any]

    def __init__(
        self,
        *,
        id: Optional[UUID] = None,
        type: str,
        fields: Optional[Mapping[str, Any]] = None,
    ):
        self.id = ensure_uuid(id)
        self.type = assert_valid_typename(type)
        self.fields = copy.deepcopy(ensure_dict(fields))


@dataclass(frozen=True)
class TapestryEdgeDoc(JsonSerializable):
    type: str
    source: UUID
    target: UUID

    def __post_init__(self):
        assert_valid_typename(self.type)


def to_set_and_duplicates(items: Iterable[V]) -> Tuple[Set[V], Set[V]]:
    results: Set[V] = set()
    duplicates: Set[V] = set()
    last_size: int = len(results)

    for x in items:
        results.add(x)
        size = len(results)
        if size == last_size:
            duplicates.add(x)

        last_size = size

    return results, duplicates


@dataclass
class TapestryGraphDoc(JsonSerializable):
    """
    Serialization format for Tapestry Graphs.
    """

    id: UUID
    nodes: List[TapestryNodeDoc]
    edges: List[TapestryEdgeDoc]

    _node_map: Dict[UUID, TapestryNodeDoc]
    _edge_set: Set[TapestryEdgeDoc]

    class Meta:
        "Marshmallow Schema control parameters."
        exclude = (
            "_node_map",
            "_edge_set",
        )

    def __init__(
        self,
        *,
        id: Optional[UUID] = None,
        nodes: Optional[Sequence[TapestryNodeDoc]] = None,
        edges: Optional[Sequence[TapestryEdgeDoc]] = None,
    ):
        self.id = ensure_uuid(id)

        self.nodes = []
        self._node_map = {}

        if nodes:
            for n in nodes:
                self.add_node(n)

        self.edges = []
        self._edge_set = set()

        if edges:
            for e in edges:
                self.add_edge(e)

    def new_id(self) -> uuid.UUID:
        """
        Allocate a new id not in use in the graph.
        """
        while True:
            id = uuid.uuid4()
            if id not in self.nodes:
                return id

    def add_node(self, node: TapestryNodeDoc) -> None:
        """
        Add a node to the graph.

        The id must not already be present.

        :param node: node to add.
        """
        if node.id in self._node_map:
            raise ValueError(f"Node already in graph: {node.id}")

        self.nodes.append(node)
        self._node_map[node.id] = node

    def remove_node(
        self,
        node: Union[uuid.UUID | TapestryNodeDoc],
        *,
        remove_links: bool = False,
    ) -> None:
        """
        Remove a node from the graph.

        :param node: the node or id to remove.
        :param remove_links: if true, removes all edges to/from the node;
          if false, it is an error to remove a node that still has edges.
        """
        if isinstance(node, TapestryNodeDoc):
            node = node.id

        if node not in self._node_map:
            raise ValueError(f"Node not in graph: {node}")

        del_list = []
        for e in self.edges:
            if node in (e.source, e.target):
                del_list.append(e)

        if del_list and remove_links == False:
            raise ValueError(f"Node still has pending links: {del_list}")

        if remove_links:
            for e in del_list:
                self.remove_edge(e)

        obj = [x for x in self.nodes if x.id == node][0]
        self.nodes.remove(obj)
        del self._node_map[node]

    def add_edge(
        self,
        edge: Optional[TapestryEdgeDoc] = None,
        *,
        type: Optional[str] = None,
        source: Optional[Union[uuid.UUID, TapestryNodeDoc]] = None,
        target: Optional[Union[uuid.UUID, TapestryNodeDoc]] = None,
    ) -> TapestryEdgeDoc:
        """
        Add an edge to the graph.

        :param edge:
        :return: the new edge.
        """
        if isinstance(source, TapestryNodeDoc):
            source = source.id
        if isinstance(target, TapestryNodeDoc):
            target = target.id

        if edge and (type or source or target):
            raise ValueError("edge and (type, source, target) are mutually exclusive.")
        if not edge:
            if type and source and target:
                edge = TapestryEdgeDoc(type=type, source=source, target=target)
            else:
                raise ValueError("(type, source, target) are all required")

        if edge.source not in self._node_map:
            raise ValueError(f"Edge source node not in graph: {edge.source}")
        if edge.target not in self._node_map:
            raise ValueError(f"Edge target node not in graph: {edge.target}")

        if edge in self._edge_set:
            raise ValueError(f"Edge already in graph: {edge}")

        self.edges.append(edge)
        self._edge_set.add(edge)

        return edge

    def find_edges(
        self,
        *,
        types: Optional[Union[str, Iterable[str]]] = None,
        sources: Optional[
            Union[
                uuid.UUID,
                TapestryNodeDoc,
                Iterable[Union[uuid.UUID, TapestryNodeDoc]],
            ]
        ] = None,
        targets: Optional[
            Union[
                uuid.UUID,
                TapestryNodeDoc,
                Iterable[Union[uuid.UUID, TapestryNodeDoc]],
            ]
        ] = None,
    ) -> List[TapestryEdgeDoc]:
        """
        Find all edges matching the parameters.

        If a parameter is omitted, will match any edge with that parameter;
        if a parameter is provided, must match one of the parameter's options.

        :param types: (Optional) the targets.
        :param sources: (Optional) the sources (ids or nodes).
        :param targets: (Optional) the targets (ids or nodes).
        :return: a list of matching nodes.
        """

        def resolve_ids(items) -> Set[uuid.UUID]:
            if items is None:
                return set()

            if isinstance(items, (TapestryNodeDoc, uuid.UUID)):
                items = [items]

            result = set()
            for item in items:
                if isinstance(item, TapestryNodeDoc):
                    item = item.id

                if not isinstance(item, uuid.UUID):
                    raise ValueError(f"Item unknown type: {type(item)}:: {item}")

                result.add(item)

            return result

        items = self.edges

        if sources:
            sources = resolve_ids(sources)
            items = [n for n in items if n.source in sources]

        if targets:
            targets = resolve_ids(targets)
            items = [n for n in items if n.target in targets]

        if types:
            if isinstance(types, str):
                types = {types}
            types = set(types)
            items = [n for n in items if n.type in types]

        return items

    def remove_edge(self, edge: TapestryEdgeDoc) -> None:
        """
        Remove an edge from the graph.

        :param edge: the edge to remove.
        """
        try:
            self.edges.remove(edge)
            self._edge_set.remove(edge)
        except ValueError:
            raise ValueError(f"Edge not in graph: {edge}")
