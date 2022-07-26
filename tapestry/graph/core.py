import re
import uuid
from dataclasses import field
from typing import Any, Dict, Iterable, Mapping, Optional, Type, TypeVar
from uuid import UUID

import marshmallow
from marshmallow_dataclass import dataclass

K = TypeVar("K")
V = TypeVar("V")


def ensure_dict(val: Optional[Mapping[K, V]]) -> Dict[K, V]:
    if val is None:
        val = {}
    return dict(val)


def ensure_uuid(val: Optional[UUID] = None) -> UUID:
    if val is None:
        return uuid.uuid4()
    return val


TYPE_NAME_PATTERN = r"^[a-zA-Z][_a-zA-Z0-9]*(\.[a-zA-Z][_a-zA-Z0-9]*)*$"
TYPE_NAME_REGEX = re.compile(TYPE_NAME_PATTERN)

C = TypeVar("C", bound="JsonSerializable")


class JsonSerializable:
    @classmethod
    def get_schema(cls) -> marshmallow.Schema:
        return getattr(cls, "Schema")()

    def to_json_data(self) -> Any:
        return self.get_schema().dump(self)

    @classmethod
    def from_json_data(cls: Type[C], data: Any) -> C:
        return cls.get_schema().load(data)


@dataclass
class TapestryNodeDoc(JsonSerializable):
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
        self.type = type
        self.fields = ensure_dict(fields)


@dataclass
class TapestryGraphDoc(JsonSerializable):
    id: UUID

    # graph meta?
    # _graph_type: TapestryType

    nodes: Dict[UUID, TapestryNodeDoc] = field(default_factory=dict)

    def __init__(
        self,
        *,
        id: Optional[UUID] = None,
        nodes: Optional[Mapping[UUID, TapestryNodeDoc]] = None,
    ):
        self.id = ensure_uuid(id)
        self.nodes = ensure_dict(nodes)

    def add_node(self, node: TapestryNodeDoc) -> None:
        if node.id in self.nodes:
            raise AssertionError(f"Node {node.id} already in graph.")

        self.nodes[node.id] = node

    def nodes_view(self) -> Iterable[TapestryNodeDoc]:
        return self.nodes.values()
