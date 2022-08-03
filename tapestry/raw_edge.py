import uuid
from dataclasses import field
from typing import Dict, List, Optional

from marshmallow_dataclass import dataclass

from tapestry.serialization.json import JsonSerializable
from tapestry.zspace import ZAffineMap, ZRange, ZRangeMap


@dataclass
class GraphNode(JsonSerializable):
    id: uuid.UUID


@dataclass
class TensorSource(GraphNode):
    pass


@dataclass
class TensorValue(TensorSource):
    pass


@dataclass
class TensorView(TensorSource):
    sources: List[TensorSource]
    concat_dim: int = 0
    coord_map: Optional[ZAffineMap] = None


@dataclass
class OperationNode(GraphNode):
    pass


@dataclass
class BlockOperation(OperationNode):
    operator: str
    id: uuid.UUID
    inputs: Dict[str, TensorSource] = field(default_factory=dict)
    outputs: Dict[str, TensorValue] = field(default_factory=dict)


def raw():
    i = ZRange([3, 4], start=[0, 1])

    m = ZAffineMap(
        projection=[[1, 0], [0, 1]],
        offset=[3, 4],
    )

    m(i.inclusive_corners())

    r = ZRangeMap(
        zaffine_map=m,
        shape=[1, 2],
    )

    w = r.range_to_range(i)
    print(w)


if __name__ == "__main__":
    raw()
