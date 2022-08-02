import uuid
from dataclasses import field
from typing import Dict, List, Optional

import numpy as np
from marshmallow import fields
from marshmallow_dataclass import NewType, dataclass

from tapestry.graph.docs import JsonSerializable


def as_zarray(arr) -> np.ndarray:
    return np.array(arr, dtype=np.int64)


def as_zvector(arr) -> np.ndarray:
    arr = as_zarray(arr)
    assert arr.ndim == 1, f"Not a z-vector: {arr}"
    return arr


def as_zmatrix(arr) -> np.ndarray:
    arr = as_zarray(arr)
    assert arr.ndim == 2, f"Not a z-matrix: {arr}"
    return arr


class ZArrayField(fields.Field):
    def __init__(self, *args, **kwargs):
        super(ZArrayField, self).__init__(*args, **kwargs)

    def _serialize(self, value: np.ndarray, *args, **kwargs):
        if value is None:
            return None

        return value.tolist()

    def _deserialize(self, value, *args, **kwargs):
        if value is None:
            return None

        return as_zarray(value)


ZArray = NewType("NdArray", np.ndarray, field=ZArrayField)


@dataclass
class ZRange(JsonSerializable):
    start: ZArray
    end: ZArray

    def __init__(self, end, *, start=None):
        end = as_zvector(end)

        if start is None:
            start = np.zeros_like(end)
        start = as_zvector(start)

        self.start = start
        self.end = end

        if not np.all(self.end >= self.start):
            raise ValueError(f"start ({self.start}) is not >= end ({self.end})")

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def shape(self) -> np.ndarray:
        return self.end - self.start

    @property
    def size(self) -> int:
        return self.shape.prod()

    @property
    def empty(self) -> bool:
        return self.size == 0

    def inclusive_corners(self) -> List[np.ndarray]:
        if self.empty:
            return []

        ndim = self.ndim
        shape = self.shape
        inclusive_end = self.end - 1

        # generate every inclusive corner once.

        corners = [np.array([], dtype=np.int64)]
        for d in range(ndim):
            old_corners = corners
            cs = [self.start[d]]
            if shape[d]:
                cs.append(inclusive_end[d])
            corners = [np.append(p, c) for p in old_corners for c in cs]

        return corners


@dataclass
class CoordMap(JsonSerializable):
    projection: ZArray
    offset: ZArray

    def __init__(self, projection, offset):
        self.projection = as_zmatrix(projection)
        self.offset = as_zvector(offset)

        if self.out_dim != self.offset.shape[0]:
            raise ValueError(
                f"Projection output shape ({self.projection.shape})"
                f" != offset shape: ({self.offset.shape})"
            )

    @property
    def in_dim(self) -> int:
        return self.projection.shape[0]

    @property
    def out_dim(self) -> int:
        return self.projection.shape[1]

    def __call__(self, coords) -> np.ndarray:
        return self.projection.dot(coords) + self.offset

    def marginal_strides(self) -> np.ndarray:
        return self(np.identity(self.in_dim, dtype=np.int64))


@dataclass
class RangeMap(JsonSerializable):
    coord_map: CoordMap
    shape: ZArray

    def point_to_range(self, coord) -> ZRange:
        start = self.coord_map(coord)
        return ZRange(start=start, end=start + self.shape)

    def marginal_strides(self) -> np.ndarray:
        return self.coord_map.marginal_strides()

    def marginal_overlap(self) -> np.ndarray:
        """
        Returns the marginal shape overlap of strides along each dim.
        """
        return (self.coord_map.marginal_strides() - self.shape).clip(min=0)

    def marginal_waste(self) -> np.ndarray:
        """
        Returns the marginal waste of strides along each dim.
        """
        return (self.coord_map.marginal_strides() - self.shape).clip(max=0).abs()

    def range_to_range(self, range: ZRange) -> ZRange:
        assert not range.empty

        # FIXME: this is dumb.
        #
        # there is _certainly_ a mechanism using marginal strides to compute
        # a fixed form of this, rather than creating and sorting every inclusive
        # bounding corner.
        corners = sorted(tuple(self.coord_map(p)) for p in range.inclusive_corners())
        least_start = np.array(corners[0])
        greatest_start = np.array(corners[-1])

        return ZRange(
            start=least_start,
            end=greatest_start + self.shape,
        )


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
    coord_map: Optional[CoordMap] = None


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

    m = CoordMap(
        projection=[[1, 0], [0, 1]],
        offset=[3, 4],
    )

    r = RangeMap(
        coord_map=m,
        shape=[1, 2],
    )

    w = r.range_to_range(i)
    print(w)


if __name__ == "__main__":
    raw()
