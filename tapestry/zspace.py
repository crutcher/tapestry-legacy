from typing import List

import numpy as np
from marshmallow import fields
from marshmallow_dataclass import NewType, dataclass

from tapestry.numpy_util import as_zarray, np_hash
from tapestry.serialization.json import JsonSerializable


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
        end = as_zarray(end, ndim=1, immutable=True)

        if start is None:
            start = np.zeros_like(end)
        start = as_zarray(start, ndim=1, immutable=True)

        self.start = start
        self.end = end

        if not np.all(self.end >= self.start):
            raise ValueError(f"start ({self.start}) is not >= end ({self.end})")

    def __eq__(self, other) -> bool:
        if not isinstance(other, ZRange):
            return False

        return all(
            (
                np.array_equal(self.start, other.start),
                np.array_equal(self.end, other.end),
            )
        )

    def __hash__(self) -> int:
        return np_hash(self.start) ^ np_hash(self.end)

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
            cs = [self.start[d]]
            if shape[d]:
                cs.append(inclusive_end[d])

            corners = [np.append(p, c) for p in corners for c in cs]

        return corners


@dataclass
class CoordMap(JsonSerializable):
    projection: ZArray
    offset: ZArray

    def __init__(self, projection, offset):
        self.projection = as_zarray(projection, ndim=2, immutable=True)
        self.offset = as_zarray(offset, ndim=1, immutable=True)

        if self.out_dim != self.offset.shape[0]:
            raise ValueError(
                f"Projection output shape ({self.projection.shape})"
                f" != offset shape: ({self.offset.shape})"
            )

    def __eq__(self, other) -> bool:
        if not isinstance(other, CoordMap):
            return False

        return all(
            (
                self.projection == other.projection,
                np.array_equal(self.offset, other.offset),
            )
        )

    def __hash__(self) -> int:
        return hash(self.projection) ^ np_hash(self.offset)

    @property
    def in_dim(self) -> int:
        return self.projection.shape[0]

    @property
    def out_dim(self) -> int:
        return self.projection.shape[1]

    def __call__(self, coords) -> np.ndarray:
        return np.matmul(coords, self.projection) + self.offset

    def marginal_strides(self) -> np.ndarray:
        return self(np.identity(self.in_dim, dtype=np.int64))


@dataclass
class RangeMap(JsonSerializable):
    coord_map: CoordMap
    shape: ZArray

    def __init__(self, coord_map: CoordMap, shape):
        self.coord_map = coord_map
        self.shape = as_zarray(shape, ndim=1, immutable=True)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RangeMap):
            return False

        return all(
            (
                self.coord_map == other.coord_map,
                np.array_equal(self.shape, other.shape),
            )
        )

    def __hash__(self) -> int:
        return hash(self.coord_map) ^ np_hash(self.shape)

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

    def range_to_range(self, zrange: ZRange) -> ZRange:
        assert not zrange.empty

        # FIXME: this is dumb.
        #
        # there is _certainly_ a mechanism using marginal strides to compute
        # a fixed form of this, rather than creating and sorting every inclusive
        # bounding corner.
        #
        # We only really care as ndim grows.

        corners = sorted(
            self.coord_map(zrange.inclusive_corners()),
            key=lambda x: x.tolist(),
        )

        least_start = np.array(corners[0])
        greatest_start = np.array(corners[-1])

        return ZRange(
            start=least_start,
            end=greatest_start + self.shape,
        )
