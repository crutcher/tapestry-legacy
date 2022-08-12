import functools
from typing import Iterable, List

from marshmallow import fields
from marshmallow_dataclass import NewType, dataclass
import numpy as np

from tapestry.numpy_utils import as_zarray, ndarray_hash, ndarray_lt
from tapestry.serialization.frozendoc import FrozenDoc


class ZArrayField(fields.Field):
    """
    Marshmallow Field type for ℤ-Space (integer) numpy NDarrays.

    Depends upon the following `setup.cfg` for mpyp:

    >>> [mypy]
    >>> plugins = marshmallow_dataclass.mypy
    """

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
"""
Marshmallow NewType for ZArrayField.

Usage:

>>> @marshmallow_dataclass.dataclass
... class Example:
...     coords: ZArray
"""


@dataclass
@functools.total_ordering
class ZRange(FrozenDoc):
    """
    [start, end) coordinate range in ℤ-Space.

    * `start` is inclusive, if the range is non-empty,
    * `end` is exclusive,
    * there are infinitely many empty ranges with the same `start`
    """

    __slots__ = ("start", "end")

    start: ZArray
    end: ZArray

    def __init__(self, end, *, start=None):
        end = as_zarray(
            end,
            ndim=1,
            immutable=True,
        )

        if start is None:
            start = np.zeros_like(end)

        start = as_zarray(
            start,
            ndim=1,
            immutable=True,
        )

        with self._thaw_context():
            self.start = start
            self.end = end

        if not np.all(self.end >= self.start):
            raise ValueError(f"start ({self.start}) is not >= end ({self.end})")

    def __hash__(self) -> int:
        return ndarray_hash(self.start) ^ ndarray_hash(self.end)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ZRange):
            return False

        return all(
            (
                np.array_equal(self.start, other.start),
                np.array_equal(self.end, other.end),
            )
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, ZRange):
            raise TypeError(f"Cannot compare ({type(self)}) and ({type(other)})")

        if ndarray_lt(self.start, other.start):
            return True

        return np.array_equal(self.start, other.start) and ndarray_lt(
            self.end, other.end
        )

    @functools.cached_property
    def ndim(self) -> int:
        "The number of dimensions of the ZSpace coordinate."
        return len(self.start)

    @functools.cached_property
    def shape(self) -> np.ndarray:
        "The shape of the range."
        return self.end - self.start

    @functools.cached_property
    def size(self) -> int:
        "The size of the range."
        return self.shape.prod()

    @functools.cached_property
    def empty(self) -> bool:
        "Is the range empty?"
        return self.size == 0

    @functools.cached_property
    def nonempty(self) -> bool:
        "Is the range non-empty?"
        return not self.empty

    @functools.cached_property
    def inclusive_corners(self) -> List[np.ndarray]:
        """
        Every inclusive corner in the range.

        * Duplicate corners will be included once,
        * Empty ranges will return []
        """
        if self.empty:
            return []

        ndim = self.ndim
        shape = self.shape
        inclusive_end = self.end - 1

        corners = [np.array([], dtype=np.int64)]
        for d in range(ndim):
            cs = [self.start[d]]
            if shape[d] > 1:
                cs.append(inclusive_end[d])

            old_corners = corners
            corners = [np.append(p, c) for p in old_corners for c in cs]

        return corners


@dataclass
class ZTransform(FrozenDoc):
    """
    Affine ℤ-Space index_map from one coordinate space to another.
    """

    __slots__ = ("projection", "offset")

    projection: ZArray
    """The projection matrix."""
    offset: ZArray
    """The offset vector."""

    @classmethod
    def identity(cls, n_dim: int, offset=None):
        """
        Construct an identity transform in the given dimensions.

        :param n_dim: number of dimensions.
        :param offset: (optional) offset, defaults to [0, ...].
        :return: a new ZTransform.
        """
        if offset is None:
            offset = np.zeros(n_dim, dtype=int)

        return ZTransform(
            projection=np.identity(n_dim, dtype=int),
            offset=offset,
        )

    def __init__(self, projection, offset):
        with self._thaw_context():
            self.projection = as_zarray(
                projection,
                ndim=2,
                immutable=True,
            )
            self.offset = as_zarray(
                offset,
                ndim=1,
                immutable=True,
            )

        if self.out_dim != self.offset.shape[0]:
            raise ValueError(
                f"Projection output shape ({self.projection.shape[1]})"
                f" != offset shape ({self.offset.shape[0]})"
            )

    def __hash__(self) -> int:
        return ndarray_hash(self.projection) ^ ndarray_hash(self.offset)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ZTransform):
            return False

        return all(
            (
                np.array_equal(self.projection, other.projection),
                np.array_equal(self.offset, other.offset),
            )
        )

    @functools.cached_property
    def in_dim(self) -> int:
        return self.projection.shape[0]

    @functools.cached_property
    def out_dim(self) -> int:
        return self.projection.shape[1]

    @functools.cached_property
    def constant(self) -> bool:
        return (self.projection == 0).all()

    def __call__(self, coords) -> np.ndarray:
        return np.matmul(coords, self.projection) + self.offset

    def marginal_strides(self) -> np.ndarray:
        "The marginal strides are the projection."
        return self.projection


@dataclass
class ZRangeMap(FrozenDoc):
    """
    Map from ZRange in in_dim, to ZRange in out_dim.
    """

    __slots__ = ("transform", "shape")

    transform: ZTransform
    shape: ZArray

    @classmethod
    def identity(cls, shape=None, *, offset=None):
        return ZRangeMap(
            transform=ZTransform.identity(
                n_dim=len(shape),
                offset=offset,
            ),
            shape=shape,
        )

    def __init__(self, transform: ZTransform, shape: Iterable[int]):
        with self._thaw_context():
            self.transform = transform
            self.shape = as_zarray(
                shape,
                ndim=1,
                immutable=True,
            )

        if self.transform.out_dim != self.shape.shape[0]:
            raise ValueError(
                f"Coord index_map out ndim ({self.transform.out_dim}) != shape ndim ({self.shape.shape[0]})"
            )

    def __hash__(self) -> int:
        return hash(self.transform) ^ ndarray_hash(self.shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ZRangeMap):
            return False

        return all(
            (
                self.transform == other.transform,
                np.array_equal(self.shape, other.shape),
            )
        )

    @functools.cached_property
    def in_dim(self) -> int:
        return self.transform.in_dim

    @functools.cached_property
    def out_dim(self) -> int:
        return self.transform.out_dim

    @functools.cached_property
    def constant(self) -> bool:
        return self.transform.constant

    def point_to_range(self, coord) -> ZRange:
        """
        Map a single point in in_dim to its bounding ZRange in out_dim.
        """
        coord = as_zarray(coord)

        if coord.shape != (self.in_dim,):
            raise ValueError(
                f"Coord shape {coord.shape} != ZRangeMap in_dim ({self.in_dim},)"
            )

        start = self.transform(coord)

        return ZRange(start=start, end=start + self.shape)

    def __call__(self, zrange: ZRange) -> ZRange:
        """
        Map a range in in_dim to the enclosing bounding range in out_dim.

        This will be the coherent union of mapping `point_to_range()` for each
        point; and may contain extra points between mapped shapes.
        """
        if zrange.ndim != self.in_dim:
            raise ValueError(
                f"ZRange ndim ({zrange.ndim}) != ZRangeMap in_dim ({self.in_dim})"
            )

        assert not zrange.empty

        # FIXME: this is dumb.
        #
        # there is _certainly_ a mechanism using marginal strides to compute
        # a fixed form of this, rather than creating and sorting every inclusive
        # bounding corner.
        #
        # We only really care as ndim grows.

        corners = sorted(
            self.transform(zrange.inclusive_corners),
            key=lambda x: x.tolist(),
        )

        least_start = np.array(corners[0])
        greatest_start = np.array(corners[-1])

        return ZRange(
            start=least_start,
            end=greatest_start + self.shape,
        )

    @functools.cache
    def marginal_overlap(self) -> np.ndarray:
        """
        Returns the marginal shape overlap of strides along each dim.
        """
        return (self.shape - self.transform.marginal_strides()).clip(min=0)

    @functools.cache
    def marginal_waste(self) -> np.ndarray:
        """
        Returns the marginal waste of strides along each dim.
        """
        return (self.transform.marginal_strides() - self.shape).clip(min=0)
