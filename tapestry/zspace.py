import enum
import functools
import math
from typing import Iterable, List, Set, Tuple

from marshmallow import fields
from marshmallow_dataclass import NewType, dataclass
import numpy as np
import numpy.ma

from tapestry.numpy_utils import (
    as_zarray,
    ndarray_aggregate_equality,
    ndarray_hash,
    ndarray_lt,
)
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

    @classmethod
    def bounds(cls, *points) -> "ZRange":
        if not len(points):
            raise AssertionError(f"bounds() called with no points.")

        ndim = len(points[0])
        for p in points:
            if len(p) != ndim:
                raise AssertionError(
                    f"Incompatible dimensions in call to bound(): {points[0]} !~ {p}",
                )

        ps = np.concatenate(points)

        return ZRange(
            start=ps.min(axis=0),
            end=ps.max(axis=0) + 1,
        )

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

    def __copy__(self) -> "ZRange":
        return self

    def __deepcopy__(self, memodict={}) -> "ZRange":
        return self

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

    def __contains__(self, item) -> bool:
        if self.is_empty:
            return False

        if isinstance(item, ZRange):
            if item.is_empty:
                return False

            return item.start in self and (item.end - 1) in self

        try:
            coords = as_zarray(item)
        except ValueError:
            return False

        if coords.ndim != 1 or len(coords) != self.ndim:
            return False

        return bool((self.start <= coords).all() and (coords < self.end).all())

    def __lt__(self, other) -> bool:
        if not isinstance(other, ZRange):
            raise TypeError(f"Cannot compare ({type(self)}) and ({type(other)})")

        if ndarray_lt(self.start, other.start):
            return True

        return np.array_equal(self.start, other.start) and ndarray_lt(
            self.end, other.end
        )

    def __sub__(self, other) -> "ZRange":
        return self + (-as_zarray(other))

    def __add__(self, other) -> "ZRange":
        offset = as_zarray(other)
        return ZRange(
            start=self.start + offset,
            end=self.end + offset,
        )

    def as_slice(self):
        return tuple(slice(s, e) for s, e in zip(self.start, self.end))

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
    def is_empty(self) -> bool:
        "Is the range empty?"
        return self.size == 0

    @functools.cached_property
    def is_nonempty(self) -> bool:
        "Is the range non-empty?"
        return not self.is_empty

    @functools.cached_property
    def inclusive_corners(self) -> List[np.ndarray]:
        """
        Every inclusive corner in the range.

        * Duplicate corners will be included once,
        * Empty ranges will return []
        """
        if self.is_empty:
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

    def split(self, *, axis: int, sections: int) -> List["ZRange"]:
        return self.multi_split((axis, sections))

    def section(self, sections) -> List["ZRange"]:
        return self.multi_split(*enumerate(as_zarray(sections)))

    def multi_split(self, *axis_splits: Tuple[int, int]) -> List["ZRange"]:
        """
        Split a ZRange along multiple axes.

        :param axis_splits: (axis, section) pairs
        :return: a list of net ZRanges.
        """
        shape = self.shape

        seen: Set[int] = set()
        for axis, sections in axis_splits:
            if sections < 1:
                raise AssertionError(
                    f"Illegal sectioning: occurs more than once: {axis_splits}"
                )

            if sections == 1:
                continue

            if axis in seen:
                raise AssertionError(f"axis occurs more than once: {axis_splits}")

            axis_size = shape[axis]
            if sections > axis_size:
                raise AssertionError(
                    f"Cannot split {shape}[{axis}] ({axis_size}) into ({sections}) non-zero sections."
                )

        parts = [self]

        for axis, sections in axis_splits:
            axis_size = shape[axis]
            step_size = math.ceil(float(axis_size) / sections)

            chunk_step = np.zeros(self.ndim, dtype=int)
            chunk_step[axis] = step_size

            last_parts = parts
            parts = []
            for zr in last_parts:
                end0 = zr.end.copy()
                end0[axis] += step_size - axis_size

                for idx in range(sections):
                    offset = chunk_step * idx
                    start = zr.start + offset
                    end = np.minimum(end0 + offset, zr.end)

                    parts.append(
                        ZRange(
                            start=start,
                            end=end,
                        )
                    )

        return parts

    def intersection(self, val: "ZRange") -> "ZRange":
        """
        Compute the intersection of two ZRanges.

        :param val: the other ZRange.
        :return: intersecting range, or ZRange([0]*ndim).
        :raises ValueError: on error.
        """
        if self.ndim != val.ndim:
            raise ValueError(
                f"Incompatible dimensions ({self.ndim} != {val.ndim})",
            )

        start = numpy.maximum(self.start, val.start)
        end = numpy.minimum(self.end, val.end)

        if (start >= end).any():
            return ZRange([0] * self.ndim)

        return ZRange(start=start, end=end)


class EmbeddingMode(enum.Enum):
    CLIP = enum.auto()
    TILE = enum.auto()


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
    def identity_transform(
        cls,
        dim: int = 1,
        *,
        offset=None,
    ) -> "ZTransform":
        """
        Construct an identity transform in the given dimensions.

        :param dim: number of dimensions.
        :param offset: (optional) offset, defaults to [0, ...].
        :return: a new ZTransform.
        """
        if offset is None:
            offset = np.zeros(dim, dtype=int)

        return ZTransform(
            projection=np.identity(dim, dtype=int),
            offset=offset,
        )

    @classmethod
    def constant_transform(
        cls,
        input_dim: int,
        *,
        out_dim: int = None,
        offset=None,
    ) -> "ZTransform":
        """
        Construct an identity transform in the given dimensions.

        :param n_dim: number of dimensions.
        :param offset: (optional) offset, defaults to [0, ...].
        :return: a new ZTransform.
        """
        if out_dim is None and offset is None:
            raise ValueError("offset and out_dim are mutually exclusive")
        if not (out_dim is None or offset is None):
            raise ValueError("One of offset or out_dim is required")

        if out_dim is not None:
            offset = np.zeros((out_dim,), dtype=int)

        return ZTransform(
            projection=np.zeros((input_dim, len(offset)), dtype=int),
            offset=offset,
        )

    def __init__(
        self,
        projection,
        offset=None,
    ):
        with self._thaw_context():
            self.projection = as_zarray(
                projection,
                ndim=2,
                immutable=True,
            )

            if offset is None:
                offset = np.zeros(self.projection.shape[1])

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

    def __copy__(self) -> "ZTransform":
        return self

    def __deepcopy__(self, memodict={}) -> "ZTransform":
        return self

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

    def embed(
        self,
        in_dim: int,
        mode: EmbeddingMode = EmbeddingMode.TILE,
    ) -> "ZTransform":
        """
        Create a new ZTransform by embedding this ZTransform in the target number of in_dims.

        The resulting ZTransform will have `t.in_dim` == `in_dim.

        If `in_dim` == `self.in_dim`; this is a no-op, and the original transform will be returned.

        If mode is `CLIP`, the additional dimensions will be discarded, and `t.out_dim` will
        remain unchanged.

        If mode is `BROADCAST`, the additional dimensions will be passed through
        as an identity, and `t.out_dim` will grow in size.

        :param in_dim: the new total in dimensions.
        :param mode: the embedding mode.
        :return: a new ZTransform.
        :raises ValueError: if `in_dim` < `self.in_dim`.
        """
        if in_dim < self.in_dim:
            raise ValueError(
                f"Cannot embed ZTransform of in_dims ({self.in_dim}) in ({in_dim}) dims"
            )

        if in_dim == self.in_dim:
            # no-op.
            return self

        new_dims = in_dim - self.in_dim

        projection = np.concatenate(
            (
                np.zeros((new_dims, self.out_dim)),
                self.projection,
            ),
        )
        offset = self.offset

        if mode is EmbeddingMode.TILE:
            projection = np.concatenate(
                (
                    np.concatenate(
                        (
                            np.identity(new_dims, dtype=int),
                            np.zeros((self.in_dim, new_dims), dtype=int),
                        )
                    ),
                    projection,
                ),
                axis=1,
            )

            offset = np.concatenate((np.zeros(new_dims, dtype=int), offset))

        return ZTransform(
            projection=projection,
            offset=offset,
        )

    @functools.cached_property
    def in_dim(self) -> int:
        return self.projection.shape[0]

    @functools.cached_property
    def out_dim(self) -> int:
        return self.projection.shape[1]

    @functools.cached_property
    def is_constant(self) -> bool:
        return (self.projection == 0).all()

    @functools.cached_property
    def is_identity(self) -> bool:
        if self.projection.shape[0] != self.projection.shape[1]:
            return False
        return (self.projection == np.identity(self.projection.shape[0])).all()

    def __call__(self, coords) -> np.ndarray:
        coords = np.asarray(coords)
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
    def identity_map(
        cls,
    ) -> "ZRangeMap":
        return ZRangeMap(
            transform=ZTransform.identity_transform(1),
            shape=[1],
        )

    @classmethod
    def constant_map(
        cls,
        input_dim: int,
        *,
        shape,
        offset=None,
    ) -> "ZRangeMap":
        if offset is None:
            offset = np.zeros(len(shape), dtype=int)

        return ZRangeMap(
            transform=ZTransform.constant_transform(
                input_dim=input_dim,
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

    def __copy__(self) -> "ZRangeMap":
        return self

    def __deepcopy__(self, memodict={}) -> "ZRangeMap":
        return self

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

    def embed(
        self,
        in_dim: int,
        mode: EmbeddingMode,
    ) -> "ZRangeMap":
        transform = self.transform.embed(in_dim, mode)
        shape = np.concatenate(
            (
                np.ones(transform.out_dim - len(self.shape), dtype=int),
                self.shape,
            )
        )
        return ZRangeMap(
            transform=transform,
            shape=shape,
        )

    @functools.cached_property
    def in_dim(self) -> int:
        return self.transform.in_dim

    @functools.cached_property
    def out_dim(self) -> int:
        return self.transform.out_dim

    @functools.cached_property
    def constant(self) -> bool:
        return self.transform.is_constant

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
        if zrange.is_empty:
            start = self.transform(zrange)
            return ZRange(
                start=start,
                end=start,
            )

        corners = [self.transform(corner) for corner in zrange.inclusive_corners]

        return ZRange.bounds(corners + [c + self.shape - 1 for c in corners])

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


def assert_shape(
    actual: np.ndarray,
    expected: np.ndarray,
    msg: str = "actual shape {actual} != expected shape {expected}",
    **kwargs,
):
    if not ndarray_aggregate_equality(actual, expected):
        raise ValueError(msg.format(actual=actual, expected=expected, **kwargs))
