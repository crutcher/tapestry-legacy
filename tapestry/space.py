from typing import Iterable, Tuple

import numpy
import torch


class ZPoint:
    """
    Immutable point in ℤN coordinate space.
    """

    @staticmethod
    def coerce(obj) -> "ZPoint":
        if isinstance(obj, ZPoint):
            return obj
        return ZPoint(obj)

    _coords: Tuple[int, ...]

    def __init__(
        self,
        coords: Iterable[int],
    ):
        self._coords = tuple(int(x) for x in coords)
        assert len(self) >= 0, "coords must be non-empty"

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, item):
        return self._coords[item]

    def __setitem__(self, key, value):
        raise TypeError(
            f"{self.__class__.__name__} object does not support item assignment"
        )

    def __delitem__(self, key):
        raise TypeError(
            f"{self.__class__.__name__} object does not support item deletion"
        )

    def __eq__(self, other) -> bool:
        return all(a == b for a, b in zip(self, ZPoint.coerce(other)))

    def __hash__(self) -> int:
        return hash(self._coords)

    def ndim(self) -> int:
        return len(self._coords)

    def as_tuple(self) -> Tuple[int, ...]:
        "Return coords as a tuple()."
        return self._coords

    def as_tensor(self) -> torch.Tensor:
        "Return coords as a torch.tensor()."
        return torch.tensor(self)

    def as_numpy(self) -> numpy.ndarray:
        "Return coords as a numpy.array()."
        return numpy.array(self)


def expect_zpoint_lt_zpoint(
    a: Iterable[int],
    b: Iterable[int],
    value_name: str = "a",
    expected_name: str = "b",
):
    a = ZPoint.coerce(a).as_tensor()
    b = ZPoint.coerce(b).as_tensor()

    if (a.shape != b.shape) or not (a <= b).all():
        raise AssertionError(
            f"Expected {value_name}:{a} to be strictly <= {expected_name}:{b}"
        )


class ZRange:
    _start: ZPoint
    _end: ZPoint

    def __init__(
        self,
        start: Iterable[int],
        end: Iterable[int],
    ):
        self._start = ZPoint.coerce(start)
        self._end = ZPoint.coerce(end)

        expect_zpoint_lt_zpoint(
            self._start,
            self._end,
            value_name="start",
            expected_name="end",
        )

    def start(self) -> ZPoint:
        return self._start

    def end(self) -> ZPoint:
        return self._end

    def shape(self) -> torch.Tensor:
        return self._end.as_tensor() - self._start.as_tensor()

    def card(self) -> int:
        return int(self.shape().prod().item())
