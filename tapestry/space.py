from typing import Iterable, Tuple

import numpy
import torch


def expect_zpoint(
    a: torch.Tensor,
    name: str = "value",
):
    if not (
        isinstance(a, torch.Tensor)
        and a.ndim == 1
        and len(a) >= 1
        and a.dtype == torch.int64
    ):
        raise AssertionError(f"Expected ZPoint, found {name}:{a}")


def expect_zpoint_lt_zpoint(
    a: torch.Tensor,
    b: torch.Tensor,
    value_name: str = "a",
    expected_name: str = "b",
):
    expect_zpoint(a, value_name)
    expect_zpoint(b, expected_name)

    if (a.shape != b.shape) or not (a <= b).all():
        raise AssertionError(
            f"Expected {value_name}:{a} to be strictly <= {expected_name}:{b}"
        )


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

    def as_tuple(self) -> Tuple[int, ...]:
        "Return coords as a tuple()."
        return self._coords

    def as_tensor(self) -> torch.Tensor:
        "Return coords as a torch.tensor()."
        return torch.tensor(self)

    def as_numpy(self) -> numpy.ndarray:
        "Return coords as a numpy.array()."
        return numpy.array(self)


class ZBox:
    _start: ZPoint
    _end: ZPoint


class ZRange:
    _start: torch.Tensor
    _end: torch.Tensor

    def __init__(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
    ):
        self._start = torch.tensor(start)
        self._end = torch.tensor(end)

        expect_zpoint_lt_zpoint(
            self._start,
            self._end,
            value_name="start",
            expected_name="end",
        )

    def start(self) -> torch.Tensor:
        return self._start

    def end(self) -> torch.Tensor:
        return self._end

    def shape(self) -> torch.Tensor:
        return self._end - self._start

    def card(self) -> int:
        return int(self.shape().prod().item())
