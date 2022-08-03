import numpy as np


def np_hash(arr: np.ndarray) -> int:
    return hash(arr.tobytes())


def np_immutable_guard(*args: np.ndarray) -> None:
    """
    Mark numpy arrays as immutable.
    """
    for x in args:
        x.flags.writeable = False


def as_zarray(val, *, ndim=None, immutable=False) -> np.ndarray:
    direct = np.array(val)
    arr = np.asarray(direct, dtype=np.int64)

    # this is slow, but it prevents dumb errors.
    if direct is not arr and np.any(direct != arr):
        raise ValueError(f"ZArray rounding error: {direct}")

    if ndim is not None:
        if direct.ndim != ndim:
            raise ValueError(
                f"Expected an ndarray of dim ({ndim}), found ({direct.ndim})"
            )

    if immutable:
        np_immutable_guard(arr)

    return arr


def _zip_lt(a, b) -> bool:
    k_a = len(a)
    k_b = len(b)
    if k_a < k_b:
        return True
    if k_a > k_b:
        return False

    for x, y in zip(a, b):
        if x < y:
            return True

    return False


def ndarry_lt(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Establish a tuple-like less than (<) ordering between numpy tuples.

    Equivalent to:

    >>> a.tolist() < b.tolist()
    """
    if a.shape != b.shape:
        return _zip_lt(a.shape, b.shape)

    return _zip_lt(a.flat, b.flat)


def _zip_le(a, b) -> bool:
    k_a = len(a)
    k_b = len(b)
    if k_a < k_b:
        return True
    if k_a > k_b:
        return False

    for x, y in zip(a, b):
        if x <= y:
            return True

    return False


def ndarry_le(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Establish a tuple-like less than or equal (<=) ordering between numpy tuples.

    Equivalent to:

    >>> a.tolist() <= b.tolist()
    """
    if a.shape != b.shape:
        return _zip_lt(a.shape, b.shape)

    return _zip_le(a.flat, b.flat)
