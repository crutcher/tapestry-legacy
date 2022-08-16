import numpy as np


def ndarray_aggregate_equality(a, b):
    """
    Aggregate, non-throwing (a == b).

    :param a: a numpy coercible array.
    :param b: a numpy coercible array.
    :return: True or False.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    try:
        return np.equal(a, b).all()
    except np.ValueError:
        return False


def ndarray_hash(arr: np.ndarray) -> int:
    return hash(arr.tobytes())


def make_ndarray_immutable(*args: np.ndarray) -> None:
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
        make_ndarray_immutable(arr)

    return arr


def ndarray_lt(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Establish a tuple-like less than (<) ordering between numpy tuples.
    """
    return a.tolist() < b.tolist()


def ndarray_le(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Establish a tuple-like less than or equal (<=) ordering between numpy tuples.
    """
    return a.tolist() <= b.tolist()
