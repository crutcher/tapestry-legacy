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
