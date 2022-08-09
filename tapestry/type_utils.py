import uuid
from typing import Any, Mapping, Optional, Union

UUIDConvertable = Union[str, uuid.UUID]


def dict_to_parameter_str(val: Mapping[str, Any]) -> str:
    """
    Format a dict as a '(a=b, c=d)' style parameter string.

    :param val: the value to format.
    :return: a str.
    """
    return ", ".join(f"{k}={repr(v)}" for k, v in val.items())


def coerce_uuid(val: UUIDConvertable) -> uuid.UUID:
    """
    Coerce a value to a uuid.UUID

    :param val: the value.
    :return: a uuid.UUID
    :raises ValueError: if the value is not coercible.
    """
    if isinstance(val, str):
        return uuid.UUID(val)
    if isinstance(val, uuid.UUID):
        return val
    raise ValueError(
        f"Unable to coerce to UUID: {type(val)}:: {val}",
    )


def ensure_uuid(val: Optional[UUIDConvertable] = None) -> uuid.UUID:
    """
    Ensure a uuid.UUID value.

    Either coerces the given value, or generates a new one on None.

    :param val: (Optional) the value.
    :return: a uuid.UUID
    :raises ValueError: if the value is not coercible.
    """
    if val is None:
        return uuid.uuid4()
    return coerce_uuid(val)
