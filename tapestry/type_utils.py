import uuid
from typing import Any, Mapping, Optional, Union

UUIDConvertable = Union[str, uuid.UUID]


def dict_to_parameter_str(d: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={repr(v)}" for k, v in d.items())


def coerce_uuid(val: UUIDConvertable) -> uuid.UUID:
    if isinstance(val, str):
        return uuid.UUID(val)
    if isinstance(val, uuid.UUID):
        return val
    raise ValueError(
        f"Unable to coerce to UUID: {type(val)}:: {val}",
    )


def ensure_uuid(val: Optional[UUIDConvertable] = None) -> uuid.UUID:
    if val is None:
        return uuid.uuid4()
    return coerce_uuid(val)
