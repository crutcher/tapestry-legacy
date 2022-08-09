from tapestry.class_utils import Frozen
from tapestry.serialization.json_serializable import JsonSerializable


class FrozenDoc(JsonSerializable, Frozen):
    """Aggregate JsonSerializable, Frozen base class."""
