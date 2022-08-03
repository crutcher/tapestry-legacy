from tapestry.class_utils import Frozen
from tapestry.serialization.json import JsonSerializable


class FrozenDoc(JsonSerializable, Frozen):
    """Aggregate JsonSerializable, Frozen base class."""
