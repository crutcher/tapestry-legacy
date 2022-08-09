from tapestry.class_utils import Frozen
from tapestry.serialization.json_serializable import JsonLoadable


class FrozenDoc(JsonLoadable, Frozen):
    """Aggregate JsonLoadable, Frozen base class."""
