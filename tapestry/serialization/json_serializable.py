import json
from typing import Any, ClassVar, Optional, Type, TypeVar

import marshmallow

C = TypeVar("C", bound="JsonLoadable")


class JsonDumpable:
    """
    Base class for marshmallow dumpable dataclasses.
    """

    class Meta:
        # marshmallow_dataclasses control class.

        ordered = True
        # order the fields by declaration order.

    def get_dump_schema(self) -> marshmallow.Schema:
        raise NotImplementedError("Subclass Implements")

    def dump_json_data(self, *, reencode: bool = True) -> Any:
        # re-export to json to remove OrderedDict
        data = self.get_dump_schema().dump(self)
        if reencode:
            data = json.loads(json.dumps(data))
        return data

    def dump_json_str(
        self,
        *,
        indent: Optional[int] = None,
    ) -> Any:
        return self.get_dump_schema().dumps(self, indent=indent)

    def __str__(self):
        return self.dump_json_str()

    def pretty(self, *, prefix=""):
        return "\n".join(
            prefix + line for line in self.dump_json_str(indent=2).splitlines()
        )


class JsonLoadable(JsonDumpable):
    """
    Base class for marshmallow loadable dataclasses.
    """

    Schema: ClassVar[Type[marshmallow.Schema]] = marshmallow.Schema

    @classmethod
    def get_load_schema(cls) -> marshmallow.Schema:
        return getattr(cls, "Schema")()

    def get_dump_schema(self) -> marshmallow.Schema:
        return self.get_load_schema()

    @classmethod
    def load_json_data(cls: Type[C], data: Any) -> C:
        return cls.get_load_schema().load(data)

    @classmethod
    def load_json_str(cls: Type[C], data: str) -> C:
        return cls.get_load_schema().loads(data)
