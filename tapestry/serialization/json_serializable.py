from typing import Any, ClassVar, Optional, Type, TypeVar

import marshmallow

C = TypeVar("C", bound="JsonLoadable")


class JsonDumpable:
    class Meta:
        ordered = True

    def get_dump_schema(self) -> marshmallow.Schema:
        raise NotImplementedError("Subclass Implements")

    def dump_json_data(self) -> Any:
        return self.get_dump_schema().dump(self)

    def dump_json_str(
        self,
        *,
        indent: Optional[int] = None,
    ) -> Any:
        return self.get_dump_schema().dumps(self, indent=indent)

    def __str__(self):
        return self.dump_json_str()

    def pretty(self):
        return self.dump_json_str(indent=2)


class JsonLoadable(JsonDumpable):
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
