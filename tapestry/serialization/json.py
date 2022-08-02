from typing import Any, Optional, Type, TypeVar

import marshmallow

C = TypeVar("C", bound="JsonSerializable")


class JsonSerializable:
    class Meta:
        ordered = True

    @classmethod
    def get_schema(cls) -> marshmallow.Schema:
        return getattr(cls, "Schema")()

    def dump_json_data(self) -> Any:
        return self.get_schema().dump(self)

    def dump_json_str(
        self,
        *,
        indent: Optional[int] = None,
    ) -> Any:
        return self.get_schema().dumps(self, indent=indent)

    @classmethod
    def load_json_data(cls: Type[C], data: Any) -> C:
        return cls.get_schema().load(data)

    @classmethod
    def load_json_str(cls: Type[C], data: str) -> C:
        return cls.get_schema().loads(data)

    def __str__(self):
        return self.dump_json_str()

    def pretty(self):
        return self.dump_json_str(indent=2)
