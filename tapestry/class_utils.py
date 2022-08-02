import contextlib


class Frozen:
    """
    Mixin class to implement making attributes un-setable.
    """

    _frozen: bool = True
    """Set to `True` to freeze attribute setting."""

    @contextlib.contextmanager
    def _thaw_context(self):
        last_value = self._frozen
        self.__dict__["_frozen"] = False
        yield
        self.__dict__["_frozen"] = last_value

    def __setattr__(self, key, value):
        if self._frozen:
            raise AssertionError(
                f'Illegal attempt to set "{key}": {type(self).__name__} is frozen'
            )
        super().__setattr__(key, value)
