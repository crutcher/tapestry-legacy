import numbers
import typing

import hamcrest
import nptyping
import numpy as np
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher

from tapestry.testlib import eggs


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    eggs.hide_module_tracebacks(globals(), mode)


hide_tracebacks(True)


class NDArrayStructureMatcher(BaseMatcher):
    expected: np.ndarray

    def __init__(self, expected: typing.Any) -> None:
        self.expected = np.asarray(expected)

    def _matches(self, item: typing.Any) -> bool:
        # Todo: structural miss-match that still shows expected ndarray.

        try:
            eggs.assert_match(
                item.shape,
                self.expected.shape,
            )
            eggs.assert_match(
                item.dtype,
                self.expected.dtype,
            )
            return True
        except AssertionError as e:
            raise e
            return False

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_ndarray_structure(
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> NDArrayStructureMatcher:
    return NDArrayStructureMatcher(expected)


def assert_ndarray_structure(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> None:
    hamcrest.assert_that(
        actual,
        expect_ndarray_structure(expected),
    )


class NDArrayMatcher(NDArrayStructureMatcher):
    close: bool = False

    def __init__(
        self,
        expected: typing.Any,
        *,
        close: bool = False,
    ) -> None:
        super().__init__(expected=expected)
        self.close = close

    def _matches(self, item: typing.Any) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            np.testing.assert_allclose(
                item,
                self.expected,
                equal_nan=True,
            )

        else:
            np.testing.assert_equal(item, self.expected)

        return True

    def describe_to(self, description: Description) -> None:
        description.append_text("\n")
        description.append_description_of(self.expected)

    def describe_match(self, item: typing.Any, match_description: Description) -> None:
        match_description.append_text("was \n")
        match_description.append_description_of(item)

    def describe_mismatch(
        self, item: typing.Any, mismatch_description: Description
    ) -> None:
        mismatch_description.append_text("was \n")
        mismatch_description.append_description_of(item)


def matches_ndarray(
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
    close: bool = False,
) -> NDArrayMatcher:
    return NDArrayMatcher(expected, close=close)


def expect_ndarray_seq(
    *expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> Matcher:
    return hamcrest.contains_exactly(*[matches_ndarray(e) for e in expected])


def assert_ndarray_equals(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
    close: bool = False,
) -> None:
    hamcrest.assert_that(
        actual,
        matches_ndarray(expected, close=close),
    )


def assert_ndarray_seq(
    actual: typing.Sequence[np.ndarray],
    *expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> None:
    hamcrest.assert_that(
        actual,
        expect_ndarray_seq(*expected),
    )


def assert_ndarray_close(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> None:
    hamcrest.assert_that(
        actual,
        matches_ndarray(expected, close=True),
    )
