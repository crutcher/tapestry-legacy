import hamcrest
from hamcrest.core.matcher import Matcher

from tapestry import space


def match_zpoint(expected) -> Matcher:
    "Hamcrest Matcher for a tapestry.space.ZPoint"
    return hamcrest.all_of(
        hamcrest.instance_of(space.ZPoint),
        hamcrest.contains_exactly(*expected),
    )
