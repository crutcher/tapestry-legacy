from unittest import TestCase

from tapestry.expression_graph import TapestryGraph


class PlacementTest(TestCase):
    def test_nothing(self) -> None:
        g = TapestryGraph()
        pass