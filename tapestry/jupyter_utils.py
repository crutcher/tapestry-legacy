from IPython.core.display import Image
from IPython.core.display_functions import display
import pydot

from tapestry.expression_graph import TapestryGraph


def display_pydot(pdot: pydot.Dot):
    plt = Image(pdot.create_png())
    display(plt)


def display_graph(graph_doc: TapestryGraph):
    display_pydot(graph_doc.to_dot())
