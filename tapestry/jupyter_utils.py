import pydot
from IPython.core.display import Image
from IPython.core.display_functions import display

from tapestry.expression_graph import GraphDoc, GraphHandle


def display_pydot(pdot: pydot.Dot):
    plt = Image(pdot.create_png())
    display(plt)


def display_graph_doc(graph_doc: GraphDoc):
    display_pydot(graph_doc.to_dot())


def display_graph_handle(graph_handle: GraphHandle):
    display_graph_doc(graph_handle.doc)
