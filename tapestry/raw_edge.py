from tapestry.attrs_docs import ExternalTensorValueAttrs, GraphDoc, TensorValueAttrs
from tapestry.expression_graph import (
    ExpressionGraph,
    ExternalTensorValue,
    NodeWrapper,
    TensorSource,
)


def raw():
    gdoc = GraphDoc()

    adoc = ExternalTensorValueAttrs(
        name="A",
        storage="pre:A",
    )
    gdoc.add_node(adoc)

    bdoc = TensorValueAttrs(
        name="B",
    )
    gdoc.add_node(bdoc)

    print(gdoc.pretty())

    g = ExpressionGraph(gdoc)
    print(g.list_nodes_of_type(ExternalTensorValue))

    print(g.get_node(bdoc.node_id, TensorSource))

    x = g.get_node(bdoc.node_id, TensorSource)
    if y := x.try_as_type(NodeWrapper):
        print("expected", y)
    if y := x.try_as_type(ExternalTensorValue):
        print("not expected", y)

    z = g.get_node(adoc.node_id, ExternalTensorValue)
    print(z.attrs.storage)


if __name__ == "__main__":
    raw()
