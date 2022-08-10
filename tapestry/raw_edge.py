from tapestry.expression_graph import EdgeAttrs, GraphDoc, NodeAttrs


def raw():
    g = GraphDoc()

    foo_node = NodeAttrs(
        display_name="foo",
    )
    g.add_node(foo_node)

    edge_node = EdgeAttrs(
        source_node_id=foo_node.node_id,
        target_node_id=foo_node.node_id,
    )
    g.add_node(edge_node)

    print(g.pretty())

    if False:
        bad_edge_node = EdgeAttrs(
            display_name="bad",
            source_node_id=edge_node.node_id,
            target_node_id=foo_node.node_id,
        )
        g.add_node(bad_edge_node)

    s = GraphDoc.build_load_schema(
        [
            NodeAttrs,
            EdgeAttrs,
        ]
    )

    g2 = s.load(g.dump_json_data())
    print(g2.pretty())


if __name__ == "__main__":
    raw()
