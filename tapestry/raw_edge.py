from tapestry.expression_graph import BlockOperation, GraphDoc, NodeAttributes
from tapestry.zspace import ZRange, ZRangeMap, ZTransform


def raw():
    g = GraphDoc()

    foo_node = NodeAttributes(
        display_name="foo",
    )
    g.add_node(foo_node)

    block_op = BlockOperation(index_space=ZRange([2, 3]))
    g.add_node(block_op)

    bind = BlockOperation.BlockInput(
        source_node_id=block_op.node_id,
        target_node_id=foo_node.node_id,
        selector=ZRangeMap(
            transform=ZTransform(
                projection=[[2, 0], [0, 1]],
                offset=[-1, 0],
            ),
            shape=[1, 1],
        ),
    )
    g.add_node(bind)

    print(g.pretty())


if __name__ == "__main__":
    raw()
