import numpy as np

from tapestry.expression_graph import (
    BlockOperation,
    ExternalTensorValue,
    GraphHandle,
    TensorResult,
    TensorValue,
)
from tapestry.zspace import ZRange, ZRangeMap, ZTransform


def f(
    x: TensorValue.Handle,
    w: TensorValue.Handle,
) -> TensorResult.Handle:
    graph = x.graph
    assert w.graph == graph

    x_shape = x.attrs.shape
    w_shape = w.attrs.shape

    assert x_shape[-1] == w_shape[0]
    y = graph.doc.add_node(
        TensorResult(
            display_name="Y",
            shape=np.array([100, 3]),
            dtype="torch.float16",
        )
    )

    op = graph.doc.add_node(
        BlockOperation(
            display_name="Linear",
            index_space=ZRange([2, 3]),
        )
    )

    graph.doc.add_node(
        BlockOperation.BlockInput(
            source_node_id=op.node_id,
            target_node_id=x.node_id,
            selector=ZRangeMap(
                transform=ZTransform(
                    projection=[[2, 0], [0, 1]],
                    offset=[-1, 0],
                ),
                shape=x.attrs.shape,
            ),
        )
    )

    graph.doc.add_node(
        BlockOperation.BlockInput(
            source_node_id=op.node_id,
            target_node_id=w.node_id,
            selector=ZRangeMap(
                transform=ZTransform(
                    projection=[[0, 0], [0, 0]],
                    offset=[0, 0],
                ),
                shape=w.attrs.shape,
            ),
        )
    )

    graph.doc.add_node(
        BlockOperation.BlockOutput(
            source_node_id=y.node_id,
            target_node_id=op.node_id,
            selector=ZRangeMap.identity(y.shape),
        ),
    )

    return TensorResult.Handle(
        graph=graph,
        attrs=y,
    )


def raw():
    g = GraphHandle()

    x_attrs = g.doc.add_node(
        ExternalTensorValue(
            display_name="X",
            shape=[100, 2],
            dtype="torch.float16",
            storage="store:x",
        )
    )

    w_attrs = g.doc.add_node(
        ExternalTensorValue(
            display_name="W",
            shape=[2, 3],
            dtype="torch.float16",
            storage="store:w",
        )
    )

    y = f(
        g.get_node(x_attrs.node_id, TensorValue.Handle),
        g.get_node(w_attrs.node_id, TensorValue.Handle),
    )

    print(g.doc.pretty())


if __name__ == "__main__":
    raw()
