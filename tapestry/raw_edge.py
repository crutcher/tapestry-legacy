from dataclasses import dataclass

import numpy as np

from tapestry.expression_graph import (
    BlockOperation,
    ExternalTensor,
    TapestryGraph,
    TensorResult,
    TensorValue,
)
from tapestry.zspace import ZRange, ZRangeMap


def f(
    x: TensorValue,
    w: TensorValue,
) -> TensorResult:
    graph = x.assert_graph()
    assert w.graph == graph

    assert x.shape[-1] == w.shape[0]

    y_shape = np.append(x.shape[:-1], w.shape[1])

    op_name = "Linear"

    op = graph.add_node(
        BlockOperation(
            name=op_name,
            index_space=ZRange(y_shape),
        )
    )

    graph.add_node(
        BlockOperation.Input(
            source_id=op.node_id,
            target_id=x.node_id,
            selector=ZRangeMap.identity_map(shape=[1, 2]),
            name="x",
        )
    )

    graph.add_node(
        BlockOperation.Input(
            source_id=op.node_id,
            target_id=w.node_id,
            selector=ZRangeMap.constant_map(2, shape=w.shape),
            name="w",
        )
    )

    y = graph.add_node(
        TensorResult(
            name=f"{op_name}:Y",
            shape=y_shape,
            dtype="torch.float16",
        )
    )

    graph.add_node(
        BlockOperation.Result(
            source_id=y.node_id,
            target_id=op.node_id,
            selector=ZRangeMap.identity_map(y.shape),
            name="y",
        ),
    )

    return y



def raw():
    g = TapestryGraph()

    x = g.add_node(
        ExternalTensor(
            name="X",
            shape=[100, 2],
            dtype="torch.float16",
            storage="store:x",
        )
    )

    w = g.add_node(
        ExternalTensor(
            name="W",
            shape=[2, 3],
            dtype="torch.float16",
            storage="store:w",
        )
    )

    y = f(x, w)

    g.validate()


if __name__ == "__main__":
    raw()
