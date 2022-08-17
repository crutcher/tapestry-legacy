from typing import Optional

import numpy as np
import torch

from tapestry.expression_graph import (
    BlockOperation,
    PinnedTensor,
    TapestryGraph,
    TensorResult,
    TensorValue,
)
from tapestry.zspace import ZRange, ZRangeMap, ZTransform, assert_shape


def _name_and_shape(val: TensorValue):
    s = "Tensor"
    if val.name:
        s += f"[{val.name}]"
    s += f"{val.shape}"
    return s


def linear_op(
    *,
    x: TensorValue,
    w: TensorValue,
    bias: Optional[TensorValue] = None,
) -> TensorResult:
    graph = x.assert_graph()
    assert w.graph == graph

    assert len(w.shape) == 2, w.shape
    in_dim = w.shape[0]
    out_dim = w.shape[1]

    assert_shape(
        x.shape[-1:],
        w.shape[:1],
        "input shape {xshape} in_dim {actual} incompatible "
        "with weight shape {wshape} in_dim {expected}",
        xshape=x.shape,
        wshape=w.shape,
    )

    index_space = ZRange(x.shape[:-1])

    op_name = "Linear"

    op = graph.add_node(
        BlockOperation(
            name=op_name,
            index_space=index_space,
        )
    )

    op.bind_tiled_input(
        name="input",
        value=x,
        projection=[[1, 0]],
        shape=[1, in_dim],
    )

    op.bind_fixed_input(name="w", value=w)

    if bias is not None:
        assert_shape(
            bias.shape,
            w.shape[-1:],
            "bias shape {actual} != weight [out_dim] {expected}",
        )

        op.bind_fixed_input(name="bias", value=bias)

    return op.bind_result(
        name="result",
        selector=ZRangeMap(
            transform=ZTransform(projection=[[1, 0]]),
            shape=[1, out_dim],
        ),
    )

def linear_op_shard_w(
        *,
        x: TensorValue,
        w: TensorValue,
        bias: Optional[TensorValue] = None,
) -> TensorResult:
    graph = x.assert_graph()
    assert w.graph == graph

    assert len(w.shape) == 2, w.shape
    in_dim = w.shape[0]
    out_dim = w.shape[1]

    assert_shape(
        x.shape[-1:],
        w.shape[:1],
        "input shape {xshape} in_dim {actual} incompatible "
        "with weight shape {wshape} in_dim {expected}",
        xshape=x.shape,
        wshape=w.shape,
    )

    index_space = ZRange(x.shape[:-1].tolist() + [out_dim])

    op_name = "Linear"

    op = graph.add_node(
        BlockOperation(
            name=op_name,
            index_space=index_space,
        )
    )

    # TODO: broadcast-one (naming?)
    # broadcasting to additional dimensions should include them.
    op.bind_tiled_input(
        name="input",
        value=x,
        projection=[[1, 0], [0, 0]],
        shape=[1, in_dim],
    )

    projection = np.zeros((index_space.ndim, 2))
    projection[-1, -1] = 1

    # TODO: broadcast-zero (naming?)
    # broadcasting to additional dimensions should ignore them.
    op.bind_tiled_input(
        name="w",
        value=w,
        projection=projection,
        shape=[in_dim, 1],
    )

    if bias is not None:
        assert_shape(
            bias.shape,
            w.shape[-1:],
            "bias shape {actual} != weight [out_dim] {expected}",
        )

        op.bind_tiled_input(
            name="bias",
            value=bias,
            projection=[[0], [1]],
            shape=[1],
        )

    return op.bind_result(
        name="result",
        selector=ZRangeMap(
            transform=ZTransform(projection=[[1, 0], [0, 1]]),
            shape=[1, 1],
        ),
    )


def relu_op(
    value: TensorValue,
) -> TensorResult:
    graph = value.assert_graph()

    index_space = ZRange(value.shape)

    op_name = "ReLU"

    op = graph.add_node(
        BlockOperation(
            name=op_name,
            index_space=index_space,
        )
    )

    selector = ZRangeMap.identity_map()

    op.bind_input(
        name="input",
        value=value,
        selector=selector,
    )

    return op.bind_result(
        name="result",
        selector=selector,
    )


def raw():
    g = TapestryGraph()

    x = g.add_node(
        PinnedTensor(
            name="X",
            shape=[20, 30, 2],
            dtype=torch.float16,
            storage="store:x",
        )
    )

    w1 = g.add_node(
        PinnedTensor(
            name="W1",
            shape=[2, 3],
            dtype=torch.float16,
            storage="store:w1",
        )
    )

    b1 = g.add_node(
        PinnedTensor(
            name="B1",
            shape=[3],
            dtype=torch.float16,
            storage="store:b1",
        )
    )

    a = linear_op_shard_w(x=x, w=w1, bias=b1)
    g.validate()
    return g

    y = relu_op(a)

    w2 = g.add_node(
        PinnedTensor(
            name="W1",
            shape=[3, 4],
            dtype=torch.float16,
            storage="store:w",
        )
    )

    z = relu_op(linear_op(x=y, w=w2))
    print(z.pretty())

    g.validate()

    return g


if __name__ == "__main__":
    raw()
