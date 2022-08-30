from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
from uuid import UUID

import torch

from tapestry.expression_graph import (
    AggregateTensor,
    BlockOperation,
    PinnedTensor,
    ReadSlice,
    TensorShard,
    TensorValue,
    WriteSlice,
)


def expect_tensor_shape_and_type(
    t: torch.Tensor,
    *,
    shape,
    dtype: torch.dtype,
    prefix: str = "Tensor ",
    suffix: str = ".",
):
    if t.shape != torch.Size(shape):
        raise AssertionError(
            f"{prefix} shape ({t.shape}) != expected shape ({shape}){suffix}",
        )
    if t.dtype != dtype:
        raise AssertionError(
            f"{prefix} dtype ({t.dtype}) != expected dtype ({dtype}){suffix}",
        )


EvEnv = Dict[Union[UUID, str], Union[torch.Tensor, Dict[UUID, torch.Tensor]]]
OpType = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


@dataclass
class Environment:
    values: Dict[UUID, torch.Tensor] = field(default_factory=dict)
    block_results: Dict[UUID, Dict[str, torch.Tensor]] = field(default_factory=dict)
    operations: Dict[str, OpType] = field(default_factory=dict)


def evaluate_tensor_value(
    node: TensorValue,
    *,
    env: Environment,
    seen: Optional[List[UUID]] = None,
) -> torch.Tensor:
    if node.node_id in env.values:
        return env.values[node.node_id]

    if seen and node.node_id in seen:
        raise ValueError(f"Cycle detected: {node.node_id} > {seen}")

    seen = [node.node_id] + (seen if seen else [])

    if isinstance(node, PinnedTensor):
        raise ValueError(f"PinnedTensor not bound in env: {node}")

    if isinstance(node, AggregateTensor):
        val = torch.zeros(tuple(node.shape), dtype=node.dtype)

        shards = node.shards()
        for shard in shards:
            val[shard.slice.as_slice()] = evaluate_tensor_value(
                shard, env=env, seen=seen
            )

        env.values[node.node_id] = val
        return val

    if isinstance(node, TensorShard):
        # assuming unevaluated TensorShards are produced by block ops.
        write_slice = node.assert_graph().get_singular_edge(
            WriteSlice,
            target_id=node.node_id,
        )
        op_shard = write_slice.source(BlockOperation.Shard)
        evaluate_op_shard(op_shard, env=env, seen=seen)

        # expect op_shard to populate this value.
        return env.values[node.node_id]

    raise ValueError(
        f"Unknown TensorValue type: {node}",
    )


def evaluate_read_slice(
    node: ReadSlice,
    *,
    env: Environment,
    seen: Optional[List[UUID]] = None,
) -> torch.Tensor:
    if node.node_id in env.values:
        return env.values[node.node_id]

    if seen and node.node_id in seen:
        raise ValueError(f"Cycle detected: {node.node_id} > {seen}")

    seen = [node.node_id] + (seen if seen else [])

    source = node.target(TensorValue)
    source_value = evaluate_tensor_value(source, env=env, seen=seen)

    zslice = node.slice

    if isinstance(source, TensorShard):
        # adjust slicing indexes for the shard slice values.
        assert node.slice in source.slice, (node.slice, source.slice)
        zslice = zslice - source.slice.start

    value = source_value[zslice.as_slice()]

    env.values[node.node_id] = value
    return value


def evaluate_op_shard(
    node: BlockOperation.Shard,
    *,
    env: Environment,
    seen: Optional[List[UUID]] = None,
) -> Dict[str, torch.Tensor]:
    if node.node_id in env.block_results:
        # TODO: check shapes
        return env.block_results[node.node_id]

    if seen and node.node_id in seen:
        raise ValueError(f"Cycle detected: {node.node_id} > {seen}")

    seen = [node.node_id] + (seen if seen else [])

    params: Dict[str, torch.Tensor] = {}

    for read_slice in node.inputs():
        val = evaluate_read_slice(
            read_slice,
            env=env,
            seen=seen,
        )
        params[read_slice.name] = val

    op = env.operations[node.operation]
    try:
        named_results = op(params)
    except Exception as e:
        raise AssertionError(
            f"BlockOperation failed:\n"
            f"{node}\n"
            f"{ {k: (v.shape, v.dtype) for k,v in params.items()} }"
        ) from e
    env.block_results[node.node_id] = named_results

    for write_slice in node.results():
        tensor_shard = write_slice.target(TensorShard)
        assert write_slice.slice == tensor_shard.slice
        env.values[tensor_shard.node_id] = named_results[write_slice.name]

    return named_results
