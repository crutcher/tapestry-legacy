from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Tuple
import uuid

import torch

from tapestry.expression_graph import (
    AggregateTensor,
    BlockOperation,
    TapestryGraph,
    TensorShard,
    TensorValue,
)
from tapestry.zspace import ZArray, ZRange

# (TapestryGraph, { BlockOperation.Shard: Placement })

# (Placement, Placement) => cost


@dataclass
class Placement:
    pass
    # id
    # mem
    # compute

    compute_width: int
    compute_delay: float


class PlacementEnv:
    pass

    # p ->


class PlacementConstraintError(Exception):
    pass


def check_placement(
    shard: BlockOperation.Shard,
    placement: Placement,
) -> None:
    raise PlacementConstraintError("TBD")


def transit_size(
    shape: ZArray,
    dtype: torch.dtype,
    *,
    overhead: float = 0.05,
) -> int:
    return int(dtype.__sizeof__() * shape.prod() * (1 + overhead))


def route_tensor_read(
    *,
    tensor_value: TensorValue,
    read_slice: ZRange,
) -> List[Tuple[TensorValue, ZRange]]:
    if isinstance(tensor_value, TensorShard):
        return [(tensor_value, read_slice)]

    elif isinstance(tensor_value, AggregateTensor):
        res = []
        for shard in tensor_value.shards():
            shard_slice = shard.slice.intersection(read_slice)
            if shard_slice.size == 0:
                continue

            res.extend(
                route_tensor_read(
                    tensor_value=shard,
                    read_slice=shard_slice,
                ),
            )

        return res

    raise AssertionError(f"Expanding unresolvable {tensor_value}")


def schedule_cost(
    g: TapestryGraph,
    schedule: Dict[uuid.UUID, Placement],
    bandwidth_bps_cb: Callable[[Placement, Placement], int],
    parallel_reads: bool = False,
):
    # metrics we care about:
    #
    # * constraints - does this placement fit on the graph?
    #
    # * wallclock time - the slowest path through the graph.
    #   compute with flood fill.
    #
    # * {mem, comp} utilization - ratio of peak needs to
    #   available needs.

    for shard in g.list_nodes(BlockOperation.Shard):
        shard_placement = schedule[shard.node_id]
        check_placement(shard=shard, placement=shard_placement)

        read_times = []

        for read_slice in shard.inputs():
            # where is the real source?

            read_source = read_slice.target(TensorValue)
            dtype = read_source.dtype

            for primary_source, route_slice in route_tensor_read(
                tensor_value=read_source,
                read_slice=read_slice.slice,
            ):
                primary_source_placement = schedule[primary_source.node_id]

                if primary_source_placement == shard_placement:
                    continue

                ts = transit_size(
                    shape=route_slice.shape,
                    dtype=dtype,
                )
                bps = bandwidth_bps_cb(
                    primary_source_placement,
                    shard_placement,
                )

                transit_time = float(ts) / float(bps)

                read_times.append(transit_time)

        if not read_times:
            load_time = 0.0
        elif parallel_reads:
            load_time = float(max(read_times))
        else:
            load_time = float(sum(read_times))

        compute_depth = shard.compute_depth
        if shard.compute_width > shard_placement.compute_width:
            compute_depth *= math.ceil(
                shard.compute_width / shard_placement.compute_width
            )

        compute_time = compute_depth * shard_placement.compute_delay

        # not modeled at this point.
        store_time = 0.0

        shard_time = load_time + compute_time + store_time

        # compute time
        # compute width
        # compute depth
