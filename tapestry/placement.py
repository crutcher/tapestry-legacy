import uuid
from dataclasses import dataclass
from typing import Dict, Callable

import torch

from tapestry.expression_graph import TapestryGraph, BlockOperation, TensorValue
from tapestry.zspace import ZRange


# (TapestryGraph, { BlockOperation.Shard: Placement })

# (Placement, Placement) => cost

@dataclass
class Placement:
    pass
    # id
    # mem
    # compute


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


def transit_size(shape: ZRange, dtype: torch.dtype, *, overhead: float = 0.05,) -> int:
    return int(dtype.__sizeof__() * shape.size * (1 + overhead))

def shard_compute_time(
        shard: BlockOperation.Shard,
        placement: Placement,
) -> float:

def k(
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
        # shard_placement = ???
        shard_placement: Placement
        check_placement(shard=shard, placement=shard_placement)

        read_times = []

        for read_slice in shard.inputs():
            # slice_placement = ???
            source_placement: Placement
            read_source = read_slice.target(TensorValue)

            if source_placement == shard_placement:
                continue

            ts = transit_size(
                shape=read_slice.shape,
                dtype=read_source.dtype,
            )
            bps = bandwidth_bps_cb(
                source_placement,
                source_placement,
            )

            transit_time = float(ts) / float(bps)

            read_times.append(transit_time)

        if not read_times:
            load_time = 0.0
        elif parallel_reads:
            load_time = float(max(read_times))
        else:
            load_time = float(sum(read_times))

        # compute time
        # compute width
        # compute depth
