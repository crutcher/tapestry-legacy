from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple
import uuid

import torch

from tapestry.expression_graph import (
    AggregateTensor,
    BlockOperation,
    NodeIdCoercible,
    TapestryGraph,
    TensorShard,
    TensorValue,
    coerce_node_id,
)
from tapestry.zspace import ZArray, ZRange


class PlacementConstraintError(Exception):
    pass


@dataclass
class Placement:
    compute_width: int
    compute_delay: float


@dataclass
class Layout:
    node_placement: Dict[uuid.UUID, Placement] = field(default_factory=dict)

    def get_placement(self, node: NodeIdCoercible) -> Placement:
        node_id = coerce_node_id(node)
        return self.node_placement[node_id]


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


@dataclass
class PlacementCostModel:
    parallel_shard_io: bool = False
    transit_compression_ratio: float = 1.05

    def wallclock_cost(
        self,
        *,
        graph: TapestryGraph,
        layout: Layout,
    ) -> float:
        op_times = {
            op_shard.node_id: self._op_shard_delay(
                op_shard=op_shard,
                layout=layout,
            )
            for op_shard in graph.list_nodes(BlockOperation.Shard)
        }

        return self._slowest_path(graph=graph, op_times=op_times)

    def _tensor_transit_size(
        self,
        *,
        shape: ZArray,
        dtype: torch.dtype,
    ) -> int:
        return int(dtype.__sizeof__() * shape.prod() * self.transit_compression_ratio)

    def _placement_transit_time(
        self,
        *,
        transit_bytes: int,
        source_location: Placement,
        target_location: Placement,
    ) -> float:
        if source_location == target_location:
            return 0

        # FIXME:
        bytes_per_second = 1024

        return float(transit_bytes) / float(bytes_per_second)

    def _op_shard_compute_time(
        self,
        *,
        op_shard: BlockOperation.Shard,
        placement: Placement,
    ) -> float:
        compute_depth = op_shard.compute_depth
        if op_shard.compute_width > placement.compute_width:
            compute_depth *= math.ceil(op_shard.compute_width / placement.compute_width)

        return compute_depth * placement.compute_delay

    def _op_shard_delay(
        self,
        *,
        op_shard: BlockOperation.Shard,
        layout: Layout,
    ):
        op_shard_placement = layout.get_placement(op_shard)

        compute_time = self._op_shard_compute_time(
            op_shard=op_shard,
            placement=op_shard_placement,
        )

        read_times = []
        for read_slice in op_shard.inputs():
            # where is the real source?

            read_source = read_slice.target(TensorValue)
            dtype = read_source.dtype

            for primary_source, route_slice in route_tensor_read(
                tensor_value=read_source,
                read_slice=read_slice.slice,
            ):
                primary_source_placement = layout.get_placement(primary_source)

                transit_bytes = self._tensor_transit_size(
                    shape=route_slice.shape,
                    dtype=dtype,
                )

                transit_time = self._placement_transit_time(
                    transit_bytes=transit_bytes,
                    source_location=primary_source_placement,
                    target_location=op_shard_placement,
                )

                read_times.append(transit_time)

        write_times = []
        for write_slice in op_shard.results():
            write_target = write_slice.target(TensorValue)
            write_target_placement = layout.get_placement(write_target)
            dtype = write_target.dtype

            transit_bytes = self._tensor_transit_size(
                shape=write_slice.slice.shape,
                dtype=dtype,
            )

            transit_time = self._placement_transit_time(
                transit_bytes=transit_bytes,
                source_location=write_target_placement,
                target_location=op_shard_placement,
            )

            write_times.append(transit_time)

        if self.parallel_shard_io:
            aggregate_read_time = float(max([0.0] + read_times))
            aggregate_write_time = float(max([0.0] + write_times))
        else:
            aggregate_read_time = float(sum([0.0] + read_times))
            aggregate_write_time = float(sum([0.0] + write_times))

        delay = aggregate_read_time + compute_time + aggregate_write_time

        return delay

    def _slowest_path(
        self,
        *,
        graph: TapestryGraph,
        op_times: Dict[uuid.UUID, float],
    ) -> float:
        path_times: Dict[uuid.UUID, float] = {}
        return max(
            self._path_time(
                graph=graph,
                node_id=node_id,
                op_times=op_times,
                path_times=path_times,
            )
            for node_id in graph.observed
        )

    def _path_time(
        self,
        *,
        graph: TapestryGraph,
        node_id: uuid.UUID,
        op_times: Dict[uuid.UUID, float],
        path_times: Dict[uuid.UUID, float],
    ) -> float:
        if node_id in path_times:
            return path_times[node_id]

        node = graph.get_node(node_id)

        if isinstance(node, BlockOperation.Shard):
            path_time = op_times[node_id] + max(
                self._path_time(
                    graph=graph,
                    node_id=read_slice.node_id,
                    op_times=op_times,
                    path_times=path_times,
                )
                for read_slice in node.inputs()
            )

        elif isinstance(node, AggregateTensor):
            path_time = max(
                self._path_time(
                    graph=graph,
                    node_id=shard.node_id,
                    op_times=op_times,
                    path_times=path_times,
                )
                for shard in node.shards()
            )

        elif isinstance(node, TensorShard):
            if op_shard := node.writer():
                path_time = self._path_time(
                    graph=graph,
                    node_id=op_shard.node_id,
                    op_times=op_times,
                    path_times=path_times,
                )
            else:
                path_time = 0.0

        else:
            raise ValueError(f"Unsupported node type: {repr(node)}")

        path_times[node_id] = path_time
        return path_time
