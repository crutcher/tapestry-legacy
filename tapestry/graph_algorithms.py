import numpy as np

from tapestry.expression_graph import (
    AggregateTensor,
    BlockOperation,
    ReadSlice,
    TapestryGraph,
    TensorValue,
)


def add_total_shards(g: TapestryGraph) -> None:
    with g.relax():
        for op in g.list_nodes(BlockOperation):
            op.add_shard(op.index_space)


def section_plan_max_dim(g: TapestryGraph, shards: int) -> None:
    with g.relax():
        planned_blocks = {e.source_id for e in g.list_tags(BlockOperation.SectionPlan)}

        for op in g.list_nodes(
            BlockOperation,
            filter=lambda n: n.node_id not in planned_blocks,
        ):
            max_dim = int(op.index_space.shape.argmax())
            k = int(min(shards, op.index_space.shape[max_dim]))

            sections = np.ones(op.index_space.ndim, dtype=int)
            sections[max_dim] = k

            op.attach_section_plan(sections)


def expand_section_plans(g: TapestryGraph, remove: bool = True) -> None:
    with g.relax():
        for section_plan in g.list_tags(BlockOperation.SectionPlan):
            op = section_plan.source(BlockOperation)

            for part in op.index_space.section(section_plan.sections):
                op.add_shard(part)

            if remove:
                g.remove_node(section_plan, remove_edges=True)


def shard_max_dim(g: TapestryGraph, shards: int) -> None:
    section_plan_max_dim(g=g, shards=shards)
    expand_section_plans(g, remove=True)

    g.validate_if_enabled()


def strip_blocks(g: TapestryGraph) -> None:
    with g.relax():
        for node in g.list_nodes(BlockOperation):
            g.remove_node(node, remove_edges=True)


def specialize_read_slices(g: TapestryGraph) -> None:
    """
    Rewrite the graph to redirect ReadSlice links from AggregateTensors
    to TensorShards, if there is a TensorShard that entirely contains them.
    """
    with g.relax():
        for read_slice in g.list_edges(ReadSlice):
            value = read_slice.target(TensorValue)
            if isinstance(value, AggregateTensor):
                for shard in value.shards():
                    if read_slice.slice in shard.slice:
                        # NOTE: changing the target node here,
                        # we don't re-check the target validity.
                        read_slice.target_id = shard.node_id
                        break


def strip_orphan_values(g: TapestryGraph) -> None:
    with g.relax():
        for tensor_value in g.list_nodes(
            TensorValue,
            restrict=AggregateTensor,
        ):
            node_id = tensor_value.node_id
            if node_id in g.observed:
                continue

            if g.list_edges(
                target_id=node_id,
                restrict=(ReadSlice, BlockOperation.Input),
            ):
                continue

            g.remove_node(tensor_value, remove_edges=True)
