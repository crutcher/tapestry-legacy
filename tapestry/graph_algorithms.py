from tapestry.expression_graph import (
    AggregateTensor,
    BlockOperation,
    PinnedTensor,
    ReadSlice,
    TapestryGraph,
    TensorValue,
)


def add_total_shards(g: TapestryGraph) -> None:
    for op in g.list_nodes(BlockOperation):
        op.add_shard(op.index_space)

    g.validate()


def shard_max_dim(g: TapestryGraph, shards: int) -> None:
    for op in g.list_nodes(BlockOperation):
        max_dim = int(op.index_space.shape.argmax())
        k = int(min(shards, op.index_space.shape[max_dim]))
        for part in op.index_space.split(axis=max_dim, sections=k):
            op.add_shard(part)

    g.validate()


def strip_blocks(g: TapestryGraph) -> None:
    for node in g.list_nodes(BlockOperation):
        g.remove_node(node, remove_edges=True)


def specialize_read_slices(g: TapestryGraph) -> None:
    """
    Rewrite the graph to redirect ReadSlice links from AggregateTensors
    to TensorShards, if there is a TensorShard that entirely contains them.
    """
    for read_slice in g.list_edges(ReadSlice):
        value = read_slice.target(TensorValue)
        if isinstance(value, AggregateTensor):
            for shard in value.shards():
                if read_slice.slice in shard.slice:
                    read_slice.target_id = shard.node_id
                    break

    g.validate_if_enabled()


def strip_orphan_values(g: TapestryGraph) -> None:
    with g.relax():
        for tensor_value in g.list_nodes(
            TensorValue,
            restrict=AggregateTensor,
            exclude=PinnedTensor,
        ):
            node_id = tensor_value.node_id
            if node_id in g.observed:
                continue

            if g.list_edges(
                target_id=node_id, restrict=(ReadSlice, BlockOperation.Input)
            ):
                continue

            g.remove_node(tensor_value, remove_edges=True)
