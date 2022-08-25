from tapestry.expression_graph import BlockOperation, TapestryGraph


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
