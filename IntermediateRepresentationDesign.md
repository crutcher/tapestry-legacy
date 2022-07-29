# Intermediate Representation Design

* [Table of Contents](README.md)
* [Previous: Block Graph](BlockGraph.md)

We need a graph IR (Intermediate Representation) implementation to describe block graphs, and the 
series of incremental transformations over them.

Since there are several layers of transformations needed to schedule block graphs; the 
amenability of the IR to serialization, debugging, and type enforcement seems valuable.

I'm taking the following requirements on this problem:

* *Python* - the goal is accessibility to the python pytorch community.
* *JSON* - the representation must be serializable to/from JSON.
* *graphviz* - we need to be able to generate visualization graphs.
* *typed nodes* - we have strong ideas about what is in a node.
* *extensible* - it should be easy to write incremental languages.

### Related Work

#### NetworkX

The [NetworkX](https://networkx.org/documentation/stable/index.html) codebase provides graph 
libraries for large graphs, and analysis over them. The focus is on the link structure, and not 
so much on the data inside a node or edge (though there's some basic support for that).

I spent some time thinking about how to make this work.

## Alternative Structures


## Fat Reified Nodes, Fat Reified Edges

One approach is to give both nodes and edges:

* a unique id (UUID?)
* a type (to index a type schema)
* data attributes (defined by type schema)

```json
{
  "nodes": [
    {
      "id": <UUID>,
      "type": <str>,
      "attributes": {
        <str>: <JSON>,
      }
    }
  ],
  "edges": [
    {
      "id": <UUID>,
      "type": <str>,
      "source": <Node UUID>,
      "target": <Node UUID>,
      "attributes": {
        <str>: <JSON>,
      }
    },
  ],
}

```

Both nodes and edges have ids here (we could use global indexes, but UUIDs work pretty well, and 
have nice history properties between graph versions).

Both nodes and edges have types, permitting us to tie them to schemas; and attributes (presumably
defined by those schemas in terms of the types).

Type + attributes is enough to bind a serialization/deserialization library for data content.

This mechanism permits multi-edges (we can have more than one edge of a given type between two 
nodes), so things like "children" are well-described; however multi edges aren't ordered; 
there's no natural way to say "the first child" in this situation.

The JSON also just doesn't *read* well.

## Fat Reified Nodes, Edges are Node Children

```json
{
  "nodes": [
    {
      "id": <UUID>,
      "type": <str>,
      "attributes": {
        <str>: <JSON>
      },
      "targets": {
        "<edge_type>": [ <UUID>, ],
      }
    }
  ],
  "edges": [
    {
      "id": <UUID>,
      "type": <str>,
      "source": <Node
      UUID>,
      "target": <Node
      UUID>,
      "attributes": {
        <str>: <JSON>
      }
    }
  ]
}

```

Both nodes and edges have ids here (we could use global indexes, but UUIDs work pretty well, and
have nice history properties between graph versions).

Both nodes and edges have types, permitting us to tie them to schemas; and attributes (presumably
defined by those schemas in terms of the types).

Type + attributes is enough to bind a serialization/deserialization library for data content.

This mechanism permits multi-edges (we can have more than one edge of a given type between two
nodes), so things like "children" are well-described; however multi edges aren't ordered;
there's no natural way to say "the first child" in this situation.

The JSON also just doesn't *read* well.

## Brand X

```json
{
  "nodes": [
    {
      "id": UUID,
      "type": STR,
      "attributes": {
        STR: JSON
      },
      "targets": {
        "<edge_type>": [
          {
            "target": UUID,
            "attributes": {
              "<str>": JSON
            }
          }
        ]
      }
    }
  ]
}

```

## Brand X
```
BlockOp:
  id
  operation_id: str
  index_space: { start, end }
  projecton_maps: { k: ZP }
  inputs: { k: REF }
  outputs: { k: REF }

```

## Next

* [Table of Contents](README.md)
