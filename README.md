# Tapestry

crutcher@meta.com, crutcher@gmail.com

## Abstract

Tapestry is a project to prototype a rewriting and scheduling framework for dense block tensor
graph expressions; "Spark, for GPU Accelerated AI/ML Tensor Algorithms".

The goal is to be able to take abstract block operation graphs over tensor spaces:

![abstract](media/graphs/graph1.dot.png)

And yield sharded and fused operation graphs which can be scheduled effectively over distributed 
resources:

![compiled](media/graphs/graph2.dot.png)

The approach is fine-grained modeling of marginal sharding at the block operation level, 
enabling aggressive semantics preserving graph rewrites.

## Discussion

Please direct questions to the github issues for this project: 

* [tapestry issues](https://github.com/crutcher/tapestry/issues)

## Background

There would be significant value in divorcing the development of tensor applications from the
scheduling and efficient execution of those applications. One problem requires specialized training
in statistics, machine learning, physics, or some other branch of math that cares about tensors;
the other requires specialized training in scheduling theory, distributed system engineering,
and compiler design.

Exiting dataflow environments are already
[embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), able to
exploit large numbers of workers simultaneously, while computing effective function dependencies
between calculations.

Most tensor algorithms used in AI/ML are, in principle, embarrassingly parallel at the level of the
functional dependencies of individual data cells, but existing tensor algorithm expression languages
make it difficult for execution environments to exploit this property.

The challenge of functionally exploiting parallelism in tensor algorithms across machines lies in:

* expressing the parallelism opportunities to the scheduler,
* while maintaining spatial and temporal locality needed to exploit CUDA/GPU/TPU environments,
* and exposing the above in a framework which remains understandable to application developers.

A significant amount of work has been done in decorating existing AI/ML algorithms with parallelism
points, either by batching workers, or splitting layers across devices, or both; and in attempting
to determine locations that those annotations can be safely auto-inserted by sufficiently smart
compilers.

Tapestry aims to demonstrate an approach to the problem from the other side, by demonstrating
extensions to the dataflow graph execution model (Parallel Collections + Parallel Operations) which
permit fine-grained description of the covariance between tensor spatial locality and CUDA kernel
time locality in operations.

If we can describe the tensor block operations in a tensor operation graph in a way which
permits mechanical sharding decisions, we can rewrite bulk abstract operation graphs into
smaller operations in mechanical ways which preserve their semantics.

If we have an execution cost model for tensor block operation graphs, and can describe the marginal
data sharing and compute costs of different choices of block split, we can search for graph
rewrites which fit target constraints (no more than 40 machines), or optimize target goals
(minimal wallclock time, or minimal power) by casting the block split / scheduling problem as a
multidimensional resource scheduling problem.

There are 4 primary components needed to demonstrate this:

* a demonstration that AI/ML algorithms can fit in dataflow languages at all,
* a demonstration of at least one working solution to space/time covariance graph description,
* a demonstration that that covariance description can be used to schedule dense operations,
* a demonstration that an api built on this can be aligned with existing AI/ML design patterns.

## Table of Contents

* [Graph Rewrites](GraphRewrites.md)  
  Describes the general problem of incremental lowering via graph rewriting.
* [API Design Overview](ApiDesign.md)  
  Gives an overview on design goals for a usable API.
* [Exploring Tensor Block Sharding Feasibility](BlockSharding.md)  
  Explores common tensor block algorithms, and feasibility on extracting sharding patterns.
* [Index Projection Design](IndexProjectionDesign.md)  
  Derives an index projection function design from index tensor block sharding exploration.
* [Block Graph](BlockGraph.md)  
  Describes the components of a block operation graph.


### Related Work

* *FAX* on *jax.pjit* \
  https://www.arxiv-vanity.com/papers/2204.06514/ \
  Appears to be pursuing similar slicing research, using *jax.pjit* as a backend.


