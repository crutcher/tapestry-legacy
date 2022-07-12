# Tapestry

crutcher@gmail.com, crutcher@meta.com

## Abstract

Tapestry is a project to prototype "Spark, for GPU Accelerated AI/ML Tensor Algorithms".

There would be significant value in divorcing the development of tensor applications from the scheduling and
efficient execution of those applications. One problem requires specialized training in statistics, machine learning,
physics, or some other branch of math that cares about tensors; the other requires specialized training in
scheduling theory, distributed system engineering, and compiler design.

Exiting dataflow environments are already
[embarasingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), able to exploit large numbers of workers
simultaneously, while computing effective function dependencies between calculations.

Most tensor algorithms used in AI/ML are, in principle, embarrassingly parallel at the level of the functional
dependencies of individual data cells, but existing tensor algorithm expression languages make it difficult for
execution environments to exploit this property.

The challenge of functionally exploiting parallelism in tensor algorithms across machines lies in:

* expressing the parallelism opportunities to the scheduler,
* while maintaining spatial and temporal locality needed to exploit CUDA/GPU/TPU environments,
* and exposing the above in a framework which remains understandable to application developers.

A significant amount of work has been done in decorating existing AI/ML algorithms with parallelism points, either
by batching workers, or splitting layers across devices, or both; and in attempting to determine locations that
those annotations can be safely auto-inserted by sufficiently smart compilers.

Tapestry aims to demonstrate an approach to the problem from the other side, by demonstrating extensions to the
dataflow graph execution model (Parallel Collections + Parallel Operations) which permit fine-grained description of
the covariance between tensor spatial locality and CUDA kernel time locality in operations.

There are 4 primary components needed to demonstrate this:

* a demonstration that AI/ML algorithms can fit in dataflow languages at all,
* a demonstration of at least one working solution to space/time covariance graph description,
* a demonstration that that covariance description can be used to schedule dense operations,
* a demonstration that an api built on this can be aligned with existing AI/ML design patterns.

## API Design Overview

Existing dataflow environments already permit massive horizontal scaling of parallel operations over parallel
collections.

For reference, see:

* [Apache Spark](https://github.com/apache/spark)
* [Apache Beam](https://beam.apache.org/)
* [Google Dataflow](https://cloud.google.com/dataflow)

These environments have several components:

* *a coordinator environment* - a traditional sequential programming environment which interacts with the external
  world, constructs operation graphs, dispatches
  those graphs to be executed, awaits their results, and potentially kicks off subsequent dependent calculations.
* *an operation environment* - constructed of idempotent parallel collections and operations which describes operations
  in a restricted embedding environment, and whose execution scheduling is managed by the scaling executor.

Writing applications in these environments does require additional training; while the runtime expectations of the
coordinator environments are relatively serial (construct an operation graph, dispatch it for execution, wait for
completion, observe the results and make further control flow choices); the runtime expectations of the operation
environment (hermetic, idempotent, no IO to other systems, no iteration or batch visibility), and the semantics
of the parallel collections in the operation environment require additional training over traditional serial/local
execution environments.

However, decades of api research in this space have produced many reusable design patterns; not only in the way to
structure and debug these APIs in usable ways, but also in approaches towards collecting them into higher-level
primitives.

As we see in dataflow languages, and can expect to see here, most users, combining pre-built layers and combinators
of standard components, should not need to know or care how the underlying covariance is described, or how the
underlying kernels are scheduled.

### Dataflow Environments use Category Theory for the Win

Dataflow languages have converged on an observation about distributed scheduling; by *strictly* restricting primitive
operations, and algorithms built up from graphs of those operations, to a few simple ideas from category
theory:

* maps (functions),
* monoids (reduces),
* and arrows (chained composition)

We can build arbitrarily aggressive compilation, scheduling, and execution environments which provably produce
the same results.

Each of these ideas come with a few laws about their operation behavior; and we can use embedding
proofs to prove that an algorithm whose component parts do not violate those rules, written in these terms of these
ideas, can be mechanically restructured to large number of equivalent algorithms in different embeddings, provided that
the embeddings maintain the invariants of those rules.

The transformed program is guaranteed to be equivalent to the source program:

![functor](media/graphs/functor.dot.png)

Armed with this proof, a large family of mechanical optimizations have already been developed which bring execution
throughput and reliability of large programs to levels unreachable by human-maintained algorithms; because the
fusion results would be too complex to maintain directly; transparent retries, operation fusion, stacked operation
rewrites, local recomputation, result caching; all in the same execution environment, *for free* from the
perspective of application developers.

Additionally, in practice, we see that these languages permit specialization of R&D streams:

* library/application developers using high-level operations to build functions, reductions, and graphs;
* category theory developers writing new composite high-level operations to expose functionality in reusable ways;
* and a (much smaller) group of system developers building and optimizing the execution / embedding environments.

*Particularly, advancement of research and development at the execution layer accelerates all programs.*

### But Does it Blend?

CUDA/GPU/TPU execution environments are fast because they can dispatch dense operations in parallel on vector unit
hardware. A large family of kernel operations have been developed to take advantage of that hardware, and at first
glance making the argument that "Everything fits in maps, monoids, and arrows" is not obvious.

Tapestry will attempt to demonstrate that the following common AI/ML components can be meaningfully embedded on a
framework of (map, monoid, arrow), and densely scheduled to memory and vector device:

* Activation - _trivial, this is just map_
* Convolution - _this is *also* map, but it's less obvious_
* Matmul / nn.Linear - _this is map if weights are small, and map+monoid if they are not_
* Sum, Variance - _this is monoid_

## Exploring Feasibility of Embedding nn.Linear

Consider [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear), which
can be described as operating on the tensors (X, W, B) and producing (Y):

    Y = X W + B

![linear.f1](media/graphs/linear.f1.dot.png)

On a single CUDA/GPU/TPU device, we can dispatch this entire operation to a matmul kernel followed by an addition
kernel, or we can dispatch to a single specialized linear or affine transform kernel.

### Rewriting Linear over i (batch)

By examination of the *Linear* operation, we can see that rewriting into smaller operations over
the *i* (batch) dimension, and mechanically merging the results, will produce the same result:

![linear.f3](media/graphs/linear.f2.dot.png)

Under fixed *W* and *B*, nn.Linear is a *map* over the *i* (*batch*) input dimension. And we can schedule this densely,

* the map shards spatially along *i* in data resources,
* the map shards temporally along *i* in compute resources,
* the component blocks still have dense CUDA kernels for efficient dispatch of their smaller data.

### Rewriting Linear over n (nodes)

Again, by examination of the *Linear* operation, we can see that rewriting into smaller operations over the *n*
(nodes) dimension, and mechanically merging the results along a different axis, will produce the same results.

![linear.f3](media/graphs/linear.f3.dot.png)

In this case, we spatially shard both *W* and *b*, but not *X*; but we still yield a map over the *n* (node) dimension.

* the map shards spatially along *n* in data resources,
* the map shards temporally along *n* in compute resources,
* the component blocks still have dense CUDA kernels for efficient dispatch of their smaller data.

### Defining an index space over Linear

Combining these observations over the operations in Linear, we can invent an *index* space which corresponds to the
functional dependiences of the result cells, and maps to slices of the input and output operations:

![linear.index1](media/graphs/linear.index1.dot.png)

This brings us to a property we'd like to preserve over index spaces, *coherency*:

* Any contiguous partitioning over the *index* space will be equivalent to some combination of rewrites over sharding 
by *i* and sharding by *n*, and will produce spatially coherent leaf operations to dispatch to CUDA/GPU/TPU kernels.

![linear.index2](media/graphs/linear.index2.dot.png)

### Rewriting Linear over m (axial sum reduction)

Examining the data dependencies of *Linear*, we see we cannot rewrite over *m* and still use the coherent leaf 
operation; each cell of output depends on an entire row of the *x* tensor, and an entire column of the *w* tensor.

But if we're willing to examine the content of the leaf, we can see that:

    y(i, n) := b(n) + sum(a=[0, m-1], x(i,a) * w(a,n))
    y(i, n) := b(n) + sum(a=[0, m-1], x(i,a) * w(a,n))
    y(i, n) := b(n) + sum(a=[0, k], x(i,a) * w(a,n)) + sum(a=[k+1, m], x(i,a) * w(a,n))

If we now introduce two alternative leaf operations *matmul* and *sum*; we can rewrite this by introducing a new
accumulation dimension *k* for the sharded partial results:

![linear.axial1](media/graphs/linear.axial1.dot.png)

This rewrite requires us to understand the relationship between alternative leaf operations, and make that visible 
to the graph scheduler.

### Summarizing Linear Rewrites

At this point, we've found a number of production rules defining equivalent embeddings of *Linear*:

* *Linear* := { *Linear* }
* *Linear* := *matmul* => *sum*
* *matmul* := { *matmul* }
* *sum* := { *sum* }

Each of these transformations will produce an operation graph with different space/time costs to evaluation; and
starting with a single *Linear* operation, we can search over execution plans to minimize that cost, without any 
change to the initial application's generation of that operation graph.

