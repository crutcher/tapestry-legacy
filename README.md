# Tapestrey
crutcher@gmail.com, crutcher@meta.com

## Terminology

### Tensor Fabrics

Lacking a better name for "a lot of CUDA-device/PUs/TPUs networked together",
I'm going to call the general idea a "Tensor Fabric", as "fabric" is a pretty
common term in networking, and distributed computing, and "tensors" are what
we're interested in.

## Goals
Tapestry is a project to derive an initial proof-of-concept for a restricted
arbitrary-scale tensor compute evaluation environment.


We'd like to be able to say, for some large family of algorithms,
that we have an embedding `E` such that embedding the data and the algorithm
into our embedding environment produces the same result, and *goes really fast*:

    data -> <algorithm> -> result

    E(data) -> E(<algorithm>) -> E(result)

The goal is to be able to write tensor algorithms in an embedding environment
accessible to programmers, which maintain enough meta-information that an execution
environment can provably scale those algorithms to arbitraryly large distributed
compute CUDA/GPU/TPU tensor-fabric resources both accurately (producing the correct
results) and efficiently (making maximum use of available compute resources).

Embedding environments are a concept from both formal semantics in computation
language design, and in category theory in the definition of Functors. They define
environments which, if we follow the operation and composition rules, we can prove
that transformations on the expressions which maintain the environment's invariants
will continue to produce the same results, or "mean" the same things.

Sometimes described using the "diagram chasing" equations of category theory;
we can assemble proofs of equivalence under transformation by establishing
that a computation performed outside the environment will arrive at the same
result as one performed by first embedding the data and the operation into the
environment:

     a    ->   op    ->   op(a)

     |         |           |
     F         F           F
     |         |           |
     v         V           V

    F(a)  ->  F(op)  ->  F(op)(b)
    

The goal of an arbitrary scale compute environment is that scale is a function
of the evaluator, not the algorithms; to the limits of
[Ahmdal's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law); this is an ambitious
project, so we start with that goal and attempt to work backwards, deriving
mechanics that do not make the problem worse.

> #### What do me mean by "derive"?
> As the goal is to produce a provable embedding environment, we start with the
> rules of such an environment, and incrementally add operations which do not
> violate those rules, attempting to work backwards from simple operations
> to an environment which is actually useful for defining real programs.
> 
> At each step, we select the simplest consistent extensions we can find
> which maintain the properties we're interested in, so in some sense the
> R&D approach resembles derivation over design.
> 
> The goal at the end of the project is to be able to hand an arbitrarily
> complex algorithm to an arbitrarily aggressive execution scheduler, and
> know, provably, that the results of execution will be identical to the
> most naive scheduler we can define over the same graph.
> 
> This approach has been very successful in the world of dataflow languages,
> SQL query planners, and the Haskell / Functional Programming stream
> fusion environments, and the research project is to apply the same ideas
> to tensor operations with kernel window visibility.

We are not going to attempt to define an execution environment which can arbitrarily
scale arbitrary programs, the halting problem proves that approach will not be successful;
we're going to attempt to restrict our embedding operations to those which can
be described in a subset of scalable operations, and build up a formal semantics
upon that framework.


## Design Overview

Existing dataflow environments already permit massive horizontal scaling of
parallel operations over parallel collections.

For reference, see:
  * [Apache Spark](https://github.com/apache/spark)
  * [Apache Beam](https://beam.apache.org/)
  * [Google Dataflow](https://cloud.google.com/dataflow)

These environments have several components:
  * A coordinator environment - a traditional sequential programming environment
    which interacts with the external world, constructs operation graphs, dispatches 
    those graphs to be executed, awaits their results, and potentially kicks off 
    subsequent dependent calculations.
  * An operation environment - constructed of idempotent parallel collections and operations
    which describes operations in a restricted embedding environment, and whose
    execution scheduling is managed by the scaling executor.

If we focus on what these environments provide, in terms of parallel collections,
we'll see that in general these collections exist lacking neighbor locality,
one point of data in a given collection has no guaranteed locality relative
to any other point of data.

Tensor algorithms frequently operate on convolutions over shared data, we could
express most tensor operations in terms of dataflow environments, at the cost
of losing all of the block data and operation acceleration which tensor fabric
processors (CUDA/GPU/TPU/etc) provide.

The design goal is to copy most of the ideas from existing dataflow languages
about the structure of a coordinator environment, while extending the notion
of *parallel collections* and *parallel operations* to describe data-locality
of the operation data needs, and time-locality of the CUDA kernels.