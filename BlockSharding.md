# Exploring Tensor Block Sharding Feasibility

* [Table of Contents](README.md)
* [Previous](ApiDesign.md)

CUDA/GPU/TPU execution environments are fast because they can dispatch dense operations in parallel
on vector unit hardware. A large family of kernel operations have been developed to take advantage
of that hardware, and at first glance making the argument that "Everything fits in maps, monoids,
and arrows" is not obvious.

Tapestry will attempt to demonstrate that the following common AI/ML components can be meaningfully
embedded on a framework of (map, monoid, arrow), and densely scheduled to memory and vector device:

* Activation - _trivial, this is just map_
* Convolution - _this is *also* map, but it's less obvious_
* Matmul / nn.Linear - _this is map if weights are small, and map+monoid if they are not_
* Sum, Variance - _this is monoid_

## Exploring Feasibility of Embedding nn.Linear

Consider [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
, which can be described as operating on the tensors (X, W, B) and producing (Y):

    Y = X W + B

![linear.f1](media/graphs/linear.f1.dot.png)

On a single CUDA/GPU/TPU device, we can dispatch this entire operation to a matmul kernel followed
by an addition kernel, or we can dispatch to a single specialized linear or affine transform kernel.

### Rewriting Linear over i (batch)

By examination of the *Linear* operation, we can see that rewriting into smaller operations over
the *i* (batch) dimension, and mechanically merging the results, will produce the same result:

![linear.f3](media/graphs/linear.f2.dot.png)

Under fixed *W* and *B*, nn.Linear is a *map* over the *i* (*batch*) input dimension. And we can
schedule this densely,

* the map shards spatially along *i* in data resources,
* the map shards temporally along *i* in compute resources,
* the component blocks still have dense CUDA kernels for efficient dispatch of their smaller data.

### Rewriting Linear over n (nodes)

Again, by examination of the *Linear* operation, we can see that rewriting into smaller operations
over the *n* (nodes) dimension, and mechanically merging the results along a different axis, will
produce the same results.

![linear.f3](media/graphs/linear.f3.dot.png)

In this case, we spatially shard both *W* and *b*, but not *X*; but we still yield a map over the *
n* (node) dimension.

* the map shards spatially along *n* in data resources,
* the map shards temporally along *n* in compute resources,
* the component blocks still have dense CUDA kernels for efficient dispatch of their smaller data.

### Defining an index space over Linear

Combining these observations over the operations in Linear, we can invent an *index* space which
corresponds to the functional dependencies of the result cells, and maps to slices of the input and
output operations:

![linear.index1](media/graphs/linear.index1.dot.png)

This brings us to a property we'd like to preserve over index spaces, *coherency*:

* Any contiguous partitioning over the *index* space will be equivalent to some combination of
  rewrites over sharding by *i* and sharding by *n*, and will produce spatially coherent leaf
  operations to dispatch to CUDA/GPU/TPU kernels.

![linear.index2](media/graphs/linear.index2.dot.png)

### Rewriting Linear over m (axial sum reduction)

Examining the data dependencies of *Linear*, we see we cannot rewrite over *m* and still use the
coherent leaf operation; each cell of output depends on an entire row of the *x* tensor, and an
entire column of the *w* tensor.

But if we're willing to examine the content of the leaf, we can see that:

    y(i, n) := b(n) + sum(a=[0, m-1], x(i,a) * w(a,n))
    y(i, n) := b(n) + sum(a=[0, m-1], x(i,a) * w(a,n))
    y(i, n) := b(n) + sum(a=[0, k], x(i,a) * w(a,n)) + sum(a=[k+1, m], x(i,a) * w(a,n))

If we now introduce two alternative leaf operations *matmul* and *sum*; we can rewrite this by
introducing a new accumulation dimension *k* for the sharded partial results:

![linear.axial1](media/graphs/linear.axial1.dot.png)

This rewrite requires us to understand the relationship between alternative leaf operations, and
make that visible to the graph scheduler.

### Summarizing Linear Rewrites

At this point, we've found a number of production rules defining equivalent embeddings of *Linear*:

* *Linear* := { *Linear* }
* *Linear* := *matmul* => *sum*
* *matmul* := { *matmul* }
* *sum* := { *sum* }

Each of these transformations will produce an operation graph with different space/time costs to
evaluation; and starting with a single *Linear* operation, we can search over execution plans to
minimize that cost, without any change to the initial application's generation of that operation
graph.

## Exploring Feasibility of Embedding Conv

To explore feasibility of embedding *torch.nn.Conv*, we have to discuss coherent overlapping view
regions.

Neighboring *Conv* result cells frequently consume overlapping input data:

![conv.f1](media/graphs/conv.f1.dot.png)

If we take our projection from *index* space to the data spaces for individual points in the *index*
space, and compute their overlap, rather than their union; we are still left with coherent blocks to
pass to the leaf operation. This gives us another constraint on our design:

* Coherent blocks of index projections should yield coherent overlapping blocks to input and output
  tensors.

Things become more complicated when we consider stride convolutions, where neighboring cells may not
consume the same data:

![conv.f2](media/graphs/conv.f2.dot.png)

Naively, our input regions are now non-coherent; and we have a design choice in this situation.

* Compute the overlapping region, which reduces data sharing between operations; or
* Pre-Slice the input tensors into strided tensors, rewrite the strides and index space provided to
  the leaf kernels.

Consider:

![conv.f3](media/graphs/conv.f3.dot.png)

This transformation has the same computational leaf cost; but permits us to recover dense neighbor
data sharing of strided conv operations; which can be useful in achieving more efficient tensor
network transmission and node memory utilization.

## Exploring Feasibility of Embedding Sum (Reduce Operations)

There's a family of operations which need to perform reductions along an entire axis.

* sum, avg
* product
* stddev, variance

Many reduction operations can be modeled as [monoids](https://en.wikipedia.org/wiki/Monoid).

To generically model as a reducible monoid, we need 4 things:

* a way to put a thing into the monoid:
    * `wrap(x) -> M[x]`
* an associative way to merge two things in the monoid:
    * `M[a] • M[b] -> M[c]`
    * `M[b] • M[a] -> M[c]`
* a zero element, that can be merged with any element as an identity:
    * `M[0] • M[a] -> M[a]`
* a way to remove things from the monoid:
    * `unwrap(M[x]) -> x`

For many operations (`sum`, `product`), `wrap()` and `unwrap()` can just be identity; the monoid
representation is the same as the input and output representation.

Other operations may require global information to complete, so their reduction representation
may be more complex. Consider `stddev`:

    @dataclass
    class PartialStats:
      n: int
      sum: float
      sum_of_squares: float

    def zero():
      return PartialStats(n=0, sum=0.0, sum_of_squares=0.0)

    def op(a, b):
      # We might even consider rescaling values to prevent overflow here.
      return PartialStats(
        n = a.n + b.n,
        sum = a.sum + b.sum,
        sum_of_squares = a.sum_of_squares + b.sum_of_squares,
      )

    def wrap(x):
      return PartialStats(n=1, sum=x, sum_of_squares=x*x)

    def wrap_all(xs):
      # equivalent to reduce(op, [wrap(x) for x in xs] + [zero()])
      return PartialStats(
        n=len(xs),
        sum=sum(xs),
        sum_of_squares=sum(x**2 for x in xs),
      )

    def unwrap_stddev(p):
      # beyond the scope of the current example, but we could just as easily
      # return several stats at once:
      #   (len, sum, avg, stddev)
      return math.sqrt((p.sum_of_squares/p.n) - (p.sum/p.n) ** 2)

We might even consider rewriting the scale (*n*) during merge to prevent value overflow.

If we've got a monoidic representation of an expression; we can rewrite arbitrarily long reductions
as a tree of smaller reductions and be certain we'll produce the same result.

In graph scheduling, we can turn an *N*-scale problem into a `log_b(N)` scale problem. If we work
with leaf operations which can perform more than one merge at a time, *b* can be quite large,
and the resulting tree graph can be very shallow.

![reduce.f1](media/graphs/reduce.f1.dot.png)

If we know that an operation has monoid characteristics on a given axis, we show that we can rewrite
nodes into *log_b(N)* reduction layers:

![reduce.f2](media/graphs/reduce.f2.dot.png)

## Exploring Feasibility of Embedding Tensor Generators (Random Numbers)

There's a case that's worth talking about, that breaks our existing models, but is extremely common;
random number generators:

    Y = X * 0.25 * rand_like(X)

Random number generators naively appear to violate our *map* assumptions; if we're concerned about
producing idempotent results, we have to generate the same values each time; but they're stateful
between cells, so slicing work units introduces a state management problem.

This is only a concern if we care that:

* nodes can be perfectly re-computed, and
* any slicing of the index space will produce the *same* random numbers.

Which in turn are properties to preserve primarily if:

* re-executing the tree, under any execution schedule, should yield the same result.

With numerical instability of floating point operations, this is a hard target to pursue;
different reduction orders or block slicing could yield different results; but it's a good
target to keep in mind while designing applications, as there are some where bit-identical
results are highly valued.

Any useful model of tensor operations will need a solution to embedding tensor generators which
remain stable under sharding.

If, at an api level, we can say "this is a random tensor, from this distribution, with this shape",
and take indexed slices of that space, the *how* of the tensor's generation becomes opaque to the
leaf computation, it's just another input.

If we can provide, to a generator, the original index of a sampled tensor space, and the seed
the tensor is being sampled at (and whatever static parameters the generator takes); we can
generate stable results for each view block.

    seed = 345
    sample_shape = [7, 8, 9]
    sample_point = [3, 1, 5]

    r = g(seed, sample_point)

One potential (horribly slow) implementation would be:

    gen = generator(seed)
    idx = (sample_point * sample_shape).sum()
    gen.skip(idx)

    r = gen.next()

This is a lot of wasted work, but is easy to define and stable, and works with most random number
generators.

We could potentially save some computation by examination of the selected region, and construction
of coherent runs on the original index space.

Alternatively, we could look for one-shot generators, which took the whole key as a seed input,
and yielded one-shot values with appropriate statistical properties.

Consider this paper on parallel random number generators, which may provide closed-form answers:

* http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

We'll need a solution to this problem space.

## Summarizing Rewrites Observations

Even under index space projection restrictions, we appear to be able to rewrite a large family
of operations:

* region mappings and matmuls (inc: Linear, Conv, ReLU)
* reductions (inc: Sum, Stddev, Avg)

This collection of operations *appears* sufficient to embed most AI/ML applications; so we we
can pivot:

* from asking "are these operations embeddable?"
* to "how do we represent index projections?":

Examining the abstract embeddings considered thus far, we can make a number of observations about
graph components needed.

* Tensor Transpose/Slice/Merge Nodes
    * Leaf operations consume and produce *slices* of their input and output tensor spaces;
      and rewrites of the leaf operations are accompanied by rewrites of their index spaces,
      but also the slices they operate on. It will be necessary to expose slice operations
      at the graph transformation layer.
* Index Projection Functions
    * Projection from leaf index spaces to tensor block regions requires some projection/slice
      function to specify operation regions. A few properties we know we'll need:
        * Coherent projections - as the leaf operations are block operations, projections to
          neighboring cells in index space should yield coherent/contiguous selections in the target
          tensors.
        * Transformable - there are rewrites we'd like to be able to describe deterministically
          which
          alter the index projection of the rewritten nodes; so it's valuable if we can transform
          those projection functions under rewrites.
        * overlapping input projections - as we wish to model convolutions, our projection
          machinery,
          and concept of "coherent" should model overlapping neighbor selection regions.
        * non-overlapping, coherent outputs - for *output* tensors, we'd like to be able to assert
          that projections don't produce overlapping regions, and fully fill a target space.
* Tensor Generators
    * Some stable solution to rand will be needed.

Tensor transposition and slicing is extensively described; it's easy to reuse existing machinery
to describe transformations to map one set of tensor indexes to another; our primary goal is to
be able to analyze and re-write those transformations. If we are only interested in
subdividing work, then we can always append further transpose/slice operations on existing view
stacks.

So we can model tensor view operations as index mapping stacks, each producing a "new" tensor,
where the intermediate tensors may never be reified.

Index projection is a more complicated case, we're not building 1:1 mapping between cell index
locations, but describing regions, and we need a mechanic which permits this, we need a
mechanism to check that this projection is valid (to prevent bad operations in the graph, and
guard against bad re-writes), and we need a way to rewrite it.

## Next

* [Table of Contents](README.md)
* [Next](IndexProjectionDesign.md)
