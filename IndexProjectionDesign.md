# Index Projection Functions

* [Table of Contents](README.md)
* [Previous](BlockSharding.md)

In seeking index projection functions, we need to establish a mapping from every point in the
abstract index space for an operation, to some coherent region in an associated input or output
tensor.

Two notable points:

* The coherent region comes from the design constraint that neighboring points in index space should
  map to coherent blocks of the associated tensors.

* We'll further restrict selection to cubic regions: points, rectangles, blocks, etc.

* We'll further assume that the selected regions are not empty; our projection need not generate
  regions with zero (or negative) contained cells.

* We'll additionally require that all projections for the same function produce regions of the
  same shape; we're targeting block operations anyway.

* To further simplify this problem, we'll make the assumption that the index projection will not
  change any aspect of the tensor's stride: dimensionality, dimension ordering, and dimension
  direction will not change under projection. We're just selecting a region in an existing layout.

To define a coherent, non-empty cubic region of constant shape in an integer coordinate system,
we could:

* Specify an inclusive (the point is inside the region) *start* and an exclusive (the point is
  outside the region) *end* point.
* Specify an inclusive *start* and an inclusive *end* point.
* Specify an inclusive *start* and a shape.
* Specify all of the corners of the region.
* etc ...

Each of these representations is equivalent, but Projecting to a dynamic *start* point, with a
fixed *shape* is simpler to specify, and does not require the well-formedness checks of the
other mechanics, so we'll use (*start*, *shape*) to describe the output of a projection.

To map a point in one integer coordinate space to a point in another integer coordinate space,
an integer affine transform is a good place to start.

Let's consider an index projection composed of:

* *projection* - an integer projection matrix
* *offset* - an integer offset vector (it moves the *start* location)
* *shape* - an integer region shape

Let's call this approach ZProjection; is this sufficient?

### Exploring ZProjections: Fixed Tensor Inputs

For a number of operations explored above (*nn.Linear*, *nn.Conv*), while we sharded along
the dimensions of some tensors, others were consumed in full, not varying at different points
in index space.

We can use a ZProjection to project to fixed views using by:

* setting the *projection* to a zero matrix,
* the *offset* to the origin of the view,
* and the *shape* to the shape of the view.

As a result, all points in index space will map to the same fixed input view.

![zprojection.fixed.f1](media/graphs/zprojection.fixed.f1.dot.png)

We can also trivially select a sub-region of the original tensor for fixed reads,
by adjusting *offset* and *shape*:

![zprojection.fixed.f2](media/graphs/zprojection.fixed.f2.dot.png)

### Exploring ZProjections: nn.Linear Strides

*nn.Linear* and *matmul* stride one-at-a-time along their input and output tensors, the
shape of the selected regions is generally a one vector.

* setting the *projection* to map adjacent cells,
* the *offset* to a zero vector,
* and the *shape* to a one vector.

Again, we have a stable stride.

![zprojection.linear.f1](media/graphs/zprojection.linear.f1.dot.png)

### Exploring ZProjections: negative nn.Linear Strides

Given that we're working with projections, there's no particular reason we can't define
a projection which counts *backwards* as index space increments.

However, when we consider dense blocks, and what it means for them to be coherent;
it is important to consider that a *subsequent* block along some dimension may
map to a *preceding* location in the target coordinate space.

![zprojection.linear.f2](media/graphs/zprojection.linear.f2.dot.png)

### Exploring ZProjections: axial reduce (nn.Sum)

Reducing all the values along a given dimension can be accomplished by:

* setting *projection* to map the first cell in that dimension,
* setting *offset* to a zero vector,
* and *shape* to the shape of that axial row (all ones, except for one dimension matching
  the length).

### Exploring ZProjections: nn.Conv Kernel Window Strides

*nn.Conv* and convolution algorithms generally need to describe a kernel window region
centered upon some notional location.

We can start defining windows by:

* setting *projection* to map the "center" cell in the region,
* setting *offset* to adjust the location of the *start* cell,
* and *shape* to the shape of the selected region.

![zprojection.conv.f1](media/graphs/zprojection.conv.f1.dot.png)

Our first mapped point produces a kernel which crosses the bounds of the space;
and we can see we'll see similar bounds crossing along all borders of the space
with this offset design; so we have to address the question:

* How we handle out-of-bounds indexes?

We could:

* forbid out-of-bounds indexes,
* treat index space as toroidal (it wraps around),
* treat negative indexes whose absolute value is less than the dimension as counting
  "backwards" from the end (this is what *numpy* does),
* treat the infinite space "around" our tensor as some form of pad-space
  (and define an approach to handling padding).

#### Offset is composite: projection offset + relative start offset

There's an additional issue with using *offset* to describe windows walking over
negative projection dimensions; which is that we're using *offset* to describe two things:

* the relative location of the start point from the mapped point,
* the bounds adjustment needed for negative indexing.

This has no impact when computing projections, but may affect authoring them (working in terms
of relative offsets may be easier), and in mechanically computing stride order changes (the
relative start would need to be recovered before reversing a negative dimension).

At issue is a question of what the "real" target of a projection covering a region is;
do we wish to mark some distinguished "center" point other than the start point?

When authoring some forms of windows (such as convolutions) we have a distinguished
central point in mind; but when authoring others (such as max downsampling by 2) we
do not.

This could be modeled by adding a list of named offsets, rather than one offset; and this could
help some debugging and authoring tooling; but if it were reified at the block level, there
would need to be some mechanism to mark some offsets as mechanically editable, and others
as not.

It's worth calling out as future work, but we'll defer modeling this for now.

## Incremental Delta Strides are Constant

Since affine projections (including integer affine projections) are linear maps,
the incremental stride along one dimension of index space will be a constant vector
across the entire index space.

    ZP(C) = C.T P + Offset

    ΔZP/Δi = ZP(C + [0, ..., 1 @i, ..., 0]) - ZP(C)
           = (C + [0, ..., 1 @i, ..., 0]).T P + Offset - C P - Offset
           = (C + [0, ..., 1 @i, ..., 0]).T P - C P
           = (C + [0, ..., 1 @i, ..., 0] - C).T P
           = [0, ..., 1 @i, ..., 0].T P

We can use this fact to conclude that:

* Coherent blocks in the index space coordinate system will map to coherent blocks in the target
  tensor coordinate systems.

## Computing Coherent Blocks

Given that we know incremental strides in index space are constant in tensor space (but may be 
negative), we can compute the bounds of the projection of a coherent block of index space into
tensor space by taking the least *start* point and the greatest *end* point generated by the block.

Without negative strides, these are always associated with the least and greatest points in the 
index space block; but with negative strides our bounds are different.

## Coherent Blocks May Be Sparse

Our guarantees about coherence of layout, and block mapping union, guarantee that the mapped 
block is laid out the same as the source block, but not that it is all used.

We may describe strides which skip data; the zprojection mechanism will include the unused 
interspacing data in the fused block, and has no mechanism for avoiding this *at this layer*.

Under certain tensor stride selection operations, we could pre-stride a space; or perform other 
space reduction or selection mechanisms before reaching this layer; that is out-of-scope of
index projection.

## Incremental Tensor Sharing is Constant

Since we know that incremental strides are constant, when examining tensor data sharing between
work units, we can compute the incremental block overlap between tensors, which shows the degree
of data sharing along an index axis.

For fixed view tensors, the overlap is total; and for standard linear maps, the overlap is zero;
for many convolution algorithms, there is partial overlap along some index dimensions (those of
the convolution kernel window), and no overlap along others (the batch dimension, or a channel
dimension).

Effective sharding depends on being able to predict marginal costs along sharding dimensions.

## Incremental Block Memory and Execution Costs

Block operations require some amount of intermediate storage and execution resources. Some may 
produce their output tensors directly, but others may have, potentially very large, intermediate 
tensors; and all blocks require some execution, which we'd like to model.

We expect that well-behaved blocks will scale those storage needs along similar incremental
dimensions; but describing only the input and output spaces of a block operation fails to
make intermediate memory utilization visible to the graph scheduler.

The prior art has a few suggestions:

* Provide a per-index-point scale factor.
  We could attach an estimated memory-per-index-point rate to all operations. While required at
  the graph planning layer, we could apply a default, or probe for one, at the API layer.
* Provide a per-index-dimension scale factor.
  Expected costs scale differently along different index dimensions; so providing a 
  per-index-dimension scale factor would allow more accurate modeling.
* Probe the operation.
  This would require (some) layer to probe the execution costs of the operation by running sample
  blocks, in order to discover cost estimates. The cost estimates for a given block would be
  relatively stable, and execution environments could have lookup pools of cost estimates.

Apache Spark assumes a standard compute cost based upon the size of the data input, and that cost
can be overridden for a given operation.

Note that, due to the parallel nature of the CUDA/GPU/TPU dispatch mechanisms, incremental
execution costs are not linear, and will have some expected cliff functions. Additionally, we may
be interested in both wall time, and power.

## Next

* [Table of Contents](README.md)
* [Block Graph](BlockGraph.md)
