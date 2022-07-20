# Index Projection Functions

* [Table of Contents](README.md)
* [Previous](ExploringTensorBlockShardingFeasibility.md)

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

## Next

* [Table of Contents](README.md)
