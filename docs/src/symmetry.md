# Exploiting Spatial Symmetries

Some systems have symmetries that apply to their spatial dimensions. For example, the Barkley model that we use an example has equations:
```math
u_t = \frac{1}{\epsilon}u(1-u)(u-\frac{v+b}{a}) + \nabla^2 u \\
v_t = u^3 - v
```
The only spatial coupling component is the Laplacian operator, ``\nabla^2``. This means that the equations of of the Barkley model have rotational symmetry with respect to space.

In principle one should be able to take advantage of these symmetries to reduce the embedded space dimension.

## Symmetries
We encode symmetries with the following types:
```@docs
Symmetry
Reflection
Rotation
```
## Symmetric Embedding
You can use the symmetries in the following embedding:
```@docs
SymmetricEmbedding
```
