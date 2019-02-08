# Spatiotemporal Timeseries Prediction

An application and extension of [local modeling](tsprediction/localmodels) to
spatiotemporal timeseries.

!!! tip "Examples"
    Several example scripts can be found in `TimeseriesPrediction/examples`. These examples are run in the [examples](stexamples.md) page.


## Spatio-Temporal Embeddings
some info here.
```@docs
TimeseriesPrediction.reconstruct
AbstractSpatialEmbedding
SpatioTemporalEmbedding
cubic_shell_embedding
light_cone_embedding
PCAEmbedding
```

Here is a function that visualizes how the `light_cone_embedding` works in 2 spatial dimensions:
```julia
using PyPlot, TimeseriesPrediction
using LinearAlgebra: norm

function explain_light_cone(;γ = 2, τ = 2, r = 1, c = 1)

    maxr = D*τ*c + r0

    figure()
    xticks(-maxr:maxr)
    yticks(-maxr:maxr)

    for i in γ:-1:0
        r = i*τ*c + r
        points = TimeseriesPrediction.indices_within_sphere(r, 2)
        radius = maximum(norm(Tuple(p)) for p in points)

        if r != 0
            x = r*cos.(range(0, stop = 2π, length = 100))
            y = r*sin.(range(0, stop = 2π, length = 100))
            plot(x,y, c = "C$i")
        end

        x = map(xy -> xy[1], points)
        y = map(xy -> xy[2], points)
        scatter(x,y, c = "C$i", s=100, zorder = 3, label = "within \$r = r + $i \\tau c\$")

    end
    title("γ = $γ, τ = $τ, r = $r, c = $c")
    PyPlot.grid(zorder = -1)
    legend(loc = "upper right")
    xlabel("x")
    ylabel("y")
    PyPlot.axis("equal")
    tight_layout()
end
```

---

Boundary conditions
```@docs
ConstantBoundary
PeriodicBoundary
```

## Prediction functions
```@docs
temporalprediction
crossprediction
```
