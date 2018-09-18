"""
Prediction of timeseries using methods of nonlinear dynamics and timeseries analysis
"""
module TimeseriesPrediction

using Reexport
@reexport using DynamicalSystemsBase

using Statistics, LinearAlgebra

include("localmodeling.jl")
include("reconstruction.jl")

#include("pca/pca.jl")
include("pcareconstruction.jl")

include("temporalprediction.jl")
include("crossprediction.jl")
include("spatiotemporalsystem.jl")
end
