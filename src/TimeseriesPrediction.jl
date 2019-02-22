"""
Prediction of timeseries using methods of nonlinear dynamics and timeseries analysis
"""
module TimeseriesPrediction

using Reexport
@reexport using MultivariateStats
@reexport using DelayEmbeddings
using Statistics, LinearAlgebra

include("localmodeling.jl")

include("spatiotemporalembedding.jl")
include("pcaembedding.jl")
include("symmetric_embedding.jl")
include("reconstruct.jl")

include("temporalprediction.jl")
include("crossprediction.jl")
end
