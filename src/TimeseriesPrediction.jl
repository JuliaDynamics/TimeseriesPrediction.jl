__precompile__()

"""
Prediction of timeseries using methods of nonlinear dynamics and timeseries analysis
"""
module TimeseriesPrediction

using Reexport
@reexport using DynamicalSystemsBase

include("localmodeling.jl")
include("reconstruction.jl")
include("st_prediction.jl")
#include("spatiotemporalsystem.jl")
end
