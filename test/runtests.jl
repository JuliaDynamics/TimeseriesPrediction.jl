using TimeseriesPrediction

using DynamicalSystemsBase

ti = time()

include("localmodeling_tests.jl")
include("st_pred_barkley.jl")
include("st_coupledhenon_tests.jl")
include("reconstruction_tests.jl")
ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
