using NearestNeighbors
export KDTree
export TemporalPrediction

function working_ts(s,em)
    L = length(s)
    τmax = get_τmax(em)
    return s[L-τmax : L]
end

function gen_queries(s,em)
    L = length(s)
    τmax = get_τmax(em)
    s_slice = view( s, L-τmax:L)
    return reconstruct(s_slice, em)
end

function convert_idx(idx, em)
    τmax = get_τmax(em)
    num_pt = get_num_pt(em)
    t = 1 + (idx-1) ÷ num_pt + get_τmax(em)
    α = 1 + (idx-1) % num_pt
    return t,α
end

cut_off_beginning!(s,em) = deleteat!(s, 1:get_τmax(em))

macro record(name, to_record)
    return esc(:(sol.runtimes[$name] = @elapsed $to_record))
end

###########################################################################################
#                        Iterated Time Series Prediction                                  #
###########################################################################################

mutable struct TemporalPrediction{T,Φ,BC,X}
    em::AbstractSpatialEmbedding{T,Φ,BC,X}
    method::AbstractLocalModel
    ntype::AbstractNeighborhood
    treetype#::NNTree what's the type here?
    timesteps::Int64

    runtimes::Dict{Symbol,Float64}
    spred::Vector{Array{T,Φ}}
end
TemporalPrediction(em::ASE{T,Φ}, method, ntype, ttype, tsteps) where {T,Φ} =
TemporalPrediction(em, method, ntype, ttype, tsteps, Dict{Symbol,Float64}(),Array{T,Φ}[])


"""
    TemporalPrediction(U, em::AbstractSpatialEmbedding, tsteps; kwargs...)
Perform a spatio-temporal time series prediction for `tsteps` iterations,
using local weighted modeling [1] give a time series of the form
`U::AbstractVector{<:AbstractArray{T, Φ}}`. The function returns a
solution struct with the prediction `spred` of the same type as `U`.

The returned data always contains the final state of `U` as starting point
(total returned length is `tsteps+1`).
The reconstruction process is defined by `em`.
For available methods and interfaces see [`AbstractSpatialEmbedding`](@ref)
## Keyword Arguments
  * `ttype = KDTree` : Type/Constructor of tree structure. So far only tested with `KDTree`.
  * `method = AverageLocalModel(ω_safe)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of [`AbstractNeighborhood`](@ref).
  * `printprogress = true` : To print progress done.
## Description
This method works similarly to [`localmodel_tsp`](@ref), by expanding the concept
of delay embedding to spatially extended systems. Instead of reconstructing
complete states of the system, local states are used. This implicitely assumes
a finite speed `c` at which information travels within the system as well as a
sufficiently fine spatial and temporal sampling such that
``\\frac{\\Delta x}{\\Delta t}\\sim c``.
See [`AbstractSpatialEmbeddingn`](@ref) for details on the embedding.
Predictions are then performed frame by frame and point py point. Once all values for a new
frame are found, the frame is added to the end of the timeseries and used to generate
new prediction queries for the next time step.
## Performance Notes
Be careful when choosing embedding parameters as memory usage and computation time
depend strongly on the resulting embedding dimension.
## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function TemporalPrediction(s,
    em::AbstractSpatialEmbedding{T,Φ},
    tsteps;
    ttype=KDTree,
    method = AverageLocalModel(ω_safe),
    ntype = FixedMassNeighborhood(3),
    progress=true) where {T,Φ}

    prelim_sol = TemporalPrediction(em, method, ntype, ttype, tsteps)
    return TemporalPrediction(prelim_sol, s; progress=progress)
end

function TemporalPrediction(sol, s; progress=true)
    progress && println("Reconstructing")
    @record :recontruct   R = reconstruct(s,sol.em)

    #Prepare tree but remove the last reconstructed states first
    progress && println("Creating Tree")
    L = length(R)
    M = get_num_pt(sol.em)

    @record :tree   tree = sol.treetype(R[1:L-M])
    TemporalPrediction(sol, s, R, tree; progress=progress)
end



function TemporalPrediction(sol, s, R, tree; progress=true) where {T, Φ, BC, X}
    em = sol.em
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])

    #End of timeseries to work with
    sol.spred = spred = working_ts(s,em)

    @record :prediction for n=1:sol.timesteps
        progress && println("Working on Frame $(n)/$(sol.timesteps)")
        queries = gen_queries(spred, em)

        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = queries[m]

            #Find neighbors
            idxs,dists = neighborhood_and_distances(q,R,tree,sol.ntype)

            xnn = R[idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                s[t+1][α]
            end
            state[m] = sol.method(q,xnn,ynn,dists)[1]
        end
        push!(spred,copy(state))
    end

    cut_off_beginning!(spred,em)
    return sol
end
