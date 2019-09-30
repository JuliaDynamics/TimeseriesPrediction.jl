using NearestNeighbors
import Parameters.@with_kw_noshow
export KDTree
export temporalprediction
export PredictionParameters

export FullSymmetry

@with_kw_noshow struct PredictionParameters{ASE <: AbstractSpatialEmbedding,
        LM <: AbstractLocalModel, NT<:AbstractNeighborhood}
    em::ASE
    method::LM = AverageLocalModel(ω_safe)
    ntype::NT = FixedMassNeighborhood(3)
    treetype = KDTree
end

struct FullSymmetry end


###########################################################################################
#                        Iterated Time Series Prediction                                  #
###########################################################################################

"""
    temporalprediction(U, em::AbstractSpatialEmbedding, tsteps; kwargs...)
Perform a spatio-temporal time series prediction for `tsteps` iterations,
using local weighted modeling [1] give a time series of the form
`U::AbstractVector{<:AbstractArray{T, Φ}}`.

The returned data always contains the final state of `U` as starting point
(total returned length is `tsteps+1`).
The reconstruction process is defined by `em`.
For available methods and interfaces see [`AbstractSpatialEmbedding`](@ref).

## Keyword Arguments
  * `ttype = KDTree` : Type/Constructor of tree structure. So far only tested with `KDTree`.
  * `method = AverageLocalModel(ω_safe)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of `AbstractNeighborhood`.
  * `initial_ts = U` : Initial states for prediction (same type as `U`).
    Must have at least as many states as the maximum delay time used. Defaults to the training set `U`.
  * `progress = true` : To print progress done.

## Description
This method works similarly to [`localmodel_tsp`](@ref), by expanding the concept
of delay embedding to spatially extended systems. Instead of reconstructing
complete states of the system, local states are used.
See [`AbstractSpatialEmbedding`](@ref) for details on the embedding.
Predictions are then performed frame by frame and point py point. Once all values for a new
frame are found, the frame is added to the end of the timeseries and used to generate
new prediction queries for the next time step.

## Performance Notes
Be careful when choosing embedding parameters as memory usage and computation time
depend strongly on the resulting embedding dimension.

## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function temporalprediction(s,
    em::AbstractSpatialEmbedding,
    tsteps;
    method = AverageLocalModel(ω_safe),
    ntype = FixedMassNeighborhood(3),
    ttype=KDTree,
    kwargs...)

    params = PredictionParameters(em, method, ntype, ttype)
    return temporalprediction(params, s, tsteps, symmetry; kwargs...)
end

function temporalprediction(params, s, tsteps; progress=true, kwargs...)
    progress && println("Reconstructing")
    R = reconstruct(view(s, 1, length(s)-1),params.em)
    progress && println("Creating Tree")
    tree = params.treetype(R)
    temporalprediction(params, s, tsteps, R, tree,symmetry; progress=progress, kwargs...)
end



function temporalprediction(params, s,tsteps, R, tree;
        initial_ts=s, #optional start for prediction
        progress=true,
        symmetry=nothing,
        kwargs...)
    em = params.em
    @assert outdim(em) == size(R,2)
    @assert length(initial_ts) > get_τmax(em)

    #Prepare starting point of prediction timeseries
    spred = working_ts(initial_ts, em)

    #Prepare potential symmetry operations
    prep = preparatory_computation(em, symmetry)
    for n=1:tsteps
        progress && println("Working on Frame $(n)/$(tsteps)")
        predict_frame!(s, spred, params, R, tree, prep; kwargs...)
    end

    cut_off_beginning!(spred,tsteps)
    return spred
end

preparatory_computation(em,::Nothing) = nothing


function predict_frame!(s, spred, params, R, tree, ::Nothing; kwargs...)
    #New state that
    state = similar(spred[1])
    em = params.em
    queries = gen_queries(spred, em)
    #Iterate over queries/ spatial points
    for m=1:get_num_pt(em)
        q = queries[m]
        #Find neighbors
        idxs,dists = neighborhood_and_distances(q,R,tree,params.ntype)

        xnn = R[idxs]
        #Retrieve ynn
        ynn = map(idxs) do idx
            #Indices idxs are indices of R. Convert to indices of s
            t,α = convert_idx(idx,em)
            s[t+1][α]
        end
        state[m] = params.method(q,xnn,ynn,dists)[1]
    end
    push!(spred,state)
end


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

cut_off_beginning!(s,tsteps) = (N=length(s); deleteat!(s, 1:N-tsteps-1))
