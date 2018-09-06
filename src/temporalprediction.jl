using NearestNeighbors
using Parameters
export KDTree
export temporalprediction
export PredictionParameters

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

@with_kw struct PredictionParameters{T,Φ,BC,X}
    em::AbstractSpatialEmbedding{T,Φ,BC,X}
    method::AbstractLocalModel = AverageLocalModel(ω_safe)
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)
    treetype = KDTree
end


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
function temporalprediction(s,
    em::AbstractSpatialEmbedding{T,Φ},
    tsteps;
    method = AverageLocalModel(ω_safe),
    ntype = FixedMassNeighborhood(3),
    ttype=KDTree,
    progress=true) where {T,Φ}

    params = PredictionParameters(em, method, ntype, ttype)
    return temporalprediction(params, s, tsteps; progress=progress)
end

function temporalprediction(params, s, tsteps; progress=true)
    progress && println("Reconstructing")
    R = reconstruct(s,params.em)

    #Prepare tree but remove the last reconstructed states first
    progress && println("Creating Tree")
    L = length(R)
    M = get_num_pt(params.em)

    tree = params.treetype(R[1:L-M])
    temporalprediction(params, s, tsteps, R, tree; progress=progress)
end



function temporalprediction(params, s,tsteps, R, tree; progress=true) where {T, Φ, BC, X}
    em = params.em
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])

    #End of timeseries to work with
    spred = working_ts(s,em)

    for n=1:tsteps
        progress && println("Working on Frame $(n)/$(tsteps)")
        queries = gen_queries(spred, em)

        #Iterate over queries/ spatial points
        for m=1:num_pt
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
        push!(spred,copy(state))
    end

    cut_off_beginning!(spred,em)
    return spred
end
