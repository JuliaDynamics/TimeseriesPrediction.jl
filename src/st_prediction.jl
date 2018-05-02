using NearestNeighbors

export localmodel_stts
export crosspred_stts

###########################################################################################
#                                     Prediction                                          #
###########################################################################################
"""
    localmodel_stts(U::AbstractVector{<:AbstractArray{T, Φ}}, D, τ, p, B, k; kwargs...)

Perform a spatio-temporal timeseries prediction for `p` iterations,
using local weighted modeling [1]. The function always returns an
object of the same type as `U`, with each entry being a predicted state.
The returned data always contains the final state of `U` as starting point
(total returned length is `p+1`).

`(D, τ, B, k)` are used to make a [`STReconstruction`](@ref) on `U`.
In most cases `k=1` and `τ=1,2,3` give best results.

## Keyword Arguments
  * `boundary, weighting` : Passed directly to [`STReconstruction`](@ref).
  * `method = AverageLocalModel(2)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of [`AbstractNeighborhood`](@ref).


## Description
This method works identically to [`localmodel_tsp`](@ref), by expanding the concept
from vector-states to general array-states.

## Performance Notes
Be careful when choosing `B` as memory usage and computation time depend strongly on it.

## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function localmodel_stts(s::AbstractVector{<:AbstractArray{T, Φ}},
    D, τ, p, B, k;
    boundary=20,
    weighting::Tuple{Real, Real} = (0,0),
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3),
    printprogress = true) where {T, Φ}
    M = prod(size(s[1]))
    L = length(s) #Number of temporal points
    printprogress && println("Reconstructing")
    R = STReconstruction(s,D,τ,B,k,boundary, weighting)
    #Prepare tree but remove the last reconstructed states first
    printprogress && println("Creating Tree")
    tree = KDTree(R[1:end-M])

    s_pred = s[L-D*τ:L]
    return _localmodel_stts(s_pred, R, tree, D, τ, p, B, k, boundary,
    weighting, method, ntype, printprogress)[D*τ+1:end]
end

function gen_qs(s_pred, D, τ, B, k, boundary, weighting)
    N = length(s_pred)
    s_slice = @view(s_pred[N-(D-1)*τ:N])
    return STReconstruction(s_slice, D, τ, B, k, boundary, weighting)
end

function _localmodel_stts(s::AbstractVector{Array{T, Φ}},
    R, tree ,D, τ, p, B, k, boundary, weighting, method, ntype,
    printprogress = true) where {T, Φ}
    M = prod(size(s[1]))
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])
    #Index of relevant element in ynn (not proven but seemingly correct)
    im = 1 + (D-1)*(2B+1)^Φ + B*sum(i -> (2B+1)^(Φ-i), 1:Φ)
    for n=1:p
        printprogress && println("Working on Frame $(n)/$p")
        qs = gen_qs(s, D, τ, B, k, boundary, weighting)
        for (m,q) ∈ enumerate(qs)
            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]
            ynn = map(y -> y[im],R[idxs+M])

            state[m] = method(q,xnn,ynn,dists)[1]
        end
        s = push!(s,copy(state))
    end
    return s
end






"""
    crosspred_stts(source_train, target_train, source_pred,
                   D, τ, B, k; kwargs...)

Perform a spatio-temporal timeseries cross-prediction for `target` from
`source`, using local weighted modeling [1]. This can be used for example
when there are coupled spatial fields and one is used to predict the other.

It is assumed that `source_train`, `target_train`, `source_pred` are all of
the same type, `AbstractVector{<:AbstractArray{T, Φ}}`.

`(D, τ, B, k)` are used to make a [`STReconstruction`](@ref) on `source_train`.
In most cases `k=1` and `τ=1,2,3` give best results.


## Keyword Arguments
  * `boundary, weighting` : Passed directly to [`STReconstruction`](@ref).
  * `method = AverageLocalModel(2)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of [`AbstractNeighborhood`](@ref).

## Description
The reconstructed state of `source_train[t][i,j,...]` is associated with the output
value `target_train[t][i,j,...]`. This establishes a "connection"
between `target` and `source`.

Taking a reconstructed state of `source_pred` as query point,
the function finds its neighbors in the reconstructed space of `source_train` using
neighborhood `ntype`. Then, the neighbor *indices* are used
to make a prediction for the corresponding value of the `target`, using the
established "connection" between fields (from the training data).

The algorithm is applied for all points in space and time of `pred_in` minus the first
`(D-1)τ` states that are needed for reconstruction.

## Performance Notes
Be careful when choosing `B` as memory usage and computation time depend strongly on it.

## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function crosspred_stts(
    train_in::AbstractVector{<:AbstractArray{T, Φ}},
    train_out::AbstractVector{<:AbstractArray{T, Φ}},
    pred_in ::AbstractVector{<:AbstractArray{T, Φ}},
    D,τ,B=1,k=1;
    boundary=20,
    weighting::Tuple{Real, Real} = (0,0),
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    R = STReconstruction(train_in,D,τ,B,k,boundary, weighting)
    tree = KDTree(R)
    return _crosspred_stts(train_out,pred_in, R, tree, D, τ, B, k, boundary,
     weighting, method, ntype)
end


function _crosspred_stts(
    train_out::AbstractVector{<:AbstractArray{T, Φ}},
    pred_in, R, tree, D, τ, B, k,
    boundary,weighting,method,ntype) where {T,Φ}

    M = prod(size(pred_in[1]))
    L = length(pred_in)

    pred_out = Vector{Array{T, Φ}}()
    state = similar(pred_in[1])

    #create all qs
    qs = STReconstruction(pred_in, D, τ, B, k, boundary, weighting)
    for n=1:L-(D-1)τ
        println("Cross-prediction frame $(n)/$(L-(D-1)τ)")
        for m=1:M
            q = qs[m + M*(n-1)]
            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]   #not used in method...
            ynn = [train_out[1 + (D-1)*τ + (ind-1)÷ M][1 + (ind-1) % M] for ind ∈ idxs]

            state[m] = method(q,xnn,ynn,dists)[1]
        end
        push!(pred_out, copy(state))
    end
    return pred_out
end
