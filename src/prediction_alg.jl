using NearestNeighbors
using TimeseriesPrediction
using Juno

export localmodel_stts
export crosspred_stts

###########################################################################################
#                                     Prediction                                          #
###########################################################################################
"""
    localmodel_stts(s::AbstractVector{Array{T, Φ}}, D, τ, p, B, k; kwargs...)

Perform a spatio-temporal timeseries prediction for `p` iterations,
using local weighted modeling [1]. The function always returns an
object of the same type as `s`, which is a vector of states. Each state is represented by
an array of the same dimension as the spatial dim. of the system.
The returned data always contains the final state of `s` as starting point.
This means that the returned data has length of `p + 1`.

Given `(s, D, τ, B, k)` a [`myReconstruction`](@ref) is performed on `s`
with `D-1` temporal neighbors, delay `τ` and `B` spatial neighbors along each direction.
`k` is analogous to `τ` but with respect to space. The total embedding dimension is then
`D * (2B + 1)^Φ` where `Φ` is the dimension of space.

## Keyword Arguments
  * `boundary = 20` : Constant boundary value used for reconstruction of states close to
    the border. Pass `false` for periodic boundary conditions.
  * `weighting = (0,0)` : Add `Φ` additional entries to rec. vectors. These are a spatial
     weighting that may be useful for considering spatially inhomogenous dynamics.
     Each entry is calculated with the given parameters `(a,b)` and
     a normalized spatial coordinate ``-1\\leq\\tilde{x}\\leq 1``:
```math
\\begin{aligned}
\\omega(\\tilde{x}) = a \\tilde{x} ^ b.
\\end{aligned}
```
  * `method = AverageLocalModel(2)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of [`AbstractNeighborhood`](@ref).


## Description
Given a rec. vector as query point, the function finds its neighbors using neighborhood
`ntype`. Then, the neighbors `xnn` and their images `ynn` are used to make
a prediction for the future of the query point, using the provided `method`.
The images `ynn` are points in the original data shifted by one state into the future.

The algorithm is applied for all points in space and iteratively repeated until a
prediction of length `p+1` has
been created, starting with the query state (last point of the original timeseries).

## Note
In many cases `k=1` and `τ=1,2,3` give best results.
Be careful when choosing `B` as memory usage and computation time depend strongly on it.

## References
[1] : U. Parlitz & C. Merkwirth, *Prediction of Spatiotemporal Time Series Based on
Reconstructed Local States*, Phys. Rev. Lett. (2000)
"""
function localmodel_stts(s::AbstractVector{Array{T, Φ}},
    D, τ, p, B, k;
    boundary=20,
    weighting::Tuple{Real, Real} = (0,0),
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    M = prod(size(s[1]))
    L = length(s) #Number of temporal points
    println("Reconstructing")
    R = myReconstruction(s,D,τ,B,k,boundary, weighting)
    #Prepare tree but remove the last reconstructed states first
    println("Creating Tree")
    tree = KDTree(R[1:end-M])

    s_pred = s[L-D*τ:L]
    return _localmodel_stts(s_pred, R, tree, D, τ, p, B, k, boundary,
    weighting, method, ntype)[D*τ+1:end]
end

function gen_qs(s_pred, D, τ, B, k, boundary, weighting)
    N = length(s_pred)
    s_slice = @view(s_pred[N-(D-1)*τ:N])
    return myReconstruction(s_slice, D, τ, B, k, boundary, weighting)
end

function _localmodel_stts(s::AbstractVector{Array{T, Φ}},
    R, tree ,D, τ, p, B, k, boundary, weighting, method, ntype) where {T, Φ}
    M = prod(size(s[1]))
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])
    #Index of relevant element in ynn (not proven but seemingly correct)
    im = 1 + (D-1)*(2B+1)^Φ + B*sum(i -> (2B+1)^(Φ-i), 1:Φ)
    @progress "Frame" for n=1:p
        println("Working on Frame $(n)/$p")
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



function crosspred_stts(
    train_in::AbstractVector{Array{T, Φ}},
    train_out::AbstractVector{Array{T, Φ}},
    pred_in ::AbstractVector{Array{T, Φ}},
    D,τ,p,B=1,k=1;
    boundary=20,
    weighting::Tuple{Real, Real} = (0,0),
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    R = myReconstruction(train_in,D,τ,B,k,boundary, weighting)
    tree = KDTree(R)
    return _crosspred_stts(train_out,pred_in, R, tree, D, τ, B, k, boundary,
     weighting, method, ntype)
end


function _crosspred_stts(
    train_out::AbstractVector{Array{T, Φ}},
    pred_in, R, tree, D, τ, B, k,
    boundary,weighting,method,ntype) where {T,Φ}

    M = prod(size(pred_in[1]))
    L = length(pred_in)

    pred_out = Vector{Array{T, Φ}}()
    state = similar(pred_in[1])

    #create all qs
    qs = myReconstruction(pred_in, D, τ, B, k, boundary, weighting)
    for n=1:L-(D-1)τ
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
