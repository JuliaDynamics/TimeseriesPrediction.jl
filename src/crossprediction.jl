export CrossPrediction

###########################################################################################
#                                  Cross Prediction                                       #
###########################################################################################

mutable struct CrossPrediction{T,Φ,BC,X}
    em::AbstractSpatialEmbedding{T,Φ,BC,X}
    method::AbstractLocalModel
    ntype::AbstractNeighborhood
    treetype#::NNTree what's the type here?
    pred_in::Vector{Array{T,Φ}}

    runtimes::Dict{Symbol,Float64}
    pred_out::Vector{Array{T,Φ}}
    CrossPrediction{T,Φ,BC,X}(em::ASE{T,Φ,BC,X}, method, ntype, ttype, pred_in
                                ) where {T,Φ,BC,X} =
                                new(em, method, ntype, ttype, pred_in,
                                Dict{Symbol,Float64}(), Array{T,Φ}[])
end

"""
    CrossPrediction(source_train, target_train, source_pred,
                   em::AbstractSpatialEmbedding; kwargs...)
Perform a spatio-temporal timeseries cross-prediction for `target` from
`source`, using local weighted modeling [1]. This can be used for example
when there are coupled spatial fields and one is used to predict the other.
It is assumed that `source_train`, `target_train`, `source_pred` are all of
the same type, `AbstractVector{<:AbstractArray{T, Φ}}`.

The spatio temporal delay embedding process is defined by `em`.
See [`AbstractSpatialEmbedding`](@ref) for available methods and interfaces.

## Keyword Arguments
  * `ttype = KDTree` : Type/Constructor of tree structure. So far only tested with `KDTree`.
  * `method = AverageLocalModel(ω_safe)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(3)` : Subtype of [`AbstractNeighborhood`](@ref).
  * `printprogress = true` : To print progress done.
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
states that are needed for reconstruction.
## Additional Interfaces
To save computation time in the case of repeated predictions with the same training set
and embedding parameters we provide an additional interface that allows the user to
provide an existing reconstruction and tree structure.
    R = reconstruct(train_in, em)
    tree = ttype(R)
    prelim_sol = CrossPrediction(em, method, ntype, ttype, pred_in)
    sol = CrossPrediction(copy(prelim_sol), train_out, R, tree; progress=true)
where `prelim_sol` is a preliminary solution struct containing all relevant parameters.

## Performance Notes
Be careful when choosing embedding parameters as memory usage and computation time
depend strongly on the resulting embedding dimension.
## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function CrossPrediction(train_in ::AbstractVector{<:AbstractArray{T, Φ}},
                        train_out::AbstractVector{<:AbstractArray{T, Φ}},
                        pred_in  ::AbstractVector{<:AbstractArray{T, Φ}},
                        em::AbstractSpatialEmbedding{T,Φ};
                        ttype = KDTree,
                        method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                        ntype::AbstractNeighborhood = FixedMassNeighborhood(3),
                        progress=true
                        ) where {T, Φ}
    prelim_sol = CrossPrediction(em, method, ntype, ttype, pred_in)
    CrossPrediction(prelim_sol, train_in, train_out; progress=true)
end

function CrossPrediction(sol, train_in, train_out; progress=true)
    progress && println("Reconstructing")
    @record :recontruct   R = reconstruct(train_in,sol.em)

    progress && println("Creating Tree")
    @record :tree         tree = sol.treetype(R)

    CrossPrediction(sol, train_out, R, tree; progress=progress)
end

function CrossPrediction(sol, train_out, R, tree; progress=true)
    em = sol.em
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(train_out[1])

    queries = reconstruct(sol.pred_in, em)
    sol.pred_out = eltype(train_out)[]

    @record :prediction for n=1:length(sol.pred_in)-get_τmax(em)
        progress && println("Working on Frame $(n)/$(length(sol.pred_in)-get_τmax(em))")
        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = queries[m+(n-1)*num_pt]

            #Find neighbors
            idxs,dists = neighborhood_and_distances(q,R,tree,sol.ntype)

            xnn = R[idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                train_out[t][α]
            end
            state[m] = sol.method(q,xnn,ynn,dists)[1]
            #won't work for lin loc model, needs Vector{SVector}
        end
        push!(sol.pred_out,copy(state))
    end
    return sol
end
