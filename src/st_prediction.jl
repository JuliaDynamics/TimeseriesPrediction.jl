using NearestNeighbors

export localmodel_stts
export crosspred_stts
export KDTree

#Return struct
struct TemporalPrediction{T,Φ}
    em::AbstractEmbedding
    treetype#::NNTree what's the type here?
    spred::Vector{Array{T,Φ}}
    #fun facts
  #  runtime::Float64
  #  reconstruction_time::Float64
  #  prediction_time::Float64
end

function localmodel_stts(s,em,timesteps; progress=true, kwargs...)

    progress && println("Reconstructing")
    R = reconstruct(s,em)

    localmodel_stts(s,em,timesteps, R; progress=progress, kwargs...)
end

function localmodel_stts(s,em,timesteps,R; progress=true, treetype=KDTree, kwargs...)
    #Prepare tree but remove the last reconstructed states first
    progress && println("Creating Tree")
    L = length(R)
    M = get_num_pt(em)
    tree = treetype(view(R,:,1:L-M))

    localmodel_stts(s,em,timesteps, R, tree; progress=progess, kwargs...)
end


#    s_pred = s[L-D*τ:L]
#    return _localmodel_stts(s_pred, R, tree, D, τ, p, B, k, boundary,
#    weighting, method, ntype, printprogress)[D*τ+1:end]
#end

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

function neighbors(point, R, tree, ntype)
    idxs,dists = knn(tree, point, ntype.K, false)
    return idxs,dists
end
function convert_idx(idx, em)
    τmax = get_τmax(em)
    num_pt = get_num_pt(em)
    t = 1 + get_τmax(em) + (idx-1) ÷ num_pt
    α = 1 + (idx-1) % num_pt
    return t,α
end

cut_off_beginning!(s,em) = deleteat!(s, 1:get_τmax(em))

function localmodel_stts(   s::AbstractVector{Array{T, Φ}},
                            em::AbstractEmbedding,
                            timesteps::Int,
                            R::AbstractMatrix{T},
                            tree::NNTree;
                            progress=true,
                            method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                            ntype::AbstractNeighborhood = FixedMassNeighborhood(3)
                        ) where {T, Φ}
    @assert outdim(em) == size(R,1)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])

    #End of timeseries to work with
    spred = working_ts(s,em)

    for n=1:timesteps
        progress && println("Working on Frame $(n)/$timesteps")
        queries = gen_queries(spred, em)
        
        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = queries[:,m]

            #Find neighbors
            #Note, not yet compatible with old `neighborhood_and_distances` functions
            idxs,dists = neighbors(q,R,tree,ntype) #call this on all queries at once?

            xnn = @view R[:, idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                s[t][α]
            end
            state[m] = method(q,xnn,ynn,dists)[1]
            #won't work for lin loc model, needs Vector{SVector}
        end
        spred = push!(spred,copy(state))
    end
    cut_off_beginning!(spred,em)

    return TemporalPrediction{T,Φ}(em, typeof(tree), spred) #funfacts runtimes
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
    method::AbstractLocalModel = AverageLocalModel(ω_safe),
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
