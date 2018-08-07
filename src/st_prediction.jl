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
struct CrossPrediction{T,Φ}
    em::AbstractEmbedding
    treetype#::NNTree what's the type here?
    pred_in::Vector{Array{T,Φ}}
    pred_out::Vector{Array{T,Φ}}
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
    t = get_τmax(em) + (idx-1) ÷ num_pt
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
            q = @view queries[:,m]

            #Find neighbors
            #Note, not yet compatible with old `neighborhood_and_distances` functions
            idxs,dists = neighbors(q,R,tree,ntype)

            xnn = @view R[:, idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                s[t+1][α]
            end
            state[m] = method(q,xnn,ynn,dists)[1]
            #won't work for lin loc model, needs Vector{SVector}
        end
        push!(spred,copy(state))
    end
    cut_off_beginning!(spred,em)

    return TemporalPrediction{T,Φ}(em, typeof(tree), spred) #funfacts runtimes
end



function crosspred_stts(    train_out::AbstractVector{<:AbstractArray{T, Φ}},
                            pred_in  ::AbstractVector{<:AbstractArray{T, Φ}},
                            em::AbstractEmbedding,
                            R::AbstractMatrix{T},
                            tree::NNTree;
                            progress=true,
                            method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                            ntype::AbstractNeighborhood = FixedMassNeighborhood(3)
                        ) where {T, Φ}
    @assert outdim(em) == size(R,1)
    @show num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(train_out[1])

    queries = reconstruct(pred_in, em)

    for n=1:length(pred_in)-get_τmax(em)
        progress && println("Working on Frame $(n)/$(length(pred_in)-get_τmax(em))")
        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = @view queries[:,m+(n-1)*num_pt]

            #Find neighbors
            #Note, not yet compatible with old `neighborhood_and_distances` functions
            idxs,dists = neighbors(q,R,tree,ntype)

            xnn = @view R[:, idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                train_out[t][α]
            end
            state[m] = method(q,xnn,ynn,dists)[1]
            #won't work for lin loc model, needs Vector{SVector}
        end
        push!(pred_out,copy(state))
    end

    return CrossPrediction{T,Φ}(em, typeof(tree), pred_in, pred_out)
     #funfacts runtimes
end
