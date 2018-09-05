using NearestNeighbors

export localmodel_stts
export crosspred_stts
export KDTree

###########################################################################################
#                        Iterated Time Series Prediction                                  #
###########################################################################################

mutable struct TemporalPrediction{T,Φ,X,BC}
    em::AbstractSpatialEmbedding{T,Φ,X,BC}
    method::AbstractLocalModel
    ntype::AbstractNeighborhood
    treetype#::NNTree what's the type here?
    timesteps::Int64

    runtimes::Dict{Symbol,Float64}
    spred::Vector{Array{T,Φ}}
end



function localmodel_stts(s,
    em::AbstractSpatialEmbedding{T,Φ,X,BC},
    tsteps;
    ttype=KDTree,
    method = AverageLocalModel(ω_safe),
    ntype = FixedMassNeighborhood(3),
    progress=true) where {T,Φ,X,BC}

    sol = TemporalPrediction{T,Φ,X,BC}(em,method, ntype,ttype,tsteps,Dict{Symbol,Float64}(),Array{T,Φ}[])

    localmodel_stts(sol, s; progress=progress)
end

function localmodel_stts(sol, s; progress=true)
    progress && println("Reconstructing")
    sol.runtimes[:recontruct] = @elapsed(
        R = reconstruct(s,sol.em)
    )

    #Prepare tree but remove the last reconstructed states first
    progress && println("Creating Tree")
    L = length(R)
    M = get_num_pt(sol.em)

    sol.runtimes[:tree] = @elapsed(
        tree = sol.treetype(R[1:L-M])
    )
    localmodel_stts(sol, s, R, tree; progress=progress)
end



function localmodel_stts(sol, s, R, tree; progress=true) where {T, Φ, BC, X}
    em = sol.em
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])

    #End of timeseries to work with
    sol.spred = spred = working_ts(s,em)

    sol.runtimes[:prediction] = @elapsed(
        for n=1:sol.timesteps
            progress && println("Working on Frame $(n)/$(sol.timesteps)")
            queries = gen_queries(spred, em)

            #Iterate over queries/ spatial points
            for m=1:num_pt
                q = queries[m]

                #Find neighbors
                #Note, not yet compatible with old `neighborhood_and_distances` functions
                idxs,dists = neighbors(q,R,tree,sol.ntype)

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
    )

    cut_off_beginning!(spred,em)
    return sol
end



struct CrossPrediction{T,Φ}
    em::AbstractSpatialEmbedding
    treetype#::NNTree what's the type here?
    pred_in::Vector{Array{T,Φ}}
    pred_out::Vector{Array{T,Φ}}

    runtimes::Dict{Symbol,Float64}
    CrossPrediction{T,Φ}() where {T,Φ} = new()
end

function crosspred_stts(    train_out::AbstractVector{<:AbstractArray{T, Φ}},
                            pred_in  ::AbstractVector{<:AbstractArray{T, Φ}},
                            em::AbstractSpatialEmbedding{T,Φ,BC,X},
                            R::AbstractDataset{X,T},
                            tree::NNTree;
                            progress=true,
                            method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                            ntype::AbstractNeighborhood = FixedMassNeighborhood(3)
                        ) where {T, Φ, BC, X}
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(train_out[1])

    queries = reconstruct(pred_in, em)
    pred_out = eltype(train_out)[]

    for n=1:length(pred_in)-get_τmax(em)
        progress && println("Working on Frame $(n)/$(length(pred_in)-get_τmax(em))")
        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = queries[m+(n-1)*num_pt]

            #Find neighbors
            #Note, not yet compatible with old `neighborhood_and_distances` functions
            idxs,dists = neighbors(q,R,tree,ntype)

            xnn = R[idxs]
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

function neighbors(point, R, tree::KDTree, ntype)
    idxs,dists = knn(tree, point, ntype.K, false)
    return idxs,dists
end
function convert_idx(idx, em)
    τmax = get_τmax(em)
    num_pt = get_num_pt(em)
    t = 1 + (idx-1) ÷ num_pt + get_τmax(em)
    α = 1 + (idx-1) % num_pt
    return t,α
end

cut_off_beginning!(s,em) = deleteat!(s, 1:get_τmax(em))
