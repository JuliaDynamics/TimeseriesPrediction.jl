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
end

function CrossPrediction(train_in ::AbstractVector{<:AbstractArray{T, Φ}},
                        train_out::AbstractVector{<:AbstractArray{T, Φ}},
                        pred_in  ::AbstractVector{<:AbstractArray{T, Φ}},
                        em::AbstractSpatialEmbedding{T,Φ,BC,X};
                        treetype = KDTree,
                        method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                        ntype::AbstractNeighborhood = FixedMassNeighborhood(3),
                        progress=true
                        ) where {T, Φ, BC, X}
    sol = CrossPrediction{T,Φ,BC,X}(em, method, ntype, treetype, pred_in,
                                    Dict{Symbol,Float64}(), Array{T,Φ}[])
    CrossPrediction(sol, train_in, train_out; progress=true)
end

function CrossPrediction(sol, train_in, train_out; progress=true)
    progress && println("Reconstructing")
    sol.runtimes[:recontruct] = @elapsed(
        R = reconstruct(train_in,sol.em)
    )

    progress && println("Creating Tree")
    sol.runtimes[:tree] = @elapsed(
        tree = sol.treetype(R)
    )
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

    sol.runtimes[:prediction] = @elapsed(
        for n=1:length(sol.pred_in)-get_τmax(em)
            progress && println("Working on Frame $(n)/$(length(sol.pred_in)-get_τmax(em))")
            #Iterate over queries/ spatial points
            for m=1:num_pt
                q = queries[m+(n-1)*num_pt]

                #Find neighbors
                #Note, not yet compatible with old `neighborhood_and_distances` functions
                idxs,dists = neighbors(q,R,tree,sol.ntype)

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
    )
    return sol
end
