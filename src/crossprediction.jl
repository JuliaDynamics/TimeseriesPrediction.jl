export crossprediction

###########################################################################################
#                                  Cross Prediction                                       #
###########################################################################################


function crossprediction(train_in ::AbstractVector{<:AbstractArray{T, Φ}},
                        train_out::AbstractVector{<:AbstractArray{T, Φ}},
                        pred_in  ::AbstractVector{<:AbstractArray{T, Φ}},
                        em::AbstractSpatialEmbedding{T,Φ};
                        ttype = KDTree,
                        method::AbstractLocalModel  = AverageLocalModel(ω_safe),
                        ntype::AbstractNeighborhood = FixedMassNeighborhood(3),
                        progress=true
                        ) where {T, Φ}
    params = PredictionParameters(em, method, ntype, ttype)
    crossprediction(params, train_in, train_out, pred_in; progress=true)
end

function crossprediction(params, train_in, train_out,pred_in; progress=true)
    progress && println("Reconstructing")
    R = reconstruct(train_in,params.em)

    progress && println("Creating Tree")
    tree = params.treetype(R)

    crossprediction(params, train_out,pred_in, R, tree; progress=progress)
end

function crossprediction(params, train_out,pred_in, R, tree; progress=true)
    em = params.em
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
            idxs,dists = neighborhood_and_distances(q,R,tree,params.ntype)

            xnn = R[idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                train_out[t][α]
            end
            state[m] = params.method(q,xnn,ynn,dists)[1]
            #won't work for lin loc model, needs Vector{SVector}
        end
        push!(pred_out,copy(state))
    end
    return pred_out
end
