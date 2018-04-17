using NearestNeighbors
using TimeseriesPrediction


###########################################################################################
#                                     Prediction                                          #
###########################################################################################

function localmodel_stts(s::AbstractVector{Array{T, Φ}},
    D,τ,p,B=1,k=1;
    boundary=20,
    weighting::Tuple{Real, Real} = (0,0),
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    M = prod(size(s[1]))
    L = length(s) #Number of temporal points
    R = myReconstruction(s,D,τ,B,k,boundary, weighting)
    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R[1:end-M])

    s_pred = s[L-D*τ:L]
    return _localmodel_stts(s_pred, R, tree, D, τ, p, B, k, boundary,
    weighting, method, ntype)[D*τ+1:end]
end

function gen_qs(s_pred, D, τ, B, k, boundary, weighting)
    N = length(s_pred)
    s_slice = @view(s_pred[N-D*τ+1:τ:N])
    return myReconstruction(s_slice, D, τ, B, k, boundary, weighting)
end

function _localmodel_stts(s::AbstractVector{Array{T, Φ}},
    R, tree ,D, τ, p, B, k, boundary, weighting, method, ntype) where {T, Φ}
    M = prod(size(s[1]))
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])
    #Index of relevant element in ynn (not proven but seemingly correct)
    im = 1 + (D-1)*(2B+1)^Φ + B*sum(i -> (2B+1)^(Φ-i), 1:Φ)
    for n=1:p
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
    pred_in ::AbstractVector{Array{T, Φ}},
    R, tree, D, τ, B, k,
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
