using NearestNeighbors
using TimeseriesPrediction


###########################################################################################
#                                     Prediction                                          #
###########################################################################################

function localmodel_stts(s::AbstractVector{Array{T, Φ}},
    D,τ,p,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    M = prod(size(s[1]))
    L = length(s) #Number of temporal points
    R = myReconstruction(s,D,τ,B,k,boundary, a, b)
    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R[1:end-M])

    s_pred = s[L-D*τ:L]
    return _localmodel_stts(s_pred, R, tree, D, τ, p, B, k, boundary, a, b)[D*τ+1:end]
end



function gen_qs(s_pred, D, τ, B, k, boundary, a, b)
    N = length(s_pred)
    s_slice = @view(s_pred[N-D*τ+1:τ:N])
    return myReconstruction(s_slice, D, τ, B, k, boundary, a, b)
end


function _localmodel_stts(s::AbstractVector{Array{T, Φ}},
    R, tree ,D, τ, p, B, k, boundary, a, b;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Φ}
    M = prod(size(s[1]))
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])
    #Index of relevant element in ynn (not proven but seemingly correct)
    im = 1 + (D-1)*(2B+1)^Φ + B*sum(i -> (2B+1)^(Φ-i), 1:Φ)
    for n=1:p
        qs = gen_qs(s, D, τ, B, k, boundary, a, b)
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






function crosspred_stts(strain::AbstractArray{T,3}, ytrain, spred,
    D,τ,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(1)) where T

    R = myReconstruction(strain,D,τ,B,k,boundary, a, b)
    X,Y,L = size(strain)

    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R)


    #Prepare s_pred with end of STTS so all initial queries can be created

    q = zeros(length(R[1]))
    N = size(spred)[end]
    ypred = zeros((X,Y,N-(D-1)*τ))
    for n=1:N-(D-1)τ
        for mx=1:X, my=1:Y
            #create q from previous predictions
            qidx = 1
            for   t=n:τ:n+(D-1)*τ, j=my-B*k:k:my+B*k, i=mx-B*k:k:mx+B*k
                if 0< i <=X && 0< j <=Y
                    q[qidx] = spred[i,j,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b
            q[qidx+1] = a*(-1+2*(my-1)/(Y-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]   #not used in method...
            ynn = ytrain[idxs+X*Y*(D-1)]    #Indeces in R are shifted by X*Y rel. to ytrain

            ypred[mx,my,n] = method(q,xnn,ynn,dists)[1]
        end
    end
    #return only the predictions without boundary and w/o old STTS
    return ypred
end

function crosspred_stts(strain::AbstractArray{T,2}, ytrain, spred,
    D,τ,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where T

    R = myReconstruction(strain,D,τ,B,k,boundary, a, b)
    X,L = size(strain)

    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R)


    #Prepare s_pred with end of STTS so all initial queries can be created

    q = zeros(length(R[1]))
    N = size(spred)[end]
    ypred = zeros((X,N-(D-1)*τ))
    for n=1:N-(D-1)τ
        for mx=1:X
            #create q from previous predictions
            qidx = 1
            for   t=n:τ:n+(D-1)*τ, i=mx-B*k:k:mx+B*k
                if 0< i <=X
                    q[qidx] = spred[i,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]   #not used in method...
            ynn = ytrain[idxs+X*(D-1)]    #Indeces in R are shifted by X*Y rel. to ytrain

            ypred[mx,n] = method(q,xnn,ynn,dists)[1]
        end
    end
    #return only the predictions without boundary and w/o old STTS
    return ypred
end
