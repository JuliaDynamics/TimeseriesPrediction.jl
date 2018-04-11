using NearestNeighbors
using TimeseriesPrediction


###########################################################################################
#                                     Prediction                                          #
###########################################################################################

function localmodel_stts(s::AbstractArray{T,Ψ},D,τ,p,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where {T, Ψ}
    Φ = Ψ-1
    R = myReconstruction(s,D,τ,B,k,boundary, a, b)
    return _localmodel_stts(s, R, D, τ, p, B, k, boundary, a, b)
end

function _localmodel_stts(s::AbstractArray{T,3},R, D, τ, p, B, k, boundary, a, b;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where T
    X,Y,L = size(s)

    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R[1:end-X*Y])

    #Prepare s_pred with end of STTS so all initial queries can be created
    s_pred = s[:,:,L-D*τ:L]

    #New state that will be predicted, allocate once and reuse
    state = zeros(X,Y)
    q = zeros(length(R[1]))
    for n=1:p
        N = size(s_pred)[end]
        for mx=1:X, my=1:Y
            #create q from previous predictions
            qidx = 1
            for   t=N-D*τ+1:τ:N, j=my-B*k:k:my+B*k, i=mx-B*k:k:mx+B*k
                if 0< i <=X && 0< j <=Y
                    q[qidx] = s_pred[i,j,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b
            q[qidx+1] = a*(-1+2*(my-1)/(Y-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]
            ynn = map(y -> y[1+(2*B+1)*B+B+(D-1)*(2*B+1)^2],R[idxs+X*Y])

            state[mx,my] = method(q,xnn,ynn,dists)[1]
        end
        s_pred = cat(3,s_pred,state)
    end
    #return only the predictions without boundary and w/o old STTS
    return s_pred[:,:,D*τ+1:end]
end

function _localmodel_stts(s::AbstractArray{T,2},R, D, τ, p, B, k, boundary, a, b;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)) where T
    X,L = size(s)

    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R[1:end-X])

    #Prepare s_pred with end of STTS so all initial queries can be created
    s_pred = s[:,L-D*τ:L]

    #New state that will be predicted, allocate once and reuse
    state = zeros(X)
    q = zeros(length(R[1]))
    for n=1:p
        N = size(s_pred)[end]
        for mx=1:X
            #create q from previous predictions
            qidx = 1
            for   t=N-D*τ+1:τ:N, i=mx-B*k:k:mx+B*k
                if 0< i <=X
                    q[qidx] = s_pred[i,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]
            ynn = map(y -> y[1+B+(D-1)*(2*B+1)],R[idxs+X])

            state[mx] = method(q,xnn,ynn,dists)[1]
        end
        s_pred = cat(2,s_pred,state)
    end
    #return only the predictions without boundary and w/o old STTS
    return s_pred[:,D*τ+1:end]
end




function crosspred_stts(strain,ytrain,spred,D,τ,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(1))

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
