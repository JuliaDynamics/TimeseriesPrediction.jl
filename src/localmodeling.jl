using NearestNeighbors, StaticArrays
using DynamicalSystemsBase

export AbstractLocalModel
export LocalAverageModel,LocalLinearModel,LocalPolynomialModel
export TSP
export MSE1,MSEp

abstract type AbstractLocalModel end

struct LocalAverageModel <: AbstractLocalModel
    n::Int #n=0,1,2,3
end
LocalAverageModel() = LocalAverageModel(1)
struct LocalLinearModel <: AbstractLocalModel
    n::Int #n=0,1,2,3
end
LocalLinearModel() = LocalLinearModel(1)

struct LocalPolynomialModel <: AbstractLocalModel
    n::Int #n=0,1,2,3 Exponent in weighting function
    m::Int #m=1,2,3,4 degree of Polynome
end

function (M::LocalAverageModel)(q,xnn,ynn,dists)
    @assert length(ynn)>0 "No Nearest Neighbors given"
    #Weight Function
    ω(r) = (1-r^M.n)^M.n
    dmax = maximum(dists)
    y_pred = zeros(size(ynn[1]))
    Ω = 0
    for (y,d) in zip(ynn,dists)
        Ω += ω2 = ω(d/dmax)^2
        y_pred += ω2*y
    end
    y_pred /= Ω
    return y_pred
end


function (M::LocalLinearModel)(
    q,
    xnn::Vector{SVector{D,T}},
    ynn,
    dists) where {D,T}

    @assert length(ynn)>0 "No Nearest Neighbors given"
    y_pred = zeros(size(ynn[1]))
    k= length(xnn) #Can this be inferred? Nope, and probably not worth it.
    #Weight Function
    ω(r) = (1-r^M.n)^M.n
    dmax = maximum(dists)
    #Create Weight Matrix
    W = diagm([ω(di/dmax) for di in dists])
    x_mean = mean(xnn)
    y_mean = mean(ynn)
    #Create X
    X = zeros(k,D)
    #X[:,1] = 1
    for i=1:k
        X[i,1:end] = xnn[i] - x_mean
    end
    #Pseudo Inverse
    U,S,V = svd(W*X)

    #Regularization
    #D+1 Singular Values
    μ = 2
    f(σ) = σ^2/(μ^2+σ^2)
    Sp = diagm([σ>0 ? f(σ)/σ : 0 for σ in S])
    Xw_inv = V*Sp*U'

    #The following code is meant for 1D ynn values
    #Repeat for all components
    for i=1:length(ynn[1])
        y = map(ynn->ynn[i], ynn)
        #Coefficient Vector
        ν = Xw_inv * W* y
        y_pred[i] = y_mean[i] + (q-x_mean)'* ν
    end

    return y_pred
end

function (M::LocalPolynomialModel)(
    q,
    xnn::Vector{SVector{D,T}},
    ynn,
    dists) where {D,T}

    @assert length(ynn)>0 "No Nearest Neighbors given"
    y_pred = zeros(size(ynn[1]))
    k= length(xnn) #Can this be inferred? Nope, and probably not worth it.
    #Weight Function
    ω(r) = (1-r^M.n)^M.n
    dmax = maximum(dists)
    #Create Weight Matrix
    W = diagm([ω(di/dmax) for di in dists])
    x_mean = mean(xnn)
    y_mean = mean(ynn)

    #Create X
    X = zeros(k,M.m*D)
    for i=1:k, j=1:M.m
        X[i,1+(j-1)*D:j*D] = (xnn[i] - x_mean).^j
    end
    #Pseudo Inverse
    U,S,V = svd(W*X)
    #println(S)
    #Regularization
    #D+1 Singular Values
    μ = 0.8
    #f(σ) = σ^2/(μ^2+σ^2)
    smin = 0.08
    smax = 1
    function f(σ)
        if σ < smin return 0
        elseif σ > smin return 1
        else return (1-((smax-σ)/(smax-smin))^2)^2
        end
    end
    Sp = diagm([σ>0 ? f(σ)/σ : 0 for σ in S])
    Xw_inv = V*Sp*U'

    #The following code is meant for 1D ynn values
    #Repeat for all components
    for i=1:length(ynn[1])
        y = map(ynn->ynn[i], ynn)
        #Coefficient Vector
        ν = Xw_inv * W* y
        #Query Matrix
        q_mat = T[]; sizehint!(q_mat, M.m*D)
        for j=1:M.m
            append!(q_mat,(q-x_mean).^j)
        end
        y_pred[i] = y_mean[i] + dot(q_mat,ν)
    end

    return y_pred
end

function neighborhood(q,tree::KDTree,method::FixedMassNeighborhood)
    idxs, dists = knn(tree, q,method.K, false)
end

function TSP(
    tree::KDTree,
    R::Reconstruction{D,T,τ}, # is τ used anywhere? if not ::AbstractDataset{D, T}
    q::SVector{D,T},
    num_points::Int,
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f) where {D,T,τ} # no reason to declare f::Function

    s_pred = T[]; sizehint!(s_pred,num_points+1) #Prepare estimated Timeseries
    push!(s_pred, q[end]) #Push query

    for n=1:num_points   #Iteratively estimate Timeseries
        idxs,dists = neighborhood(q,tree,method)
        xnn = R[idxs]
        ynn = R[f(idxs)]
        q = LocalModel(q,xnn,ynn, dists)
        push!(s_pred, q[end])
    end
    return s_pred
end

TSP(tree,R,num_points,LocalModel,method,f) = TSP(tree,R,R[end], num_points,LocalModel,method,f)

#####################################################################################
#                                  Error Measures                                   #
#####################################################################################

"""mean squared error of one step each
Needs as input a KDTree and a suffiently long timeseries.
(length >> dim*τ)
"""
function MSE1(tree::KDTree,
    R::Reconstruction{D,T,τ},
    s_test::Vector{T},
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f::Function) where {D,T,τ}

    R_test = Reconstruction(s_test,D,τ)
    y_test = map(q-> q[end], R_test[2:end])
    y_pred = T[]; sizehint!(y_pred, length(y_test))
    for q in R_test[1:end-1] # Remove last state,
                            #because there is nothing to compare the preciction to
        idxs,dists = neighborhood(q,tree,method)
        xnn = R[idxs]
        ynn = R[f(idxs)]
        push!(y_pred,LocalModel(q,xnn,ynn, dists)[end])
    end
    return norm(y_test-y_pred)^2/length(y_test)
end
#FIXME: I shouldn't have to square the norm... What is the solution?



"""mean squared error of iterated predictions of length p
Needs as input a Tree and a suffiently long timeseries.
(length >> dim*τ)
"""
function MSEp(tree::KDTree,
    R::Reconstruction{D,T,τ},
    s_test::Vector{T},
    p::Int,
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f::Function) where {D,T,τ}

    @assert length(s_test)>(p+(D-1)*τ) "Given Timeseries is too short"
    R_test  = Reconstruction(s_test, D, τ)
    y_test = map(q-> q[end], R_test[2:end])

    Tref = (length(R_test)-p-1)
    error = 0
    for t =1:Tref
        y_pred = TSP(tree,R,R_test[t], p, LocalModel, method,f)
        error += norm(y_test[t:t+p]-y_pred)^2 /Tref/p
    end
    return error
end
#FIXME: I shouldn't have to square the norm... What is the solution?
