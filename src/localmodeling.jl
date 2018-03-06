using NearestNeighbors, StaticArrays
using DynamicalSystemsBase

export AbstractLocalModel
export LocalAverageModel,LocalLinearModel,LocalPolynomialModel
export TSP
export MSE1,MSEp
export estimate_param
abstract type AbstractLocalModel end


"""
    LocalAverageModel  <: AbstractLocalModel
A Model functor that computes an estimate `y_pred` for a query `q`.
Given the nearest neighbors `xnn` and their images `ynn`,
it averages over `ynn` weighted by the distances of the `xnn` to `q`.

```math
\\begin{aligned}
y\_{pred} = \\frac{\\sum(ω_i^2) y_{nn,i}}{\\sum{ω_i^2}}
\\end{aligned}
```

```math
\\begin{aligned}
ω_i = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^n\\right]^n
\\end{aligned}
```
"""
struct LocalAverageModel <: AbstractLocalModel
    n::Int #n=0,1,2,3
end
LocalAverageModel() = LocalAverageModel(1)
struct LocalLinearModel{F} <: AbstractLocalModel
    n::Int #n=0,1,2,3
    f::F
end

LocalLinearModel() = LocalLinearModel(1, 2.0)
svchooser_default(σ, μ) = σ^2/(μ^2 + σ^2)

LocalLinearModel(n::Int, μ::Real) =
LocalLinearModel(n, (σ) -> svchooser_default(σ, μ))


struct LocalPolynomialModel <: AbstractLocalModel
    n::Int #n=0,1,2,3 Exponent in weighting function
    m::Int #m=1,2,3,4 degree of Polynome
end

function (M::LocalAverageModel)(q,xnn,ynn,dists)
    @assert length(ynn)>0 "No Nearest Neighbors given"
    dmax = maximum(dists)
    y_pred = zeros(typeof(ynn[1]))
    Ω = 0.
    for (y,d) in zip(ynn,dists)
        Ω += ω2 = (1-(d/dmax)^M.n)^2M.n
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
    Sp = diagm([σ>0 ? M.f(σ)/σ : 0 for σ in S])
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
    #Regularization
    #μ = 0.8
    #f(σ) = σ^2/(μ^2+σ^2)

    #This could be a field "Regularization function"
    #in LocalPolynomialModel
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


"""
`TSP(tree,R,p,LocalModel,method,f)`

This function is a simple tool for Time Series Prediction.
Makes steps as defined in given function `f` i.e. `f(idx)=idx+1`.

Finds nearest neighbors of query point in the given `KDTree` with the supplied method
(`FixedMassNeighborhood`, `FixedSizeNeighborhood`). The nearest neighbors xnn and the points
they map to (ynn) are used to make a prediction.
(With the provided `LocalModel <: AbstractLocalModel`)

This method is applied iteratively until a prediction time series of length num_points has
been created.

Call it with code that might look like this
```
ds = DynamicalSystemsBase.Systems.roessler()
data = trajectory(ds,200)
Ntraining = 10000
p = 1000 # Points to predict
s = data[1:Ntraining,1] # Select first component as Timeseries
dim = 3 # Embedding dimension and delay
τ = 50
R = Reconstruction(s,dim,τ)
tree = KDTree(R.data[1:end-50]) # Leave off a few points at the end so that there
                                # will always be a "next" point in the Reconstruction
f(i) = i+1 # This means step size = 1
method = FixedMassNeighborhood(5) # Always find 5 nearest neighbors
LocalModel = LocalAverageModel(2) #Use local averaging and a biquadratic weight function

s_pred = TSP(tree,R,p,LocalModel,method,f)
```
"""
function TSP(
    tree::KDTree,
    R::AbstractDataset{D,T},
    q::SVector{D,T},
    num_points::Int,
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f) where {D,T}
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

TSP(tree,R,num_points,LocalModel,method,f) =
    TSP(tree,R,R[end], num_points,LocalModel,method,f)

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
Needs as input a Tree and a suffiently Reconstructed long timeseries.
(length >> dim*τ)
"""
function MSEp(tree::KDTree,
    R::AbstractDataset{D,T},
    R_test::AbstractDataset{D,T},
    p::Int,
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f::Function) where {D,T}

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


"""
    estimate_param(s::AbstractVector,dims,delay,K,N; valid_len=100, num_tries=50)

Brute Force approach to finding good parameters for the model.
Takes as arguments the timeseries `s` and all the parameters to try.
Therefore `dims`,`delay`,`K`,`N` need to be Iterables of Integers,
where `K` is the number of nearest neighbors and `N` is the degree
of the weighting function. (See `LocalAverageModel`)

Optional keyword arguments are the Validation length `valid_len`
which is the number of predicted points in MSEp and the number `num_tries`
of how many different starting points for the prediction should be used.

Returns the optimal parameter set found
and a dictionary of all parameters and their respective errors.
"""
function estimate_param(s::AbstractVector,
    dims,delay,K,N; valid_len=100, num_tries=50)
    Result = Dict{SVector{4,Int},Float64}()
    f(i) = i+1
    for n ∈ N
        LocalModel = LocalAverageModel(n)
        for D ∈ dims, τ ∈ delay
            s_train = @view s[1:end-D*τ-valid_len-num_tries-50]
            s_test = @view s[end-(D-1)*τ-valid_len-num_tries:end]
            R = Reconstruction(s_train,D,τ)
            R_test = Reconstruction(s_test,D,τ)
            tree = KDTree(R[1:end-1])
            for k ∈ K
                method = FixedMassNeighborhood(k)
                Result[@SVector([D,τ,k,n])] =
                MSEp(tree,R,R_test,valid_len,LocalModel,method,f)
            end
        end
    end
    best_param = collect(keys(Result))[findmin(values(Result))[2]]
    return best_param, Result
end
