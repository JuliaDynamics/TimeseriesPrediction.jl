using NearestNeighbors, StaticArrays
using DynamicalSystemsBase

export AbstractLocalModel
export LocalAverageModel,LocalLinearModel,LocalPolynomialModel
export predict
export MSE1,MSEp
export estimate_param

"""
    AbstractLocalModel
Supertype of methods for making a prediction `y` for the query `q` given it's nearest
neighbors `xnn` and their respective images `ynn`. The distances between `q` and the `xnn`
are given in `dists` to allow for a weighted model.

Concrete subtypes:
  * `LocalAverageModel(n::Int)`   : Compute a weighted average over the `ynn`.
  * `LocalLinearModel(n::Int, μ::Real)` or
    `LocalLinearModel(n::Int, s_min::Real, s_max::Real)` :
    Compute a weighted linear regression over
    the given nearest neighbors.
    Parameters `μ`,`s_min`,`s_max` decide which method of regularization is used.
    (See [`LocalLinearModel`](@ref))
"""
abstract type AbstractLocalModel end

"""
    LocalAverageModel(n::Int = 2)

Return an estimate `y_pred` for a query point `q`.

## Description
Given the nearest neighbors `xnn` and their images `ynn`,
average over `ynn` weighted by the distances of the `xnn` to `q`:
```math
\\begin{aligned}
y\_{pred} = \\frac{\\sum{ω_i^2 y_{nn,i}}}{\\sum{ω_i^2}}
\\end{aligned}
```
where `y_pred` and `ynn[i]` may be vectors themselves.

The weighting parameter for each neighbor is
```math
\\begin{aligned}
ω_i = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^n\\right]^n
\\end{aligned}
```
with ``d_i = ||x_{nn,i} -q||_2`` and degree `n` a property of `LocalAverageModel`
"""
struct LocalAverageModel <: AbstractLocalModel
    n::Int #n=0,1,2,3
end
LocalAverageModel() = LocalAverageModel(2)


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


"""
    LocalLinearModel (n::Int, μ::Real)
    LocalLinearModel (n::Int, s_min::Real, s_max::Real)

Return an estimate `y_pred` for a query point `q`.

## Description
Given the nearest neighbors `xnn` and their images `ynn`,
perform a weighted linear regression over `xnn` and `ynn`.
The method employed is stated in [1].

The weighting parameter for each neighbor is
```math
\\begin{aligned}
ω_i = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^n\\right]^n
\\end{aligned}
```
with ``d_i = ||x_{nn,i} -q||_2`` and degree `n` a property of `LocalLinearModel`

Giving either `μ` or `s_min` and `s_max` determines which type of regularization is applied.
  * `μ` : Ridge Regression
    ```math
    \\begin{aligned}
    f(σ) = \\frac{σ^2}{μ^2 + σ^2}
    \\end{aligned}
    ```
  *  `s_min`, `s_max` : Soft Threshold
    ```math
    \\begin{aligned}
    f(σ) = \\begin{cases} 0, &σ < s_{min}\\\\
    \\left(1 - \\left( \\frac{s_{max}-σ}{s_{max}-s_{min}}\\right)^2 \\right)^2, &s_{min} \\leq
    σ \\leq s_{max} \\\\
    1, &σ > s_{max}\\end{cases}
    \\end{aligned}
    ```
## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*, VCH-Wiley, pp 39-65
(2006)
"""
struct LocalLinearModel{F} <: AbstractLocalModel
    n::Int #n=0,1,2,3
    f::F
end

LocalLinearModel() = LocalLinearModel(2, 2.0)
ridge_reg(σ, μ) = σ^2/(μ^2 + σ^2)
function mcnames_reg(σ,smin,smax)
    if σ < smin return 0
    elseif σ > smin return 1
    else return (1-((smax-σ)/(smax-smin))^2)^2
    end
end
LocalLinearModel(n::Int, μ::Real) =
LocalLinearModel(n, (σ) -> ridge_reg(σ, μ))

LocalLinearModel(n::Int, s_min::Real, s_max::Real) =
LocalLinearModel(n, (σ) -> mcnames_reg(σ, s_min, s_max))

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
struct LocalPolynomialModel <: AbstractLocalModel
    n::Int #n=0,1,2,3 Exponent in weighting function
    m::Int #m=1,2,3,4 degree of Polynome
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
    predict(R::AbstractDataset{D,T}, [q=R[end],] p::Int, LocalModel, method, f)

A simple tool for timeseries prediction.

## Description
Finds nearest neighbors of query point `q` in the given Reconstruction `R` with the
supplied method (`FixedMassNeighborhood`, `FixedSizeNeighborhood`).
The nearest neighbors `xnn` and their images `ynn`, determined through `f`,
are used to make a prediction.
(With the provided `LocalModel <: AbstractLocalModel`)

This method is applied iteratively until a prediction timeseries of length `p` has
been created. This method is described in [1].

## Arguments
  * `R::AbstractDataset{D, T}` : Reconstructed state space (See [`Reconstruction`](@ref))
  * `q::SVector{D,T}` : Optional query point.
    Defaults to the last reconstructed state `R[end]`
  * `p::Int` : Number of points to predict
  * `LocalModel` : Either [`LocalAverageModel`](@ref) or [`LocalLinearModel`](@ref)
  * `method` : Either `FixedMassNeighborhood` (fixed number of neighbors),
    `FixedSizeNeighborhood` (fixed size of neighborhood)
  * `f` : Maps indices of `xnn` neighbors to their images `ynn`.
    ``f(i) = i+1`` is usually a good choice


## Examples
```julia
ds = DynamicalSystemsBase.Systems.roessler()
data = trajectory(ds,200)
p = 1000 # Number of points to predict
s = data[:,1] # Select first component as timeseries
dim = 3 # Embedding dimension and delay
τ = 150
R = Reconstruction(s,dim,τ)
tree = KDTree(R.data[1:end-50])
# Leave off a few points at the end so that there
# will always be a "next" point in the Reconstruction
f(i) = i+1 # This means step size = 1
method = FixedMassNeighborhood(2) # Always find 2 nearest neighbors
LocalModel = LocalAverageModel(2) #Use local averaging and a biquadratic weight function

s_pred = predict(R,p,LocalModel,method,f)
```
## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*, VCH-Wiley, pp 39-65
(2006)

"""
function predict(
    tree::KDTree,
    R::AbstractDataset{D,T},
    q::SVector{D,T},
    num_points::Int,
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f) where {D,T}
    s_pred = T[]; sizehint!(s_pred,num_points+1) #Prepare estimated timeseries
    push!(s_pred, q[end]) #Push query

    for n=1:num_points   #Iteratively estimate timeseries
        idxs,dists = neighborhood(q,tree,method)
        xnn = R[idxs]
        ynn = R[f(idxs)]
        q = LocalModel(q,xnn,ynn, dists)
        push!(s_pred, q[end])
    end
    return s_pred
end

predict(tree::KDTree, R, num_points, LocalModel, method, f) =
    predict(tree, R, R[end], num_points, LocalModel, method, f)
predict(R, num_points, LocalModel, method, f) =
    predict(KDTree(R[1:end-30]), R, R[end], num_points, LocalModel, method, f)
predict(R, q, num_points, LocalModel, method, f) =
    predict(KDTree(R[1:end-30]), R, q, num_points, LocalModel, method, f)


#####################################################################################
#                                  Error Measures                                   #
#####################################################################################

"""
    MSE1(R::AbstractDataset{D,T}, LocalModel, method, f) -> error

Compute mean squared error of single predictions using test set `R_test`.

## Description
This error measure, as described in [1], takes in a prediction model consisting of `tree`,
`R`, `LocalModel`, `method` and `f` and evaluates its performance.
The test set `R_test` is a delay reconstruction with the same delay `τ` and dimension `D` as
`R`.
For every point in `R_test` (except for the last) the image `y` is predicted.
The model error is defined as
```math
\\begin{aligned}
MSE_1 = \\frac{1}{|T_{ref}|}\\sum_{t\\in T_{ref}} \\left(y_{t} - y_{pred,t} \\right)^2
\\end{aligned}
```
where ``|T_{ref}|`` is the total number of predictions made.

## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*, VCH-Wiley, pp 39-65
(2006)
"""
function MSE1(tree::KDTree,
    R::AbstractDataset{D,T},
    R_test::AbstractDataset{D,T},
    LocalModel::AbstractLocalModel,
    method::AbstractNeighborhood,
    f::Function) where {D,T}

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

MSE1(R, R_test, LocalModel, method, f) =
MSE1(KDTree(R[1:end-30]), R, R_test, LocalModel, method, f)


"""
    MSEp(R::AbstractDataset{D,T}, R_test, p, LocalModel, method, f) -> error

Compute mean squared error of iterated predictions of length `p` using test set `R_test`.

## Description
This error measure, as described in [1], takes in a prediction model consisting of `tree`,
`R`, `LocalModel`, `method` and `f` and evaluates its performance. The test set `R_test` is
a delay reconstruction with the same delay `τ` and dimension `D` as `R`.
For each subset of `R_test` with length `p` it calls `predict` to predict the timeseries.
The model error is then defined as
```math
\\begin{aligned}
MSE_p = \\frac{1}{p|T_{ref}|}\\sum_{t\\in T_{ref}}\\sum_{i=1}^{p} \\left(y_{t+i} - y_{pred,t+i} \\right)^2
\\end{aligned}
```
where ``|T_{ref}|`` is the number of subsets of `R_test` used.
## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*, VCH-Wiley, pp 39-65
(2006)
"""
function MSEp(tree::KDTree,  #_
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
        y_pred = predict(tree,R,R_test[t], p, LocalModel, method,f)
        error += norm(y_test[t:t+p]-y_pred)^2 /Tref/p
    end
    return error
end
#FIXME: I shouldn't have to square the norm... What is the solution?
MSEp(R, R_test,p, LocalModel, method, f) =
MSEp(KDTree(R[1:end-30]), R, R_test,p, LocalModel, method, f)


"""
    estimate_param(s::AbstractVector, dims, delay, K, N; kwargs...) -> (D, τ, k, n)

Brute Force approach to finding good parameters for the model.

## Description
Takes as arguments the timeseries `s` and all the parameters to try.
Therefore `dims`,`delay`,`K`,`N` need to be Iterables of Integers,
where `K` is the number of nearest neighbors and `N` is the degree
of the weighting function. (See [`LocalAverageModel`](@ref))

Create Delay `Reconstruction`s and `KDTree`s for all parameter combinations.
Evaluate Models by calling `MSEp` and return best parameter set found.

## Keyword Arguments
  * valid_len=100 : Validation length - Number of prediction points used for error
    calculation.
  * num_tries=50  : Number of different starting queries for error calculation.
"""
function estimate_param(s::AbstractVector,
    dims, delay, K, N; valid_len=100, num_tries=50)
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
    return best_param
end
