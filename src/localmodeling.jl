using NearestNeighbors, StaticArrays
using DynamicalSystemsBase

export AbstractLocalModel
export LocalAverageModel,LocalLinearModel
export predict_timeseries
export MSE1,MSEp

"""
    AbstractLocalModel
Supertype of methods for making a prediction of a query point `q` using local models,
following the methods of [1]. Concrete subtypes are `LocalAverageModel` and
`LocalLinearModel`.

All models weight neighbors with the following weight function
```math
\\begin{aligned}
ω_i = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^n\\right]^n
\\end{aligned}
```
with ``d_i = ||x_{nn,i} -q||_2`` and degree `n`,
to ensure smoothness of interpolation.

### Local Average Model
    LocalAverageModel(n::Int)

The prediction is simply the weighted average of the images of
the neighbors ``y_{nn, i}`` of the query point `q`:
```math
\\begin{aligned}
y\_{pred} = \\frac{\\sum{ω_i^2 y_{nn,i}}}{\\sum{ω_i^2}}
\\end{aligned}
```

### Local Linear Model
    LocalLinearModel(n::Int, μ::Real)
    LocalLinearModel(n::Int, s_min::Real, s_max::Real)

The prediction is a weighted linear regression over the neighbors of
the query ``x_{nn, i}`` and their images ``y_{nn,i}`` as shown in [1].

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
abstract type AbstractLocalModel end

"""
    LocalAverageModel(n::Int = 2)
See [`AbstractLocalModel`](@ref).
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
        ω2 = (1-(d/dmax)^M.n)^2M.n
        Ω += ω2
        y_pred += ω2*y
    end
    y_pred /= Ω
    return y_pred
end


"""
    LocalLinearModel (n::Int, μ::Real)
    LocalLinearModel (n::Int, s_min::Real, s_max::Real)
See [`AbstractLocalModel`](@ref).
"""
struct LocalLinearModel{F} <: AbstractLocalModel
    n::Int #n=0,1,2,3
    f::F
end

LocalLinearModel() = LocalLinearModel(2, 2.0)
#Regularization functions
ridge_reg(σ, μ) = σ^2/(μ^2 + σ^2)
function mcnames_reg(σ,smin,smax)
    if σ < smin return 0
    elseif σ > smax return 1
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
    ynn::Vector{SVector{D,T}},
    dists) where {D,T}

    @assert length(ynn)>0 "No Nearest Neighbors given"
    y_pred = zeros(size(ynn[1]))
    k= length(xnn)
    #Weight Function
    ω(r) = (1-r^M.n)^M.n
    dmax = maximum(dists)
    #Create Weight Matrix
    W = diagm([ω(di/dmax) for di in dists])
    x_mean = mean(xnn)
    y_mean = mean(ynn)
    #Create X
    X = zeros(k,D)
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


#=
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
=#




function neighborhood(q,tree::KDTree,ntype::FixedMassNeighborhood)
    idxs, dists = knn(tree, q,ntype.K, false)
end

"""
    predict_timeseries(
        s::AbstractVector, D::Int, τ::T, p::Int;
        method::AbstractLocalModel = LocalAverageModel(2)
        ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
        step::Int = 1) where {T}

Timeseries prediction using local weighted modeling.
First argument can be either `s, D, τ` or a reconstructed state space `R`.
(See [`Reconstruction`](@ref)).

## Description
Finds nearest neighbors of query point `q` in the reconstruction `R` with the
supplied `ntype` (`FixedMassNeighborhood`, `FixedSizeNeighborhood`).
The nearest neighbors `xnn` and their images `ynn` are used to make a prediction,
with the provided `method <: AbstractLocalModel`.
The images `ynn` are the futures of `xnn` shifted by `step` into the future.

This method is applied iteratively until a prediction timeseries of length `p` has
been created. This method is described in [1].

## Arguments
  * `s:AbstractVector` : Input time series
  * `D::Int` : Delay embedding dimension
  * `τ::T` : Delay time, either `Int` or `Vector{Int}`. See [`Reconstruction`](@ref)
  * `p::Int` : Number of points to predict
  * `method` : Subtype of [`AbstractLocalModel`](@ref)
  * `ntype` : Subtype of [`AbstractNeighborhood`](@ref)
  * `step` : Prediction step size. `step=1` is usually a good choice.

## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*, VCH-Wiley, pp 39-65
(2006)
"""
function predict_timeseries(
    R::AbstractDataset{D,T},
    tree::KDTree,
    q::SVector{D,T},
    p::Int;
    method::AbstractLocalModel = LocalAverageModel(2),
    ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
    step::Int = 1) where {D,T}
    s_pred = T[]; sizehint!(s_pred,p+1) #Prepare estimated timeseries
    push!(s_pred, q[end]) #Push query

    for n=1:p   #Iteratively estimate timeseries
        idxs,dists = neighborhood(q,tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs+step]
        q = method(q, xnn, ynn, dists)
        push!(s_pred, q[end])
    end
    return s_pred
end

function predict_timeseries(
    s::AbstractVector, D::Int, τ::T, p::Int;
    method::AbstractLocalModel = LocalAverageModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
    step::Int = 1) where {T}

    R = Reconstruction(s, D, τ)
    tree = KDTree(R[1:end-step])
    #Still take away step elements so that y = R[i+step] is always defined

    return predict_timeseries(R, tree, R[end], p, method, ntype, step)
end
predict_timeseries(R::AbstractDataset, p::Int;
    method::AbstractLocalModel = LocalAverageModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
    step::Int = 1) =
predict_timeseries(R, KDTree(R[1:end-step]), R[end], p, method, ntype, step)


#####################################################################################
#                                  Error Measures                                   #
#####################################################################################

"""
    MSE1(R::AbstractDataset{D,T},R_test, method, ntype, step) -> error

Compute mean squared error of single predictions using test set `R_test`.

## Description
This error measure, as described in [1], takes in a prediction model consisting of
`R`, `method`, `ntype` and `step` and evaluates its performance.
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
function MSE1(
    R::AbstractDataset{D,T},
    tree::KDTree,
    R_test::AbstractDataset{D,T};
    method::AbstractLocalModel = LocalAverageModel(2),
    ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
    step::Int = 1) where {D,T}

    y_test = map(q-> q[end], R_test[2:end])
    y_pred = T[]; sizehint!(y_pred, length(y_test))
    for q in R_test[1:end-1] # Remove last state,
                            #because there is nothing to compare the preciction to
        idxs,dists = neighborhood(q,tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs+step]
        push!(y_pred,method(q,xnn,ynn, dists)[end])
    end
    return norm(y_test-y_pred)^2/length(y_test)
end
#FIXME: I shouldn't have to square the norm... What is the solution?

MSE1(R, R_test; kwargs...) =
MSE1(R, KDTree(R), R_test; kwargs...)


"""
    MSEp(R::AbstractDataset{D,T}, R_test, p, method, ntype, step) -> error

Compute mean squared error of iterated predictions of length `p` using test set `R_test`.

## Description
This error measure, as described in [1], takes in a prediction model consisting of `R`,
 `method`, `ntype` and `step` and evaluates its performance. The test set `R_test` is
a delay reconstruction with the same delay `τ` and dimension `D` as `R`.
For each subset of `R_test` with length `p` it calls `predict_timeseries`.
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
function MSEp(
    R::AbstractDataset{D,T},
    tree::KDTree,
    R_test::AbstractDataset{D,T},
    p::Int;
    kwargs...) where {D,T}

    y_test = map(q-> q[end], R_test[2:end])

    Tref = (length(R_test)-p-1)
    error = 0
    for t =1:Tref
        y_pred = predict_timeseries(R,tree,R_test[t], p; kwargs...)
        error += norm(y_test[t:t+p]-y_pred)^2 /Tref/p
    end
    return error
end
#FIXME: I shouldn't have to square the norm... What is the solution?
MSEp(R, R_test, p; kwargs...) =
MSEp(R, KDTree(R[1:end-1]), R_test, p; kwargs...)


"""
    estimate_param(s::AbstractVector, dims, delay, K, N; kwargs...) -> (D, τ, k, n)

Brute Force approach to finding good parameters for the model.

## Description
Takes as arguments the timeseries `s` and all the parameters to try.
Therefore `dims`,`delay`,`K`,`N` need to be Iterables of Integers,
where `K` is the number of nearest neighbors and `N` is the degree
of the weighting function. (See [`AbstractLocalModel`](@ref))

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
    step = 1
    for n ∈ N
        method = LocalAverageModel(n)
        for D ∈ dims, τ ∈ delay
            s_train = @view s[1:end-D*τ-valid_len-num_tries-50]
            s_test = @view s[end-(D-1)*τ-valid_len-num_tries:end]
            R = Reconstruction(s_train,D,τ)
            R_test = Reconstruction(s_test,D,τ)
            tree = KDTree(R[1:end-1])
            for k ∈ K
                ntype = FixedMassNeighborhood(k)
                Result[@SVector([D,τ,k,n])] =
                MSEp(R, tree, R_test, valid_len, method, ntype, step)
            end
        end
    end
    best_param = collect(keys(Result))[findmin(values(Result))[2]]
    return best_param
end
