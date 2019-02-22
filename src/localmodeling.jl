using NearestNeighbors, StaticArrays, Statistics
using DelayEmbeddings
using DelayEmbeddings: AbstractDataset

export AbstractLocalModel
export AverageLocalModel,LinearLocalModel
export localmodel_tsp, localmodel_cp
export MSEp

"""
    AbstractLocalModel
Supertype of methods for making a prediction of a query point `q` using local models,
following the methods of [1]. Concrete subtypes are `AverageLocalModel` and
`LinearLocalModel`.

All models weight neighbors with a chosen function, so that distant neighbors
have smaller impact on the prediction and so that the interpolation
is smooth. The default weighting function we use is
```math
\\begin{aligned}
ω_i(d_i,d_{max}) = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^2\\right]^4
\\end{aligned}
```
with ``d_i = ||x_{nn,i} -q||_2`` being the distance of each neighbor from
the query point.

You can also provide your own function or give
`ω_safe(d, dmax) = dmax > 0 ? (1.1 - (d/dmax)^2)^4 : 1.0`
for a safe version of ``ω`` that takes into acount edge cases. Finally
you can also give `nothing` in place of `ω`. In that case no weighting is done
and direct average of neighbors is returned.

### Average Local Model

    AverageLocalModel(ω)

The prediction is simply the weighted average of the images ``y_{nn, i}`` of
the neighbors ``x_{nn, i}`` of the query point `q`, weighting using given
function `ω`
```math
\\begin{aligned}
y_{pred} = \\frac{\\sum{\\omega_i y_{nn,i}}}{\\sum{\\omega_i}}
\\end{aligned}
```

### Linear Local Model

    LinearLocalModel([ω ], μ::Real=2.])
    LinearLocalModel([ω ], s_min::Real, s_max::Real)

The prediction is a weighted linear regression over the neighbors ``x_{nn, i}`` of
the query and their images ``y_{nn,i}`` as shown in [1].

Giving either `μ` or `s_min` and `s_max` determines which type of regularization
is applied.
  * `μ` : Ridge Regression
    ```math
    \\begin{aligned}
    f(\\sigma) = \\frac{\\sigma^2}{\\mu^2 + \\sigma^2}
    \\end{aligned}
    ```
  *  `s_min`, `s_max` : Soft Threshold
    ```math
    \\begin{aligned}
    f(\\sigma) = \\begin{cases} 0, & \\sigma < s_{min}\\\\
    \\left(1 - \\left( \\frac{s_{max}-\\sigma}{s_{max}-s_{min}}\\right)^2 \\right)^2,
    &s_{min} \\leq \\sigma \\leq s_{max} \\\\
    1, & \\sigma > s_{max}\\end{cases}
    \\end{aligned}
    ```

## References
[1] : D. Engster & U. Parlitz, *Handbook of Time Series Analysis* Ch. 1,
VCH-Wiley (2006)
"""
abstract type AbstractLocalModel end

ω_safe(d, dmax) = dmax > 0 ? (x=d/dmax; y= 1.1-x*x; y*y*y*y) : 1.
ω_unsafe(d, dmax) = (x=d/dmax; y= 1-x*x; y*y*y*y)

"""
    AverageLocalModel(ω::Function = ω_unsafe)
See [`AbstractLocalModel`](@ref).
"""
struct AverageLocalModel{F} <: AbstractLocalModel
    ω::F
end
AverageLocalModel() = AverageLocalModel(ω_unsafe)

function (M::AverageLocalModel)(q,xnn,ynn,dists)
    if length(ynn) > 1 && (dmax = maximum(dists)) > 0
        y_pred = zeros(typeof(ynn[1]))
        Ω = zero(typeof(dmax))
        for (y,d) in zip(ynn,dists)
            ω2 = M.ω.(d, dmax)
            Ω += ω2
            y_pred = y_pred .+ ω2*y
        end
        y_pred /= Ω
        return y_pred
    end
    return ynn[1]
end

(M::AverageLocalModel{Nothing})(q,xnn,ynn,dists) = sum(ynn)/length(ynn)



"""
    LinearLocalModel([ω ], μ::Real=2.])
    LinearLocalModel([ω ], s_min::Real, s_max::Real)
See [`AbstractLocalModel`](@ref).
"""
struct LinearLocalModel{Ω, F} <: AbstractLocalModel
    ω::Ω # weighting
    f::F # regularization
end

LinearLocalModel(μ::Real=2.) = LinearLocalModel(ω_unsafe, μ)

#Regularization functions
ridge_reg(σ, μ) = σ^2/(μ^2 + σ^2)
function mcnames_reg(σ,smin,smax)
    if σ < smin return 0
    elseif σ > smax return 1
    else return (1-((smax-σ)/(smax-smin))^2)^2
    end
end

LinearLocalModel(ω,μ::Real=2.) =
LinearLocalModel(ω,(σ) -> ridge_reg(σ, μ))

LinearLocalModel(ω, s_min::Real, s_max::Real) =
LinearLocalModel(ω, (σ) -> mcnames_reg(σ, s_min, s_max))

function (M::LinearLocalModel)(
    q,
    xnn::Vector{SVector{L,T}},
    ynn::Vector{TT},
    dists) where {L,T,TT}

    @assert length(ynn)>0 "No Nearest Neighbors given"
    k= length(xnn)
    #Weight Function
    dmax = maximum(dists)
    #Create Weight Matrix
    W = Diagonal([M.ω.(di,dmax) for di in dists])
    x_mean = mean(xnn)
    y_mean = mean(ynn)
    #Create X
    X = zeros(k,L)
    for i=1:k
        X[i,1:end] = xnn[i] .- x_mean
    end
    #Pseudo Inverse
    U,S,V = svd(W*X)

    #Regularization
    #D+1 Singular Values
    Sp = Diagonal([σ>0 ? M.f(σ)/σ : 0 for σ in S])
    Xw_inv = V*Sp*U'
    #The following code is meant for 1D ynn values
    #Repeat for all components
    y_pred = map(eachindex(ynn[1])) do i
        y = map(ynn->ynn[i], ynn)
        #Coefficient Vector
        ν = Xw_inv * W* y
        y_mean[i] + (q-x_mean)'* ν
    end

    return y_pred
end


#= # THE FOLLOWING IS THE POLYNOMIAL MODEL, WHICH WE DID NOT IMPLEMENT
# BECAUSE IT WAS NOT NEARLY AS GOOD AS THE OTHER TWO.
# SOMEONE MIGHT WANT TO TAKE A LOOK AT ANY POINT IN THE FUTURE
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


function neighborhood_and_distances(point::AbstractVector,R::AbstractDataset, tree,
                      ntype::FixedMassNeighborhood, n::Int, w::Int = 1)
    idxs,dists = knn(tree, point, ntype.K, false, i -> abs(i-n) < w)
    return idxs,dists
end
function neighborhood_and_distances(point::AbstractVector,R, tree,
    ntype::FixedMassNeighborhood)
    idxs,dists = knn(tree, point, ntype.K, false)
    return idxs,dists
end

function neighborhood_and_distances(point::AbstractVector,R::AbstractDataset, tree,
                      ntype::FixedSizeNeighborhood, n::Int, w::Int = 1)
    idxs = inrange(tree, point, ntype.ε)
    filter!((el) -> abs(el - n) ≥ w, idxs)
    dists = [norm(x-point) for x in R[idxs]]   #Note: this is not an SVector!
    return idxs,dists
end
function neighborhood_and_distances(point::AbstractVector,R::AbstractDataset,
                      tree, ntype::FixedSizeNeighborhood)
    idxs = inrange(tree, point, ntype.ε)
    dists = [norm(x-point) for x in R[idxs]]   #Note: this is not an SVector!
    return idxs,dists
end


function _localmodel_tsp(R::AbstractDataset{D,T},
                        tree::KDTree,
                        q::SVector{D,T},
                        p::Int;
                        method::AbstractLocalModel = AverageLocalModel(),
                        ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
                        stepsize::Int = 1) where {D,T}


    s_pred = Vector{SVector{D, T}}(undef, p+1)
    s_pred[1] = q
    for n=2:p+1   #Iteratively estimate timeseries
        idxs,dists = neighborhood_and_distances(q,R, tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs.+stepsize]
        q = method(q, xnn, ynn, dists)
        s_pred[n] = q
    end
    return Dataset(s_pred)
end

"""
    localmodel_tsp(s, γ::Int, τ, p::Int; method, ntype, stepsize)
    localmodel_tsp(s, p::Int; method, ntype, stepsize)

Perform a timeseries prediction for `p` points,
using local weighted modeling [1]. The function always returns an
object of the same type as `s`, which can be either a timeseries (vector) or an
`AbstractDataset` (trajectory), and the returned data
always contains the final point of `s` as starting point. This means that the
returned data has length of `p + 1`.

If given `(s, γ, τ)`, it first calls `reconstruct` (from `DelayEmbeddings`) on `s`
with `(γ, τ)`. If given only `s` then no reconstruction is done.

## Keyword Arguments
  * `method = AverageLocalModel(ω_unsafe)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(2)` : Subtype of `AbstractNeighborhood` (from
    `DelayEmbeddings`).
  * `stepsize = 1` : Prediction step size.

## Description
Given a query point, the function finds its neighbors using neighborhood `ntype`.
Then, the neighbors `xnn` and their images `ynn` are used to make a prediction for
the future of the query point, using the provided `method`.
The images `ynn` are the points `xnn` shifted by `stepsize` into the future.

The algorithm is applied iteratively until a prediction of length `p` has
been created, starting with the query point to be the last point of the timeseries.

## References
[1] : D. Engster & U. Parlitz, *Handbook of Time Series Analysis* Ch. 1,
VCH-Wiley (2006)
"""
function localmodel_tsp(R::AbstractDataset{B}, p::Int;
    method::AbstractLocalModel = AverageLocalModel(),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
    stepsize::Int = 1) where B
    B > 1 || throw(ArgumentError("Dataset Dimension needs to be >1! ",
    "Alternatively pass embedding parameters."))
    return _localmodel_tsp(R, KDTree(R[1:end-stepsize]), R[end], p;
    method=method, ntype=ntype, stepsize=stepsize)
end

function localmodel_tsp(
    s::AbstractVector, γ::Int, τ::T, p::Int; kwargs... ) where {T}
    localmodel_tsp(reconstruct(s, γ, τ), p; kwargs...)[:,γ+1]
end

function localmodel_tsp(
    ss::AbstractDataset{B}, γ::Int, τ::T, p::Int; kwargs...) where {B,T}
    sind = SVector{B, Int}(((γ+1)*B - i for i in B-1:-1:0)...)
    localmodel_tsp(reconstruct(ss, γ, τ), p; kwargs...)[:,sind]
end





#####################################################################################
#                                  Cross Prediction                                   #
#####################################################################################

"""
    localmodel_cp(source_pool, target_pool, source_pred,  γ, τ; kwargs...)

Perform a cross prediction from  _source_ to _target_,
using local weighted modeling [1]. `source_pred` is the input for the prediction
and `source_pool` and `target_pool` are used as pooling/training data for the predictions.
The function always returns an object of the same type as `target_pool`,
which can be either a timeseries (vector) or an `AbstractDataset` (trajectory).

## Keyword Arguments
  * `method = AverageLocalModel(ω_unsafe)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(2)` : Subtype of `AbstractNeighborhood` (from
    `DelayEmbeddings`).
  * `stepsize = 1` : Prediction step size.

Instead of passing `γ` & `τ` for reconstruction one may also give
existing `Dataset`s as `source_pool` and `source_pred`.
In this case an additional keyword argument `y_idx_shift::Int=0` may be necessary
to account for the index shift introduced in the reconstruction process.

## Description
Given a query point, the function finds its neighbors using neighborhood `ntype`.
Then, the neighbors `xnn` and their images `ynn` are used to make a prediction for
the image of the query point, using the provided `method`.

## References
[1] : D. Engster & U. Parlitz, *Handbook of Time Series Analysis* Ch. 1,
VCH-Wiley (2006)
"""
function localmodel_cp(R::AbstractDataset{D,T},
                       target_train,
                       source_pred::AbstractDataset{D,T},
                       tree::KDTree;
                       method::AbstractLocalModel = AverageLocalModel(),
                       ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
                       y_idx_shift::Int=0) where {D,T}

    N = length(source_pred)
    target_pred = typeof(target_train)(undef, N)
    for n=1:N   #Iteratively estimate timeseries
        q = source_pred[n]
        idxs,dists = neighborhood_and_distances(q,R, tree,ntype)
        xnn = R[idxs]
        ynn = target_train[idxs .+ y_idx_shift]
        target_pred[n] = method(q, xnn, ynn, dists)[1]
    end
    return target_pred
end


function localmodel_cp(
    source_train::AbstractDataset{B},
    target_train,
    source_pred::AbstractDataset{B};kwargs...) where B
    B > 1 || throw(ArgumentError("Dataset Dimension needs to be >1! ",
    "Alternatively pass embedding parameters."))
    return localmodel_cp(source_train, target_train, source_pred,KDTree(source_train); kwargs... )
end

function localmodel_cp(
    source_train,
    target_train,
    source_pred,
     γ::Int, τ::Int; kwargs... )
    localmodel_cp(reconstruct(source_train, γ, τ),
                    target_train,
                    reconstruct(source_pred, γ, τ);
                    y_idx_shift=γ*τ, kwargs...)
end
function localmodel_cp(
    source_train,
    target_train,
    source_pred,
     γ::Int, τ::T; kwargs... ) where {T}
    localmodel_cp(reconstruct(source_train, γ, τ),
                    target_train,
                    reconstruct(source_pred, γ, τ);
                    y_idx_shift=maximum(τ), kwargs...)
end

#####################################################################################
#                                  Error Measures                                   #
#####################################################################################


function MSE1(
    R::AbstractDataset{D,T},
    tree::KDTree,
    R_test::AbstractDataset{D,T};
    method::AbstractLocalModel = AverageLocalModel(),
    ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
    stepsize::Int = 1) where {D,T}

    y_test = map(q-> q[end], R_test[2:end])
    y_pred = T[]; sizehint!(y_pred, length(y_test))
    for q in R_test[1:end-1] # Remove last state,
                            #because there is nothing to compare the preciction to
        idxs,dists = neighborhood_and_distances(q, R, tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs.+stepsize]
        push!(y_pred,method(q,xnn,ynn, dists)[end])
    end
    return norm(y_test-y_pred)^2/length(y_test)
end
#FIXME: I shouldn't have to square the norm... What is the solution?

"""
    MSEp(R::AbstractDataset{D,T}, R_test, p; method, ntype, stepsize) -> error

Compute mean squared error of iterated predictions of length `p`
using test set `R_test`.

## Description
This error measure takes in a prediction model consisting of `R`,
`method`, `ntype` and `stepsize` and evaluates its performance. The test set
`R_test` is
a delay reconstruction with the same delay `τ` and dimension `D` as `R`.
For each subset of `R_test` with length `p` it calls `localmodel_tsp`.
The model error is then defined as
```math
\\begin{aligned}
MSE_p = \\frac{1}{p|T_{ref}|}\\sum_{t\\in T_{ref}}\\sum_{i=1}^{p} \\left(y_{t+i}
- y_{pred,t+i} \\right)^2
\\end{aligned}
```
where ``|T_{ref}|`` is the number of subsets of `R_test` used.
## References
See [`localmodel_tsp`](@ref).
"""
function MSEp(
    R::AbstractDataset{D,T},
    tree::KDTree,
    R_test::AbstractDataset{D,T},
    p::Int;
    kwargs...) where {D,T}
    if p == 1
        return MSE1(R,tree,R_test;kwargs...)
    end
    Tref = (length(R_test)-p-1)
    error = 0
    for t =1:Tref
        R_pred = _localmodel_tsp(R,tree,R_test[t], p; kwargs...)
        error += norm(R_test[t:t+p]-R_pred.data)^2 /Tref/p
    end
    return error
end
#FIXME: I shouldn't have to square the norm... What is the solution?
MSEp(R, R_test, p; method::AbstractLocalModel = AverageLocalModel(),
ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
stepsize::Int = 1) =
MSEp(R, KDTree(R[1:end-stepsize]), R_test, p; method=method,
ntype=ntype, stepsize=stepsize)
