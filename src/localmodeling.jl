using NearestNeighbors, StaticArrays
using DynamicalSystemsBase

export AbstractLocalModel
export AverageLocalModel,LinearLocalModel
export localmodel_tsp
export MSE1,MSEp

"""
    AbstractLocalModel
Supertype of methods for making a prediction of a query point `q` using local models,
following the methods of [1]. Concrete subtypes are `AverageLocalModel` and
`LinearLocalModel`.

All models weight neighbors with the following weight function
```math
\\begin{aligned}
ω_i = \\left[ 1- \\left(\\frac{d_i}{d_{max}}\\right)^n\\right]^n
\\end{aligned}
```
with ``d_i = ||x_{nn,i} -q||_2`` and degree `n`,
to ensure smoothness of interpolation.

### Average Local Model
    AverageLocalModel(n::Int)

The prediction is simply the weighted average of the images ``y_{nn, i}`` of
the neighbors ``x_{nn, i}`` of the query point `q`:
```math
\\begin{aligned}
y\_{pred} = \\frac{\\sum{ω_i^2 y_{nn,i}}}{\\sum{ω_i^2}}
\\end{aligned}
```

### Linear Local Model
    LinearLocalModel(n::Int, μ::Real)
    LinearLocalModel(n::Int, s_min::Real, s_max::Real)

The prediction is a weighted linear regression over the neighbors ``x_{nn, i}`` of
the query and their images ``y_{nn,i}`` as shown in [1].

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
    AverageLocalModel(n::Int = 2)
See [`AbstractLocalModel`](@ref).
"""
struct AverageLocalModel <: AbstractLocalModel
    n::Int #n=0,1,2,3
end
AverageLocalModel() = AverageLocalModel(2)


function (M::AverageLocalModel)(q,xnn,ynn,dists)
    if length(xnn) > 1
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
    return ynn[1]
end


"""
    LinearLocalModel(n::Int, μ::Real)
    LinearLocalModel(n::Int, s_min::Real, s_max::Real)
See [`AbstractLocalModel`](@ref).
"""
struct LinearLocalModel{F} <: AbstractLocalModel
    n::Int #n=0,1,2,3
    f::F
end

LinearLocalModel() = LinearLocalModel(2, 2.0)
#Regularization functions
ridge_reg(σ, μ) = σ^2/(μ^2 + σ^2)
function mcnames_reg(σ,smin,smax)
    if σ < smin return 0
    elseif σ > smax return 1
    else return (1-((smax-σ)/(smax-smin))^2)^2
    end
end

LinearLocalModel(n::Int, μ::Real) =
LinearLocalModel(n, (σ) -> ridge_reg(σ, μ))

LinearLocalModel(n::Int, s_min::Real, s_max::Real) =
LinearLocalModel(n, (σ) -> mcnames_reg(σ, s_min, s_max))

function (M::LinearLocalModel)(
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
function neighborhood_and_distances(point::AbstractVector,R, tree, ntype::FixedMassNeighborhood)
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
                        method::AbstractLocalModel = AverageLocalModel(2),
                        ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
                        stepsize::Int = 1) where {D,T}


    s_pred = Vector{SVector{D, T}}(p+1)
    s_pred[1] = q
    for n=2:p+1   #Iteratively estimate timeseries
        idxs,dists = neighborhood_and_distances(q,R, tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs+stepsize]
        q = method(q, xnn, ynn, dists)
        s_pred[n] = q
    end
    return Dataset(s_pred)
end

"""
    localmodel_tsp(s, D::Int, τ, p::Int; method, ntype, stepsize)
    localmodel_tsp(s, p::Int; method, ntype, stepsize)

Perform a timeseries prediction for `p` points,
using local weighted modeling [1]. The function always returns an
object of the same type as `s`, which can be either a timeseries (vector) or an
`AbstractDataset` (trajectory), and the returned data
always contains the final point of `s` as starting point. This means that the
returned data has length of `p + 1`.

If given `(s, D, τ)`, then a [`Reconstruction`](@ref) is performed on `s`
with dimension `D` and delay `τ`. If given only `s` then no [`Reconstruction`](@ref)
is done. Keep in mind that the intented behavior of the algorithm is to work with
a reconstruction, and not "raw" data.

## Keyword Arguments
  * `method = AverageLocalModel(2)` : Subtype of [`AbstractLocalModel`](@ref).
  * `ntype = FixedMassNeighborhood(2)` : Subtype of [`AbstractNeighborhood`](@ref).
  * `stepsize = 1` : Prediction step size.

## Description
Given a query point, the function finds its neighbors using neighborhood `ntype`.
Then, the neighbors `xnn` and their images `ynn` are used to make a prediction for
the future of the query point, using the provided `method`.
The images `ynn` are the points `xnn` shifted by `stepsize` into the future.

The algorithm is applied iteratively until a prediction of length `p` has
been created, starting with the query point to be the last point of the timeseries.

## References
[1] : Eds. B. Schelter *et al.*, *Handbook of Time Series Analysis*,
VCH-Wiley, pp 39-65 (2006)
"""
function localmodel_tsp(R::AbstractDataset{B}, p::Int;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
    stepsize::Int = 1) where B
    B > 1 || throw(ArgumentError("Dataset Dimension needs to be >1! ",
    "Alternatively pass embedding parameters."))
    return _localmodel_tsp(R, KDTree(R[1:end-stepsize]), R[end], p;
    method=method, ntype=ntype, stepsize=stepsize)
end

function localmodel_tsp(s::AbstractVector, D::Int, τ::T, p::Int; kwargs... ) where {T}
    localmodel_tsp(Reconstruction(s, D, τ), p; kwargs...)[:,D]
end

function localmodel_tsp(ss::AbstractDataset{B}, D::Int, τ::T, p::Int; kwargs...) where {B,T}
    sind = SVector{B, Int}((D*B - i for i in B-1:-1:0)...)
    localmodel_tsp(Reconstruction(ss, D, τ), p; kwargs...)[:,sind]
end



#####################################################################################
#                                  Error Measures                                   #
#####################################################################################

"""
    MSE1(R::AbstractDataset{D,T},R_test, method, ntype, stepsize) -> error

Compute mean squared error of single predictions using test set `R_test`.

## Description
This error measure takes in a prediction model consisting of
`R`, `method`, `ntype` and `stepsize` and evaluates its performance.
The test set `R_test` is a delay reconstruction with the same delay `τ` and
dimension `D` as `R`.
For every point in `R_test` (except for the last) the image `y` is predicted.
The model error is defined as
```math
\\begin{aligned}
MSE_1 = \\frac{1}{|T_{ref}|}\\sum_{t\\in T_{ref}} \\left(y_{t} - y_{pred,t} \\right)^2
\\end{aligned}
```
where ``|T_{ref}|`` is the total number of predictions made.

## References
See [`localmodel_tsp`](@ref).
"""
function MSE1(
    R::AbstractDataset{D,T},
    tree::KDTree,
    R_test::AbstractDataset{D,T};
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood  = FixedMassNeighborhood(2),
    stepsize::Int = 1) where {D,T}

    y_test = map(q-> q[end], R_test[2:end])
    y_pred = T[]; sizehint!(y_pred, length(y_test))
    for q in R_test[1:end-1] # Remove last state,
                            #because there is nothing to compare the preciction to
        idxs,dists = neighborhood_and_distances(q, R, tree,ntype)
        xnn = R[idxs]
        ynn = R[idxs+stepsize]
        push!(y_pred,method(q,xnn,ynn, dists)[end])
    end
    return norm(y_test-y_pred)^2/length(y_test)
end
#FIXME: I shouldn't have to square the norm... What is the solution?

MSE1(R, R_test; method::AbstractLocalModel = AverageLocalModel(2),
ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
stepsize::Int = 1) =
MSE1(R, KDTree(R[1:end-stepsize]), R_test;  method=method, ntype=ntype, stepsize=stepsize)


"""
    MSEp(R::AbstractDataset{D,T}, R_test, p; method, ntype, stepsize) -> error

Compute mean squared error of iterated predictions of length `p` using test set `R_test`.

## Description
This error measure takes in a prediction model consisting of `R`,
 `method`, `ntype` and `stepsize` and evaluates its performance. The test set `R_test` is
a delay reconstruction with the same delay `τ` and dimension `D` as `R`.
For each subset of `R_test` with length `p` it calls `localmodel_tsp`.
The model error is then defined as
```math
\\begin{aligned}
MSE_p = \\frac{1}{p|T_{ref}|}\\sum_{t\\in T_{ref}}\\sum_{i=1}^{p} \\left(y_{t+i} - y_{pred,t+i} \\right)^2
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

    Tref = (length(R_test)-p-1)
    error = 0
    for t =1:Tref
        R_pred = _localmodel_tsp(R,tree,R_test[t], p; kwargs...)
        error += norm(R_test[t:t+p]-R_pred.data)^2 /Tref/p
    end
    return error
end
#FIXME: I shouldn't have to square the norm... What is the solution?
MSEp(R, R_test, p; method::AbstractLocalModel = AverageLocalModel(2),
ntype::AbstractNeighborhood = FixedMassNeighborhood(2),
stepsize::Int = 1) =
MSEp(R, KDTree(R[1:end-stepsize]), R_test, p; method=method, ntype=ntype, stepsize=stepsize)
