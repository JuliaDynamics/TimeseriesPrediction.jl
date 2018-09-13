using Statistics
using LinearAlgebra

export PCA



# choose the first k values and columns
#
# S must have fields: values & vectors

function extract_kv(fac::Factorization{T}, ord::AbstractVector{Int}, k::Int
	) where {T}
    si = ord[1:k]
    vals = fac.values[si]::Vector{T}
    vecs = fac.vectors[:, si]::Matrix{T}
    return (vals, vecs)
end


#### PCA type

mutable struct PCA{T<:AbstractFloat}
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    proj::Matrix{T}       # projection matrix: of size d x p
    prinvars::Vector{T}   # principal variances: of length p
    tprinvar::T           # total principal variance, i.e. sum(prinvars)
    tvar::T               # total input variance
end

## constructor

function PCA(mean::Vector{T}, proj::Matrix{T}, pvars::Vector{T}, tvar::T
	) where {T<:AbstractFloat}
    d, p = size(proj)
    (isempty(mean) || length(mean) == d) ||
        throw(DimensionMismatch("Dimensions of mean and proj are inconsistent."))
    length(pvars) == p ||
        throw(DimensionMismatch("Dimensions of proj and pvars are inconsistent."))
    tpvar = sum(pvars)
    tpvar <= tvar || isapprox(tpvar,tvar) || throw(ArgumentError("principal variance cannot exceed total variance."))
    PCA(mean, proj, pvars, tpvar, tvar)
end

## properties

indim(M::PCA) = size(M.proj, 1)
outdim(M::PCA) = size(M.proj, 2)

principalvar(M::PCA, i::Int) = M.prinvars[i]
principalvars(M::PCA) = M.prinvars
principalratio(M::PCA) = M.tprinvar / M.tvar

## show

function Base.show(io::IO, M::PCA)
    pr = round(principalratio(M), digits=5)
    println(io, "PCA(indim = $(indim(M)), outdim = $(outdim(M)), principalratio = $pr)")
end

#### PCA Training

## auxiliary

const default_pca_pratio = 0.99

function check_pcaparams(d::Int, mean::Vector{T}, md::Int, pr::AbstractFloat
	) where {T<:AbstractFloat}
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Incorrect length of mean."))
    md >= 1 || error("maxoutdim must be a positive integer.")
    0.0 < pr <= 1.0 || throw(ArgumentError("pratio must be a positive real value with pratio ≤ 1.0."))
end


function choose_pcadim(v::AbstractVector{T}, ord::Vector{Int}, vsum::T, md::Int, pr::AbstractFloat) where {T<:AbstractFloat}
    md = min(length(v), md)
    k = 1
    a = v[ord[1]]
    thres = vsum * pr
    while k < md && a < thres
        a += v[ord[k += 1]]
    end
    return k
end


## core algorithms

function pcacov(C::AbstractMatrix{T}, mean::Vector{T};
                maxoutdim::Int=size(C,1),
                pratio::AbstractFloat=default_pca_pratio) where {T<:AbstractFloat}

    check_pcaparams(size(C,1), mean, maxoutdim, pratio)
    Eg = eigen(Symmetric(C))
    ev = Eg.values
    ord = sortperm(ev; rev=true)
    vsum = sum(ev)
    k = choose_pcadim(ev, ord, vsum, maxoutdim, pratio)
    v, P = extract_kv(Eg, ord, k)
    PCA(mean, P, v, vsum)
end
