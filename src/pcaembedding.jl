using MultivariateStats: PCA, pcacov
import IterTools: takenth, product
export PCAEmbedding

"""
	PCAEmbedding(s, em::SpatioTemporalEmbedding; kwargs...) → embedding
A spatio temporal delay coordinates structure with
Principal Component Analysis as a means of dimension reduction,
`embedding` can be used as a functor:
```julia
embedding(rvec, s, t, α)
```
which operates inplace on `rvec` and reconstructs vector from spatial time series `s` at
timestep `t` and cartesian index `α`.

To instantiate this `embedding`, give the data to be reconstructed `s` as well as an
instance of [`SpatioTemporalEmbedding`](@ref) to `PCAEmbedding`.

## Keyword Arguments
* `pratio = 0.99` : Ratio of variances that needs to be preserved in low-dimensional
  PCA-reconstruction.
* `maxoutdim = 25`: Upper limit for output dimension. May break `pratio` criterion.
* `every_t = 1` : Speed up computation by only using every n-th point in time.
* `every_α = 1` : Speed up computation further by only using every n-th point in space
  (linear indexing).

To set the output dimension to a certain value `X`, pass `pratio=1, maxoutdim=X`.
"""
struct PCAEmbedding{T,Φ,BC,X,Y} <: AbstractSpatialEmbedding{Φ,BC,X}
	stem::SpatioTemporalEmbedding{Φ,BC,Y}
	meanv::T
	covmat::Matrix{T}
	drmodel::PCA{T}
	tmp::Vector{T} #length Y
end


function PCAEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		stem::SpatioTemporalEmbedding{Φ,BC,Y};
		pratio   = 0.99,
		maxoutdim= 25,
		every_t::Int    = 1,
		every_α::Int    = 1) where {T,Φ,BC,Y}
	meanv = mean(mean.(s))
	tsteps = (length(s) - get_τmax(stem))
	usable_idxs = get_usable_idxs(stem)
	num_pt = length(usable_idxs)
	subset = Iterators.product(takenth(usable_idxs, every_α), 1:every_t:tsteps)
	L      = length(subset)

	recv = zeros(T,Y)
	covmat = zeros(T,Y,Y)
	for (α, t) ∈ subset
		stem(recv, s,t,α)
		recv  .-= meanv
		#covmat .+= recv' .* recv / L
		# Compute only  upper triangular half
		@inbounds for j=1:Y, i=1:j
			covmat[i,j] += recv[i]*recv[j] / L
		end
	end
	#Copy upper half to bottom
	LowerTriangular(covmat) .= UpperTriangular(covmat)'
	drmodel = compute_pca(covmat,pratio, maxoutdim)
	X = size(drmodel.proj,2)
	tmp = zeros(Y)
	return PCAEmbedding{T,Φ,BC,X,Y}(stem, meanv, covmat, drmodel,tmp)
end

function (r::PCAEmbedding{T,Φ,BC,X,Y})(data, s, t, α) where {T,Φ,BC,X,Y}
	r.stem(r.tmp, s,t,α)
	r.tmp .-= r.meanv
	mul!(data,transpose(r.drmodel.proj),r.tmp)
end



Base.show(io::IO, em::PCAEmbedding) = (show(io, em.stem); show(io, em.drmodel))

compute_pca(covmat::Matrix{T}, pratio, maxoutdim) where T=
	pcacov(covmat, T[]; maxoutdim=maxoutdim, pratio=pratio)

get_τmax(em::PCAEmbedding) = get_τmax(em.stem)
get_num_pt(em::PCAEmbedding) = get_num_pt(em.stem)

outdim(em::PCAEmbedding{T,Φ,BC,X}) where {T,Φ,BC,X} = X
