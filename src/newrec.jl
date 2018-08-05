using Statistics
export STDelayEmbedding, PCAEmbedding, reconstruct

#Rewrite this properly
function inner_field(βs::Vector{CartesianIndex{Φ}}, fsize) where Φ
	ranges = Vector{UnitRange{Int64}}(undef,Φ)
	for φ = 1:Φ
		js = map(β -> β[φ], βs) # jth entries
		mi,ma = extrema(js)
		mi = 1 - min(mi, 0)
		ma = fsize[φ] - max(ma, 0)
		ranges[φ] = mi:ma
	end
	return CartesianIndices((ranges...,))
end

#Very odd behavior here. em.β[10] allocates if <:AbstractEmbedding
struct STDelayEmbedding{T,Φ,X,R1,R2} <: AbstractEmbedding
  	τ::Vector{Int64}
	β::Vector{CartesianIndex{Φ}}
 	boundary::Float64
	inner::CartesianIndices{Φ,R1}  #inner field far from boundary
	outer::CartesianIndices{Φ,R2}	#whole field
	function STDelayEmbedding{T,Φ,X}(τ,β,boundary,fsize) where {T,Φ,X}
		inner = inner_field(β, fsize)
		outer = CartesianIndices(fsize)
		R1 = typeof(inner.indices)
		R2 = typeof(outer.indices)
		return new{T,Φ,X,R1,R2}(τ,β,boundary,inner,outer)
	end
end

function STDelayEmbedding(s::AbstractArray{<:AbstractArray{T,Φ}},
	D,τ,B,k,c) where {T,Φ}
	X = (D+1)*(2B+1)^Φ
	τs = Vector{Int64}(undef,X)
	βs = Vector{CartesianIndex{Φ}}(undef,X)
	n = 1
	for d=0:D, α = Iterators.product([-B*k:k:B*k for φ=1:Φ]...)
		τs[n] = d*τ
		βs[n] = CartesianIndex(α)
		n +=1
	end
	return STDelayEmbedding{T,Φ,X}(τs, βs, c, size(s[1]))
end


function Base.summary(::IO, ::STDelayEmbedding{T,Φ,X}) where {T,Φ,X}
	println("$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
end
function Base.in(α::CartesianIndex{Φ}, fsize) where {T,Φ}
	for φ=1:Φ
		0 < α[φ] <= fsize[φ] || return false
	end
	return true
end

function impl_rec(::Val{T}, ::Val{X}) where {T,X}
	gens = [:( 	if α + r.β[$n] in r.outer
					s[ t + r.τ[$n] ][ α + r.β[$n] ]
				else
					r.boundary
				end )  for n=1:X]
	gens_inner = [:( s[t+r.τ[$n]][α+r.β[$n]] )  for n=1:X]

	quote
		#inline_meta
		#Base.@_inline_meta
		if α in r.inner
			@inbounds return SVector{$X,$T}($(gens_inner...))
		else
			@inbounds return SVector{$X,$T}($(gens...))
		end
	end
end
#This function is not safe. If you call it directly with bad params - can fail
@generated function (r::STDelayEmbedding{T,Φ,X})(
		s::AbstractArray{<:AbstractArray{T,Φ}},t,α) where {T,Φ,X}
	impl_rec(Val(T),Val(X))
end
# #while this thing does not allocate
# @generated function rec_vec(s::AbstractArray{<:AbstractArray{T,Φ}},t,α, r::STDelayEmbedding{T,Φ,X}) where {T,Φ,X}
# 	impl_rec(Val(T),Val(X))
# end



using PrincipalComponentAnalysis

struct PCAEmbedding{T,Φ,X,Y,R1,R2} <: AbstractEmbedding
	stem::STDelayEmbedding{T,Φ,Y,R1,R2}
	meanv::T
	covmat::Matrix{T}
	drmodel::PCA{T}
end

compute_pca(covmat::Matrix{T}, pratio, maxoutdim) where T=
	PrincipalComponentAnalysis.pcacov(covmat, T[];
										maxoutdim=maxoutdim,
										pratio=pratio)
function recompute_pca(em::PCAEmbedding{T,Φ,X,Y,R1,R2}, pratio, maxoutdim) where {T,Φ,X,Y,R1,R2}
	drmodel = compute_pca(em.covmat, pratio, maxoutdim)
	Ynew = outdim(drmodel)
	PCAEmbedding{T,Φ,X,Ynew,R1,R2}(em.stemb, em.meanv, em.covmat, drmodel)
end


function PCAEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		stem::STDelayEmbedding{T,Φ,Y,R1,R2};
		pratio   = 0.99,
		maxoutdim= 25,
		every::Int    = 1) where {T,Φ,Y,R1,R2}
	meanv = Statistics.mean(Statistics.mean.(s))
	tsteps = (length(s) - maximum(stem.τ))
	num_pt = length(stem.inner)
	L      = tsteps*num_pt

	recv = zeros(T,Y)
	covmat = zeros(T,Y,Y)
	for n = 1:every:L
		t = 1 + (n-1) ÷ num_pt
		α = stem.inner[n-(t-1)*num_pt]
		recv .= stem(s,t,α) .- meanv
		#covmat .+= recv' .* recv / L
		@inbounds for j=1:Y, i=1:Y
			covmat[i,j] += recv[i]*recv[j] / L
		end
	end
	drmodel = compute_pca(covmat,pratio, maxoutdim)
	#X = size(drmodel.proj)[2]
	X = PrincipalComponentAnalysis.outdim(drmodel)
	return PCAEmbedding{T,Φ,X,Y,R1,R2}(stem, meanv, covmat, drmodel)
end

#This allocates the resulting vector. Also this is Mat*SVector
#which is slower than Mat*Vector even if you add preallocated tmp .= SVector
#This should be alleviated via preallocated temporary vectors
function (r::PCAEmbedding{T,X,Φ})(s, t, α) where {T,Φ,X}
	transpose(r.drmodel.proj) * (r.stem(s,t,α) - r.meanv)
end

function Base.show(io::IO, em::PCAEmbedding{T,X,Φ}) where {T,X,Φ}
	Base.show(io, em.stem)
	Base.show(io, em.drmodel)
end


get_tmax(em::STDelayEmbedding) = maximum(em.τ)
get_tmax(em::PCAEmbedding) = get_tmax(em.stem)


function reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}},
	stem::Union{STDelayEmbedding{T,Φ,X}, PCAEmbedding{T,Φ,X,Y}}
	) where {T<:Number,Φ,X,Y}
	timesteps = (length(s) - get_tmax(stem))
	num_pt    = prod(size(s[1]))
	L         = timesteps*num_pt


	pt_in_space = CartesianIndices(s[1])
	lin_idxs    = LinearIndices(s[1])
	data = Matrix{T}(undef,X,L)

	@inbounds for t in 1:timesteps, α in pt_in_space
		n = (t-1)*num_pt+lin_idxs[α]
		data[:,n] .= stem(s,t,α)
	end
	return data
end
