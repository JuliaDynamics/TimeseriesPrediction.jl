using Statistics
export STDelayEmbedding, PCAEmbedding, reconstruct
using LinearAlgebra

struct Region{Φ}
	mini::NTuple{Φ,Int64}
	maxi::NTuple{Φ,Int64}
end

function Base.in(idx, r::Region{Φ}) where Φ
	#all(re.mini .<= α.I .<= re.maxi)
	for φ=1:Φ
		r.mini[φ] <= idx[φ] <= r.maxi[φ] || return false
 	end
 	return true
end

Base.length(r::Region{Φ}) where Φ = prod(r.maxi .- r.mini .+1)

Base.CartesianIndices(r::Region{Φ}) where Φ =
	 CartesianIndices{Φ,NTuple{Φ,UnitRange{Int64}}}(
	 ([r.mini[φ]:r.maxi[φ] for φ=1:Φ]...,))

function inner_field(βs::Vector{CartesianIndex{Φ}}, fsize) where Φ
	mini = Int[]
	maxi = Int[]
	for φ = 1:Φ
		js = map(β -> β[φ], βs) # jth entries
		mi,ma = extrema(js)
		push!(mini, 1 - min(mi, 0))
		push!(maxi,fsize[φ] - max(ma, 0))
	end
	return Region{Φ}((mini...,), (maxi...,))
end

struct STDelayEmbedding{T,Φ,X} <: AbstractEmbedding
  	τ::Vector{Int64}
	β::Vector{CartesianIndex{Φ}}
 	boundary::Float64
	inner::Region{Φ}  #inner field far from boundary
	outer::Region{Φ}	#whole field
	function STDelayEmbedding{T,Φ,X}(τ,β,boundary,fsize) where {T,Φ,X}
		inner = inner_field(β, fsize)
		outer = Region((ones(Int,Φ)...,), fsize)
		return new{T,Φ,X}(τ,β,boundary,inner,outer)
	end
end

function STDelayEmbedding(
	s::AbstractArray{<:AbstractArray{T,Φ}},
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

#This function is not safe. If you call it directly with bad params - can fail
function (r::STDelayEmbedding{T,Φ,X})(rvec,
		s::AbstractArray{<:AbstractArray{T,Φ}},t,α) where {T,Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			rvec[n] = s[ t + r.τ[n] ][ α + r.β[n] ]
		end
	else
		@inbounds for n=1:X
			rvec[n] = 	if α + r.β[n] in r.outer
							s[ t + r.τ[n] ][ α + r.β[n] ]
						else
							r.boundary
						end
		end
	end
	return nothing
end



using PrincipalComponentAnalysis

struct PCAEmbedding{T,Φ,X,Y} <: AbstractEmbedding
	stem::STDelayEmbedding{T,Φ,Y}
	meanv::T
	covmat::Matrix{T}
	drmodel::PCA{T}
	tmp::Vector{T} #length Y
end

compute_pca(covmat::Matrix{T}, pratio, maxoutdim) where T=
	PrincipalComponentAnalysis.pcacov(covmat, T[];
										maxoutdim=maxoutdim,
										pratio=pratio)
function recompute_pca(em::PCAEmbedding{T,Φ,X,Y}, pratio, maxoutdim) where {T,Φ,X,Y}
	drmodel = compute_pca(em.covmat, pratio, maxoutdim)
	Ynew = outdim(drmodel)
	PCAEmbedding{T,Φ,X,Ynew}(em.stemb, em.meanv, em.covmat, drmodel)
end


function PCAEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		stem::STDelayEmbedding{T,Φ,Y};
		pratio   = 0.99,
		maxoutdim= 25,
		every::Int    = 1) where {T,Φ,Y}
	meanv = Statistics.mean(Statistics.mean.(s))
	tsteps = (length(s) - maximum(stem.τ))
	num_pt = length(stem.inner)
	L      = tsteps*num_pt

	recv = zeros(T,Y)
	covmat = zeros(T,Y,Y)
	inner_idxs = CartesianIndices(stem.inner)
	for n = 1:every:L
		t = 1 + (n-1) ÷ num_pt
		α = inner_idxs[n-(t-1)*num_pt]
		stem(recv, s,t,α)
		recv  .-= meanv
		#covmat .+= recv' .* recv / L
		@inbounds for j=1:Y, i=1:Y
			covmat[i,j] += recv[i]*recv[j] / L
		end
	end
	drmodel = compute_pca(covmat,pratio, maxoutdim)
	X = PrincipalComponentAnalysis.outdim(drmodel)
	tmp = zeros(Y)
	return PCAEmbedding{T,Φ,X,Y}(stem, meanv, covmat, drmodel,tmp)
end

function (r::PCAEmbedding{T,X,Φ})(data, s, t, α) where {T,Φ,X}
	r.stem(r.tmp, s,t,α)
	r.tmp .-= r.meanv
	mul!(data,transpose(r.drmodel.proj),r.tmp)
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
	recv = zeros(T,X)
	@inbounds for t in 1:timesteps, α in pt_in_space
		n = (t-1)*num_pt+lin_idxs[α]
		#Maybe unsafe array views here
		#recv = view(data,:,n)
		stem(recv,s,t,α)
		#very odd. data[:,n] .= recv allocates
		for i=1:X data[i,n] = recv[i] end
	end
	return data
end
