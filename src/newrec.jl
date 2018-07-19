using Statistics
export STDelayEmbedding, PCAEmbedding, reconstruct

function spatial_extent(delays)
	Φ = length(delays[1].β)
	spex = zeros(Int,Φ,2)
	for φ = 1:Φ
		spex[φ,:] .= abs.(extrema(map(idx -> idx.β[φ], delays)))
	end
	return spex
end

struct STDelayEmbedding{X,Φ} <: AbstractEmbedding
  	delays::Vector{NamedTuple{(:τ,:β), Tuple{Int64,CartesianIndex{Φ}}}}
	#weighting #Implement this properly
 	boundary
	tmax
	spex
	function STDelayEmbedding{X,Φ}(delays,c) where {X,Φ}
		tmax = maximum(map(idx -> idx.τ,delays))
		spex = spatial_extent(delays)
		return new{X,Φ}(delays,c,tmax,spex)
	end
end
#TODO: Implement w!=0
function STDelayEmbedding(D,τ,B,k,c, Φ)
	X = (D+1)*(2B+1)^Φ #+ Φ*(any(w .!= 0))
	idxs = Vector{NamedTuple{(:τ,:β),Tuple{Int64,CartesianIndex{Φ}}}}(undef,X)
	n = 1
	for d=0:D, α = Iterators.product([-B*k:k:B*k for φ=1:Φ]...)
		idxs[n] = (τ=d*τ, β=CartesianIndex(α))
		n +=1
	end
	@assert X==length(idxs)
	return STDelayEmbedding{X,Φ}(idxs,c)
end

function dump(io::IO, r::STDelayEmbedding{X,Φ}) where {X,Φ}
	show(io,r)
	println("Spatial Extent: ", r.spex)
	println("Time Interval: ", r.tmax)
end
function Base.show(::IO, r::STDelayEmbedding{X,Φ}) where {X,Φ}
	println("$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
	for n=1:X println(r.delays[n]) end
end
function Base.summary(::IO, ::STDelayEmbedding{X,Φ}) where {X,Φ}
	println("$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
end

#Add another abstraction layer
#at the boundary
#This function is not safe. If you call it directly with bad params - can fail
@generated function (r::STDelayEmbedding{X,Φ})(
		s::AbstractArray{<:AbstractArray{T,Φ}},t,α,pt_in_space) where {T,Φ,X}
	gens = [:(  de = r.delays[$n];
			 	if α + de.β in pt_in_space
					s[t+de.τ][α + de.β]
				else
					r.boundary
				end)
			for n=1:X]
    quote
        @inbounds return SVector{$X,$T}($(gens...))
    end
end


using PrincipalComponentAnalysis

struct PCAEmbedding{T,Y,Φ} <: AbstractEmbedding
	stemb::STDelayEmbedding{X,Φ} where X#Regular Embedding w/o PCA
	meanv::T
	drmodel::PCA
end


function PCAEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		stemb::STDelayEmbedding{X,Φ};
		pratio   = 0.99,
		maxoutdim= 25,
		every::Int    = 1) where {T,X,Φ}
	meanv = Statistics.mean(Statistics.mean.(s))
	tsteps = (length(s) - stemb.tmax)
	fsize= size(s[1])
	num_pt    = prod(fsize)
	L         = tsteps*num_pt

	#exclude boundary points right here
	#ugly...this creates a CartesianIndices iterator that only iterates
	# over points far enough from the boundary to not encounter boundary
	#values
	middle_field = ([1+stemb.spex[i,1] : fsize[i]-stemb.spex[i,2] for i=1:Φ]...,)
	pt_in_space = CartesianIndices(middle_field)
	lin_idxs    = LinearIndices(fsize)

	recv = zeros(X)
	covmat = zeros(X,X)
	for t in 1:tsteps, α in pt_in_space
		( (t-1)*num_pt+lin_idxs[α] ) % every == 0 || (continue)

		recv .= stemb(s,t,α,pt_in_space)
		covmat .+= kron(recv', recv) ./ L
	end
	drmodel = PrincipalComponentAnalysis.pcacov(covmat, T[];
												maxoutdim=maxoutdim,
												pratio=pratio)
	Y = PrincipalComponentAnalysis.outdim(drmodel)
	return PCAEmbedding{T,Y,Φ}(stemb, meanv, drmodel)
end
function (r::PCAEmbedding{T,X,Φ})(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		t, α, pt_in_space) where {T,Φ,X}
		return transform(r.drmodel, r.stemb(s,t,α,pt_in_space))
end

function Base.show(io::IO, em::PCAEmbedding{T,X,Φ}) where {T,X,Φ}
	Base.show(io, em.stemb)
	Base.show(io, em.drmodel)
end


get_tmax(r::STDelayEmbedding) = r.tmax
get_tmax(r::PCAEmbedding) = r.stemb.tmax

#Need Outdi
function reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}},
	stemb::Union{STDelayEmbedding{X,Φ}, PCAEmbedding{T,X,Φ}}
	) where {T<:Number,Φ,X}
	timesteps = (length(s) - get_tmax(stemb))
	field_size= size(s[1])
	num_pt    = prod(field_size)
	L         = timesteps*num_pt


	pt_in_space = CartesianIndices(field_size)
	lin_idxs    = LinearIndices(field_size)
#	data = Matrix{T}(undef,X,L)
	data = Matrix{T}(undef,X,L)

	for t in 1:timesteps, α in pt_in_space
		n = (t-1)*num_pt+lin_idxs[α]
		data[:,n] .= stemb(s,t,α,pt_in_space)
	end
	return Dataset(data)
end
