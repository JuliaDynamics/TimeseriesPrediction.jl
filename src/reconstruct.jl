import DelayEmbeddings: reconstruct

#####################################################################################
#                                   Reconstruction                                  #
#####################################################################################
"""
	reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}}, em)
Reconstruct the spatial timeseries `s` represented by a `Vector` of `AbstractArray`
states using the embedding struct `em` of type [`AbstractSpatialEmbedding`](@ref).

Returns the reconstruction in the form of a `Dataset` (from `DelayEmbeddings`) where each
row is a reconstructed state and they are ordered first through linear
indexing into each state and then incrementing in time.
"""
function reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}},
	em::AbstractSpatialEmbedding{Φ,BC,X}
	) where {T<:Number,Φ,BC,X}
	timesteps = (length(s) - get_τmax(em))
	num_pt    = get_num_pt(em)
	L         = timesteps*num_pt

	pt_in_space = CartesianIndices(s[1])
	lin_idxs    = LinearIndices(s[1])
	data = Vector{SVector{X,T}}(undef,L)
	recv = zeros(T,X)
	@inbounds for t in 1:timesteps, α in pt_in_space
		n = (t-1)*num_pt+lin_idxs[α]
		em(recv,s,t,α)
		data[n] = recv
	end
	return Dataset(data)
end
