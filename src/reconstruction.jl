using Statistics
using LinearAlgebra
export AbstractSpatialEmbedding
export SpatioTemporalEmbedding, STE
export outdim
export AbstractBoundaryCondition, PeriodicBoundary, ConstantBoundary

import DynamicalSystemsBase: reconstruct

#####################################################################################
#                 Spatio Temporal Delay Embedding Reconstruction                    #
#####################################################################################
"""
    AbstractSpatialEmbedding <: AbstractEmbedding
Super-type of spatiotemporal embedding methods.
Use `subtypes(AbstractSpatialEmbedding)` for available methods.
"""
abstract type AbstractSpatialEmbedding{T,Φ,BC,X} <: AbstractEmbedding end
const ASE = AbstractSpatialEmbedding
"""
    AbstractBoundaryCondition
Super-type of boundary conditions for [`SpatioTemporalEmbedding`](@ref).
Use `subtypes(AbstractBoundaryCondition)` for available methods.
"""
abstract type AbstractBoundaryCondition end

"""
	ConstantBoundary(c)
Constant boundary condition type. Enforces constant boundary conditions
when passed to [`SpatioTemporalEmbedding`](@ref)
by filling missing out-of-bounds values in the reconstruction with
parameter `c`.
"""
struct ConstantBoundary{T} <: AbstractBoundaryCondition
    c::T
end

"""
	PeriodicBoundary
Periodic boundary condition struct. Enforces periodic boundary conditions
when passed to [`SpatioTemporalEmbedding`](@ref) in the reconstruction.
"""
struct PeriodicBoundary    <: AbstractBoundaryCondition end


"""
	Region{Φ}
Internal struct for efficiently keeping track of region far from boundaries of field.
Used to speed up reconstruction process.
"""
struct Region{Φ}
	mini::NTuple{Φ,Int}
	maxi::NTuple{Φ,Int}
end

Base.length(r::Region{Φ}) where Φ = prod(r.maxi .- r.mini .+1)
function Base.in(idx, r::Region{Φ}) where Φ
	for φ=1:Φ
		r.mini[φ] <= idx[φ] <= r.maxi[φ] || return false
 	end
 	return true
end
Base.CartesianIndices(r::Region{Φ}) where Φ =
	 CartesianIndices{Φ,NTuple{Φ,UnitRange{Int}}}(([r.mini[φ]:r.maxi[φ] for φ=1:Φ]...,))


function inner_region(βs::Vector{CartesianIndex{Φ}}, fsize) where Φ
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

function project_inside(α::CartesianIndex{Φ}, r::Region{Φ}) where Φ
	CartesianIndex(mod.(α.I .-1, r.maxi).+1)
end

"""
	 SpatioTemporalEmbedding{T,Φ,BC,X} <: AbstractSpatialEmbedding{T,Φ,BC,X} → `embedding`
A spatio temporal delay coordinates structure to be used as a functor.

	embedding(rvec, s, t, α)
Operates inplace on `rvec` and reconstructs vector from spatial timeseries `s` at
timestep `t` and cartesian index `α`. Note that there are no bounds checks for `t`.

## Constructors
The structure can be created directly by calling

	SpatioTemporalEmbedding{T,Φ,X}(τ,β,bc,fsize)
where `T` is the `eltype` of the timeseries, `Φ` the spatial dimension of the system,
`bc` the boundary condition type and `X` the length of reconstructed vectors.
Arguments `τ` and `β` are Vectors of `Int` and `CartesianIndex` that contain
all points to be included in the reconstruction in *relative* coordinates
and `fsize` is the size of each state in the timeseries.

Note that this is a forward embedding and `τ` needs to be sorted in ascending order.
The most recent values in the reconstruction are located at the end of each vector.

A simpler constructor for convenience

	SpatioTemporalEmbedding(s, D, τ, B, k, ::Type{<:AbstractBoundaryCondition})
Takes as arguments the spatial timeseries `s` and reconstructs
`B` spatial shells separated by `k` points around each point
and repeats this for `D` past timesteps separated by `τ` each.
"""
struct SpatioTemporalEmbedding{T,Φ,BC,X} <: AbstractSpatialEmbedding{T,Φ,BC,X}
  	τ::Vector{Int}
	β::Vector{CartesianIndex{Φ}}
	inner::Region{Φ}  #inner field far from boundary
	whole::Region{Φ}
    boundary::BC

	function SpatioTemporalEmbedding{T,Φ,X}(τ,β,bc::BC,fsize) where {T,Φ,BC,X}
		inner = inner_region(β, fsize)
		whole = Region((ones(Int,Φ)...,), fsize)
		return new{T,Φ,BC,X}(τ,β,inner,whole, bc)
	end
end
const STE = SpatioTemporalEmbedding

function SpatioTemporalEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		D, τ, B, k, boundary::BC
		) where {T,Φ, BC<:AbstractBoundaryCondition}
	@assert issorted(τ) "Delays need to be sorted in ascending order"
    #"ConstantBoundary condition value C needs to be the same type as values in s"
    @assert BC <: PeriodicBoundary || typeof(boundary.c) == T "typeof(boundary.c) == eltype(s[1])"
	X = (D+1)*(2B+1)^Φ
	τs = Vector{Int}(undef,X)
	βs = Vector{CartesianIndex{Φ}}(undef,X)
	n = 1
	for d=0:D, α = Iterators.product([-B*k:k:B*k for φ=1:Φ]...)
		τs[n] = d*τ
		βs[n] = CartesianIndex(α)
		n +=1
	end
	return SpatioTemporalEmbedding{T,Φ,X}(τs, βs, boundary, size(s[1]))
end



#This function is not safe. If you call it directly with bad params - can fail
function (r::SpatioTemporalEmbedding{T,Φ,ConstantBoundary{T},X})(rvec,s,t,α) where {T,Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			rvec[n] = s[ t + r.τ[n] ][ α + r.β[n] ]
		end
	else
		@inbounds for n=1:X
			rvec[n] = α + r.β[n] in r.whole ? s[ t+r.τ[n] ][ α+r.β[n] ] : r.boundary.c
		end
	end
	return nothing
end

function (r::SpatioTemporalEmbedding{T,Φ,PeriodicBoundary,X})(rvec,s,t,α) where {T,Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			rvec[n] = s[ t + r.τ[n] ][ α + r.β[n] ]
		end
	else
		@inbounds for n=1:X
			rvec[n] = s[ t + r.τ[n] ][ project_inside(α + r.β[n], r.whole) ]
		end
	end
	return nothing
end



get_num_pt(em::SpatioTemporalEmbedding) = prod(em.whole.maxi)
get_τmax(em::SpatioTemporalEmbedding{T,Φ,BC,X}) where {T,Φ,BC,X} = em.τ[X]
outdim(em::SpatioTemporalEmbedding{T,Φ,BC,X}) where {T,Φ,BC,X} = X




function Base.show(io::IO, em::SpatioTemporalEmbedding{T,Φ,BC, X}) where {T,Φ,BC,X}
	print(io, "$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
    if BC == PeriodicBoundary
        println(io, " and PeriodicBoundary condition.")
    else
        println(io, " and ConstantBoundary condition with c = $(em.boundary.c).")
    end
    println(io, "The included neighboring points are (forward embedding):")
    for (τ,β) in zip(em.τ,em.β)
        println(io, "τ = $τ , β = $(β.I)")
    end
end


"""
	reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}}, em)
Reconstruct the spatial timeseries `s` represented by a `Vector` of `AbstractArray`
states using the embedding struct `em` of type [`AbstractSpatialEmbedding`](@ref).

Returns the reconstruction in the form of a [`Dataset`](@ref) where each row is a
reconstructed state and they are ordered first through linear indexing into each state
and then incrementing in time.
"""
function reconstruct(s::AbstractArray{<:AbstractArray{T,Φ}},
	em::AbstractSpatialEmbedding{T,Φ,BC,X}
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
