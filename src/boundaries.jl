"""
    AbstractSpatialEmbedding <: AbstractEmbedding
Super-type of spatiotemporal embedding methods. Valid subtypes:
* `SpatioTemporalEmbedding`
* `PCAEmbedding`
"""
abstract type AbstractSpatialEmbedding{Φ,BC,X} <: AbstractEmbedding end
const ASE = AbstractSpatialEmbedding

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


"""
    AbstractBoundaryCondition
Super-type of boundary conditions for [`SpatioTemporalEmbedding`](@ref).
Use `subtypes(AbstractBoundaryCondition)` for available methods.
"""
abstract type AbstractBoundaryCondition end

"""
	ConstantBoundary(b) <: AbstractBoundaryCondition
Constant boundary condition type. Enforces constant boundary conditions
when passed to [`SpatioTemporalEmbedding`](@ref)
by filling missing out-of-bounds values in the reconstruction with
parameter `b`.
"""
struct ConstantBoundary{T} <: AbstractBoundaryCondition
    b::T
end

"""
	PeriodicBoundary <: AbstractBoundaryCondition
Periodic boundary condition struct. Enforces periodic boundary conditions
when passed to [`SpatioTemporalEmbedding`](@ref) in the reconstruction.
"""
struct PeriodicBoundary <: AbstractBoundaryCondition end



struct BoundaryWrapper{BC <: AbstractBoundaryCondition, T, Φ, V}
    boundary::BC
    region::Region{Φ}
    data::V
end
function BoundaryWrapper(r::AbstractSpatialEmbedding{Φ,BC}, data) where {Φ,BC}
	T = eltype(data[1][1])
	BoundaryWrapper{BC,T,Φ,typeof(data)}(r.boundary, r.whole, data)
end
Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{<:ConstantBoundary}, t::Int ,α::CartesianIndex)
    if α in A.region
        A.data[t][α]
    else
        A.boundary.b
    end
end

Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{<:ConstantBoundary} ,α::CartesianIndex)
    α in A.region ? A.data[α] : A.boundary.b
end

function project_inside(α::CartesianIndex{Φ}, r::Region{Φ}) where Φ
	CartesianIndex(mod.(α.I .-1, r.maxi).+1)
end


Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{PeriodicBoundary}, t::Int, α::CartesianIndex)
    if α in A.region
        A.data[t][α]
    else
        A.data[t][project_inside(α, A.region)]
    end
end

Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{PeriodicBoundary}, α::CartesianIndex)
    α in A.region ? A.data[α] :  A.data[project_inside(α, A.region)]
end

#Base.@propagate_inbounds function Base.getindex(
#			A::BoundaryWrapper{PeriodicBoundary}, I::AbstractRange, J::AbstractRange)
#    [
#		(α = CartesianIndex(i,j);
#		α in A.region ? A.data[α] :  A.data[project_inside(α, A.region)])
#		for i in I, j in J
#	]
#end


Base.eltype(bw::BoundaryWrapper{BC, T}) where {BC, T} = T
