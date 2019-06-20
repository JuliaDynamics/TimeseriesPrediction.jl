

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



struct BoundaryWrapper{BC <: AbstractBoundaryCondition, R ,V}
    boundary::BC
    region::R
    data::V
end
BoundaryWrapper(r, data) = BoundaryWrapper(r.boundary, r.whole, data)

Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{ConstantBoundary}, t::Int ,α::CartesianIndex)
    if α in A.region
        A.data[t][α]
    else
        boundary.b
    end
end

Base.@propagate_inbounds function Base.getindex(
			A::BoundaryWrapper{ConstantBoundary} ,α::CartesianIndex)
    α in A.region ? A.data[α] : boundary.b

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
    α in A.region ? A.data[t][α] :  A.data[project_inside(α, A.region)]
