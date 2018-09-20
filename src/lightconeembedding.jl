using LinearAlgebra


function indices_within(radius, dimension)
    #Return indices β within hypersphere with radius
    #Floor radius to nearest integer. Does not lose points
    r = floor(Int,radius)
    #Hypercube of indices
    hypercube = CartesianIndices((repeat([-r:r], dimension)...,))
    #Select subset of hc which is in Hypersphere

    βs = [ β for β ∈ hypercube if norm(β.I) <= radius ]
end

"""
    LightConeEmbedding(s, timesteps, stepsize, speed, boundary) → SpatioTemporalEmbedding
Create a [`SpatioTemporalEmbedding`](@ref) struct that
includes spatial and temporal neighbors of a point based on the notion of
a _sphere of influence_.
As an example, in a one-dimensional system with `timesteps=2`, `stepsize=2`, and `speed=1`
the resulting embedding might look like:

    __________o__________
    _________xxx_________
    _____________________
    _______xxxxxxx_______
where `o` is the point that is to be predicted.
An optional keyword argument `offset` allows moving the origin of the
cone along the time axis.
Above example with `offset = 1` (left) and `offset = -1` (right) becomes

    __________o__________    __________o__________
    ________xxxxx________    __________x__________
    _____________________    _____________________
    ______xxxxxxxxx______    ________xxxxx________
"""
function LightConeEmbedding(
    s::AbstractArray{<:AbstractArray{T,Φ}},
    timesteps,
    stepsize,
    speed,
    boundary::BC;
    offset = 0
    ) where {T,Φ, BC<:AbstractBoundaryCondition}
    τs = Int[]
    βs = CartesianIndex{Φ}[]
    maxτ = stepsize*timesteps
    for τ = stepsize*(timesteps:-1:1) # Backwards for forward embedding
        radius = speed*τ + offset
        β = indices_within(radius, Φ)
        push!(βs, β...)
        push!(τs, repeat([maxτ - τ], length(β))...)
    end
    X = length(τs)
    return SpatioTemporalEmbedding{X}(τs, βs, boundary, size(s[1]))
end
