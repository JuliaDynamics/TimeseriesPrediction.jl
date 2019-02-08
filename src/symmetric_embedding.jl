export SymmetricEmbedding
"""
	SymmetricEmbedding(ste::SpatioTemporalEmbedding, sym)

A `SymmetricEmbedding` is intended as a means of dimension reduction
for a [`SpatioTemporalEmbedding`](@ref) by exploiting symmetries in the system.
All points at a time step equivalent to each other according to the symmetries
passed in `sym` will be averaged to a single entry.
Parameter `sym` has to be passed as a vector of independent symmetries.

A few examples for clarification:
 * 2D space with mirror symmetry along first dimension : `sym = [ [1] ]` → maps points  to `(s[t][+n, m] + s[t][-n, m])/2`
 * 2D space with mirror symmetry along each dimension : `sym = [ [1], [2] ]` → averages over points with `(+- n, +- m)`
 * 2D space with point symmetry in both dimensions : `sym = [ [1,2] ]` → groups points with equal distance to the origin such as `(1,2), (-1,2), (2,1),...`
 * 3D space with point symmetry in dim 1 & 3 and mirror in 2: `sym = [ [1,3], [2]]` → groups points with equal distance to the origin along dimension 1 & 3 and +-2

To further explain the last example: The following points form one such group
`(+-2,1,0), (0,1,+-2), (-2,-1,0), (0,-1,-2)`

The resulting structure can be used for reconstructing datasets in
the same way as a [`SpatioTemporalEmbedding`](@ref).
"""
struct SymmetricEmbedding{Φ,BC,X} <: AbstractSpatialEmbedding{Φ,BC,X}
    #em::SpatioTemporalEmbedding
    τ::Vector{Int}
    β::Vector{Vector{CartesianIndex{Φ}}}
	inner::Region{Φ}
	whole::Region{Φ}
    boundary::BC
end

function SymmetricEmbedding(ste::SpatioTemporalEmbedding{Φ,BC}, sym) where {Φ,BC}
    #Symmetries of shape
    # ( (1,), (2,4), (3,) )

    # Mapping dimension for dimension
    # single dim symmetry : x₁ → |x₁|
    # two dim symmetry : (x₁, x₂) → (y₁, y₂) = (|x₁|, |x₂|)
    # (y₁, y₂) → y₁ > y₂ ? id : (y₂, y₁)

    # arbitrary dim symmetry : x = (x₁, ...) → sort(abs.(x), reverse=true)
    τ = Vector{Int}()
    β_groups = []
	#CartesianIndices are nice but difficult to manipulate
	β_old = map(x-> x.I ,ste.β)
    #Repeat for every timestep
    for t in unique(ste.τ)
		#Filter for β at delay t
        β = β_old[ ste.τ .== t ]
		#Map selected β to their symmetry reduced space
        #and combine into groups according to their mapped value
        new_points = groupbymap(x->fold_up(x,sym), β)
		#Append results to new lists
        append!(τ, fill(t, length(new_points)))
        append!(β_groups, new_points)
    end
	# convert β_groups back to cartesian indices
	β_new = map(β_groups) do b
		map(x-> CartesianIndex(x...), b)
	end
	X = length(τ)
    return SymmetricEmbedding{Φ,BC,X}(τ, β_new, ste.inner, ste.whole, ste.boundary)
end


function fold_up(β, sym)
    b = [β...]
    for s in sym
        b[s] .= sort(abs.(b[s]), rev=true)
    end
    return tuple(b...)
end

function groupbymap(f, v; by=norm)
    d = Dict()
    for e ∈ v
        w = f(e)
        haskey(d, w) ? push!(d[w], e) : d[w] = [e]
    end
    map(k-> d[k], sort!(collect(keys(d)); by=by) )
end

function Base.show(io::IO, em::SymmetricEmbedding{Φ,BC, X}) where {Φ,BC,X}
	print(io, "$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
    if BC == PeriodicBoundary
        println(io, " and PeriodicBoundary condition.")
    else
        println(io, " and ConstantBoundary condition with c = $(em.boundary.c).")
    end
    println(io, "The included neighboring points are (forward embedding):")
    for (τ,β) in zip(em.τ,em.β)
        println(io, "τ = $τ , β = ", getproperty.(β,:I))
    end
end


function (r::SymmetricEmbedding{Φ,ConstantBoundary{T},X})(rvec, s, t, α) where {T,Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			β = r.β[n]
			s_t = s[n + r.τ[n]]
			rvec[n] = zero(T)
			for m in eachindex(β)
				rvec[n] += s_t[ α + β[m] ]
			end
			rvec[n] /= length(β)
		end
	else
		@inbounds for n=1:X
			β = r.β[n]
			s_t = s[n + r.τ[n]]
			rvec[n] = zero(T)
			for m in eachindex(β)
				rvec[n] += α + β[m] in r.whole ? s_t[ α + β[m] ] : r.boundary.c
			end
			rvec[n] /= length(β)
		end
	end
	return nothing
end
