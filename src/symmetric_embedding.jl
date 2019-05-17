using InteractiveUtils
export SymmetricEmbedding
export Reflection, Rotation, Symmetry

"""
	Symmetry
Supertype of all symmetries used in [`SymmetricEmbedding`](@ref).
All symmetries are initialized like `Symmetry(n1, n2, ...)` with
`ni` being the indices of the spatial dimensions that have the said symmetry.

E.g. `Reflection(1,2)` means that spatial dimensions 1 and 2 have a reflection
symmetry, while `Rotation(1,3)` means that spatial dimensions 1 and 3 have a
(joint) rotation symmetry.
"""
abstract type Symmetry end

"""
	Rotation <: Symmetry
Rotation symmetry (index sets at equal distance from the center point are
equivalent), which is a joint symmetry between all input dimensions.
"""
struct Rotation <: Symmetry
	d::Vector{Int}
	function Rotation(args::Vector{Int})
		@assert length(args) > 1 "Rotation symmetry needs at least 2 dimensions"
		new(args)
	end
end

"""
	Reflection <: Symmetry
Reflection symmetry: x and -x are equavalent (for the given dimension).
"""
struct Reflection <: Symmetry
	d::Vector{Int}
end

for sym in Symbol.(subtypes(Symmetry))
	@eval $(sym)(x::Int, args...) = $(sym)([x, args...])
end

# Overload `NTuple{N,T} where {N,T<:A}` for `show`
_smallstr(::Rotation) = "rot"
_smallstr(::Reflection) = "refl"
function Base.show(io::IO, sym::Symmetry)
	s = _smallstr(sym)*string(sym.d)
	print(io, s)
end

# internal translation to the nested vectors of integers
# that will remain until we have time to properly re-write
# (if we ever encounter a different symmetry we want to use)
function _nestedvec(syms::Tuple)
	# This part of the code ensures no duplicate symmetries
	allidx = vcat(s.d for s in syms)
	@assert unique(allidx) == allidx
	# Now convert to nestedvec
	v = Vector{Vector{Int}}()
	for s in syms
		push!(v, s.d)
	end
	return v
end

"""
	SymmetricEmbedding(ste::SpatioTemporalEmbedding, sym::Tuple)

A `SymmetricEmbedding` is intended as a means of dimension reduction
for a [`SpatioTemporalEmbedding`](@ref) by exploiting spatial symmetries in
the system, listed as a `Tuple` of `<:Symmetry` (see [`Symmetry`](@ref)) for
all possible symmetries.

All points at a time step equivalent to each other according to the symmetries
passed in `sym` will be **averaged** to a single entry! For example,
the symmetry `Reflection(2)` means that the embedding won't have two entries
`u[i, j+1], u[i, j-1]` but instead a single entry
`(u[i, j+1] + u[i, j-1])/2`, with `i,j` being indices *relative to the
central point of the embedding*. (the same process is
done for any index offset `j+2, j+3`, etc., depending on how large the
spatial radius `r` is)

The resulting structure from `SymmetricEmbedding`
can be used for reconstructing datasets in
the same way as a [`SpatioTemporalEmbedding`](@ref).
"""
struct SymmetricEmbedding{Φ,BC,X} <: AbstractSpatialEmbedding{Φ,BC,X}
    τ::Vector{Int}
    β_groups::Vector{Vector{CartesianIndex{Φ}}}
	inner::Region{Φ}
	whole::Region{Φ}
    boundary::BC
end

function SymmetricEmbedding(ste::SpatioTemporalEmbedding, sym::Tuple)
	nv = _nestedvec(sym)
	_SymmetricEmbedding(ste, nv)
end

# Internal function, uses the nested vectors until we change it /
# clean it up
function _SymmetricEmbedding(ste::SpatioTemporalEmbedding{Φ,BC}, sym) where {Φ,BC}
	check_symmetry(sym, Φ)
    # Mapping dimension for dimension
    # single dim symmetry : x₁ → |x₁|
    # two dim symmetry : (x₁, x₂) → (y₁, y₂) = (|x₁|, |x₂|)
    # (y₁, y₂) → y₁ > y₂ ? id : (y₂, y₁)

    # arbitrary dim symmetry : x = (x₁, ...) → sort(abs.(x), reverse=true)
    τ = Vector{Int}()
    β_groups = []
	#CartesianIndices are nice but difficult to manipulate
	β = map(x-> x.I ,ste.β)
    #Repeat for every timestep
    for t in unique(ste.τ)
		#Filter for β at delay t
        β_t = β[ ste.τ .== t ]
		#Map selected β to their symmetry reduced space
        #and combine into groups according to their mapped value
        groups_t = groupbymap( x->fold_up(x, sym) , β_t)
		#Append results to new lists
        append!(τ, fill(t, length(groups_t)))
        append!(β_groups, groups_t)
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

function check_symmetry(sym, Φ)
	s = vcat(sym...)
	unique(s) == s || throw(ArgumentError("Dimension may not be passed more than once."))
	maximum(s) > Φ && throw(ArgumentError("Invalid dimension was passed."))
	minimum(s) < 1 && throw(ArgumentError("Invalid dimension was passed."))
end

function Base.show(io::IO, em::SymmetricEmbedding{Φ,BC, X}) where {Φ,BC,X}
	print(io, "$(Φ)D Spatio-Temporal Delay Embedding with $X Entries")
    if BC == PeriodicBoundary
        println(io, " and PeriodicBoundary condition.")
    else
        println(io, " and ConstantBoundary condition with c = $(em.boundary.c).")
    end
    println(io, "The included neighboring points are (forward embedding):")
    for (τ,β) in zip(em.τ,em.β_groups)
        println(io, "τ = $τ , β = ", getproperty.(β,:I))
    end
end


function (r::SymmetricEmbedding{Φ,ConstantBoundary{T},X})(rvec, s, t, α) where {T,Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			β = r.β_groups[n]
			s_t = s[t + r.τ[n]]
			rvec[n] = zero(T)
			for m in eachindex(β)
				rvec[n] += s_t[ α + β[m] ]
			end
			rvec[n] /= length(β)
		end
	else
		@inbounds for n=1:X
			β = r.β_groups[n]
			s_t = s[t + r.τ[n]]
			rvec[n] = zero(T)
			for m in eachindex(β)
				rvec[n] += α + β[m] in r.whole ? s_t[ α + β[m] ] : r.boundary.b
			end
			rvec[n] /= length(β)
		end
	end
	return nothing
end

function (r::SymmetricEmbedding{Φ,PeriodicBoundary,X})(rvec, s, t, α) where {Φ,X}
	if α in r.inner
		@inbounds for n=1:X
			β = r.β_groups[n]
			s_t = s[t + r.τ[n]]
			rvec[n] = zero(eltype(rvec))
			for m in eachindex(β)
				rvec[n] += s_t[ α + β[m] ]
			end
			rvec[n] /= length(β)
		end
	else
		@inbounds for n=1:X
			β = r.β_groups[n]
			s_t = s[t + r.τ[n]]
			rvec[n] = zero(eltype(rvec))
			for m in eachindex(β)
				rvec[n] = s_t[ project_inside(α + β[m], r.whole) ]
			end
			rvec[n] /= length(β)
		end
	end
	return nothing
end
