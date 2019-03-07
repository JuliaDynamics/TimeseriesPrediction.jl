export InhomogeneousEmbedding
"""
    InhomogeneousEmbedding(em::AbstractSpatialEmbedding, f)

An `InhomogeneousEmbedding` can be used as a wrapper around an existing
[`AbstractSpatialEmbedding`](@ref) to account for potential inhomogeneities
in the system.

It does so by reconstructing the local state as defined in `em`
and appending the result of `f(α)` where `α` is the spatial `CartesianIndex`
to the reconstructed vector.

The inhomogeneity function `f` can be any function that takes as
argument a `CartesianIndex{Φ}` and returns a `Tuple` or `Vector`
of numbers to be added to the end of the local state vector.

A convenience constructor for such a function is implemented in
[`linear_weights`](@ref).
"""
struct InhomogeneousEmbedding{Φ, BC, X, Y, ASE <: AbstractSpatialEmbedding{Φ, BC, Y}, F} <: AbstractSpatialEmbedding{Φ,BC,X}
    em::ASE
    f::F
end

function InhomogeneousEmbedding(em::AbstractSpatialEmbedding{Φ,BC,Y}, f) where {Φ,BC,Y}
    #Determine output dimension of f
    l = length(f(CartesianIndex(zeros(Int,Φ)...)))
    X = Y + l
    return InhomogeneousEmbedding{Φ,BC,X,Y, typeof(em), typeof(f)}(em,f)
end

function (r::InhomogeneousEmbedding{Φ,BC,X,Y})(rvec, s, t ,α) where {Φ,BC,X,Y}
    #give the internal embedding a view of rvec
    r.em(view(rvec,1:Y), s, t, α)
    rvec[Y+1:X] .= r.f(α)
    return nothing
end

get_num_pt(em::InhomogeneousEmbedding) = get_num_pt(em.em)
get_τmax(em::InhomogeneousEmbedding) = get_τmax(em.em)
get_usable_idxs(em::InhomogeneousEmbedding) = get_usable_idxs(em.em)

"""
    linear_weights(fsize::NTuple{Φ,Int}, φs = 1:Φ, scalefactor = fill(one(T),length(φs))) where {Φ}

Convenience constructor for a linear inhomogeneity function.
Returns a function `f(α::CartesianIndex{Φ})` that computes a spatial weighting
for `α`.

 `fsize` the size of the domain that
is needed for normalizing the values to +-1.
`φs` and `scalefactor` are optional arguments that allow to
restrict the weighting to some subset of the spatial dimensions
and to rescale the values with scalefactor.
"""
function linear_weights(fsize::NTuple{Φ,Int}, φs = 1:Φ, weights = fill(1,length(φs))) where {Φ}
    function f(α::CartesianIndex)
        map(φs,weights) do φ, w
            i = α.I[φ]
            s = fsize[φ]
            w*(2(i-1)/(s-1) - 1)
        end
    end
    return f
end
