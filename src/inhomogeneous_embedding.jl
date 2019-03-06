export InhomogeneousEmbedding
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
