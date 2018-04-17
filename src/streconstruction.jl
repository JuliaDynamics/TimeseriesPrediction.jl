using StaticArrays
using IterTools
###########################################################################################
#                       Better generated Reconstruction                                   #
###########################################################################################

function make_gens(::Type{Val{Φ}}, ::Type{Val{lims}},::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}, ::Type{Val{false}}
    ) where {Φ, lims, D, B, k}
    gens = Expr[]
    for d=0:D-1, lidx ∈ product([-B*k:k:B*k for i=1:Φ]...)
        x = [:(mod(-1 + midx[$i]+$(lidx[i]), $(lims[i])) +1) for i=1:Φ]
        push!(gens, :( s[t + $d*τ][$(x...)]))
    end
    return gens
end

function make_gens(::Type{Val{Φ}}, ::Type{Val{lims}},::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}, ::Type{Val{boundary}}
    ) where {Φ, lims, D, B, k, boundary}
    gens = Expr[]
    for d=0:D-1, lidx ∈ product([-B*k:k:B*k for i=1:Φ]...)
        cond = :(0 < midx[1] + $(lidx[1]) <= $(lims[1]))
        for i=2:Φ cond = :($cond && 0 < midx[$i] + $(lidx[i]) <= $(lims[i])) end

        push!(gens, :( $cond ?  s[t + $d*τ][(midx .+ $lidx)...] : $boundary))
    end
    return gens
end

function my_reconstruct_impl(::Type{Val{Φ}}, ::Type{Val{lims}},::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}, ::Type{Val{weighting}}, ::Type{Val{boundary}}
    ) where {Φ, lims, D, B, k, weighting, boundary}

    gens = make_gens(Val{Φ}, Val{lims}, Val{D}, Val{B},Val{k}, Val{boundary})
    if ((a,b) = weighting) != (0,0)
        w = Φ
        append!(gens, [:($a*(-1+2*(midx[$i]-1)/($(lims[i]-1)))^$b)  for i=1:Φ])
    else
        w = 0
    end

    midxs = product([1:lims[i] for i=1:Φ]...)
    quote
        M = prod(size(s[1]))
        L = length(s) - $(D-1)*τ
        T = eltype(s[1][1])
        data = Vector{SVector{$D*(2*$B + 1)^$Φ+$w, T}}(L*M)

        for t ∈ 1:L
            for (n,midx) ∈ enumerate($(midxs))
                data[n+(t-1)*M] = SVector{$D*(2*$B + 1)^Φ+$w, T}($(gens...))
            end
        end
        data
    end
end

@generated function my_reconstruct(
    ::Type{Val{Φ}},::Type{Val{lims}}, s,
    ::Type{Val{D}}, ::Type{Val{B}}, τ, ::Type{Val{k}},
    ::Type{Val{weighting}}, ::Type{Val{boundary}}
    ) where {Φ,lims, D, B, k, weighting, boundary}
     my_reconstruct_impl(Val{Φ}, Val{lims}, Val{D}, Val{B}, Val{k}, Val{weighting},
     Val{boundary})
end

function myReconstruction(
    s::AbstractVector{Array{T, Φ}}, D, τ::DT, B, k, boundary, weighting
    ) where {T, Φ, DT}
    lims = size(s[1])
    w = Φ*(weighting != (0,0))
    Reconstruction{D*(2B+1)^Φ+w,T,DT}(
    my_reconstruct(Val{Φ},Val{lims}, s, Val{D},
     Val{B},τ,Val{k},Val{weighting}, Val{boundary}), τ)
end
