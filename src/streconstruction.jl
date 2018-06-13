using StaticArrays
using IterTools

export STReconstruction

#in v0.7 there is Base.IteratorsMD.LinearIndices until then
function to_linear(lims, idx)
    n = 1
    for i=1:length(idx)
        n += (idx[i]-1)*prod(lims[1:i-1])
    end
    n
end


function unroll(lidx)
    [:(midx[$i] + $(lidx[i]) ) for i=1:length(lidx) ]
end

function make_gens(::Type{Val{Φ}}, ::Type{Val{lims}},::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}, ::Type{Val{boundary}}
    ) where {Φ, lims, D, B, k, boundary}
    gens = Expr[]
    for d=0:D-1, lidx ∈ product([-B*k:k:B*k for i=1:Φ]...)
        cond = :(0 < midx[1] + $(lidx[1]) <= $(lims[1]))
        for i=2:Φ cond = :($cond && 0 < midx[$i] + $(lidx[i]) <= $(lims[i])) end

        push!(gens, :( $cond ?  s[t + $d*τ][$(unroll(lidx)...)] : $boundary))
    end
    return gens
end

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

function make_middle_gens(::Type{Val{Φ}}, ::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}
    ) where {Φ, D, B, k}
    gens = [:(s[t + $d*τ][$(unroll(lidx)...)]) for
    lidx ∈ product([-B*k:k:B*k for i=1:Φ]...), d=0:D-1]

    return reshape(gens,:)
end

function my_reconstruct_impl(::Type{Val{Φ}}, ::Type{Val{lims}},::Type{Val{D}},
    ::Type{Val{B}}, ::Type{Val{k}}, ::Type{Val{weighting}}, ::Type{Val{boundary}}
    ) where {Φ, lims, D, B, k, weighting, boundary}

    gens = make_gens(Val{Φ}, Val{lims}, Val{D}, Val{B},Val{k}, Val{boundary})
    middle_gens = make_middle_gens(Val{Φ}, Val{D}, Val{B},Val{k})
    if ((a,b) = weighting) != (0,0)
        w = Φ
        append!(gens, [:($a*(-1+2*(midx[$i]-1)/($(lims[i]-1)))^$b)  for i=1:Φ])
        append!(middle_gens, [:($a*(-1+2*(midx[$i]-1)/($(lims[i]-1)))^$b)  for i=1:Φ])
    else
        w = 0
    end

    midxs = product([1:lims[i] for i=1:Φ]...)

    #is boundary
    is_middle = :($B*$k < midx[1] <= $(lims[1]) - $B*$k )
    for i=2:Φ is_middle = :($is_middle && $B*$k < midx[$i] <= $(lims[i]) - $B*$k)
    end

    quote
        M = prod(size(s[1]))
        L = length(s) - $(D-1)*τ
        T = eltype(s[1][1])
        data = Array{T}($D*(2*$B + 1)^$Φ+$w,L*M)

        @inbounds for t ∈ 1:L
            for (n,midx) ∈ enumerate($(midxs))
                if $is_middle
                    data[:,n+(t-1)*M] .= SVector($(middle_gens...))
                    #this should be alloc free
                else
                    data[:,n+(t-1)*M] .= SVector($(gens...))
                end
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




"""
    STReconstruction(U::AbstractVector{<:AbstractArray}, D, τ, B, k, boundary, weighting)
     <: AbstractDataset
Perform spatio-temporal(ST) delay-coordinates reconstruction from a ST-timeseries `s`.

## Description
An extension of [`Reconstruction`](@ref) to support inclusion of spatial neighbors into
reconstructed vectors.
`B` is the number of spatial neighbors along each direction to be included.
The parameter `k` indicates the spatial sampling density as
described in [1].

To better understand `B`, consider a system of 2 spatial dimensions, where the
state is a `Matrix`, and choose a point of the matrix to reconstruct.
Giving `B = 1` will choose the current point and 8 points around it,
*not* 4! For `Φ` dimensions,
the number of spatial points is then given by `(2B + 1)^Φ`.
Temporal embedding via `D` and `τ` is treated analogous to [`Reconstruction`](@ref).
Therefore the total embedding dimension is `D*(2B + 1)^Φ`.

## Other Parameters
* `boundary = 20` : Constant boundary value used for reconstruction of states close to
  the border. Pass `false` for periodic boundary conditions.
* `weighting = (a,b)` or `nothing` : If given numbers `(a, b)`,
  adds `Φ` additional entries to reconstructed states.
  These are a spatial weighting that may be useful for considering spatially
  inhomogenous dynamics.
  Each entry is a normalized spatial coordinate ``-1\\leq\\tilde{x}\\leq 1``:
  ```math
  \\begin{aligned}
  \\omega(\\tilde{x}) = a \\tilde{x} ^ b.
  \\end{aligned}
  ```

## References
[1] : U. Parlitz & C. Merkwirth, [Phys. Rev. Lett. **84**, pp 1890 (2000)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1890)
"""
function STReconstruction(
    s::AbstractVector{<:AbstractArray{T, Φ}}, D, τ::DT, B, k, boundary, weighting,
    ) where {T, Φ, DT}
    lims = size(s[1])
    w = Φ*(weighting != (0,0))
    data = my_reconstruct(Val{Φ},Val{lims}, s, Val{D},
            Val{B},τ,Val{k},Val{weighting}, Val{boundary})
    return Reconstruction{D*(2B+1)^Φ+w,T,DT}(reinterpret(Dataset,data).data, τ)
end
