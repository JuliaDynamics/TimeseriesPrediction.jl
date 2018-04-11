using StaticArrays
###########################################################################################
#                       Better generated Reconstruction                                   #
###########################################################################################
function my_reconstruct_impl(::Type{Val{2}}, ::Val{D}, ::Val{B}, ::Val{k}) where {D, B, k}
    gens = [:(0 < mx + $i<=X && 0 < my + $j <= Y ?
     s[mx + $i, my + $j, t + $d*τ] : boundary)
      for i=-B*k:k:B*k, j=-B*k:k:B*k, d=0:D-1]
    quote
        X,Y,L = size(s)
        L -= $(D-1)*τ
        T = eltype(s)
        data = Vector{SVector{$D*(2*$B + 1)^2+2, T}}(L*X*Y)
        for t ∈ 1:L
            for my ∈ 1:Y, mx ∈ 1:X
                wx = a*(-1+2*(mx-1)/(X-1))^b
                wy = a*(-1+2*(my-1)/(Y-1))^b
                n = mx + X*(my-1) + Y*X*(t-1)
                data[n] = SVector{$D*(2*$B + 1)^2+2, T}($(gens...),wx, wy)
            end
        end
        data
    end
end

function my_reconstruct_impl(::Type{Val{1}}, ::Val{D}, ::Val{B}, ::Val{k}) where {D, B, k}
    gens = [:(0 < mx + $i<=X  ? s[mx + $i, t + $d*τ] : boundary)
    for i=-B*k:k:B*k, d=0:D-1]
        quote
            X,L = size(s)
            L -= $(D-1)*τ
            T = eltype(s)
            data = Vector{SVector{$D*(2*$B + 1)+1, T}}(L*X)
            for t ∈ 1:L
                for mx ∈ 1:X
                    wx = a*(-1+2*(mx-1)/(X-1))^b
                    n = mx + X*(t-1)
                    data[n] = SVector{$D*(2*$B + 1)+1, T}($(gens...),wx)
                end
            end
            data
        end
    end

@generated function my_reconstruct(::Val{Φ}, s, ::Val{D}, ::Val{B}, τ,::Val{k},
     boundary, a, b) where {Φ, D, B, k}
    my_reconstruct_impl(Val{Φ}, Val{D}(), Val{B}(), Val{k}())
end

function myReconstruction(s::AbstractArray{T,Ψ}, D,τ::DT,B=1,k=1,boundary=10, a=1,b=1
    ) where {T, Ψ, DT}
    Φ = Ψ-1
    Reconstruction{D*(2B+1)^Φ+Φ,T,DT}(
    my_reconstruct(Val{Φ}(),s, Val{D}(), Val{B}(),τ,Val{k}(),boundary,a,b), τ)
end
