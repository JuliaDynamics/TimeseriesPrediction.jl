export SpatioTemporalSystem
using DynamicalSystemsBase: isinplace


struct SpatioTemporalProblem{Φ, T, F, P}
    f::F      # eom, but same syntax as ODEProblem
    u0::Array{T, Φ}     # initial state
    p::P      # parameter container
    t0::Int   # initial time
end

const STP = SpatioTemporalProblem

"""
    SpatioTemporalSystem(eom, state::Array, p; t0::Int = 0)
A spatio-temporal system that accomodates general `Array` as state.
"""
struct SpatioTemporalSystem{Φ, T, F, P} #<: DynamicalSystem{true, Array{T, Φ}, Φ, F, P, Void, Void, Void}
    prob::STP{Φ, T, F, P}
end

const STS = SpatioTemporalSystem

function SpatioTemporalSystem(
    eom::F, s::Array{T, Φ}, p::P; t0::Int = 0) where {F, T, Φ, P}
    IIP = isinplace(eom, 4)
    IIP || throw(ArgumentError("SpatioTemporalSystem only supports in-place eom."))
    prob = STP(eom, s, p, t0)
    S = typeof(prob.u0)
    D = length(s)
    return STS{Φ, T, F, P}(prob)
end

Base.summary(ds::STS{Φ, T, F, P}) where {Φ, T, F, P} =
"spatio-temporal dynamical system with $Φ spatial dimensions"

function Base.show(io::IO, ds::STS)
    ps = 14
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state size: ", ps)*"$(size(get_state(ds)))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" parameters: ", ps)*"$(ds.prob.p)\n"
    )
end



#####################################################################################
#                                  Trajectory                                       #
#####################################################################################
function DynamicalSystemsBase.trajectory(ds::STS{Φ, T, F, P}, t, u = ds.prob.u0;
    dt::Int = 1) where {Φ, T, F, P}
    ti = ds.prob.t0
    tvec = ti:dt:t
    L = length(tvec)
    data = Vector{Array{T, Φ}}(L)
    uprev = copy(ds.prob.u0)
    unext = copy(uprev)
    data[1] = copy(uprev)
    for i in 2:L
        uprev, unext = unext, uprev
        ds.prob.f(unext, uprev, ds.prob.p, tvec[i])
        data[i] = copy(unext)
    end
    return data
end
