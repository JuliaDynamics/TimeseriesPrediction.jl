using OrdinaryDiffEq
using Plots
#using PyPlot
#pyplot()


##This Algorithm is taken from
#  http://www.scholarpedia.org/article/Barkley_model

# Simulation is super fast but plotting/animating sucks....

function barkley2(Nx=100, Ny=100)
    a = 0.75
    b = 0.02
    ε = 0.02
    T = 500
    V = zeros(Nx, Ny, T)
    u = zeros(Nx, Ny)
    u[50:end,49] = 0.1
    u[50:end,50] = 0.5
    u[50:end,51] = 5
    v = zeros(Nx, Ny)
    v[50:end,48] = 1
    #u[50:52, 47:48] = 1


    h = 0.75
    Δt = 0.1
    δ = 0.01
    Σ = zeros(Nx, Ny, 2)
    r = 1
    s = 2
    function F(u, uth)
        if u < uth
            u/(1-(Δt/ε)*(1-u)*(u-uth))
        else
            (u + (Δt/ε)*u*(u-uth))/(1+(Δt/ε)*u*(u-uth))
        end
    end

    for m=1:T
        for i=1:Nx, j=1:Ny
            if u[i,j] < δ
                u[i,j] = Δt/h^2 * Σ[i,j,r]
                v[i,j] = (1 - Δt)* v[i,j]
            else
                uth = (v[i,j] + b)/a
                v[i,j] = v[i,j] + Δt*(u[i,j] - v[i,j])
                u[i,j] = F(u[i,j], uth) + Δt/h^2 *Σ[i,j,r]
                Σ[i,j,s] -= 4u[i,j]
                i > 1  ? Σ[i-1,j,s] += u[i,j] : nothing
                i < Nx ? Σ[i+1,j,s] += u[i,j] : nothing
                j > 1  ? Σ[i,j-1,s] += u[i,j] : nothing
                j < Ny ? Σ[i,j+1,s] += u[i,j] : nothing
            end
            Σ[i,j,r] = 0
        end
        r,s = s,r
        V[:,:,m] .= v
        #plot(p)
    end

    @gif for i=1:T
        plot(@view(V[:,:,i]), st=[:contourf])
    end every 10
    #return V
end
