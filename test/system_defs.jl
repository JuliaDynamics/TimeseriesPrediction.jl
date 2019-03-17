function coupled_henon1D(M,N, u0=rand(M), v0=rand(M))
    function henon(U,V)
        Un = copy(U)
        Vn = copy(V)
        Un[1] = Un[M] = 0.5
        Vn[1] = Vn[M] = 0
        for m=2:M-1
            Un[m] = 1 - 1.45*(.5*U[m]+ .25*U[m-1] + .25U[m+1])^2 + 0.3*V[m]
            Vn[m] = U[m]
        end
        return Un, Vn
    end

    U = Vector{Vector{Float64}}(undef,N)
    V = Vector{Vector{Float64}}(undef,N)
    U[1] = u0
    V[1] = v0
    for n = 2:N
        U[n],V[n] = henon(U[n-1],V[n-1])
    end
    return U,V
end

function coupled_henon2D(X,Y,N,u0=rand(X,Y), v0=rand(X,Y))
    function henon(U,V)
        Un = copy(U)
        Vn = copy(V)
        Un[1,:] = Un[:,1] = Un[X, :] = Un[:, Y] .= 0.5
        Vn[1,:] = Vn[:,1] = Vn[X, :] = Vn[:, Y] .= 0
        for mx=2:X-1 , my=2:Y-1
            Un[mx,my] = 1 - 1.45*(.5*U[mx,my]+ .125*U[mx-1,my] +
             .125U[mx+1,my]+ .125U[mx, my-1]+ .125U[mx,my+1])^2 + 0.3*V[mx,my]
            Vn[mx,my] = U[mx,my]
        end
        return Un, Vn
    end
    U = Vector{Matrix{Float64}}(undef,N)
    V = Vector{Matrix{Float64}}(undef,N)
    U[1] = u0
    V[1] = v0
    for n = 2:N
        U[n],V[n] = henon(U[n-1],V[n-1])
    end
    return U,V

end

"""
```
barkley(T;
        tskip=0,
        periodic=true,
        ssize=(50,50),
        a=0.75, b=0.06, ε=0.08, D=1/50, h=0.1, Δt=0.1)
```
Simulate the Barkley model (nonlinear `u^3` term).
"""
function barkley(T;
                tskip=0,
                periodic=true,
                ssize=(50,50),
                a=0.75, b=0.06, ε=0.08, D=1/50, h=0.1, Δt=0.1,
                init = rand(10,10))

    Nx, Ny = ssize
    @assert Nx ≥ 40
    @assert Ny ≥ 40
    U = Vector{Array{Float64,2}}()
    V = Vector{Array{Float64,2}}()

    u = zeros(Nx,Ny)
    v = zeros(Nx,Ny)

    αα, ββ = size(init)

    v .= u .= repeat(init, inner=(Nx÷αα,Ny÷ββ))
    v .= v.>0.2
    Σ = zeros(Nx, Ny, 2)
    r,s = 1,2

    function F(u, uth)
        if u < uth
            u/(1-(Δt/ε)*(1-u)*(u-uth))
        else
            (u + (Δt/ε)*u*(u-uth))/(1+(Δt/ε)*u*(u-uth))
        end
    end

    function periodic_step(u,v,Σ,a,b,Nx,Ny,s,r,D,Δt,h)
        for j=1:Ny, i=1:Nx
            uth = (v[i,j] + b)/a
            v[i,j] = v[i,j] + Δt*(u[i,j]^3 - v[i,j])
            u[i,j] = F(u[i,j], uth) + D * Δt/h^2 *Σ[i,j,r]
            Σ[i,j,s] -= 4u[i,j]
            Σ[  mod(i-1-1,Nx)+1,j,s] += u[i,j]
            Σ[  mod(i+1-1,Nx)+1,j,s] += u[i,j]
            Σ[i,mod(j-1-1,Ny)+1,  s] += u[i,j]
            Σ[i,mod(j+1-1,Ny)+1,  s] += u[i,j]
            Σ[i,j,r] = 0
        end
    end
    function constant_step(u,v,Σ,a,b,Nx,Ny,s,r,D,Δt,h)
        for i=1:Nx, j=1:Ny
            uth = (v[i,j] + b)/a
            v[i,j] = v[i,j] + Δt*(u[i,j] - v[i,j])
            u[i,j] = F(u[i,j], uth) + Δt/h^2 *Σ[i,j,r]
            Σ[i,j,s] -= 4u[i,j]
            i > 1  && (Σ[i-1,j,s] += u[i,j])
            i < Nx && (Σ[i+1,j,s] += u[i,j])
            j > 1  && (Σ[i,j-1,s] += u[i,j])
            j < Ny && (Σ[i,j+1,s] += u[i,j])
            Σ[i,j,r] = 0
        end
    end

    for m=1:T+tskip
        periodic ? periodic_step(u,v,Σ,a,b,Nx,Ny,s,r,D,Δt,h) :
                   constant_step(u,v,Σ,a,b,Nx,Ny,s,r,D,Δt,h)
        r,s = s,r
        if m > tskip
            push!(U,copy(u))
            push!(V,copy(v))
        end
    end
    return U,V
end
