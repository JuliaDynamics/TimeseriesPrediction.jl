function coupled_henon1D(M)
    function henon(du, u, p, t)
        du[1,1] = du[M,1] = 0.5
        du[1,2] = du[M,2] = 0
        for m=2:M-1
            du[m,1] = 1 - 1.45*(.5*u[m,1]+ .25*u[m-1,1] + .25u[m+1,1])^2 + 0.3*u[m,2]
            du[m,2] = u[m,1]
        end
        return nothing
    end
    return SpatioTemporalSystem(henon,rand(M,2), nothing; t0=0)
end

function coupled_henon2D(X,Y)
    function henon(du, u, p, t)
        du[1,:,1] = du[:,1,1] = du[X, :, 1] = du[:, Y, 1] = 0.5
        du[1,:,2] = du[:,1,2] = du[X, :, 2] = du[:, Y ,2] = 0
        for mx=2:X-1 , my=2:Y-1
            du[mx,my,1] = 1 - 1.45*(.5*u[mx,my,1]+ .125*u[mx-1,my,1] +
             .125u[mx+1,my,1]+ .125u[mx, my-1,1]+ .125u[mx,my+1,1])^2 + 0.3*u[mx,my,2]
            du[mx,my,2] = u[mx,my,1]
        end
        return nothing
    end
    return SpatioTemporalSystem(henon,rand(X,Y,2), nothing; t0=0)
end


function barkley_const_boundary(T, Nx, Ny)
    a = 0.75
    b = 0.02
    ε = 0.02

    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V = Vector{Array{Float64,2}}()

    #Initial state that creates spirals
    u[40:end,34] .= 0.1
    u[40:end,35] .= 0.5
    u[40:end,36] .= 5
    v[40:end,34] .= 1

    u[1:10,14] .= 5
    u[1:10,15] .= 0.5
    u[1:10,16] .= 0.1
    v[1:10,17] .= 1

    u[27:36,20] .= 5
    u[27:36,19] .= 0.5
    u[27:36,18] .= 0.1
    v[27:36,17] .= 1



    h = 0.75 #/ sqrt(2)
    Δt = 0.1 #/ 2
    δ = 0.001
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
                i > 1  && (Σ[i-1,j,s] += u[i,j])
                i < Nx && (Σ[i+1,j,s] += u[i,j])
                j > 1  && (Σ[i,j-1,s] += u[i,j])
                j < Ny && (Σ[i,j+1,s] += u[i,j])
            end
            Σ[i,j,r] = 0
        end
        r,s = s,r
        #V[:,:,m] .= v
        #U[:,:,m] .= u
        push!(U,copy(u))
        push!(V,copy(v))
    end
    return U,V
end

function barkley_periodic_boundary(T, Nx, Ny)
    a = 0.75
    b = 0.02
    ε = 0.02

    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V = Vector{Array{Float64,2}}()

    #Initial state that creates spirals
    u[40:end,34] .= 0.1
    u[40:end,35] .= 0.5
    u[40:end,36] .= 5
    v[40:end,34] .= 1

    u[1:10,14] .= 5
    u[1:10,15] .= 0.5
    u[1:10,16] .= 0.1
    v[1:10,17] .= 1

    u[27:36,20] .= 5
    u[27:36,19] .= 0.5
    u[27:36,18] .= 0.1
    v[27:36,17] .= 1

    h = 0.75 #/ sqrt(2)
    Δt = 0.1 #/ 2
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
        for j=1:Ny, i=1:Nx
            if u[i,j] < δ
                u[i,j] = Δt/h^2 * Σ[i,j,r]
                v[i,j] = (1 - Δt)* v[i,j]
            else
                uth = (v[i,j] + b)/a
                v[i,j] = v[i,j] + Δt*(u[i,j] - v[i,j])
                u[i,j] = F(u[i,j], uth) + Δt/h^2 *Σ[i,j,r]
                Σ[i,j,s] -= 4u[i,j]
                Σ[  mod(i-1-1,Nx)+1,j,s] += u[i,j]
                Σ[  mod(i+1-1,Nx)+1,j,s] += u[i,j]
                Σ[i,mod(j-1-1,Ny)+1,  s] += u[i,j]
                Σ[i,mod(j+1-1,Ny)+1,  s] += u[i,j]
            end
            Σ[i,j,r] = 0
        end
        r,s = s,r
        #V[:,:,m] .= v
        #U[:,:,m] .= u
        push!(U,copy(u))
        push!(V,copy(v))
    end
    return U,V
end


function barkley_periodic_boundary_nonlin(T, Nx, Ny)
    a = 0.75
    b = 0.02
    ε = 0.02

    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V = Vector{Array{Float64,2}}()

    #Initial state that creates spirals
    u[35:end,34] .= 1
    u[35:end,35] .= 1
    u[35:end,36] .= 1
    v[35:end,37] .= 1
    v[35:end,38] .= 1
    v[35:end,39] .= 1


    u[1:20,14] .= 1
    u[1:20,15] .= 1
    u[1:20,16] .= 1
    v[1:20,17] .= 1
    v[1:20,18] .= 1
    v[1:20,19] .= 1
    v[1:20,20] .= 1


    u[27:36,20] .= 1
    u[27:36,19] .= 1
    u[27:36,18] .= 1
    v[27:36,17] .= 1
    v[27:36,16] .= 1
    v[27:36,15] .= 1

    h = 0.75 #/ sqrt(2)
    Δt = 0.1 #/ 2
    δ = 0.001
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
        for j=1:Ny, i=1:Nx
            if u[i,j] < δ
                u[i,j] = Δt/h^2 * Σ[i,j,r]
                v[i,j] = (1 - Δt)* v[i,j]
            else
                uth = (v[i,j] + b)/a
                v[i,j] = v[i,j] + Δt*(u[i,j]^3 - v[i,j])
                u[i,j] = F(u[i,j], uth) + Δt/h^2 *Σ[i,j,r]
                Σ[i,j,s] -= 4u[i,j]
                Σ[  mod(i-1-1,Nx)+1,j,s] += u[i,j]
                Σ[  mod(i+1-1,Nx)+1,j,s] += u[i,j]
                Σ[i,mod(j-1-1,Ny)+1,  s] += u[i,j]
                Σ[i,mod(j+1-1,Ny)+1,  s] += u[i,j]
            end
            Σ[i,j,r] = 0
        end
        r,s = s,r
        #V[:,:,m] .= v
        #U[:,:,m] .= u
        push!(U,copy(u))
        push!(V,copy(v))
    end
    return U,V
end
