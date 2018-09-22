using Plots
using TimeseriesPrediction

# This Algorithm is taken from
#  http://www.scholarpedia.org/article/Barkley_model

# Simulation is super fast but plotting/animating sucks....

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


########################################################################################
#               Example starting from here                                             #
########################################################################################
Nx = 50
Ny = 50
Tskip = 100
Ttrain = 200
Ttest = 100
T = Tskip +Ttrain + Ttest

U,V = barkley_periodic_boundary_nonlin(T, Nx, Ny)

D = 2
τ = 1
B = 1
k = 1
BC = PeriodicBoundary()


Vtest  = V[Tskip + Ttrain  - (D-1)τ + 1:  T]
Vtrain = V[Tskip + 1:Tskip + Ttrain]
Vtest  = V[Tskip + Ttrain :  T]

em = cubic_shell_embedding(Vtrain, D,τ,B,k,BC)
Vpred = temporalprediction(Vtrain, em, Ttest)

err = [abs.(Vtest[i]-Vpred[i]) for i=1:Ttest+1]
fname = "periodic_barkley_ts_Train=$(Ttrain)_D=$(D)_τ=$(τ)_B=$(B)_k=$(k)"

@time anim = @animate for i=2:length(Vtest)
    l = @layout([a b c])
    p1 = plot(Vtest[i],
    title = "Barkley Model",
    xlabel = "X",
    ylabel = "Y",
    clims=(0,0.75),
    cbar = false,
    aspect_ratio=1,
    st=:heatmap)

    p2 = plot(Vpred[i],
    title = "Prediction",
    xlabel = "X",
    #ylabel = "Y",
    clims=(0,0.75),
    aspect_ratio=1,
    st=:heatmap)

    p3 = plot(err[i],
    title = "Absolute Error",
    xlabel = "X",
    #ylabel = "Y",
    clims=(0,0.1),
    aspect_ratio=1,
    st=:heatmap,
    seriescolor=:viridis)


    plot(p1,p2,p3, layout=l, size=(600,170))
end

gif(anim, fname * ".gif")
