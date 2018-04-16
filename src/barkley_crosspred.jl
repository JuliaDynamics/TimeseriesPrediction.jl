using Plots

include("streconstruction.jl")
include("prediction_alg.jl")

function barkley(T, Nx=100, Ny=100)
    a = 0.75
    b = 0.02
    ε = 0.02

    Φ = 2
    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V =Vector{Array{Float64,2}}()

    #Initial state that creates spirals
    u[35:end,34] = 0.1
    u[35:end,35] = 0.5
    u[35:end,36] = 5
    v[35:end,34] = 1

    u[1:15,14] = 5
    u[1:15,15] = 0.5
    u[1:15,16] = 0.1
    v[1:15,17] = 1


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


Nx = 50
Ny = 50
Tskip = 100
Ttrain = 100
Ttest = 10
T = Tskip +Ttrain + Ttest
D = 3
τ = 1
B = 1
k = 1
a = 1
b = 1
boundary = 20

U,V = barkley(T, Nx, Ny)
Utrain = U[Tskip + 1:Tskip + Ttrain]
Vtrain = V[Tskip + 1:Tskip + Ttrain]
Utest  = U[Tskip + Ttrain - (D-1)τ + 1:  T]
Vtest  = V[Tskip + Ttrain  - (D-1)τ + 1:  T]







Upred = crosspred_stts(Vtrain,Utrain,Vtest, D, τ, B, k, a, b)
err = [abs.(Utest[1+(D-1)τ:end][i]-Upred[i]) for i=1:Ttest]

# Animation (takes forever)
@time @gif for i=2:length(Upred)
    l = @layout([a b; c d])
    p1 = plot(Vtest[i+(D-1)τ], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    plot!(title = "Barkley Model")
    p2 = plot(Utest[i+(D-1)τ], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    title!("original U")
    p3 = plot(Upred[i], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    title!("Cross-Pred U")
    p4 = plot(err[i],clims=(0,0.1),aspect_ratio=1,
    st=:heatmap,seriescolor=:viridis)
    title!("Absolute Error")

    plot(p1,p2,p3,p4, layout=l, size=(600,600))
end




Vpred = crosspred_stts(Utrain,Vtrain,Utest, D, τ, B, k, a, b)
err = [abs.(Vtest[1+(D-1)τ:end][i]-Vpred[i]) for i=1:Ttest]

# Animation (takes forever)
@time @gif for i=2:length(Vpred)
    l = @layout([a b; c d])
    p1 = plot(Utest[i+(D-1)τ], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    plot!(title = "Barkley Model")
    p2 = plot(Vtest[i+(D-1)τ], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    title!("original V")
    p3 = plot(Vpred[i], clims=(0,0.75),aspect_ratio=1,st=:heatmap)
    title!("Cross-Pred V")
    p4 = plot(err[i],clims=(0,0.1),aspect_ratio=1,
    st=:heatmap,seriescolor=:viridis)
    title!("Absolute Error")

    plot(p1,p2,p3,p4, layout=l, size=(600,600))
end
