using Plots
using TimeseriesPrediction

#using PyPlot
#pyplot()

##This Algorithm is taken from
#  http://www.scholarpedia.org/article/Barkley_model

# Simulation is super fast but plotting/animating sucks....


function barkley(T, Nx=50, Ny=50)
    a = 0.75
    b = 0.02
    ε = 0.02

    Φ = 2
    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V =Vector{Array{Float64,2}}()

    #Initial state that creates spirals
    u[40:end,34] = 0.1
    u[40:end,35] = 0.5
    u[40:end,36] = 5
    v[40:end,34] = 1

    u[1:10,14] = 5
    u[1:10,15] = 0.5
    u[1:10,16] = 0.1
    v[1:10,17] = 1

    u[27:36,20] = 5
    u[27:36,19] = 0.5
    u[27:36,18] = 0.1
    v[27:36,17] = 1

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


###########################################################################################
#               Example starting from here                                                #
###########################################################################################



Nx = 50
Ny = 50
Tskip = 100
Ttrain = 50
Ttest = 10
T = Tskip +Ttrain + Ttest

U,V = barkley(T, Nx, Ny)


D = 2
τ = 1
B = 1
k = 1
c = false
w = (0,0)



Utrain = U[Tskip + 1:Tskip + Ttrain]
Vtrain = V[Tskip + 1:Tskip + Ttrain]
Utest  = U[Tskip + Ttrain - (D-1)τ + 1:  T]
Vtest  = V[Tskip + Ttrain  - (D-1)τ + 1:  T]
Upred = crosspred_stts(Vtrain,Utrain,Vtest, D, τ, B, k; boundary=c)#, drtype=PCA)
err = [abs.(Utest[1+(D-1)τ:end][i]-Upred[i]) for i=1:Ttest]


fname = "pca_bark_cross_Ttrain$(Ttrain)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w)"

# Animation (takes forever)
cd();cd("Documents/Bachelorarbeit/STTS/PeriodicSystems")
@time anim = @animate for i=2:length(Upred)
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
end; mp4(anim, fname * ".mp4")







Vtrain = V[Tskip + 1:Tskip + Ttrain]
Vtest  = V[Tskip + Ttrain :  T]
Vpred = localmodel_stts(Vtrain, D, τ, Ttest, B, k; boundary=c)#, drtype=PCA)
err = [abs.(Vtest[i]-Vpred[i]) for i=1:Ttest+1]
fname = "bark_ts_Ttrain$(Ttrain)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w)" * randstring()

# Animation (takes forever)
cd();cd("Documents/Bachelorarbeit/STTS/PeriodicSystems")
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
    clims=(0,0.02),
    aspect_ratio=1,
    st=:heatmap,
    seriescolor=:viridis)


    plot(p1,p2,p3, layout=l, size=(600,170))
end; mp4(anim, fname * ".mp4")
