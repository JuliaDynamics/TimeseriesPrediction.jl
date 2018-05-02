using Plots
using TimeseriesPrediction

# This Algorithm is taken from
# http://www.scholarpedia.org/article/Barkley_model

function barkley(T, Nx, Ny)
    a = 0.75
    b = 0.02
    ε = 0.02

    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    U = Vector{Array{Float64,2}}()
    V = Vector{Array{Float64,2}}()

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


###########################################################################################
#                      Example starting from here                                         #
###########################################################################################
Nx = 50
Ny = 50
Tskip = 200
Ttrain = 1000
p = 200
T = Tskip + Ttrain + p

U,V = barkley(T, Nx, Ny)
Vtrain = V[Tskip + 1:Tskip + Ttrain]
Vtest  = V[Tskip + Ttrain :  T]

D = 2
τ = 1
B = 2
k = 1
c = 200
w = (0,0)

@time Vpred = localmodel_stts(Vtrain, D, τ, p, B, k; boundary=c, weighting=w)
err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]

fname = "barkley_ts_Train=$(Ttrain)_p=$(p)_D=$(D)_τ=$(τ)_B=$(B)_k=$(k)"

cd(); mkpath("tspred_examples"); cd("tspred_examples")

@time anim = @animate for i=5:2:length(Vtest)
    l = @layout([a b c])
    p1 = Plots.plot(Vtest[i],
    title = "Barkley, t=$(i)",
    xlabel = "X",
    ylabel = "Y",
    clims=(0,0.75),
    aspect_ratio=1,
    st=:heatmap)

    p2 = Plots.plot(Vpred[i],
    title = "Prediction",
    xlabel = "X",
    #ylabel = "Y",
    clims=(0,0.75),
    aspect_ratio=1,
    st=:heatmap)

    p3 = Plots.plot(err[i],
    title = "Absolute Error",
    xlabel = "X",
    #ylabel = "Y",
    clims=(0,0.1),
    aspect_ratio=1,
    st=:heatmap,
    seriescolor=:viridis)

    Plots.plot(p1,p2,p3, layout=l, size=(600,170))
end

gif(anim, fname * ".gif", fps = 5)

println("DONE")
