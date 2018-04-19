using Plots
using TimeseriesPrediction

#using PyPlot
#pyplot()

##This Algorithm is taken from
#  http://www.scholarpedia.org/article/Barkley_model

# Simulation is super fast but plotting/animating sucks....


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


###########################################################################################
#               Example starting from here                                                #
###########################################################################################



Nx = 36
Ny = 36
Tskip = 100
Ttrain = 50
p = 30
T = Tskip + Ttrain + p

U,V = barkley(T, Nx, Ny)
Vtrain = V[Tskip + 1:Tskip + Ttrain]
Vtest  = V[Tskip + Ttrain :  T]



D = 2
τ = 1
B = 1
k = 1

c = 20
w = (0,0)



@profiler Vpred = localmodel_stts(Vtrain, D, τ, p, B, k; boundary=c, weighting=w)
err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
ε = map(s -> sum(s), err)


# Animation (takes forever)
@gif for i=2:Base.size(Vtest)[1]
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

#
# p = plot(@view(Vtest[:,:,1]), st=:heatmap,seriescolor=:viridis, cb=false, xticks=false,
# yticks=false, size=(200,200))
# savefig(p, "/home/jonas/Documents/Bachelorarbeit/Presentation/frame1.png")
# p = plot(@view(Vtest[:,:,21]), st=:heatmap, seriescolor=:viridis,cb=false, xticks=false,
# yticks=false, size=(200,200))
# savefig(p, "/home/jonas/Documents/Bachelorarbeit/Presentation/frame2.png")
# p = plot(@view(Vtest[:,:,41]), st=:heatmap,seriescolor=:viridis, cb=false, xticks=false,
# yticks=false, size=(200,200))
# savefig(p, "/home/jonas/Documents/Bachelorarbeit/Presentation/frame3.png")
