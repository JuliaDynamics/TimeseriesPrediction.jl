using Plots

include("streconstruction.jl")
include("prediction_alg.jl")

function barkley(T, Nx=100, Ny=100)
    a = 0.75
    b = 0.02
    ε = 0.02
    U = zeros(Nx, Ny, T)
    V = zeros(Nx, Ny, T)
    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)

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
        V[:,:,m] .= v
        U[:,:,m] .= u
    end
    return U,V
end


Nx = 50
Ny = 50
Tskip = 100
Ttrain = 100
Ttest = 5
T = Tskip + Ttrain + Ttest
D = 2
τ = 1
B = 1
k = 1
a = 1
b = 1
boundary = 20

U,V = barkley(T, Nx, Ny)
Utrain = U[:,:,Tskip + 1:Tskip + Ttrain]
Vtrain = V[:,:,Tskip + 1:Tskip + Ttrain]
Utest  = U[:,:,Tskip + Ttrain + (D-1)τ:  T]
Vtest  = V[:,:,Tskip + Ttrain :  T]







Upred = crosspred_stts(Vtrain,Utrain,Vtest, D, τ, B, k, a, b)
error = abs.(Utest-Upred)
ε = sum(error, (1,2))[:]


# Animation (takes forever)
@time @gif for i=2:Base.size(Utest)[3]
    l = @layout([a b; c d])
    p1 = plot(@view(Vtest[:,:,i+(D-1)τ]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    plot!(title = "Barkley Model")
    p2 = plot(@view(Utest[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("U component")
    p3 = plot(@view(Upred[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("U Prediction")
    p4 = plot(@view(error[:,:,i]),clims=(0,0.1),aspect_ratio=1,st=[:heatmap])
    title!("Model Error")

    plot(p1,p2,p3,p4, layout=l, size=(600,600))
end





# Utest  = U[:,:,Tskip + Ttrain :  T]
# Vtest  = V[:,:,Tskip + Ttrain + (D-1)τ:  T]
#
#
#
#
#
#
#
# Vpred = crosspred_stts(Utrain,Vtrain,Utest, D, τ, B, k, a, b)
# error = abs.(Vtest-Vpred)
# ε = sum(error, (1,2))[:]
#
#
# # Animation (takes forever)
# @time @gif for i=2:Base.size(Utest)[3]
#     l = @layout([a b; c d])
#     p1 = plot(@view(Vtest[:,:,i+(D-1)τ]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     plot!(title = "Barkley Model")
#     p2 = plot(@view(Utest[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     title!("U component")
#     p3 = plot(@view(Vpred[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     title!("V Prediction")
#     p4 = plot(@view(error[:,:,i]),clims=(0,0.1),aspect_ratio=1,st=[:heatmap])
#     title!("Model Error")
#
#     plot(p1,p2,p3,p4, layout=l, size=(600,600))
# end




#experiment: separate training and test
#
# U,V = barkley(T, Nx, Ny)
# Utrain = U[:,:,Tskip + 1:Tskip + Ttrain]
# Vtrain = V[:,:,Tskip + 1:Tskip + Ttrain]
# Utest  = U[:,:,Tskip + Ttrain + (D-1)τ:  T]
# Vtest  = V[:,:,Tskip + Ttrain :  T]
#
#
#
#
#
#
#
# Upred = crosspred_stts(Vtrain,Utrain,Vtest, D, τ, B, k, a, b)
# error = abs.(Utest-Upred)
# ε = sum(error, (1,2))[:]
#
#
# # Animation (takes forever)
# @time @gif for i=2:Base.size(Utest)[3]
#     l = @layout([a b; c d])
#     p1 = plot(@view(Vtest[:,:,i+(D-1)τ]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     plot!(title = "Barkley Model")
#     p2 = plot(@view(Utest[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     title!("U component")
#     p3 = plot(@view(Upred[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
#     title!("U Prediction")
#     p4 = plot(@view(error[:,:,i]),clims=(0,0.1),aspect_ratio=1,st=[:heatmap])
#     title!("Model Error")
#
#     plot(p1,p2,p3,p4, layout=l, size=(600,600))
# end
