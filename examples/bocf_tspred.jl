#yusing PyPlot
using TimeseriesPrediction
using FileIO
using Plots
# using PyCall
# @pyimport numpy as np
# cd()
# cd("Documents/Bachelorarbeit/rcp_spatio_temporal-master/src/bocf/")
# U = np.load("bocf_data_u.npy")


# include("streconstruction.jl")
# include("prediction_alg.jl")

cd()
cd("Documents/Bachelorarbeit/STTS/BOCF")
U = load("bocf_u1_5k-10k.jld", "U")

# @gif for i=1:500
#     plot(data[i,:,:],clims=(0,1.45), aspect_ratio=1,st=:heatmap)
# end


Tskip = 500
Ttrain = 50
p = 10
T = Tskip + Ttrain + p

Utrain = [U[t, :, :] for t ∈ Tskip + 1:Tskip + Ttrain]
Utest  = [U[t, :, :] for t ∈ Tskip + Ttrain :  T]
U = nothing; gc()

#save("test.jld", "Utest", Utest)

D = 2
τ = 1
B = 1
k = 2
c = 20
w = (0,0)
@time Upred = localmodel_stts(Utrain, D, τ, p, B, k; weighting=w, boundary=c)

isdir("Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w)") ||
mkdir("Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w)")
cd(   "Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w)")
save("Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w).jld2",
 "Upred", Upred)

#Upred = load("Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w).jld2",
  "Upred")
err = [abs.(Utest[i]-Upred[i]) for i=1:p+1]
ε = map(s -> sum(s), err)


# Animation (takes forever)
@time anim = @animate for i=2:Base.size(Utest)[1]
    l = @layout([a b c])
    p1 = plot(Utest[i],
    title = "BOCF Model",
    xlabel = "X",
    ylabel = "Y",
    clims=(0,0.75),
    cbar = false,
    aspect_ratio=1,
    st=:heatmap)

    p2 = plot(Upred[i],
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


    plot(p1,p2,p3, layout=l, size=(900,300))
end
gif(anim, "Tskip$(Tskip)_Ttrain$(Ttrain)_p$(p)_D$(D)_τ$(τ)_B$(B)_k$(k)_c$(c)_w$(w).gif")
