using PyPlot
using DynamicalSystemsBase
using StaticArrays
using NearestNeighbors
using TimeseriesPrediction


include("streconstruction.jl")
include("prediction_alg.jl")


function coupled_henon(M=100)
    function henon(du, u, p, t)
        du[1,1] = du[M,1] = 0.5
        du[1,2] = du[M,2] = 0
        for m=2:M-1
            du[m,1] = 1 - 1.45*(.5*u[m,1]+ .25*u[m-1,1] + .25u[m+1,1])^2 + 0.3*u[m,2]
            du[m,2] = u[m,1]
        end
        return nothing
    end
    return DiscreteDynamicalSystem(henon,rand(M,2), nothing; t0=0)
end


M=100
ds = coupled_henon(M)
N_train = 100
p = 20
data = trajectory(ds,N_train+p)
#Reconstruct this #
mat = Matrix(data[1:N_train,SVector(1:M...)])'


begin
    ax1 = subplot(311)
    #Original
    img = Matrix(data[N_train:N_train+p,SVector(1:M...)])
    pcolormesh(img)
    colorbar()
    # Make x-tick labels invisible
    setp(ax1[:get_xticklabels](), visible=false)
    title("1D Coupled Henon Map")


    #Prediction
    ax2 = subplot(312, sharex = ax1, sharey = ax1)
    s_pred = localmodel_stts(mat,2,1,p,1,1,20, 0,0)
    pred_mat = reshape(s_pred, (M,p+1))'
    pcolormesh(pred_mat)
    colorbar()
    setp(ax2[:get_xticklabels](), visible=false)
    title("prediction ( N_train=$N_train)")
    ylabel("t")

    #Error
    ax3 = subplot(313, sharex = ax1, sharey = ax1)
    ε = abs.(img-pred_mat)
    pcolormesh(ε, cmap="inferno")
    colorbar()
    title("absolute error")
    xlabel("i = (x, y)")
    tight_layout()

end
