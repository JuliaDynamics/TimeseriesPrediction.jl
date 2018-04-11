using PyPlot
using DynamicalSystemsBase
using StaticArrays
using NearestNeighbors
using TimeseriesPrediction



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
N_train = 500
p = 30
data = trajectory(ds,N_train+p)
#Reconstruct this #
s = Matrix(data[1:N_train,SVector(1:M...)])'



begin
    subplot(311)
    img = Matrix(data[N_train:N_train+p,SVector(1:M...)])
    pcolormesh(img)
    xticks([])
    colorbar()
    subplot(312)
    s_pred = localmodel_stts(s,2,1,p,1,1,20)
    pcolormesh(s_pred')
    colorbar()
    xticks([])

    subplot(313)
    #show diff
    ε = abs.(img-pred_mat')
    pcolormesh(ε)
    colorbar()
end
