using PyPlot
using DynamicalSystemsBase



include("streconstruction.jl")
include("prediction_alg.jl")






function coupled_henon(X=10,Y=10)
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
    return DiscreteDynamicalSystem(henon,rand(X,Y,2), nothing; t0=0)
end


#Size
X=10
Y=10
ds = coupled_henon(X,Y)
N_train = 1000
p = 30
data = trajectory(ds,N_train+p)
#Reconstruct this #
s = data[1:N_train,SVector(1:X*Y...)]
mat = convert(Matrix,s)'
mat = reshape(mat, (X,Y,N_train))

begin
    ax1 = subplot(311)
    #Original
    img = Matrix(data[N_train:N_train+p,SVector(1:X*Y...)])
    pcolormesh(img)
    colorbar()
    # Make x-tick labels invisible
    setp(ax1[:get_xticklabels](), visible=false)
    title("original")


    #Prediction
    ax2 = subplot(312, sharex = ax1, sharey = ax1)
    s_pred = localmodel_stts(mat,2,1,p,1,1,10, 1,1)
    pred_mat = reshape(s_pred, (X*Y,p+1))'
    pcolormesh(pred_mat)
    colorbar()
    setp(ax2[:get_xticklabels](), visible=false)
    title("prediction")
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
