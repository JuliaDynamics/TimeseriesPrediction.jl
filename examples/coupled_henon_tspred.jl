using PyPlot
using TimeseriesPrediction
using StaticArrays


function coupled_henon2D(X=10,Y=10)
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
    return SpatioTemporalSystem(henon,rand(X,Y,2), nothing; t0=0)
end

#Size
X=7
Y=7
ds = coupled_henon2D(X,Y)
N_train = 1000
p = 50
data = trajectory(ds,N_train+p)

xdata = [d[:,:,1] for d in data]

# make data 1D form, 2nd D to be time
function makeinto1D(data)
    img = zeros(p, X*Y)
    for i in 1:p
        for k in 1:X*Y
            img[i, k] = data[i][k]
        end
    end
    return img
end

ax1 = subplot(311)
img = makeinto1D(xdata[N_train+1:end])
pcolormesh(img)
colorbar()
# Make x-tick labels invisible
setp(ax1[:get_xticklabels](), visible=false)
title("2D Coupled Henon Map")


#Prediction
ax2 = subplot(312, sharex = ax1, sharey = ax1)
s_pred = localmodel_stts(xdata[1:N_train],3,1,p,1,1) # ;weighting=(1,1)
pred = makeinto1D(s_pred)
pcolormesh(pred)
colorbar()
setp(ax2[:get_xticklabels](), visible=false)
title("Prediction (Training $N_train)")
ylabel("t")

#Error
ax3 = subplot(313, sharex = ax1, sharey = ax1)
ε = abs.(img-pred)
pcolormesh(ε, cmap="inferno")
colorbar()
title("Absolute Error")
xlabel("i = (x, y)")
tight_layout()
