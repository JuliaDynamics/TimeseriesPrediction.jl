using PyPlot
using TimeseriesPrediction
using StaticArrays


function coupled_henon2D(X,Y,N,u0=rand(X,Y), v0=rand(X,Y))
    function henon(U,V)
        Un = copy(U)
        Vn = copy(V)
        Un[1,:] = Un[:,1] = Un[X, :] = Un[:, Y] .= 0.5
        Vn[1,:] = Vn[:,1] = Vn[X, :] = Vn[:, Y] .= 0
        for mx=2:X-1 , my=2:Y-1
            Un[mx,my] = 1 - 1.45*(.5*U[mx,my]+ .125*U[mx-1,my] +
             .125U[mx+1,my]+ .125U[mx, my-1]+ .125U[mx,my+1])^2 + 0.3*V[mx,my]
            Vn[mx,my] = U[mx,my]
        end
        return Un, Vn
    end
    U = Vector{Matrix{Float64}}(undef,N)
    V = Vector{Matrix{Float64}}(undef,N)
    U[1] = u0
    V[1] = v0
    for n = 2:N
        U[n],V[n] = henon(U[n-1],V[n-1])
    end
    return U,V

end

#Size
X=5
Y=5
N_train = 1000
p = 25
U,V = coupled_henon2D(X,Y,N_train+p,rand(X,Y), rand(X,Y))

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

# Plot system
figure()
ax1 = subplot(311)
img = makeinto1D(U[N_train+1:end])
pcolormesh(img)
colorbar()
# Make x-tick labels invisible
setp(ax1[:get_xticklabels](), visible=false)
title("2D Coupled Henon, field X")

# Prediction
ax2 = subplot(312, sharex = ax1, sharey = ax1)
D = 3; τ = 1; B = 1; k = 1
em = SpatioTemporalEmbedding(U,D,τ,B,k,ConstantBoundary(10.))
s_pred = temporalprediction(U,em,p)

pred = makeinto1D(s_pred)
pcolormesh(pred)
colorbar()
setp(ax2[:get_xticklabels](), visible=false)
title("Prediction (Training $N_train)")
ylabel("t")

# Error
ax3 = subplot(313, sharex = ax1, sharey = ax1)
ε = abs.(img-pred)
pcolormesh(ε, cmap="inferno")
colorbar()
title("Absolute Error")
xlabel("i = (x, y)")
tight_layout()
