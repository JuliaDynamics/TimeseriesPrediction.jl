using PyPlot
using StaticArrays
using TimeseriesPrediction

function coupled_henon1D(M,N, u0=rand(M), v0=rand(M))
    function henon(U,V)
        Un = copy(U)
        Vn = copy(V)
        Un[1] = Un[M] = 0.5
        Vn[1] = Vn[M] = 0
        for m=2:M-1
            Un[m] = 1 - 1.45*(.5*U[m]+ .25*U[m-1] + .25U[m+1])^2 + 0.3*V[m]
            Vn[m] = U[m]
        end
        return Un, Vn
    end

    U = Vector{Vector{Float64}}(undef,N)
    V = Vector{Vector{Float64}}(undef,N)
    U[1] = u0
    V[1] = v0
    for n = 2:N
        U[n],V[n] = henon(U[n-1],V[n-1])
    end
    return U,V
end


M=100
N_train = 1000
p = 20
U,V = U, V = coupled_henon1D(M, N_train+p)

#Reconstruct this #
utrain = U[1:N_train]
vtrain = V[1:N_train]
utest  = U[N_train:N_train+p]
vtest  = V[N_train:N_train+p]

#Prediction
D = 2; τ = 1; B = 1; k = 1;
em = SpatioTemporalEmbedding(utrain,D,τ,B,k,ConstantBoundary(10.))
s_pred = temporalprediction(utrain,em, p)

begin
    figure()
    ax1 = subplot(311)
    #Original
    pcolormesh(utest)
    colorbar()
    # Make x-tick labels invisible
    setp(ax1[:get_xticklabels](), visible=false)
    title("1D Coupled Henon Map")


    ax2 = subplot(312, sharex = ax1, sharey = ax1)
    pcolormesh(s_pred)
    colorbar()
    setp(ax2[:get_xticklabels](), visible=false)
    title("Prediction ( Training States $N_train)")
    ylabel("t")

    #Error
    ax3 = subplot(313, sharex = ax1, sharey = ax1)
    ε = [abs.(utest[i]-s_pred[i]) for i=1:p+1]
    pcolormesh(ε, cmap="inferno")
    colorbar()
    title("Absolute Error")
    xlabel("m")
    tight_layout()
end
