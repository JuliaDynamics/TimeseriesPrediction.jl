using TimeseriesPrediction
using PyPlot
using PyCall
@pyimport numpy as np

cd(@__DIR__);

#Generate data using https://github.com/jswhit/pyks
# ( after adding the relevant lines for saving results to file)
u = np.load("KSdata.npy")'
Ntrain = 10000
Ntest = 150
N = Ntrain + Ntest

utrain = [u[:,i] for i ∈ 1:Ntrain]
utest  = [u[:,i] for i ∈ Ntrain:N]

D = 1
τ = 1
B = 5
k = 1
c = false
w = (0,0)
nn=4; ntype = FixedMassNeighborhood(nn)


nw=3;method = AverageLocalModel(nw)

@time upred = localmodel_stts(utrain, D, τ, Ntest, B, k;
 ntype=ntype, method=method, boundary=c, weighting=w)

fname = "KS$(Ntest)_tr$(Ntrain)_D$(D)_τ$(τ)_B$(B)_k$(k)_w$(w)"
function meshgr(x,y)
    X = repmat(1.:x, 1, y)
    Y = repmat(1.:y, 1, x)'
    return X,Y
end
begin
    X,Y = meshgr(length(utest[1]),length(utest))
    Y .*= 0.25 * 0.1
    figure()
    ax1 = subplot(311)
    #Original
    pcolormesh(X,Y,hcat(utest...))
    colorbar()
    # Make x-tick labels invisible
    setp(ax1[:get_xticklabels](), visible=false)
    title("1D Kuramoto Sivashinsky")


    #Prediction
    ax2 = subplot(312, sharex = ax1, sharey = ax1)
    pcolormesh(X,Y,hcat(upred...))
    colorbar()
    setp(ax2[:get_xticklabels](), visible=false)
    title("Prediction ( Training States $Ntrain)")
    ylabel("Λ t")

    #Error
    ax3 = subplot(313, sharex = ax1, sharey = ax1)
    ε = [abs.(utest[i]-upred[i]) for i=1:Ntest+1]
    pcolormesh(X,Y,hcat(ε...), cmap="inferno")
    colorbar()
    title("Absolute Error")
    xlabel("m")
    tight_layout()
    savefig(fname *".png")
end












# function kur_siv(M, T)
#     Δx = 0.1
#     Δt = 1
#     y = rand(M)
#     y -= mean(y)
#     tr = [copy(y)]
#     Σ1 = zeros(M,2)
#     Σ2 = zeros(M,2)
#     Σ4 = zeros(M,2)
#
#     r = 1
#     s = 2
#     for t ∈ 1:T
#         Σ1[:,s] .= 0
#         Σ2[:,s] .= 0
#         Σ4[:,s] .= 0
#         for m ∈ 1:M
#             Σ1[mod(m-1-1, M) + 1, s] += y[m]/2 / Δx
#             Σ1[mod(m-1+1, M) + 1, s] -= y[m]/ 2 / Δx
#
#             Σ2[mod(m-1-1, M) + 1, s] += y[m]/Δx^2
#             Σ2[mod(m-1+0, M) + 1, s] -= 2y[m]/Δx^2
#             Σ2[mod(m-1+1, M) + 1, s] += y[m]/Δx^2
#
#             Σ4[mod(m-1-2, M) + 1, s] += y[m]/Δx^4
#             Σ4[mod(m-1-1, M) + 1, s] -= 4y[m]/Δx^4
#             Σ4[mod(m-1+0, M) + 1, s] += 6y[m]/Δx^4
#             Σ4[mod(m-1+1, M) + 1, s] -= 4y[m]/Δx^4
#             Σ4[mod(m-1+2, M) + 1, s] += y[m]/Δx^4
#
#             #new state
#             y[m] -= Δt * (Σ4[m, r] + 0.2Σ2[m, r] + 0.1Σ1[m, r] *  y[m])
#         end
#         s, r = r, s
#         push!(tr, copy(y))
#     end
#     return tr
# end
#
# tr = kur_siv(100, 10)
# mat = hcat(tr...)
# pcolormesh(tr)
