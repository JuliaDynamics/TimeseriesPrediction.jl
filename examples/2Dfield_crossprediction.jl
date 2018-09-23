#=
This example cross-predicts a field U from a field V.
Both fields have to be represented as vectors of matrices.
Where the fields come from does not matter, but to make the example
runnable we load one of the test systems of TimeseriesPrediction.

This example uses cubic shell embedding and a linear Barkley model.

Importantly, the results are compared with the "real" evolution of the
system.
=#

# %% Simulate a test System OR provide U, V
using PyPlot
using TimeseriesPrediction

testdir = dirname(dirname(pathof(TimeseriesPrediction)))*"/test"
@assert isdir(testdir)
include(testdir*"/system_defs.jl")

Nx = 50
Ny = 50
Tskip = 100
Ttrain = 500
Ttest = 10
T = Tskip +Ttrain + Ttest

U, V = barkley_const_boundary(T, Nx, Ny)

# %% Cross predict field U from field V
D = 2
τ = 1
B = 2
k = 1
bc = ConstantBoundary(20.0)

source_train = V[Tskip + 1:Tskip + Ttrain]
target_train = U[Tskip + 1:Tskip + Ttrain]
source_pred  = V[Tskip + Ttrain  - D*τ + 1:  T]
target_test  = U[Tskip + Ttrain  - D*τ + 1:  T]

em = cubic_shell_embedding(source_train, D,τ,B,k,bc)
pcaem = PCAEmbedding(source_train, em) # PCA speeds things up!

@time target_pred = crossprediction(source_train, target_train, source_pred, em)

err = [abs.(target_test[i]-target_pred[i]) for i=1:Ttest]

# %% Plot prediction

# Deduce field maximum values
source_max = maximum(maximum(s) for s in source_pred)
target_max = max(maximum(maximum(s) for s in target_test),
                 maximum(maximum(s) for s in target_pred))
source_min = minimum(minimum(s) for s in source_pred)
target_min = min(minimum(minimum(s) for s in target_test),
                 minimum(minimum(s) for s in target_pred))


for i in [1, length(err)÷2, length(err)]

    fig = figure(figsize=(10,10))
    ax1 = subplot2grid((2,2), (0,0))
    ax2 = subplot2grid((2,2), (0,1))
    ax3 = subplot2grid((2,2), (1,0))
    ax4 = subplot2grid((2,2), (1,1))
    im1 = ax1[:imshow](source_pred[i], cmap="viridis", vmin = source_min, vmax = source_max)
    im2 = ax2[:imshow](target_test[i], cmap="cividis", vmin = target_min, vmax = target_max)
    im3 = ax3[:imshow](target_pred[i], cmap="cividis", vmin = target_min, vmax = target_max)
    im4 = ax4[:imshow](err[i], cmap="inferno", vmin = 0, vmax = target_max - target_min)
    for (im, ax) in zip([im1,im2,im3,im4], [ax1,ax2,ax3,ax4])
        ax[:get_xaxis]()[:set_ticks]([])
        ax[:get_yaxis]()[:set_ticks]([])
        colorbar(im, ax = ax, fraction=0.046, pad=0.04)#, format="%.1f")
        # ax[:minorticks_off]()
    end
    ax1[:set_title]("Source")
    ax2[:set_title]("Target Test")
    ax3[:set_title]("Target Cross-Pred.")
    ax4[:set_title]("absolute error")
    tight_layout(w_pad=0.6, h_pad=0.00001)
    suptitle("frame $i")
end
