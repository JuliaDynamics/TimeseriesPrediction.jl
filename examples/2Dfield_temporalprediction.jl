# This example predicts the temporal evolution of a field U,
# which has to be represented as vectors of matrices.
# Where the field comes from does not matter, but to make the example
# runnable we load one of the test systems of TimeseriesPrediction.
#
# This example uses light cone embedding and a nonlinear Barkley model.
#
# Importantly, the results are compared with the "real" evolution of the
# system.

# ### Simulate a test system
using PyPlot
using TimeseriesPrediction

testdir = dirname(dirname(pathof(TimeseriesPrediction)))*"/test"
@assert isdir(testdir)
include(testdir*"/system_defs.jl")

Ttrain = 600
Ttest = 10
T = Ttrain + Ttest

U, V = barkley(T; tskip=100, size=(50,50))
summary(U)

# ### Temporal prediction of field U
D = 2
τ = 1
r₀ = 1
c = 1
bc = PeriodicBoundary()

pool = U[1 : Ttrain]
test  = U[ Ttrain : T]

em = light_cone_embedding(pool, D,τ,r₀,c,bc)
pcaem = PCAEmbedding(pool, em; maxoutdim=5) # PCA speeds things up!

@time pred = temporalprediction(pool, em, Ttest; progress = false)

err = [abs.(test[i]-pred[i]) for i=1:Ttest+1]

println("Maximum error: ", maximum(maximum(e) for e in err))

# ### Plot prediction

# Deduce field maximum values
vmax = max(maximum(maximum(s) for s in test),
           maximum(maximum(s) for s in pred))
vmin = min(minimum(minimum(s) for s in test),
           minimum(minimum(s) for s in pred))

# plot plot plot
for i in [1, length(err)÷2, length(err)]

    fig = figure(figsize=(10,3))
    ax1 = subplot2grid((1,3), (0,0))
    ax2 = subplot2grid((1,3), (0,1))
    ax3 = subplot2grid((1,3), (0,2))

    im1 = ax1[:imshow](pred[i], cmap="viridis", vmin = vmin, vmax = vmax)
    im2 = ax2[:imshow](test[i], cmap="viridis", vmin = vmin, vmax = vmax)
    im3 = ax3[:imshow](err[i], cmap="inferno", vmin = 0, vmax = vmax-vmin)
    for (im, ax) in zip([im1,im2,im3], [ax1,ax2,ax3])
        ax[:get_xaxis]()[:set_ticks]([])
        ax[:get_yaxis]()[:set_ticks]([])
        colorbar(im, ax = ax, fraction=0.046, pad=0.04)#, format="%.1f")
    end
    ax1[:set_title]("Prediction")
    ax2[:set_title]("Real evolution")
    ax3[:set_title]("Absolute error")
    suptitle("frame $i")
    tight_layout(w_pad=0.6, h_pad=0.00001)
    subplots_adjust(top=0.75)
end
#md savefig("barkley_tempo.png"); nothing # hide
#md # ![barkley_tempo](barkley_tempo.png)
