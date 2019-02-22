# This example predicts the temporal evolution of a one-dimensional field
# U, along with a time vector T,
# which has to be represented as vectors of vectors.
# Where the field comes from does not matter, but to make the example
# runnable we load one of the test systems of `TimeseriesPrediction`.
#
# In this example we use the solution of Kuramoto Sivashinsky equation.
#
# Importantly, the results are compared with the "real" evolution of the
# system.
#
# In the plots, the x axis is space and y axis is time.

# ### Produce field U (Kuramoto Sivashinsky)
using PyPlot
using TimeseriesPrediction

testdir = dirname(dirname(pathof(TimeseriesPrediction)))*"/test"
@assert isdir(testdir)
include(testdir*"/ks_solver.jl")

Ntrain = 10000
p = 100
N = Ntrain + p

U, T = KuramotoSivashinsky(64, 22, N÷4, 0.25)
summary(U)

# ### Temporal prediction of field U
Q = length(U[1]) # spatial length
pool = U[1:Ntrain]
test = U[Ntrain:N]

γ = 10
τ = 1
B = 10
k = 1
ntype = FixedMassNeighborhood(4)
method = AverageLocalModel()

em = cubic_shell_embedding(pool, γ,τ,B,k,PeriodicBoundary())
pcaem= PCAEmbedding(pool,em)

@time pred = temporalprediction(pool,pcaem, p;ntype=ntype, method=method, progress = false)

err = [abs.(test[i]-pred[i]) for i=1:p+1]
println("Maximum error: ", maximum(maximum(e) for e in err))


# ### Plot the result

# Deduce field extremal values
vmax = max(maximum(maximum(s) for s in test),
           maximum(maximum(s) for s in pred))
vmin = min(minimum(minimum(s) for s in test),
           minimum(minimum(s) for s in pred))

# Transform data for imshow
ptest = cat(test..., dims = 2)
ppred = cat(pred..., dims = 2)
perr = cat(err..., dims = 2)

# plot plot plot
fig = figure(figsize=(8,8))
ax1 = subplot2grid((3,1), (0,0))
ax2 = subplot2grid((3,1), (1,0))
ax3 = subplot2grid((3,1), (2,0));

im1 = ax1[:imshow](ppred, cmap="viridis", vmin = vmin, vmax = vmax,
aspect = "auto", extent = (T[Ntrain], T[N], 1, Q))
im2 = ax2[:imshow](ptest, cmap="viridis", vmin = vmin, vmax = vmax,
aspect = "auto", extent = (T[Ntrain], T[N], 1, Q))
im3 = ax3[:imshow](perr, cmap="inferno", vmin = 0, vmax = vmax-vmin,
aspect = "auto", extent = (T[Ntrain], T[N], 1, Q))

for (j, (im, ax)) in enumerate(zip([im1,im2,im3], [ax1,ax2,ax3]))
    colorbar(im, ax = ax, fraction=0.04, pad=0.01)# format="%.1f")
    if j < 3
        ax[:set_xticklabels]([])
    end
end
ax1[:set_title]("Prediction")
ax2[:set_title]("Real evolution")
ax3[:set_title]("Absolute error")

ax2[:set_ylabel]("space")
ax3[:set_xlabel]("time")
tight_layout(w_pad=0.1, h_pad=0.00001)
#md savefig("ksprediction.png"); nothing # hide
#md # ![ksprediction](ksprediction.png)
