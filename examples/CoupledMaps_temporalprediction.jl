# This example predicts the temporal evolution of a field
# consisting of coupled discrete systems, U,
# which has to be represented as vectors of vectors.
# Where the field comes from does not matter, but to make the example
# runnable we load one of the test systems of TimeseriesPrediction.
#
# In this example we use many coupled henon maps, which can be thought of
# as a field, where each map is a point in space.
#
# Importantly, the results are compared with the "real" evolution of the
# system.
#
# In the plots, the x axis is space and y axis is time.

# ### Simulate a test system
using PyPlot
using TimeseriesPrediction

testdir = dirname(dirname(pathof(TimeseriesPrediction)))*"/test"
@assert isdir(testdir)
include(testdir*"/system_defs.jl")

M = 100 # how many maps to couple, i.e. how much "space" to cover
N = 1000
p = 20
U,V = coupled_henon1D(M, N+p)

# ### Temporal prediction of field U
pool = U[1:N]
test  = U[N:N+p]

γ = 2; τ = 1; B = 1; k = 1;
em = cubic_shell_embedding(pool,γ,τ,B,k,ConstantBoundary(10.))

pred = temporalprediction(pool,em, p; progress = false)

err = [abs.(test[i]-pred[i]) for i=1:p+1]

println("Maximum error: ", maximum(maximum(e) for e in err))

# ### Plot prediction

# Deduce field maximum values
vmax = max(maximum(maximum(s) for s in test),
           maximum(maximum(s) for s in pred))
vmin = min(minimum(minimum(s) for s in test),
           minimum(minimum(s) for s in pred))

# plot plot plot
fig = figure(figsize=(8,8))
ax1 = subplot2grid((3,1), (0,0))
ax2 = subplot2grid((3,1), (1,0))
ax3 = subplot2grid((3,1), (2,0))

ptest = reverse(Matrix(cat(test..., dims = 2)'), dims = 1)
ppred = reverse(Matrix(cat(pred..., dims = 2)'), dims = 1)
perr = reverse(Matrix(cat(err..., dims = 2)'), dims = 1)

im1 = ax1[:imshow](ppred, cmap="viridis", vmin = vmin, vmax = vmax)
im2 = ax2[:imshow](ptest, cmap="viridis", vmin = vmin, vmax = vmax)
im3 = ax3[:imshow](perr, cmap="inferno", vmin = 0, vmax = vmax-vmin)

for (im, ax) in zip([im1,im2,im3], [ax1,ax2,ax3])
    colorbar(im, ax = ax, fraction=0.02, pad=0.01)# format="%.1f")
end
ax1[:set_title]("Prediction")
ax2[:set_title]("Real evolution")
ax3[:set_title]("Absolute error")

ax2[:set_ylabel]("time")
ax3[:set_xlabel]("space")
tight_layout(w_pad=0.1, h_pad=0.00001)
