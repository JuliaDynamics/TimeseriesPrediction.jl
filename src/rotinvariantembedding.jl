using Setfield
export RotationallyInvariantEmbedding

flip(α::CartesianIndex, dim::Int) = @set α.I[dim] *= -1
flip(α::CartesianIndex) = -α

function swap(α::CartesianIndex, dim1, dim2)
	tmp = α.I[dim1]
	α = @set α.I[dim1] = α.I[dim2]
	@set α.I[dim2] = tmp
end

all_orientations(βs::Vector{CartesianIndex{1}}) = [ βs, flip.(βs)]

all_orientations(βs::Vector{CartesianIndex{2}}) =
	[ βs, flip.(βs, 1),
			flip.(βs, 2),
			flip.(βs),
			swap.(βs, 1, 2),
			map(β -> swap( flip(β, 1), 1,2), βs),
			map(β -> swap( flip(β, 2), 1,2), βs),
			map(β -> swap( flip(β), 1,2), βs)]



struct RotationallyInvariantEmbedding{Φ, BC, X} <: AbstractSpatialEmbedding{Φ, BC, X}
	τ::Vector{Int}
	β_groups::Vector{Vector{CartesianIndex{Φ}}}
	inner::Region{Φ}
	whole::Region{Φ}
	boundary::BC
	# Order of symmetry ops in β_groups for 2D, for 1D only no 1 & 2
	# 1 : identity
	# 2 : flip dim 1
	# 3 : flip dim 2
	# 4 : flip dim 1 & 2
	# 5 : swap (1,2) → (2,1)
	# 6 : (a,b) → (b, -a)
	# 7 : (a,b) → (-b, a)
	# 8 : (a,b) → -(b, a)

end

function RotationallyInvariantEmbedding(em::SpatioTemporalEmbedding{Φ,BC,X}) where {Φ,BC,X}
	β_groups = all_orientations(em.β)
	RotationallyInvariantEmbedding{Φ,BC,X}(em.τ, β_groups, em.inner, em.whole, em.boundary)
end


function compute_gradient_at(U, α::CartesianIndex{1}) where T
	(U[α+CartesianIndex(1)] - U[α-CartesianIndex(1)])
end

function compute_gradient_at(U, α::CartesianIndex{2})
	g1 = g2 = zero(eltype(U))
	kernel1 = @SMatrix [ -1 -1 -1; 0 0 0; 1 1 1]
	kernel2 = @SMatrix [ -1 0 1; -1 0 1; -1 0 1]
	@inbounds for j = -1:1, i = -1:1
		u = U[α + CartesianIndex(i,j)]
		g1 += u*kernel1[i+2,j+2]
		g2 += u*kernel2[i+2,j+2]
	end
	return g1, g2
end
# function compute_gradient_at(U, α::CartesianIndex{2}, kr=1)
# 	a1, a2 = α.I
# 	window = U[a1-kr:a1+kr, a2-kr:a2+kr]
# 	kernel1 = @SMatrix [ 1 0 -1; 1 0 -1; 1 0 -1]
# 	kernel2 = @SMatrix [ 1 1 1; 0 0 0; -1 -1 -1]
# 	g1 = sum(kernel1 .* window)
# 	g2 = sum(kernel2 .* window)
# 	return g1, g2
# end
# function compute_gradient_at(U, α::CartesianIndex{2})
# 	offsets1 = (CartesianIndex(1,1), CartesianIndex(0,1), CartesianIndex(-1,1))
# 	offsets2 = (CartesianIndex(1,1), CartesianIndex(1,0), CartesianIndex(1,-1))
# 	g1 = zero(eltype(U))
# 	g2 = zero(eltype(U))
# 	foreach(idx -> g1 += U[α+idx] - U[α-idx], offsets1)
# 	foreach(idx -> g2 += U[α+idx] - U[α-idx], offsets2)
# 	return g1, g2
# end

choose_orientation(em, gradient::Real) = gradient > 0 ? em.β_groups[1] : em.β_groups[2]

# Choose correct orientation with just 3 comparisons
function choose_orientation(em, gradient::NTuple)
	g1, g2 = gradient
	idx = 1
	g1 < 0  && ( idx += 1 ; g1 = -g1 )
	g2 < 0  && ( idx += 2 ; g2 = -g2 )
	g1 < g2 && ( idx += 4 )
	return em.β_groups[idx]
end

# function (r::RotationallyInvariantEmbedding{Φ,BC,X})(rvec,s,t,α) where {Φ,BC,X}
# 	bw = BoundaryWrapper(r, s)
# 	# Find correct orientation
# 	bw_t = BoundaryWrapper(r, s[t])
# 	gradient = compute_gradient_at(bw_t, α)
# 	βs = choose_orientation(r, gradient)
# 	if α in r.inner
# 		@inbounds for n=1:X
# 			rvec[n] = s[ t + r.τ[n] ][ α + βs[n] ]
# 		end
# 	else
# 		@inbounds for n=1:X
# 			rvec[n] = bw[ t + r.τ[n], α + βs[n]]
# 		end
# 	end
# 	return nothing
# end
function (r::RotationallyInvariantEmbedding{Φ,BC,X})(rvec,s,t,α) where {Φ,BC,X}
	bw = BoundaryWrapper(r, s)
	if α in r.inner
		gradient = compute_gradient_at(s[t], α)
		βs = choose_orientation(r, gradient)
		@inbounds for n=1:X
			rvec[n] = s[ t + r.τ[n] ][ α + βs[n] ]
		end
	else
		# Find correct orientation
		bw_t = BoundaryWrapper(r, s[t])
		gradient = compute_gradient_at(bw_t, α)
		βs = choose_orientation(r, gradient)
		@inbounds for n=1:X
			rvec[n] = bw[ t + r.τ[n], α + βs[n]]
		end
	end
	return nothing
end
