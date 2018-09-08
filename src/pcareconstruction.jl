export PCAEmbedding


struct PCAEmbedding{T,Φ,BC,X,Y} <: AbstractSpatialEmbedding{T,Φ,BC,X}
	stem::SpatioTemporalEmbedding{T,Φ,BC,Y}
	meanv::T
	covmat::Matrix{T}
	drmodel::PCA{T}
	tmp::Vector{T} #length Y
end

Base.show(io::IO, em::PCAEmbedding) = (show(io, em.stem); show(io, em.drmodel))

compute_pca(covmat::Matrix{T}, pratio, maxoutdim) where T=
	pcacov(covmat, T[]; maxoutdim=maxoutdim, pratio=pratio)

get_τmax(em::PCAEmbedding) = get_τmax(em.stem)
get_num_pt(em::PCAEmbedding) = get_num_pt(em.stem)

get_usable_idxs(em::SpatioTemporalEmbedding{T,Φ,PeriodicBoundary,X}) where {T,Φ,X} =
			CartesianIndices(em.whole)
get_usable_idxs(em::SpatioTemporalEmbedding{T,Φ,ConstantBoundary{C},X}) where {T,Φ,C,X} =
			CartesianIndices(em.inner)

outdim(em::PCAEmbedding{T,Φ,BC,X}) where {T,Φ,BC,X} = X

function PCAEmbedding(
		s::AbstractArray{<:AbstractArray{T,Φ}},
		stem::SpatioTemporalEmbedding{T,Φ,BC,Y};
		pratio   = 0.99,
		maxoutdim= 25,
		every::Int    = 1) where {T,Φ,BC,Y}
	meanv = mean(mean.(s))
	tsteps = (length(s) - get_τmax(stem))
	usable_idxs = get_usable_idxs(stem)
	num_pt = length(usable_idxs)
	L      = tsteps*num_pt

	recv = zeros(T,Y)
	covmat = zeros(T,Y,Y)
	for n = 1:every:L
		t = 1 + (n-1) ÷ num_pt
		α = usable_idxs[n-(t-1)*num_pt]
		stem(recv, s,t,α)
		recv  .-= meanv
		#covmat .+= recv' .* recv / L
		@inbounds for j=1:Y, i=1:Y
			covmat[i,j] += recv[i]*recv[j] / L
		end
	end
	drmodel = compute_pca(covmat,pratio, maxoutdim)
	X = size(drmodel.proj,2)
	tmp = zeros(Y)
	return PCAEmbedding{T,Φ,BC,X,Y}(stem, meanv, covmat, drmodel,tmp)
end

function (r::PCAEmbedding{T,Φ,BC,X,Y})(data, s, t, α) where {T,Φ,BC,X,Y}
	r.stem(r.tmp, s,t,α)
	r.tmp .-= r.meanv
	mul!(data,transpose(r.drmodel.proj),r.tmp)
end
