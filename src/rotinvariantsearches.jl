function preparatory_computation(em::PCAEmbedding, symmetry::Type{FullSymmetry})
    #precompute all ems and return vector of them
    stems = prepare_for_symmetry(em.stem, symmetry)
    map(stems) do stem
        typeof(em)(stem, em.meanv, em.covmat, em.drmodel, copy(em.tmp))
    end
end

function prepare_for_symmetry(em::SpatioTemporalEmbedding, symmetry::Type{FullSymmetry})
    βs = all_orientations(em.β)
    map(βs) do β
        newem = deepcopy(em)
        newem.β .= β
        newem
    end
end


function predict_frame!(s, spred, params, R, tree, ems::AbstractVector{<: AbstractSpatialEmbedding}; kwargs...)
    #New state that
    state = similar(spred[1])

    queries = gen_queries.(Ref(spred), ems) #now many queries for each point
    #Iterate over queries/ spatial points
    Threads.@threads for m=1:get_num_pt(ems[1])
        qs = map(queries -> queries[m], queries) #multiple queries

        #Find neighbors
        idxs_n,dists_n = neighborhood_and_distances(qs,R,tree,params.ntype)

        idxs = collect(Iterators.flatten(idxs_n))
        dists = collect(Iterators.flatten(dists_n))
        xnn = R[idxs]
        #Retrieve ynn
        ynn = map(idxs) do idx
            #Indices idxs are indices of R. Convert to indices of s
            t,α = convert_idx(idx,params.em)
            s[t+1][α]
        end
        state[m] = params.method(qs[1],xnn,ynn,dists)[1]
    end
    push!(spred,state)
end
