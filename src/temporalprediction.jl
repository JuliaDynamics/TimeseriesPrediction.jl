using NearestNeighbors
using Parameters
export KDTree
export temporalprediction

function working_ts(s,em)
    L = length(s)
    τmax = get_τmax(em)
    return s[L-τmax : L]
end

function gen_queries(s,em)
    L = length(s)
    τmax = get_τmax(em)
    s_slice = view( s, L-τmax:L)
    return reconstruct(s_slice, em)
end

function convert_idx(idx, em)
    τmax = get_τmax(em)
    num_pt = get_num_pt(em)
    t = 1 + (idx-1) ÷ num_pt + get_τmax(em)
    α = 1 + (idx-1) % num_pt
    return t,α
end

cut_off_beginning!(s,em) = deleteat!(s, 1:get_τmax(em))

@with_kw struct PredictionParameters{T,Φ,BC,X}
    em::AbstractSpatialEmbedding{T,Φ,BC,X}
    method::AbstractLocalModel = AverageLocalModel(ω_safe)
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3)
    treetype::DataType = KDTree
end


###########################################################################################
#                        Iterated Time Series Prediction                                  #
###########################################################################################


function temporalprediction(s,
    em::AbstractSpatialEmbedding{T,Φ},
    tsteps;
    ttype=KDTree,
    method = AverageLocalModel(ω_safe),
    ntype = FixedMassNeighborhood(3),
    progress=true) where {T,Φ}

    params = PredictionParameters(em, method, ntype, ttype)
    return temporalprediction(params, s, tsteps; progress=progress)
end

function temporalprediction(params, s, tsteps; progress=true)
    progress && println("Reconstructing")
    R = reconstruct(s,params.em)

    #Prepare tree but remove the last reconstructed states first
    progress && println("Creating Tree")
    L = length(R)
    M = get_num_pt(params.em)

    tree = params.treetype(R[1:L-M])
    temporalprediction(params, s, tsteps, R, tree; progress=progress)
end



function temporalprediction(params, s,tsteps, R, tree; progress=true) where {T, Φ, BC, X}
    em = params.em
    @assert outdim(em) == size(R,2)
    num_pt = get_num_pt(em)
    #New state that will be predicted, allocate once and reuse
    state = similar(s[1])

    #End of timeseries to work with
    spred = working_ts(s,em)

    for n=1:sol.timesteps
        progress && println("Working on Frame $(n)/$(sol.timesteps)")
        queries = gen_queries(spred, em)

        #Iterate over queries/ spatial points
        for m=1:num_pt
            q = queries[m]

            #Find neighbors
            idxs,dists = neighborhood_and_distances(q,R,tree,params.ntype)

            xnn = R[idxs]
            #Retrieve ynn
            ynn = map(idxs) do idx
                #Indices idxs are indices of R. Convert to indices of s
                t,α = convert_idx(idx,em)
                s[t+1][α]
            end
            state[m] = params.method(q,xnn,ynn,dists)[1]
        end
        push!(spred,copy(state))
    end

    cut_off_beginning!(spred,em)
    return spred
end
