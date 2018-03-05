using DynamicalSystemsBase

export E1,E2

function E(s::AbstractVector{T},D,τ) where T
    #Sum over all a(i,d) of the Ddim Reconstructed space
    method = FixedMassNeighborhood(1)
    R1 = Reconstruction(s,D+1,τ)
    tree1 = KDTree(R1)
    R2 = Reconstruction(s,D,τ)

    e = 0.
    for i=1:length(R1)
        #Find nearest neighbor of R1[i]
        j = DynamicalSystemsBase.neighborhood(i,R1[i], tree1,method)[1]
        e += norm(R1[i]-R1[j], Inf) / norm(R2[i]-R2[j], Inf) / length(R1)
    end
    return e
end

function E1(s,D,τ)
    return E(s,D+1,τ)/E(s,D,τ)
end


function E2(s::AbstractVector{T},D,τ) where T
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    method = FixedMassNeighborhood(1)

    #Calculate E* for Dimension D+1
    R1 = Reconstruction(s,D+1,τ)
    tree1 = KDTree(R1[1:end-1-τ])
    Es1 = 0.
    for i=1:length(R1)-τ
        j = DynamicalSystemsBase.neighborhood(i,R1[i], tree1,method)[1][1]
        #Es1 += abs(R1[i+D*τ][1] - R1[j+D*τ][1]) / length(R1)
        Es1 += abs(R1[i+τ][end] - R1[j+τ][end]) / length(R1)
        #The second approach is equivalent to the first (only 1 τ)
        #But does not require to shorten the KDTree as much.
    end

    #Calculate E* for Dimension D
    R2 = Reconstruction(s,D,τ)
    tree2 = KDTree(R2[1:end-1-τ])
    Es2 = 0.
    for i=1:length(R2)-τ
        j = DynamicalSystemsBase.neighborhood(i,R2[i], tree2,method)[1][1]
        #Es2 += abs(R2[i+D*τ][1] - R2[j+D*τ][1]) / length(R2)
        Es2 += abs(R2[i+τ][end] - R2[j+τ][end]) / length(R2)
    end
    return Es1/Es2
end
