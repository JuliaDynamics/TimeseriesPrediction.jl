using Test
using TimeseriesPrediction
using Statistics, LinearAlgebra
import MultivariateStats
const MS = MultivariateStats

println("Reconstruction Tests")
@testset "cubic_shell_embedding" begin
    for γ=[0,4], τ ∈ [1,20], B=[1,3], k=[1,3], Φ=1:4
        BC=ConstantBoundary(10.)
        @testset "γ=$γ, τ=$τ, B=$B, k=$k, Φ=$Φ" begin
                #Ugly way of creating Φ dim array
                s = [rand(Float64,([10 for i=1:Φ]...,)) for i=1:10]
                emb = cubic_shell_embedding(s,γ,τ,B,k,BC)
                @test emb ==  SpatioTemporalEmbedding(s,(γ=γ,τ=τ,B=B,k=k,bc=BC))
                #Check Embedding Dimension X
                X = (γ+1)*(2B+1)^Φ
                @test typeof(emb) <: SpatioTemporalEmbedding{Φ,ConstantBoundary{Float64},X}
                @test length(emb.τ) == X
                @test length(emb.β) == X

                #check spatial extent
                @test emb.inner == TimeseriesPrediction.Region(
                                ([B*k+1 for i=1:Φ]...,),
                                ([10-B*k for i=1:Φ]...,))

                @test emb.whole == TimeseriesPrediction.Region((ones(Int,Φ)...,),(10ones(Int,Φ)...,) )
        end
    end

    #Checking a few individual Reconstructions
    @testset "Order of rec. points" begin
        γ=1; τ=1; B=1; k=1; c=10; BC=ConstantBoundary(c); Φ=2
        data = [reshape(1+i:9+i, 3,3) for i∈[0,9]]
        emb = cubic_shell_embedding(data, γ,τ,B,k,BC)
        t = 1
        α = CartesianIndex(2,2)
        rvec = zeros(18);
        emb(rvec, data, t, α)
        @test 1:18 == rvec
        α = CartesianIndex(1,1)
        rec = [c,c,c,c,1,2,c,4,5,c,c,c,c,10,11,c,13,14]
        emb(rvec, data, t, α)
        @test rec == rvec
    end
end

@testset "light_cone_embedding 1D" begin
    Φ = 1
    bc=ConstantBoundary(10.)
    s = [rand(Float64,([10 for i=1:Φ]...,)) for i=1:10]
    CI = CartesianIndex

    em = light_cone_embedding(s, 0, 1, 0, 1, bc)
    @test em.β == [CI(0)]

    em = light_cone_embedding(s, 0, 1, 1, 1, bc)
    @test em.β == CI.(-1:1)

    em = light_cone_embedding(s, 0, 1, 2, 1, bc)
    @test em.β == CI.(-2:2)

    em = light_cone_embedding(s, 0, 1, 2, 2, bc)
    @test em.β == CI.(-2:2)

    em = light_cone_embedding(s, 0, 1, 0, 1, bc)
    @test em.β == [CI(0)]

    em = light_cone_embedding(s, 1, 1, 2, 1, bc)
    @test em.β == vcat(CI.(-3:3), CI.(-2:2))

    em = light_cone_embedding(s, 1, 2, 2, 1, bc)
    @test em.β == vcat(CI.(-4:4), CI.(-2:2))

    em = light_cone_embedding(s, 1, 1, 2, 2, bc)
    @test em.β == vcat(CI.(-4:4), CI.(-2:2))

    em = light_cone_embedding(s, 2, 2, 2, 1, bc)
    @test em.β == vcat(CI.(-6:6), CI.(-4:4), CI.(-2:2))

    @test light_cone_embedding(s, 2, 2, 2, 1, bc) == STE(s,(γ=2, τ=2, r=2, c=1, bc=bc))
end

@testset "light_cone_embedding 2D" begin
    Φ = 2
    bc=ConstantBoundary(10.)
    s = [rand(Float64,([10 for i=1:Φ]...,)) for i=1:10]

    CI = CartesianIndex
    em = light_cone_embedding(s, 0, 1, 0, 0, bc)
    @test em.β == [CI(0,0)]

    em = light_cone_embedding(s, 0, 1, 0, 1, bc)
    @test em.β == [CI(0,0)]

    em = light_cone_embedding(s, 0, 20, 1, 10, bc)
    @test CI(0,1) ∈ em.β
    @test CI(0,0) ∈ em.β
    @test CI(0,-1) ∈ em.β
    @test CI(1,0) ∈ em.β
    @test CI(-1,0) ∈ em.β

    em = light_cone_embedding(s, 0, 5, 1.5, 17, bc)
    @test CI(0,1) ∈ em.β
    @test CI(0,0) ∈ em.β
    @test CI(0,-1) ∈ em.β
    @test CI(1,0) ∈ em.β
    @test CI(-1,0) ∈ em.β
    @test CI(-1,-1) ∈ em.β
    @test CI(-1,1) ∈ em.β
    @test CI(1,1) ∈ em.β
    @test CI(1,-1) ∈ em.β
end

include("system_defs.jl")

println("Testing PCA Functions")
@testset "PCAEmbedding" begin
    U,V = barkley(400;tskip=200)
    let γ=5, τ=1, B=1,k=1
        BC = PeriodicBoundary()
        em = cubic_shell_embedding(U,γ,τ,B,k,BC)
        RR = reconstruct(U, em)
        meanv = mean(mean.(U))
        R = reshape(reinterpret(Float64, RR.data), (size(RR)[2], size(RR)[1]))
        R .-= meanv
        covmat = Statistics.cov(R,dims =2)
        pcaem = PCAEmbedding(U,em)
        #Compare cov matrices
        @test norm(pcaem.covmat-covmat) < 1e-4
        #Compare Outdim
        drmodel = MS.fit(MultivariateStats.PCA,R)
        @test MS.outdim(drmodel) == MS.outdim(pcaem.drmodel)
        #Compare singular values
        @test MS.principalvars(drmodel) ≈ MS.principalvars(pcaem.drmodel) atol=0.5

    end
end
