using Test
using TimeseriesPrediction
#using PrincipalComponentAnalysis
using Statistics, LinearAlgebra

println("Reconstruction Tests")
@testset "SpatioTemporalEmbedding" begin
    for D=[0,4], τ ∈ [1,20], B=[1,3], k=[1,3], Φ=1:4
        BC=ConstantBoundary{10}
        @testset "D=$D, τ=$τ, B=$B, k=$k, Φ=$Φ" begin
                #Ugly way of creating Φ dim array
                s = [rand(Float64,([10 for i=1:Φ]...,)) for i=1:10]
                emb = SpatioTemporalEmbedding(s,D,τ,B,k,BC)
                #Check Embedding Dimension X
                X = (D+1)*(2B+1)^Φ
                @test typeof(emb) <: SpatioTemporalEmbedding{Float64,Φ,BC,X}
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
        D=1; τ=1; B=1; k=1; c=10; BC=ConstantBoundary{c}; Φ=2
        data = [reshape(1+i:9+i, 3,3) for i∈[0,9]]
        emb = SpatioTemporalEmbedding(data, D,τ,B,k,BC)
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


include("system_defs.jl")

println("Testing PCA Functions")
@testset "PCAEmbedding" begin
    U,V = barkley_periodic_boundary_nonlin(500,50,50)
    for D=5, τ=1, B=1,k=1
        BC = PeriodicBoundary
        em = SpatioTemporalEmbedding(U,D,τ,B,k,BC)
        R = TimeseriesPrediction.reconstruct(U, em)
        meanv = mean(mean.(U))
        R .-= meanv
        covmat = Statistics.covzm(R,2)
        pcaem = PCAEmbedding(U,em)
        #Compare cov matrices
        @test norm(pcaem.covmat-covmat) < 1e-2
        #Compare Outdim
        drmodel = fit(PCA,R; mean=0)
        @test outdim(drmodel) == outdim(pcaem.drmodel)
        #Compare singular values
        @test all(isapprox.(TimeseriesPrediction.principalvars(drmodel), TimeseriesPrediction.principalvars(pcaem.drmodel), atol=0.1))

    end
end
