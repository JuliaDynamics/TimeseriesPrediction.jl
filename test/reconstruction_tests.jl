using Test
using TimeseriesPrediction
using PrincipalComponentAnalysis

println("Reconstruction Tests")
@testset "STDelayEmbedding" begin
    for D=[0,4], τ ∈ [1,20], B=[1,3], k=[1,3], c=10, Φ=1:4
        @testset "D=$D, τ=$τ, B=$B, k=$k, Φ=$Φ" begin
                #Ugly way of creating Φ dim array
                s = [rand(Float64,([10 for i=1:Φ]...,)) for i=1:10]
                emb = STDelayEmbedding(s,D,τ,B,k,c)
                #Check Embedding Dimension X
                X = (D+1)*(2B+1)^Φ
                @test typeof(emb) <: STDelayEmbedding{Float64,Φ,X}
                @test length(emb.τ) == X
                @test length(emb.β) == X

                #check spatial extent
                @test emb.inner == TimeseriesPrediction.Region(
                                ([B*k+1 for i=1:Φ]...,),
                                ([10-B*k for i=1:Φ]...,))

                @test emb.outer == TimeseriesPrediction.Region((ones(Int,Φ)...,),(10ones(Int,Φ)...,) )

                #boundary value
                @test emb.boundary == c
        end
    end

    #Checking a few individual Reconstructions
    @testset "Order of rec. points" begin
        D=1; τ=1; B=1; k=1; c=10; Φ=2
        data = [reshape(1+i:9+i, 3,3) for i∈[0,9]]
        emb = STDelayEmbedding(data, D,τ,B,k,c)
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
    for D=10, τ=1, B=1,k=1, c=false, Φ=2
        em = STDelayEmbedding(U,D,τ,B,k,c)
        R = DynamicalSystemsBase.reconstruct(U, em)
        Rmat = Matrix(R)
        drmodel = fit(PCA,Rmat)
        pcaem = PCAEmbedding(U,em)
        #Compare Outdim
        @test outdim(drmodel) == outdim(pcaem.drmodel)
        #Compare singular values
        @test isapprox(principalvars(drmodel), principalvars(pcaem.drmodel), atol=5e-2)
         #Compare Projection Matrices
        @test isapprox(drmodel.proj, pcaem.drmodel.proj, atol=5e-2)
    end
end
