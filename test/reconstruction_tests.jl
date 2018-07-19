using Test
using TimeseriesPrediction

println("Reconstruction Tests")
@testset "STDelayEmbedding" begin
    for D=0:4, τ ∈ [1,10,20], B=1:3, k=1:3, c=10, Φ=1:4
        @testset "D=$D, τ=$τ, B=$B, k=$k, Φ=$Φ" begin
                emb = STDelayEmbedding(D,τ,B,k,c, Φ)
                #Check Embedding Dimension X
                X = (D+1)*(2B+1)^Φ
                @test typeof(emb) == STDelayEmbedding{X,Φ}
                @test length(emb.delays) == X

                #Check tmax
                @test emb.tmax == D*τ

                #check spatial extent
                @test emb.spex == B*k * ones(Φ,2)

                #boundary value
                @test emb.boundary == c
        end
    end

    #Checking a few individual Reconstructions
    @testset "Order of rec. points" begin
        D=1; τ=1; B=1; k=1; c=10; Φ=2
        data = [reshape(1+i:9+i, 3,3) for i∈[0,9]]
        emb = STDelayEmbedding(D,τ,B,k,c,Φ)

        pt_in_space = CartesianIndices((3,3))
        t = 1
        α = CartesianIndex(2,2)
        @test 1:18 == emb(data, t, α, pt_in_space)
        α = CartesianIndex(1,1)
        rec = [c,c,c,c,1,2,c,4,5,c,c,c,c,10,11,c,13,14]
        @test rec == emb(data, t, α, pt_in_space)
    end
end
