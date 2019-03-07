using Test
using TimeseriesPrediction

@testset "InhomogeneousEmbedding" begin
    #Test reconstruction
    γ = 0; τ = 1; r = 2; c = 0; bc = ConstantBoundary(10.);
    dummy_data = [rand(10,10) for _ in 1:10]
    em = light_cone_embedding(dummy_data, γ, τ, r, c, bc)

    ff = TimeseriesPrediction.linear_weights(size(dummy_data[1]))
    iem = InhomogeneousEmbedding(em, ff)

    R1 = reconstruct(dummy_data, em)
    R2 = reconstruct(dummy_data, iem)
    for (r1,r2) in zip(R1,R2)
        @test r1 == r2[1:end-2]
    end

    #Test linear weights function

    ff = TimeseriesPrediction.linear_weights(size(dummy_data[1]))
    iem = InhomogeneousEmbedding(em, ff)
    @test outdim(iem) == outdim(em)+2
    @test [-1, 1] == ff(CartesianIndex(1,10))
    ff = TimeseriesPrediction.linear_weights(size(dummy_data[1]),2)
    iem = InhomogeneousEmbedding(em, ff)
    @test outdim(iem) == outdim(em)+1
    @test [1] == ff(CartesianIndex(1,10))
    @test [-1] == ff(CartesianIndex(1,1))

end
