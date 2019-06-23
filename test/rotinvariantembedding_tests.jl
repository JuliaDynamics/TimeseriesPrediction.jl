using Test
using TimeseriesPrediction
using TimeseriesPrediction: all_orientations, flip, swap, choose_orientation, compute_gradient_at



################################################################################
#               Testing the reorientations as func of gradient                 #
################################################################################

@testset "Testing Reorientation" begin
    #1D
    β = [CartesianIndex(3), CartesianIndex(4)]
    em_test = (β_groups=all_orientations(β),)
    @test choose_orientation(em_test, 1) == β
    @test choose_orientation(em_test, -1) == -1 .* β
    # 2D
    β = [ CartesianIndex(1,2), CartesianIndex(2,1) ]
    em_test = (β_groups=all_orientations(β),)

    @test choose_orientation(em_test, ( 2, 1)) == β
    @test choose_orientation(em_test, (-2, 1)) == [CartesianIndex(-1, 2), CartesianIndex(-2, 1)]
    @test choose_orientation(em_test, ( 2,-1)) == [CartesianIndex( 1,-2), CartesianIndex( 2,-1)]
    @test choose_orientation(em_test, (-2,-1)) == [CartesianIndex(-1,-2), CartesianIndex(-2,-1)]
    @test choose_orientation(em_test, ( 1, 2)) == [CartesianIndex( 2, 1), CartesianIndex( 1, 2)]
    @test choose_orientation(em_test, (-1, 2)) == [CartesianIndex( 2,-1), CartesianIndex( 1,-2)]
    @test choose_orientation(em_test, ( 1,-2)) == [CartesianIndex(-2, 1), CartesianIndex(-1, 2)]
    @test choose_orientation(em_test, (-1,-2)) == [CartesianIndex(-2,-1), CartesianIndex(-1,-2)]
end


@testset "Computing Gradients" begin
    @testset "1D Gradients" begin
        X = -5:0.1:5
        U = 0.1 * X.^2 .- 0.5

        for n = 2:21
            @test compute_gradient_at(U, CartesianIndex(n)) ≈ U[n+1] - U[n-1]
        end
    end
    @testset "2D Gradients" begin
        X = range(0., stop=2π, length=50)
        U = sin.(X) .* cos.(X')
        for j = 2:49, i = 2:49
            @test compute_gradient_at(U, CartesianIndex(i,j))[1] ≈
                (U[i+1, j-1] + U[i+1, j] + U[i+1, j+1] - U[i-1, j-1] - U[i-1, j] - U[i-1, j+1])
            @test compute_gradient_at(U, CartesianIndex(i,j))[2] ≈
                 (U[i-1, j+1] + U[i, j+1] + U[i+1, j+1] - U[i-1, j-1] - U[i, j-1] - U[i+1, j-1])
        end
    end
end


@testset "Reconstruction with Rotation Invariance" begin
    @testset "1D Fields" begin
        X = collect(range(0., stop=1, length=50))
        data = [X for _ = 1:4]

        em = light_cone_embedding(data, 3, 1, 2, 0, ConstantBoundary(10.))
        rem = RotationallyInvariantEmbedding(em)
        R1 = reconstruct(data, em)
        R2 = reconstruct(data, rem)

        @test R1[1] != R2[1]   #Left boundary gets flipped
        for n = 2:length(R1)    #All others should stay unchanged
            @test R1[n] == R2[n]
        end
    end
    @testset "2D Fields" begin
        X = collect(range(0., stop=1, length=50))
        Y = 0.5X'

        data = [X.+Y for _ = 1:4]

        em = light_cone_embedding(data, 3, 1, 1, 0, ConstantBoundary(10.))
        rem = RotationallyInvariantEmbedding(em)
        R1 = reconstruct(data, em)
        R2 = reconstruct(data, rem)
        for n = 1:length(R1)    #All others should stay unchanged
            if n <= 50 || 2500 > n > 2450
                @test R1[n] != R2[n]   #Left and right boundary gets flipped
            elseif (n-1)% 50 == 0
                @test R1[n] != R2[n]  #Top boundary gets flipped
            else
                @test R1[n] == R2[n]
            end
        end
    end
end
