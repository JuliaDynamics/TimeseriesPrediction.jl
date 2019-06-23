using Test
using TimeseriesPrediction
using TimeseriesPrediction: BoundaryWrapper



@testset "BoundaryWrapper" begin
    @testset "1D Field" begin
        data = [rand(10) for _ = 1:10]
        em = light_cone_embedding(data, 3,2,1,0, PeriodicBoundary())
        bw = BoundaryWrapper(em, data)

        @test bw[1, CartesianIndex(10)] == data[1][10] # Inbounds
        @test bw[1, CartesianIndex(12)] == data[1][2] # out of bounds
        @test bw[1, CartesianIndex(-3)] == data[1][7] # out of bounds
        bw_t = BoundaryWrapper(em, data[5])

        @test bw_t[CartesianIndex(2)] == data[5][2]
        @test bw_t[CartesianIndex(-2)] == data[5][8]

        em = light_cone_embedding(data, 3,2,1,0, ConstantBoundary(10.))
        bw = BoundaryWrapper(em, data)

        @test bw[1, CartesianIndex(10)] == data[1][10] # Inbounds
        @test bw[1, CartesianIndex(12)] == 10. # out of bounds
        @test bw[1, CartesianIndex(-3)] == 10. # out of bounds
        bw_t = BoundaryWrapper(em, data[5])
        @test bw_t[CartesianIndex(2)] == data[5][2]
        @test bw_t[CartesianIndex(-2)] == 10.
    end
    
    @testset "2D Field" begin
        data = [rand(10,10) for _ = 1:10]
        em = light_cone_embedding(data, 3,2,1,0, PeriodicBoundary())
        bw = BoundaryWrapper(em, data)

        @test bw[1, CartesianIndex(10,10)] == data[1][10,10] # Inbounds
        @test bw[1, CartesianIndex(12,5)] == data[1][2, 5] # out of bounds
        @test bw[1, CartesianIndex(-3, 1)] == data[1][7,1] # out of bounds
        @test bw[1, CartesianIndex(-3, -1)] == data[1][7,9] # out of bounds
        bw_t = BoundaryWrapper(em, data[5])

        @test bw_t[CartesianIndex(2, 4)] == data[5][2, 4]
        @test bw_t[CartesianIndex(-2, 5)] == data[5][8, 5]

        em = light_cone_embedding(data, 3,2,1,0, ConstantBoundary(10.))
        bw = BoundaryWrapper(em, data)

        @test bw[1, CartesianIndex(10, 3)] == data[1][10, 3] # Inbounds
        @test bw[1, CartesianIndex(12, 2)] == 10. # out of bounds
        @test bw[1, CartesianIndex(-3, 70)] == 10. # out of bounds
        bw_t = BoundaryWrapper(em, data[5])
        @test bw_t[CartesianIndex(2,3)] == data[5][2,3]
        @test bw_t[CartesianIndex(-2, 7)] == 10.
    end
end
