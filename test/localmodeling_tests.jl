using DynamicalSystemsBase
using Base.Test
using TimeseriesPrediction


println("Testing TSP")

ds = Systems.roessler()
data = trajectory(ds, 210;dt=0.01)
N_train = 20000
s_train = data[1:N_train, 1];
s_test  = data[N_train:N_train+200,1]

@testset "TSP" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [50,60]
        R = Reconstruction(s_train,D,τ)
        tree = KDTree(R[1:end-100])
        num_points = 50
        LocalModel = LocalAverageModel(2)
        method = FixedMassNeighborhood(5)
        f(i) = i+1
        s_pred = TSP(tree,R,num_points,LocalModel,method,f)
        @test length(s_pred) == num_points+1
        @test norm(s_test[1:num_points+1] - s_pred)/num_points < 5e-2
    end
end
