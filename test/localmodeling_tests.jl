using DynamicalSystemsBase
using Base.Test
using TimeseriesPrediction


println("Testing TSP")

ds = Systems.roessler()
data = trajectory(ds, 210;dt=0.01)
N_train = 20000
s_train = data[1:N_train, 1]
s_test  = data[N_train:end,1]

@testset "TSP" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [140,150]
        R = Reconstruction(s_train,D,τ)
        tree = KDTree(R[1:end-100])
        num_points = 50
        LocalModel = LocalAverageModel(2)
        method = FixedMassNeighborhood(2)
        f(i) = i+1
        s_pred = TSP(tree,R,num_points,LocalModel,method,f)
        @test length(s_pred) == num_points+1
        @test norm(s_test[1:num_points+1] - s_pred)/num_points < 5e-2
    end
end



println("Testing MSEp")

@testset "MSEp" begin
    @testset "p=$p" for p ∈ [50,100]
        D = 3; τ = 150;
        R = Reconstruction(s_train,D,τ)
        tree = KDTree(R)
        R_test = Reconstruction(s_test[end-D*τ-p-50:end],D,τ)
        LocalModel = LocalAverageModel(2)
        method = FixedMassNeighborhood(2)
        f(i) = i+1
        @test MSEp(tree,R,R_test,p,LocalModel,method,f) < 5e-2
    end
end
