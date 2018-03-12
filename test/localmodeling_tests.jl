using DynamicalSystemsBase
using Base.Test
using TimeseriesPrediction
using StaticArrays

println("\nTesting local models...")

ds = Systems.roessler()
data = trajectory(ds, 5000; dt=0.1)
N_train = 40000
s_train = data[1:N_train, 1]
s_test  = data[N_train:end,1]

@testset "localmodel_tsp" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = AverageLocalModel(2)
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, D, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2

        #Repeat with reconstruction given
        R = Reconstruction(s_train, D, τ)
        s_pred = localmodel_tsp(R, p; method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2
    end
end

@testset "FixedSizeNeighborhood" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [14,15]
        p = 25
        method = AverageLocalModel(2)
        ntype = FixedSizeNeighborhood(0.5)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, D, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 10e-2
    end
end

@testset "Multivariate Input predict" begin
    sm_train = data[1:N_train,SVector(1,2)]
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ=15
        R = Reconstruction(sm_train,D,τ)
        num_points = 50
        method = AverageLocalModel(2)
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        pred = localmodel_tsp(R,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test size(pred) == (num_points+1, 2)
        @test norm(s_test[1:num_points+1] - pred[:, 1])/num_points < 5e-2
        @test norm(data[N_train:N_train+num_points, 2] - pred[:, 2])/num_points < 5e-2

        pred2 = localmodel_tsp(sm_train, D,τ,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test norm(pred.data-pred2.data) < 1e-10
    end
    @testset "D=3 and multi τ" begin
    sm_train = data[1:N_train,SVector(1,2)]

        D = 3;
        τ=[15 15; 30 29; 45 45]#[14 15; 29 30; 45 47]
        R = Reconstruction(sm_train,D,τ)
        num_points = 25
        method = AverageLocalModel(2)
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        pred = localmodel_tsp(R,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test size(pred) == (num_points+1, 2)
        @test norm(s_test[1:num_points+1] - pred[:, 1])/num_points < 1e-1
        @test norm(data[N_train:N_train+num_points, 2] - pred[:, 2])/num_points < 1e-1

        pred2 = localmodel_tsp(sm_train, D,τ,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test norm(pred.data-pred2.data) < 1e-10
    end
end

@testset "MSE" begin
    @testset "p=$p" for p ∈ [50,100]
        D = 3; τ = 15;
        R = Reconstruction(s_train,D,τ)
        R_test = Reconstruction(s_test[end-D*τ-p-50:end],D,τ)
        method = AverageLocalModel(2)
        ntype = FixedMassNeighborhood(2)
        stepsize = 1

        @test MSEp(R,R_test,p; method=method,ntype=ntype,stepsize=stepsize)/p < 5e-2
        @test MSE1(R,R_test; method=method,ntype=ntype,stepsize=stepsize) < 5e-2
    end
end
