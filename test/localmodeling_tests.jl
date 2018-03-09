using DynamicalSystemsBase
using Base.Test
using TimeseriesPrediction
using StaticArrays

println("Testing predict")

ds = Systems.roessler()
data = trajectory(ds, 5000;dt=0.1)
N_train = 40000
s_train = data[1:N_train, 1]
s_test  = data[N_train:end,1]

@testset "predict_timeseries" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = LocalAverageModel(2)
        ntype = FixedMassNeighborhood(2)
        step = 1
        s_pred = predict_timeseries(s_train, D, τ, p;
         method=method, ntype=ntype, step=step)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2

        #Repeat with reconstruction given
        R = Reconstruction(s_train, D, τ)
        s_pred = predict_timeseries(R, p; method=method, ntype=ntype, step=step)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2
    end
end

@testset "FixedSizeNeighborhood" begin
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = LocalAverageModel(2)
        ntype = FixedSizeNeighborhood(0.5)
        step = 1
        s_pred = predict_timeseries(s_train, D, τ, p;
         method=method, ntype=ntype, step=step)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2
    end
end


@testset "Multivariate Input predict" begin
    sind = SVector(2,1)
    sm_train = data[1:N_train,sind]
    @testset "D=$D and τ=$τ" for D ∈ [3,4], τ ∈ [14,15]
        R = Reconstruction(sm_train,D,τ)
        num_points = 50
        method = LocalAverageModel(2)
        ntype = FixedMassNeighborhood(2)
        step = 1
        s_pred = predict_timeseries(R,num_points; method=method,ntype=ntype,step=step)
        @test length(s_pred) == num_points+1
        @test norm(s_test[1:num_points+1] - s_pred)/num_points < 5e-2
    end
end



println("Testing MSEp")

@testset "MSE" begin
    @testset "p=$p" for p ∈ [50,100]
        D = 3; τ = 15;
        R = Reconstruction(s_train,D,τ)
        R_test = Reconstruction(s_test[end-D*τ-p-50:end],D,τ)
        method = LocalAverageModel(2)
        ntype = FixedMassNeighborhood(2)
        step = 1

        @test MSEp(R,R_test,p; method=method,ntype=ntype,step=step)/p < 5e-2
        @test MSE1(R,R_test; method=method,ntype=ntype,step=step) < 5e-2
    end
end
