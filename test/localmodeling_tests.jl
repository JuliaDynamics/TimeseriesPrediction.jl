using Test
using TimeseriesPrediction
using StaticArrays
using LinearAlgebra
import Statistics:var
println("\nTesting local models...")

import Random
Random.seed!(42)

ds = Systems.roessler()
data = trajectory(ds, 5000; dt=0.1)
N_train = 45000
s_train = data[1:N_train, 1]
s_test  = data[N_train:end,1]

@testset "ALM localmodel_tsp" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [2,3], τ ∈ [14,15]
        p = 50
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, γ, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2

        #Repeat with reconstruction given
        R = reconstruct(s_train, γ, τ)
        s_pred = localmodel_tsp(R, p; method=method, ntype=ntype, stepsize=stepsize)[:,γ+1]
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2
    end
end

@testset "LinearLocalModel Ridge Reg" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = LinearLocalModel()
        ntype = FixedMassNeighborhood(3)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, γ, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 6e-2

        #Repeat with reconstruction given
        R = reconstruct(s_train, γ, τ)
        s_pred = localmodel_tsp(R, p; method=method, ntype=ntype, stepsize=stepsize)[:,γ+1]
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 6e-2
    end
end
@testset "LinearLocalModel McNames Reg" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = LinearLocalModel(TimeseriesPrediction.ω_safe, 0.1,1.)
        ntype = FixedMassNeighborhood(3)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, γ, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2

        #Repeat with reconstruction given
        R = reconstruct(s_train, γ, τ)
        s_pred = localmodel_tsp(R, p; method=method, ntype=ntype, stepsize=stepsize)[:,γ+1]
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 5e-2
    end
end

@testset "FixedSizeNeighborhood" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [4], τ ∈ [14,15]
        p = 25
        method = AverageLocalModel()
        ntype = FixedSizeNeighborhood(0.5)
        stepsize = 1
        s_pred = localmodel_tsp(s_train, γ, τ, p;
        method=method, ntype=ntype, stepsize=stepsize)
        @test length(s_pred) == p+1
        @test norm(s_test[1:p+1] - s_pred)/p < 0.2
    end
end

@testset "Multivariate Input predict" begin
    sm_train = data[1:N_train,SVector(1,2)]
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ=15
        R = reconstruct(sm_train,γ,τ)
        num_points = 50
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        sind = SVector{2, Int}(((γ+1)*2 - i for i in 2-1:-1:0)...)
        pred = localmodel_tsp(R,num_points; method=method,ntype=ntype,stepsize=stepsize)[:,sind]
        @test size(pred) == (num_points+1, 2)
        @test norm(s_test[1:num_points+1] - pred[:, 1])/num_points < 5e-2
        @test norm(data[N_train:N_train+num_points, 2] - pred[:, 2])/num_points < 5e-2

        pred2 = localmodel_tsp(sm_train, γ,τ,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test norm(pred.data-pred2.data) < 1e-10
    end
    @testset "γ=3 and multi τ" begin
    sm_train = data[1:N_train,SVector(1,2)]

        γ = 3;
        τ=[15 15; 30 29; 45 45]#[14 15; 29 30; 45 47]
        R = reconstruct(sm_train,γ,τ)
        num_points = 25
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        stepsize = 1
        svind = SVector{2, Int}(7,8)
        pred = localmodel_tsp(R,num_points; method=method,ntype=ntype,stepsize=stepsize)[:,svind]
        @test size(pred) == (num_points+1, 2)
        @test norm(s_test[1:num_points+1] - pred[:, 1])/num_points < 1e-1
        @test norm(data[N_train:N_train+num_points, 2] - pred[:, 2])/num_points < 1e-1

        pred2 = localmodel_tsp(sm_train,γ,τ,num_points; method=method,ntype=ntype,stepsize=stepsize)
        @test norm(pred.data-pred2.data) < 1e-10
    end
end

@testset "MSE" begin
    @testset "p=$p" for p ∈ [50,100]
        γ = 3; τ = 15;
        R = reconstruct(s_train,γ,τ)
        R_test = reconstruct(s_test[end-γ*τ-p-50:end],γ,τ)
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        stepsize = 1

        @test MSEp(R,R_test,p; method=method,ntype=ntype,stepsize=stepsize)/p < 5e-2
        @test MSEp(R,R_test,1; method=method,ntype=ntype,stepsize=stepsize) < 5e-2
    end
end

@testset "Maps" begin
    ds = Systems.standardmap()
    data = trajectory(ds,50000; dt=1)
    N_train = 49900
    p = 25
    stepsize = 1
    s = data[1:N_train,1]
    s_real = data[N_train:N_train+p*stepsize,1]
    method = AverageLocalModel()
    ntype = FixedMassNeighborhood(2)
    s_pred = localmodel_tsp(s,2,1,p; method=method, ntype=ntype)
    @test norm(s_pred -s_real)/var(s)/p < 0.2
end


#####################################################################################
#                                  Cross Prediction                                   #
#####################################################################################

ds = Systems.roessler()
data = trajectory(ds, 5000; dt=0.1)
N_train = 45000
N_test = 1000
source_train = data[1:N_train, 1]
target_train = data[1:N_train, 2]
source_pred = data[N_train+1:N_train+N_test, 1]
target_test = data[N_train+1:N_train+N_test, 2]


@testset "ALM localmodel_cp" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ ∈ [14,15]
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred, γ, τ;
                                        method=method, ntype=ntype)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3

        #Repeat with reconstruction given
        R      = reconstruct(source_train, γ, τ)
        R_pred = reconstruct(source_pred, γ, τ)
        target_pred = localmodel_cp(
                        R, target_train, R_pred;
                        method=method, ntype=ntype, y_idx_shift=γ*τ)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3
    end
end

@testset "LinearLocalModel Ridge Reg" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ ∈ [14,15]
        method = LinearLocalModel()
        ntype = FixedMassNeighborhood(3)
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred, γ, τ;
                                        method=method,
                                        ntype=ntype)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3

        #Repeat with reconstruction given
        R      = reconstruct(source_train, γ, τ)
        R_pred = reconstruct(source_pred, γ, τ)
        target_pred = localmodel_cp(
                        R, target_train, R_pred;
                        method=method, ntype=ntype, y_idx_shift=γ*τ)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3
    end
end
@testset "LinearLocalModel McNames Reg" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ ∈ [14,15]
        p = 50
        method = LinearLocalModel(TimeseriesPrediction.ω_safe, 0.1,1.)
        ntype = FixedMassNeighborhood(3)
        stepsize = 1
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred, γ, τ;
                                        method=method,
                                        ntype=ntype)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3

        #Repeat with reconstruction given
        R      = reconstruct(source_train, γ, τ)
        R_pred = reconstruct(source_pred, γ, τ)
        target_pred = localmodel_cp(
                        R, target_train, R_pred;
                        method=method, ntype=ntype, y_idx_shift=γ*τ)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3
    end
end

@testset "FixedSizeNeighborhood" begin
    @testset "γ=$γ and τ=$τ" for γ ∈ [4], τ ∈ [14,15]
        method = AverageLocalModel()
        ntype = FixedSizeNeighborhood(0.5)
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred, γ, τ;
                                        method=method,
                                        ntype=ntype)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3
    end
end

@testset "Multivariate Input predict" begin
    sm_train = data[1:N_train,SVector(1,2)]
    sm_pred = data[1+N_train:N_train+N_test,SVector(1,2)]
    @testset "γ=$γ and τ=$τ" for γ ∈ [3,4], τ=15
        source_train = reconstruct(sm_train,γ,τ)
        source_pred = reconstruct(sm_pred,γ,τ)
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred;
                                        method=method,
                                        ntype=ntype,
                                        y_idx_shift=γ*τ)
        @test length(target_pred) == N_test - γ*τ
        @test norm(target_test[1+γ*τ:end] .- target_pred)/N_test < 5e-3
    end
    @testset "γ=3 and multi τ" begin
        γ = 3;
        τ=[15 15; 30 29; 45 45]
        source_train = reconstruct(sm_train,γ,τ)45000,
        source_pred = reconstruct(sm_pred,γ,τ)
        method = AverageLocalModel()
        ntype = FixedMassNeighborhood(2)
        target_pred = localmodel_cp(    source_train,
                                        target_train,
                                        source_pred;
                                        method=method,
                                        ntype=ntype,
                                        y_idx_shift=maximum(τ))
        @test length(target_pred) == N_test - maximum(τ)
        @test norm(target_test[1+maximum(τ):end] .- target_pred)/N_test < 5e-3
    end
end
