using TimeseriesPrediction
using Test
import Random: MersenneTwister

include("system_defs.jl")

@testset "Coupled Henon 1D" begin
    M = 100
    ds = coupled_henon1D(M,rand(MersenneTwister(42), M,2))
    N_train = 2000
    p = 5
    data = trajectory(ds,N_train+p)
    U = [d[:,1] for d in data]
    V = [d[:,2] for d in data]
    #Reconstruct this #
    utrain = U[1:N_train]
    vtrain = V[1:N_train]
    utest  = U[N_train:N_train+p]
    vtest  = V[N_train:N_train+p]

    @testset "ALM D=$D, B=$B" for D=2:3, B=1:2
        em = SpatioTemporalEmbedding(utrain,D,1,B,1,ConstantBoundary(10.))
        upred = temporalprediction(utrain,em, p)

        @test upred[1] == utrain[end]
        @test sum(abs.(utest[2]-upred[2]))/M/p < 0.05
        ε = [sum(abs.(utest[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/M / p < 0.15
    end
    @testset "LLM D=$D, B=$B" for D=2:3, B=1:2
        method = LinearLocalModel(TimeseriesPrediction.ω_safe, 0.001, 1.)
        em = SpatioTemporalEmbedding(utrain, D, 1, B,1, ConstantBoundary(10.))
        upred = temporalprediction(utrain,em,p; method=method)

        @test upred[1] == utrain[end]
        @test sum(abs.(utest[2]-upred[2]))/M/p < 0.05
        ε = [sum(abs.(utest[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/M / p < 0.15
    end
end


@testset "Coupled Henon 2D" begin
    #Size
    X=5
    Y=5
    ds = coupled_henon2D(X,Y, rand(MersenneTwister(42), X,Y,2))
    N_train = 1000
    p = 5
    data = trajectory(ds,N_train+p)
    U = [d[:,:,1] for d in data]
    V = [d[:,:,2] for d in data]
    #Reconstruct this #
    utrain = U[1:N_train]
    vtrain = V[1:N_train]
    utest  = U[N_train:N_train+p]
    vtest  = V[N_train:N_train+p]

    @testset "D=$D, B=$B" for D=2:3, B=1:2
        em = SpatioTemporalEmbedding(utrain,D,1,B,1,ConstantBoundary(10.))
        upred = temporalprediction(utrain,em,p)

        @test upred[1] == utrain[end]
        ε = [sum(abs.(utest[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/X/Y / p < 0.15
    end
end
