using TimeseriesPrediction
using Test
import Random

Random.seed!(42)
include("system_defs.jl")

@testset "Coupled Henon 1D" begin
    M = 100
    N_train = 2000
    p = 5
    U, V = coupled_henon1D(M,N_train+p+100, rand(M), rand(M))
    #Reconstruct this #
    utrain = U[1:N_train]
    vtrain = V[1:N_train]
    utest  = U[N_train:N_train+p]
    vtest  = V[N_train:N_train+p]

    @testset "ALM D=$D, B=$B" for D=2:3, B=1:2
        em = cubic_shell_embedding(utrain,D,1,B,1,ConstantBoundary(10.))
        upred = temporalprediction(utrain,em, p)

        @test upred[1] == utrain[end]
        @test sum(abs.(utest[2]-upred[2]))/M/p < 0.05
        ε = [sum(abs.(utest[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/M / p < 0.15
    end
    @testset "1D Henon with offset start of pred" begin
        D=3; B=2
        em = cubic_shell_embedding(utrain,D,1,B,1,ConstantBoundary(10.))
        ustart = U[N_train+100-4:N_train+100]
        utest_off  = U[N_train+100:N_train+p+100]
        upred = temporalprediction(utrain,em, p; initial_ts=ustart)

        @test upred[1] == ustart[end]
        @test sum(abs.(utest_off[2]-upred[2]))/M/p < 0.05
        ε = [sum(abs.(utest_off[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/M / p < 0.15
    end

    @testset "LLM D=$D, B=$B" for D=2:3, B=1:2
        method = LinearLocalModel(TimeseriesPrediction.ω_safe, 0.001, 1.)
        em = cubic_shell_embedding(utrain, D, 1, B,1, ConstantBoundary(10.))
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
    N_train = 1000
    p = 5
    U,V = coupled_henon2D(X,Y,N_train+p,rand(X,Y), rand(X,Y))
    #Reconstruct this #
    utrain = U[1:N_train]
    vtrain = V[1:N_train]
    utest  = U[N_train:N_train+p]
    vtest  = V[N_train:N_train+p]

    @testset "D=$D, B=$B" for D=2:3, B=1:2
        em = cubic_shell_embedding(utrain,D,1,B,1,ConstantBoundary(10.))
        upred = temporalprediction(utrain,em,p)

        @test upred[1] == utrain[end]
        ε = [sum(abs.(utest[i]-upred[i])) for i=1:p+1]
        @test sum(ε)/X/Y / p < 0.15
    end
end
