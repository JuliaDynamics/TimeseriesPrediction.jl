using TimeseriesPrediction
import Statistics: mean
using Test

import Random
Random.seed!(42)

include("system_defs.jl")

@testset "Barkley Const Boundary" begin
    kwargs = ( tskip = 200,
               size = (50,50),
               periodic= false,
               params =(a=0.75, b=0.02, ε=0.02, D=1, h=0.75, Δt=0.1))
    Ttrain = 400
    p = 20
    T = Ttrain + p
    U, V = barkley(T; kwargs... )
    Vtrain = V[1:Ttrain]
    Utrain = U[1:Ttrain]


    τ = 1
    k = 1
    BC = ConstantBoundary(20.)

    @testset "V, D=$D, B=$B" for D=10, B=1
        Vtest  = V[Ttrain :  T]
        em = cubic_shell_embedding(Vtrain,D,τ,B,k,BC)
        em = PCAEmbedding(Vtrain, em)
        Vpred = temporalprediction(Vtrain,em,p)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.2
            @test minimum(err[i]) < 0.1
        end
    end
    @testset "V LightCone" begin
        Vtest  = V[Ttrain :  T]
        em = SpatioTemporalEmbedding(Vtrain, (D=1,τ=1,r₀=1,c=0.5,bc=BC))
        Vpred = temporalprediction(Vtrain,em,p)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.2
            @test minimum(err[i]) < 0.1
        end
    end
    @testset "U, D=2, B=1" begin
        D=3; B=1; τ=4
        #high embedding dim, so predict fewer points to save time
        p = 10
        Utest  = U[Ttrain :  T]
        em = cubic_shell_embedding(Utrain,D,τ,B,k,BC)
        Upred = temporalprediction(Utrain,em,p)
        @test Upred[1] == Utrain[end]
        err = [abs.(Utest[i]-Upred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.2
            @test minimum(err[i]) < 0.1
        end
    end
    @testset "crosspred V → U" begin
        D = 4; B = 1
        p=10
        Utest  = U[Ttrain + 1:  T]
        Vtest  = V[Ttrain  - D*τ+ 1:  T]
        em = cubic_shell_embedding(Vtrain, D,τ,B,k,BC)
        Upred = crossprediction(Vtrain,Utrain,Vtest, em)
        err = [abs.(Utest[i]-Upred[i]) for i=1:p-1]
        for i in 1:length(err)
            @test maximum(err[i]) < 0.2
            @test minimum(err[i]) < 0.1
            @test mean(err[i]) < 0.1
        end
    end
end

@testset "Periodic Barkley" begin
    kwargs = ( tskip = 200,
               size = (50,50),
               periodic= true,
               params =(a=0.75, b=0.02, ε=0.02, D=1, h=0.75, Δt=0.1))
    Ttrain = 400
    p = 20
    T = Ttrain + p
    U, V = barkley(T; kwargs... )
    Vtrain = V[1:Ttrain]
    Utrain = U[1:Ttrain]
    τ = 1
    k = 1
    BC = PeriodicBoundary()

    @testset "crosspred V → U" begin
        D = 3; B = 1
        Utest  = U[Ttrain + 1:  T]
        Vtest  = V[Ttrain  - D*τ+ 1:  T]
        em = cubic_shell_embedding(Vtrain, D,τ,B,k,BC)
        Upred = crossprediction(Vtrain,Utrain,Vtest, em)
        err = [abs.(Utest[i]-Upred[i]) for i=1:p-1]
        for i in 1:length(err)
            @test maximum(err[i]) < 0.2
            @test mean(err[i]) < 0.1
            @test minimum(err[i]) < 0.1
        end
    end

    @testset "Periodic, D=$D, B=$B" for D=10, B=1
        Vtest  = V[Ttrain :  T]
        em = cubic_shell_embedding(Vtrain, D,τ,B,k,BC)
        em = PCAEmbedding(Vtrain, em)
        Vpred = temporalprediction(Vtrain, em, p)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.1
            @test minimum(err[i]) < 0.1
        end
    end

    @testset "Periodic diff. inital, D=$D, B=$B" for D=10, B=1
        U, V = barkley(T; kwargs... )
        Vtrain = V[1:Ttrain]
        Vtest  = V[Ttrain :  T]
        em = cubic_shell_embedding(Vtrain, D,τ,B,k,BC)
        em = PCAEmbedding(Vtrain, em)
        Vpred = temporalprediction(Vtrain, em, p)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.2
            @test minimum(err[i]) < 0.1
        end
    end
end
