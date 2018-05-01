using TimeseriesPrediction
using Base.Test

include("system_defs.jl")

@testset "Barkley STTS" begin
    Nx = 50
    Ny = 50
    Tskip = 100
    Ttrain = 300
    p = 20
    T = Tskip + Ttrain + p
    U, V = barkley_const_boundary(T, Nx, Ny)


    τ = 1
    k = 1
    c = 200
    w = (0,0)

    @testset "V, D=$D, B=$B" for D=2:3, B=1:2
        Vtrain = V[Tskip + 1:Tskip + Ttrain]
        Vtest  = V[Tskip + Ttrain :  T]
        Vpred = localmodel_stts(Vtrain, D, τ, p, B, k; boundary=c, weighting=w)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.1
        end
    end
    @testset "U" begin
        D=2; B=1
        Utrain = U[Tskip + 1:Tskip + Ttrain]
        Utest  = U[Tskip + Ttrain :  T]
        Upred = localmodel_stts(Utrain, D, τ, p, B, k; boundary=c, weighting=w)
        @test Upred[1] == Utrain[end]
        err = [abs.(Utest[i]-Upred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.2
        end
    end
    # @testset "weighting" begin
    #     D=2; B=1
    #     w = (0.5, 4)
    #     Utrain = U[Tskip + 1:Tskip + Ttrain]
    #     Utest  = U[Tskip + Ttrain :  T]
    #     Upred = localmodel_stts(Utrain, D, τ, p, B, k; boundary=c, weighting=w)
    #     @test Upred[1] == Utrain[end]
    #     err = [abs.(Utest[i]-Upred[i]) for i=1:p+1]
    #     for i in 1:p
    #         @test maximum(err[i]) < 0.1
    #     end
    # end
    @testset "Periodic, D=$D, B=$B" for D=2:3, B=1:2
        U,V = barkley_periodic_boundary(T, Nx, Ny)
        c = false
        w = (0, 0)
        Vtrain = V[Tskip + 1:Tskip + Ttrain]
        Vtest  = V[Tskip + Ttrain :  T]
        Vpred = localmodel_stts(Vtrain, D, τ, p, B, k; boundary=c, weighting=w)
        @test Vpred[1] == Vtrain[end]
        err = [abs.(Vtest[i]-Vpred[i]) for i=1:p+1]
        for i in 1:p
            @test maximum(err[i]) < 0.1
        end
    end
end
