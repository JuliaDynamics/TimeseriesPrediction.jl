using TimeseriesPrediction
using Test
using Statistics, LinearAlgebra

@testset "SymmetricEmbedding" begin
    @testset "Ordering in β_groups" begin
        γ = 0; τ = 1; r = 4; c = 0; bc = ConstantBoundary(10.);
        dummy_data = [rand(10,10,10) for _ in 1:10]
        em = light_cone_embedding(dummy_data, γ, τ, r, c, bc)

        symmetries = [(Reflection(1),), (Reflection(2),), (Rotation(1,3),),
        (Rotation(1,2,3),), (Rotation(1,3), Reflection(2))]

        # symmetries = [ [[1]], [[2]], [[1,3]], [[1,2,3]], [[3,1], [2]] ]
        @testset "Symmetry $sym" for sym in symmetries
            sem = SymmetricEmbedding(em, sym)
            # Check that first entry is always origin
            @test sem.β_groups[1] == [CartesianIndex(0,0,0)]

            # Check that all points within group have the same distance to origin
            # Check that β groups are sorted
            let dist = -Inf
                for group in sem.β_groups
                    @test dist <= norm(group[1].I)
                    dist = norm(group[1].I)
                    @test 1 == length(unique(norm(getproperty.(group, :I))))
                end
            end
        end

    end
    @testset "Ordering in τ (γ=$γ)" for γ ∈ [5,10]
        #Entries are ordered with increasing τ
        τ = 1; r = 4; c = 0; bc = ConstantBoundary(10.);
        dummy_data = [rand(10,10,10,10) for _ in 1:10]
        em = light_cone_embedding(dummy_data, γ, τ, r, c, bc)

        symmetries = [(Rotation(1,4),), (Rotation(1,2,3,4),),
                    (Rotation(1,3), Reflection(2))]

        # symmetries = [ [[1,4]], [[1,2,3,4]], [[3,1], [2]] ]
        @testset "Symmetry $sym" for sym in symmetries
            sem = SymmetricEmbedding(em, sym)

            @test issorted(sem.τ)
            @test all( sem.τ .∈ Ref(em.τ))

            #Check that points are not accidentally moved to wrong timstep
            for (t,group) in zip(sem.τ, sem.β_groups)
                for g in group
                    @test (t,g) in zip(em.τ, em.β)
                end
            end
        end
    end
    @testset "None are lost" begin
        #Check that no points go missing in the reduction
        γ = 1; τ = 1; r = 3; c = 0; bc = PeriodicBoundary();
        dummy_data = [rand(10,10,10) for _ in 1:10]
        em = light_cone_embedding(dummy_data, γ, τ, r, c, bc)

        symmetries = [(Reflection(1),), (Reflection(2),), (Rotation(1,3),),
                      (Rotation(1,2,3),), (Rotation(1,3), Reflection(2))]
        # symmetries = [ [[1]], [[2]], [[1,3]], [[1,2,3]], [[3,1], [2]] ]
        @testset "Symmetry $sym" for sym in symmetries
            sem = SymmetricEmbedding(em, sym)

            all_points = vcat(sem.β_groups...)
            @test all(all_points .∈ Ref(em.β))
            @test all(em.β .∈ Ref(all_points))
        end
    end

    @testset "Parameter checking" begin
        γ = 0; τ = 1; r = 2; c = 0; bc = ConstantBoundary(10.);
        dummy_data = [rand(10,10) for _ in 1:10]
        em = light_cone_embedding(dummy_data, γ, τ, r, c, bc)

        symmetries = [(Reflection(0),), (Reflection(3),), (Rotation(1,3),),
                      (Rotation(2,0),), (Rotation(1,2), Reflection(2))]
        # symmetries = [ [[0]], [[3]], [[1,3]], [[2,0]], [[2,1], [2]] ]

        @testset "Symmetry $sym" for sym in symmetries
            @test_throws ArgumentError SymmetricEmbedding(em, sym)
        end
    end

    @testset "Reconstruction" begin
        #Quality of results depends on the system..
        @testset "$bc" for bc in [ConstantBoundary(10.), PeriodicBoundary()]
            γ = 2; τ = 5; r = 1; c = 0;
            data = [rand(25,25) for _ in 1:50]
            em = light_cone_embedding(data, γ, τ, r, c, bc)
            sem = SymmetricEmbedding(em, (Rotation(1,2),))
            # sem = SymmetricEmbedding(em, [[1,2]])

            #Check wether this works in principle
            r1 = reconstruct(data, sem);
            @test outdim(sem) == size(r1,2)
            #Check wether PCA works
            pcaem = PCAEmbedding(data, sem)
            r2 = reconstruct(data, sem);
            @test outdim(pcaem) == size(r2,2)

            @test length(r1) == length(r2)
        end
    end
end
