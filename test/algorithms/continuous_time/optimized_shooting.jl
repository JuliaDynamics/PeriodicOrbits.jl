using Test
using PeriodicOrbits
using LinearAlgebra: norm

@inbounds function lorenz_rule(u, p, t)
    du1 = p[1] * (u[2] - u[1])
    du2 = u[1] * (p[2] - u[3]) - u[2]
    du3 = u[1] * u[2] - p[3] * u[3]
    return SVector{3}(du1, du2, du3)
end

@testset "Optimized shooting" begin
    igs = [InitialGuess(SVector(1.0, 2.0, 5.0), 4.2), InitialGuess(SVector(1.0, 2.0, 5.0), 4.2)]
    ig = igs[1]
    alg = OptimizedShooting(Δt=1e-4, n=3, nonlinear_solve_kwargs=(abstol=1e-7, reltol=1e-7))
    ds = CoupledODEs(lorenz_rule, [0.0, 10.0, 0.0], [10.0, 28.0, 8 / 3]; diffeq=(abstol=1e-11, reltol=1e-11))

    res = periodic_orbit(ds, alg, ig)

    @test !isnothing(res)
    if !isnothing(res)
        u0 = res.points[1]
        T = res.T
        reinit!(ds, u0)
        step!(ds, T, true)
        @test norm(current_state(ds) - u0) <= 1e-6
    end

    res = periodic_orbits(ds, alg, igs)
    @test length(res) == 2

    for po in res
        u0 = po.points[1]
        T = po.T
        reinit!(ds, u0)
        step!(ds, T, true)
        @test norm(current_state(ds) - u0) <= 1e-6
    end
end