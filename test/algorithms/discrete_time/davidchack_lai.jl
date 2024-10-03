using PeriodicOrbits
using Test

@testset "DavidchackLai" begin
    henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
    xs = range(0, stop = 2π, length = 5); ys = copy(xs)
    ics = InitialGuess[InitialGuess(SVector(x,y), nothing) for x in xs for y in ys]
    o = 10
    m = 6
    alg = DavidchackLai(n=o, m=m, disttol=1e-12, abstol=1e-8)
    pos = periodic_orbits(ds, alg, ics)
    tol = 1e-12
    for po in pos
        x0 = po.points[1]
        set_state!(ds, x0)
        step!(ds, po.T)
        xn = current_state(ds)
        @test isapprox(x0, xn; atol = tol)
    end
end
