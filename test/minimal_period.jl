using Test, PeriodicOrbits

@inbounds function lorenz_rule(u, p, t)
    du1 = p[1]*(u[2]-u[1])
    du2 = u[1]*(p[2]-u[3]) - u[2]
    du3 = u[1]*u[2] - p[3]*u[3]
    return SVector{3}(du1, du2, du3)
end

function logistic(x0=0.4; r = 4.0)
    return DeterministicIteratedMap(logistic_rule, SVector(x0), [r])
end
logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))

logistic_ = logistic()
period3window = StateSpaceSet([SVector{1}(x) for x in [0.15933615523767342, 0.5128107111364378, 0.9564784814729845]])

@testset "minimal_period discrete" begin
    r = 1+sqrt(8)
    set_parameters!(logistic_, [r])
    k = 4
    po = PeriodicOrbit(logistic_, period3window[1], k*3)
    po = minimal_period(logistic_, po)
    @test po.T == 3 == length(po.points)
end

@testset "minimal_period continuous" begin
    ds = CoupledODEs(lorenz_rule, [0.0, 10.0, 0.0], [10.0, 28.0, 8 / 3], diffeq=(abstol=1e-10, reltol=1e-10))
    u0 = [-4.473777426249161, -8.595978309705247, 8.410608458823141]
    T = 4.534075383577647
    n = 10
    po = PeriodicOrbit(ds, u0, n*T; Δt = 0.01)
    minT_po = minimal_period(ds, po)
    @test length(minT_po.points) == PeriodicOrbits.default_Δt_partition
    @test (ismissing(po.stable) && ismissing(minT_po.stable)) || (po.stable == minT_po.stable)
    @test isapprox(T, minT_po.T; atol=1e-4)

    Dt = 0.01
    minT_po = minimal_period(ds, po; Δt=Dt)
    @test length(minT_po.points) == floor(minT_po.T / Dt)
end

function normalhopf(u, p, t)
    # https://en.wikipedia.org/wiki/Hopf_bifurcation
    x, y = u
    μ, ω = p
    return SVector(
        (μ - x^2 - y^2)*x - ω*y,
        (μ - x^2 - y^2)*y + ω*x
    )
end

@testset "Known minimal period continuous" begin
    x, y = [1.0, 0.0] # point on a stable limit cycle
    μ = x^2 + y^2 # the radius of the orbit is sqrt(μ)
    ω = 1.1 # angular frequency
    T = 2*π/ω # period of the stable limit cycle
    ds = CoupledODEs(normalhopf, [x, y], [μ, ω])
    t = T
    step_ = 0.01
    traj, t = trajectory(ds, t; Dt=step_)

    noise = 25.2 # distort the period on purpose
    po = PeriodicOrbit(ds, [x, y], T+noise; Δt = 0.01)
    minT_po = minimal_period(ds, po)
    @test isapprox(minT_po.T, T; atol=1e-5)
end