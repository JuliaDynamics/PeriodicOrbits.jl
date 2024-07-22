using Test, PeriodicOrbits

@inbounds function lorenz_rule(u, p, t)
    du1 = p[1]*(u[2]-u[1])
    du2 = u[1]*(p[2]-u[3]) - u[2]
    du3 = u[1]*u[2] - p[3]*u[3]
    return SVector{3}(du1, du2, du3)
end

@testset "minimal_period discrete" begin
    r = 1+sqrt(8)
    set_parameters!(logistic_, [r])
    k = 4
    po = PeriodicOrbit(logistic_, period3window[1], k*3; jac=logistic_jacob)
    po = minimal_period(logistic_, po)
    @test po.T == 3 == length(po.points)
end

@testset "minimal_period continuous" begin
    ds = CoupledODEs(lorenz_rule, [0.0, 10.0, 0.0], [10.0, 28.0, 8 / 3], diffeq=(abstol=1e-10, reltol=1e-10))
    u0 = [-4.473777426249161, -8.595978309705247, 8.410608458823141]
    T = 4.534075383577647
    n = 10
    po = PeriodicOrbit(ds, u0, n*T, 0.01; jac=nothing)
    minT_po = minimal_period(ds, po)
    @test length(po.points) == length(minT_po.points)
    @test length(po.stable) == length(minT_po.stable)
    @test isapprox(T, minT_po.T; atol=1e-4)
end