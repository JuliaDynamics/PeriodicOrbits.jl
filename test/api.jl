using Test
using PeriodicOrbits

function logistic(x0=0.4; r = 4.0)
    return DeterministicIteratedMap(logistic_rule, SVector(x0), [r])
end
logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
logistic_jacob(x, p, n) = @inbounds SMatrix{1,1}(p[1]*(1 - 2x[1]))

function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CoupledODEs(lorenz_rule, u0, [σ, ρ, β])
end
@inbounds function lorenz_rule(u, p, t)
    du1 = p[1]*(u[2]-u[1])
    du2 = u[1]*(p[2]-u[3]) - u[2]
    du3 = u[1]*u[2] - p[3]*u[3]
    return SVector{3}(du1, du2, du3)
end
@inbounds function lorenz_jacob(u, p, t)
        return SMatrix{3,3}(-p[1], p[2] - u[3], u[2], p[1], -1.0, u[1], 0.0, -u[1], -p[3])
end


const logistic_ = logistic()
const lorenz_ = lorenz()
const period3window = Dataset([SVector{1}(x) for x in [0.15933615523767342, 0.5128107111364378, 0.9564784814729845]])

@testset "constructors of InitialGuess" begin
    # TODO
end

@testset "constructors of PeriodicOrbit" begin
    # TODO
end

@testset "complete orbit" begin
    r = 1+sqrt(8)
    set_parameters!(logistic_, [r])
    completed_orbit = complete_orbit(logistic_, period3window[1], 3)
    @test completed_orbit == period3window
    @test typeof(completed_orbit) <: StateSpaceSet
end

@testset "minimal_period" begin
    r = 1+sqrt(8)
    set_parameters!(logistic_, [r])
    k = 4
    po = PeriodicOrbit(logistic_, period3window[1], k*3; jac=logistic_jacob)
    po = minimal_period(logistic_, po)
    @test po.T == 3 == length(po.points)
end

@testset "unique POs" begin
    r = 1+sqrt(8)
    set_parameters!(logistic_, [r])
    po1 = PeriodicOrbit(logistic_, period3window[1], 3; jac=logistic_jacob)
    po2 = po1
    po3 = PeriodicOrbit(logistic_, SVector{1}([(r-1)/r]), 1; jac=logistic_jacob)

    # use Set to neglect order
    uniquepo = Set(uniquepos([po1, po2, po3], 1e-4))
    reference = Set([po1, po3])
    @test uniquepo == reference
end

@testset "PO equality & distance" begin
    set_parameters!(logistic_, [3.5])
    po1 = PeriodicOrbit(logistic_, SVector(0.3), 3; jac=logistic_jacob)
    po2 = PeriodicOrbit(logistic_, SVector(0.6), 3; jac=logistic_jacob)

    @test poequal(po1, po1) == true
    @test poequal(po1, po2) == false

    @test podistance(po1, po1) ≈ 0.0
    @test podistance(po1, po2) > 0.0
end

@testset "PO type" begin
    r = 1.0
    po = PeriodicOrbit(logistic_, SVector{1}([(r-1)/r]), 1; jac=logistic_jacob)
    @test isdiscretetime(po) == true

    po = PeriodicOrbit(lorenz_, current_state(lorenz_), 1.0; jac=lorenz_jacob)
    @test isdiscretetime(po) == false
end