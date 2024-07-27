using Test, PeriodicOrbits

logistic_ruleOOP(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
logistic_jacobOOP(x, p, n) = @inbounds SMatrix{1,1}(p[1]*(1 - 2x[1]))
function logistic_ruleIIP(du, u, p, n)
    du[:] = logistic_ruleOOP(u, p, n)
    return nothing
end
function logistic_jacobIIP(du, u, p, n)
    du[:] = logistic_jacobOOP(u, p, n)
    return nothing
end
henon_ruleOOP(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon_jacobOOP(x, p, n) = SMatrix{2,2}(-2*p[1]*x[1], p[2], 1.0, 0.0)
function henon_ruleIIP(du, u, p, n)
    du[:] = henon_ruleOOP(u, p, n)
    return nothing
end
function henon_jacobIIP(du, u, p, n)
    du[:] = henon_jacobOOP(u, p, n)
    return nothing
end


@testset "stability discrete 1D" begin
    for (rule, jacob) in [
            (logistic_ruleOOP, logistic_jacobOOP), 
            (logistic_ruleIIP, logistic_jacobIIP), 
            (logistic_ruleOOP, nothing), 
            (logistic_ruleIIP, nothing),
        ]
        ds = DeterministicIteratedMap(rule, [0.4], [4.0])
        if isnothing(jacob)
            jacob = jacobian(ds)    
        end
        # not using StaticArrays to work with IIP
        period3window = [[x] for x in [0.15933615523767342, 0.5128107111364378, 0.9564784814729845]]
        r = 2.3
        set_parameters!(ds, [r])
        fp = isstable(ds, [(r-1)/r], 1, jacob)
        @test fp == true
        @test typeof(fp) == Bool

        r = 3.3
        set_parameters!(ds, [r])
        @test isstable(ds, [(r-1)/r], 1, jacob) == false

        r = 1+sqrt(8)
        set_parameters!(ds, [r])
        @test isstable(ds, period3window[1], 3, jacob) == true
    end
end

@testset "stability discrete 2D" begin
    for (rule, jacob) in [
            (henon_ruleOOP, henon_jacobOOP), 
            (henon_ruleIIP, henon_jacobIIP), 
            (henon_ruleOOP, nothing), 
            (henon_ruleIIP, nothing),
        ]
        

        ds = DeterministicIteratedMap(rule, [0.0, 0.0], [1.4, 0.3])
        if isnothing(jacob)
            jacob = jacobian(ds)    
        end
        # not using StaticArrays to work with IIP
        unstable_period2 = [
            [0.88389, 0.88389],
            [-0.66612, 1.36612]
        ]
        fp = isstable(ds, unstable_period2[1], 2, jacob)
        @test fp == false
        @test typeof(fp) == Bool

        fp = isstable(ds, unstable_period2[2], 2, jacob)
        @test fp == false
    end
end

# https://en.wikipedia.org/wiki/Hopf_bifurcation
function superhopfOOP(u, p, t)
    x, y = u
    μ, ω = p
    return SVector(
        (μ - x^2 - y^2)*x - ω*y,
        (μ - x^2 - y^2)*y + ω*x
    )
end
function superhopfOOPjac(u, p, t)
    x, y = u
    μ, ω = p
    return SMatrix{2,2}(
        [
            (μ - y^2 - 3*x^2) (-2*x*y -ω);
            (-2*x*y + ω) (μ - x^2 - 3*y^2)
        ]
    )
end
function superhopfIIP(du, u, p, t)
    du[:] = superhopfOOP(u, p, t)
    return nothing
end
function superhopfIIPjac(du, u, p, t)
    du[:, :] = superhopfOOPjac(u, p, t)
    return nothing
end

function subhopfOOP(u, p, t)
    x, y = u
    μ, ω = p
    return SVector(
        -(μ - x^2 - y^2)*x - ω*y,
        -(μ - x^2 - y^2)*y + ω*x
    )
end
function subhopfOOPjac(u, p, t)
    x, y = u
    μ, ω = p
    return SMatrix{2,2}(
        [
            (-μ + y^2 + 3*x^2) (2*x*y - ω);
            (2*x*y + ω) (-μ + x^2 + 3*y^2)
        ]
    )
    return SVector(
        -(μ - x^2 - y^2)*x - ω*y,
        -(μ - x^2 - y^2)*y + ω*x
    )
end
function subhopfIIP(du, u, p, t)
    du[:] = subhopfOOP(u, p, t)
    return nothing
end
function subhopfIIPjac(du, u, p, t)
    du[:] = subhopfOOPjac(u, p, t)
    return nothing
end

@testset "stability continuous" begin
    for (rule, jac, result) in [
            # unfortunately many combinations
            (superhopfOOP, superhopfOOPjac, true),
            (superhopfIIP, superhopfIIPjac, true),
            (subhopfOOP, subhopfOOPjac, false),
            (subhopfIIP, subhopfIIPjac, false),
            (superhopfOOP, nothing, true), 
            (superhopfIIP, nothing, true), 
            (subhopfOOP, nothing, false), 
            (subhopfIIP, nothing, false)
        ]
        x, y = [1.0, 0.0] # point on a stable limit cycle
        μ = x^2 + y^2 # the radius of the orbit is sqrt(μ)
        ω = 1.1 # angular frequency
        T = 2*π/ω # period of the stable limit cycle
        ds = CoupledODEs(rule, [x, y], [μ, ω])
        if isnothing(jac)
            jac = jacobian(ds)    
        end
        @test isstable(ds, current_state(ds), T, jac) ==  result
    end
end