export periodic_orbit, periodic_orbits, OptimizedShooting

using NonlinearSolve


"""
    OptimizedShooting(; kwargs...)

A shooting method [Dednam2014](@cite) combined with Levenberg-Marquardt optimization 
to find periodic orbits of continuous-time dynamical systems.

## Keyword arguments
- `Δt::Float64 = 1e-6`: step between the points in the residual `R`. See below for details.
- `n::Int64 = 2`: `n*dimension(ds)` is the number of points in the residual `R`. See below 
  for details.
- `nonlinear_solve_kwargs = (reltol=1e-6, abstol=1e-6, maxiters=1000)`: keyword arguments 
  to pass to the `solve` function from 
  [`NonlinearSolve.jl`](https://github.com/SciML/NonlinearSolve.jl). For details on the 
  keywords see the respective package documentation.

## Description

Let us consider the following continuous-time dynamical system

```math
\\frac{dx}{dt} = f(x, p, t)
```

Dednam and Botha [Dednam2014](@cite) suggest to minimize the residual ``R`` defined as

```math
R = (x(T)-x(0), x(T+\\Delta t)-x(\\Delta t), \\dots, 
x(T+(n-1)\\Delta t)-x((n-1)\\Delta t))
```
where ``T`` is unknown period of a periodic orbit and ``x(t)`` is a solution at time ``t`` 
given some unknown initial point. Initial guess of the period ``T`` and the initial point 
is optimized by the Levenberg-Marquardt algorithm.

In our implementation keyword argument `n` corresponds to ``n`` in the residual ``R``. 
The keyword argument `Δt` corresponds to ``\\Delta t`` in the residual ``R``.
"""
@kwdef struct OptimizedShooting{T} <: PeriodicOrbitFinder
    Δt::Float64 = 1e-6
    n::Int64 = 2
    nonlinear_solve_kwargs::T = (reltol=1e-6, abstol=1e-6, maxiters=1000)
end

function periodic_orbit(ds::CoupledODEs, alg::OptimizedShooting, ig::InitialGuess)
    D = dimension(ds)
    f = (err, v, p) -> begin
        if isinplace(ds) 
            u0 = @view v[1:D]
        else
            u0 =  SVector{D}(v[1:D])
        end
        T = v[end]

        bounds = zeros(eltype(v), alg.n*2)
        for i in 0:alg.n-1
            bounds[i+1] = i*alg.Δt
            bounds[i+alg.n+1] = T + i*alg.Δt
        end
        tspan = (0.0, T + alg.n*alg.Δt)

        sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, 
        tspan=tspan); DynamicalSystemsBase.DEFAULT_DIFFEQ..., ds.diffeq..., saveat=bounds)
        if (length(sol.u) == alg.n*2)
            for i in 1:alg.n
                err[D*i-(D-1):D*i] = (sol.u[i] - sol.u[i+alg.n])
            end
        else
            fill!(err, Inf)
        end
    end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), [ig.u0..., ig.T])

    sol = solve(prob, NonlinearSolve.LevenbergMarquardt(); alg.nonlinear_solve_kwargs...)
    if sol.retcode == ReturnCode.Success
        u0 = sol.u[1:end-1]
        T = sol.u[end]
        Δt = 0.1
        return PeriodicOrbit(ds, u0, T, Δt)
    else
        return nothing
    end
end

function periodic_orbits(ds::CoupledODEs, alg::OptimizedShooting, igs::Vector{<:InitialGuess})
    pos = []
    for ig in igs
        res = periodic_orbit(ds, alg, ig)
        if !isnothing(res)
            push!(pos, res)
        end
    end
    return pos
end