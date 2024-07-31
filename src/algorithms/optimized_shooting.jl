export periodic_orbit, periodic_orbits, OptimizedShooting, periodic_orbit2, periodic_orbits2

using LeastSquaresOptim: optimize, LevenbergMarquardt
using NonlinearSolve


"""
    OptimizedShooting(; kwargs...)

A shooting method [Dednam2014](@cite) combined with Levenberg-Marquardt optimization 
to find periodic orbits of continuous-time systems.

## Keyword arguments
- `Δt::Float64 = 1e-6`: step in the residual `R` between points. See below.
- `n::Int64 = 2`: `n*dimension(ds)` is the number of points in the residual `R`. See below.
- `optim_kwargs::NamedTuple = (x_tol=1e-10,)`: keyword arguments to pass to the optimizer. 
  The optimizer used is the `optimize` from 
  [`LeastSquaresOptim.jl`](https://github.com/matthieugomez/LeastSquaresOptim.jl). 
  For details on the keywords see the respective package documentation.
- `abstol::Float64 = 1e-3` : absolute tolerance for sum of squares `ssr` of the 
  residual `R`. The method converged if `ssr <= abstol`.

## Description

Continuous dynamical system 

```math
\\frac{dx}{dt} = f(x, p, t)
```

can be rewritten in terms of the dimensionless time ``\\tau = t/T`` as

```math
\\frac{dx}{d\\tau} = Tf(x, p, T\\tau)
```
where ``T`` is a period of some periodic orbit. The boundary conditions for the 
periodic orbit now are ``x(\\tau = 0) = x(\\tau = 1)``. Dednam and Botha [Dednam2014](@cite) 
suggest minimizing the residual ``R`` defined as 

```math
R = (x(1)-x(0), x(1+\\Delta \\tau)-x(\\Delta \\tau), \\dots, 
x(1+(n-1)\\Delta \\tau)-x((n-1)\\Delta \\tau))
```

 with respect to ``x`` and ``T`` using Levenberg-Marquardt optimization.

In our implementation keyword argument `n` corresponds to ``n`` in the residual ``R``. 
The keyword argument `Δt` corresponds to ``\\Delta \\tau`` in the residual ``R``.
"""
@kwdef struct OptimizedShooting{T} <: PeriodicOrbitFinder
    Δt::Float64 = 1e-6
    n::Int64 = 2
    optim_kwargs::T = (x_tol=1e-8, g_tol=1e-3, f_tol=1e-6)
    abstol::Float64 = 1e-3
end

function periodic_orbit(ds::CoupledODEs, alg::OptimizedShooting, ig::InitialGuess)
    prob = NonlinearSolve.NonlinearLeastSquaresProblem(
        NonlinearSolve.NonlinearFunction(costfunc(ds, alg), resid_prototype = zeros(alg.n*dimension(ds))), [ig.u0..., ig.T])
    res = NonlinearSolve.solve(prob, NonlinearSolve.LevenbergMarquardt(), reltol = 1e-10, abstol = 1e-10)
    if res.retcode == 1
        u0 = res.u[1:dimension(ds)]
        T = res.u[end]
        return PeriodicOrbit(ds, u0, T, 0.01; jac=nothing)
    end
    return nothing
end

function periodic_orbit2(ds::CoupledODEs, alg::OptimizedShooting, ig::InitialGuess)
    R1 = zeros(alg.n*dimension(ds))
    R2 = copy(R1)
    err = copy(R1)
    cf = CostFunc(R1, R2, err)
    res = optimize(v -> cf(v, ds, alg), [ig.u0..., ig.T], LevenbergMarquardt(); alg.optim_kwargs...)
    if res.ssr <= alg.abstol
        u0 = res.minimizer[1:dimension(ds)]
        T = res.minimizer[end]
        return PeriodicOrbit(ds, u0, T, 0.01; jac=nothing)
    end
    return nothing
end

function periodic_orbits(ds::CoupledODEs, alg::OptimizedShooting, igs::Vector{<:InitialGuess})
    # TODO: annotate `pos` with correct type
    pos = []
    for ig in igs
        res = periodic_orbit(ds, alg, ig)
        if !isnothing(res)
            push!(pos, res)
        end
    end
    return pos
end

function periodic_orbits2(ds::CoupledODEs, alg::OptimizedShooting, igs::Vector{<:InitialGuess})
    # TODO: annotate `pos` with correct type
    pos = []
    for ig in igs
        res = periodic_orbit2(ds, alg, ig)
        if !isnothing(res)
            push!(pos, res)
        end
    end
    return pos
end

struct CostFunc{T}
    R1::Vector{T}
    R2::Vector{T}
    err::Vector{T}
end

function (cf::CostFunc)(v, ds, alg)
    (; R1, R2, err) = cf
    u0 = @view v[1:dimension(ds)]
    T = v[end]
    reinit!(ds, u0)
    compute_residual!(R1, ds, u0, T*alg.Δt, alg.n, 0)
    step!(ds, T - current_time(ds), true)
    compute_residual!(R2, ds, u0, T*alg.Δt, alg.n, T)
    err .= R2 .- R1
    return err
end

function costfunc(ds, alg)
    return (err, v, p) -> begin
        u0 = v[1:dimension(ds)]
        T = v[end]
        R1 = compute_residual(ds, u0, T*alg.Δt, alg.n, 0)
        R2 = compute_residual(ds, u0, T*alg.Δt, alg.n, T)
        err .= R2 .- R1
        return nothing
    end
end

function compute_residual(ds, u0, Δt, n, t0)
    len = n * dimension(ds)
    R = zeros(len)

    reinit!(ds, u0)
    step!(ds.integ, t0, true)
    for i in 0:n-1
        R[i*dimension(ds)+1:(i+1)*dimension(ds)] .= current_state(ds)
        step!(ds.integ, Δt, true)
    end
    return R
end

function compute_residual!(R, ds, u0, Δt, n, t0)
    for i in 0:n-1
        R[i*dimension(ds)+1:(i+1)*dimension(ds)] .= current_state(ds)
        step!(ds, Δt, true)
    end
end