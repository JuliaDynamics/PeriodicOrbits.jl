# Algorithms

## Optimized Shooting Method
```@docs
OptimizedShooting
```

## Example
```@example MAIN
using PeriodicOrbits
using CairoMakie
using OrdinaryDiffEq

@inbounds function roessler_rule(u, p, t)
    du1 = -u[2]-u[3]
    du2 = u[1] + p[1]*u[2]
    du3 = p[2] + u[3]*(u[1] - p[3])
    return SVector{3}(du1, du2, du3)
end

function plot_result(res, ds; azimuth = 1.3 * pi, elevation = 0.3 * pi)
    traj, t = trajectory(ds, res.T, res.points[1]; Dt = 0.01)
    fig = Figure()
    ax = Axis3(fig[1,1], azimuth = azimuth, elevation=elevation)
    lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)
    scatter!(ax, res.points[1])
    return fig
end

a = 0.15; b=0.2; c=3.5
ig = InitialGuess(SVector(2.0, 5.0, 10.0), 10.2)
alg = OptimizedShooting(Δt=1/(2^6), n=3)
ds = CoupledODEs(roessler_rule, [1.0, -2.0, 0.1], [a, b, c]; diffeq = (abstol = 1e-14, reltol = 1e-14))
res = periodic_orbit(ds, alg, ig)
plot_result(res, ds; azimuth = 1.3pi, elevation=0.1pi)
```