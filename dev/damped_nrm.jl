using PeriodicOrbits
using CairoMakie

function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CoupledODEs(lorenz_rule, u0, [σ, ρ, β], diffeq = (abstol = 1e-16, reltol = 1e-16))
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

#%%
ds = lorenz()
alg = DampedNewtonRaphsonMees(δ=2^(-2), J=lorenz_jacob, maxiter=8000, disttol=1e-2)
traj, t = trajectory(ds, 4; Dt=1.0)
igs = InitialGuess[InitialGuess(x, 10*rand()) for x in traj]
pos = periodic_orbits(ds, alg, igs)
reinit!(ds, traj[end])
display(pos)