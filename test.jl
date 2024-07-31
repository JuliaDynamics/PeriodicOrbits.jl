using PeriodicOrbits
using LinearAlgebra: norm
using BenchmarkTools

@inbounds function lorenz_rule(u, p, t)
    du1 = p[1] * (u[2] - u[1])
    du2 = u[1] * (p[2] - u[3]) - u[2]
    du3 = u[1] * u[2] - p[3] * u[3]
    return SVector{3}(du1, du2, du3)
end

#%%
ig = InitialGuess(SVector(1.0, 2.0, 5.0), 4.2)
alg = OptimizedShooting(Δt=1e-3, n=3, abstol=1e-6, optim_kwargs=(x_tol=1e-8, g_tol=1e-3, f_tol=1e-6))
ds = CoupledODEs(lorenz_rule, [0.0, 10.0, 0.0], [10.0, 28.0, 8 / 3]; diffeq=(abstol=1e-14, reltol=1e-14))

# @btime res = periodic_orbit(ds, alg, ig)
@profview for i in 1:20; periodic_orbit2(ds, alg, ig);end
@code_warntype periodic_orbit2(ds, alg, ig)
@btime periodic_orbit2(ds, alg, ig)


#%%
using LeastSquaresOptim
function rosenbrock(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(2)
optimize(rosenbrock, x0, Dogleg())
optimize(rosenbrock, x0, LevenbergMarquardt())
