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


using LeastSquaresOptim
#%%
function rosenbrock(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(2)
@time LeastSquaresOptim.optimize(rosenbrock, x0, LeastSquaresOptim.LevenbergMarquardt())



#%%
ig = InitialGuess(SVector(1.0, 2.0, 5.0), 4.2)
alg = OptimizedShooting(Δt=1e-3, n=3, abstol=1e-6, optim_kwargs=(x_tol=1e-8, g_tol=1e-3, f_tol=1e-6))
ds = CoupledODEs(lorenz_rule, [0.0, 10.0, 0.0], [10.0, 28.0, 8 / 3]; diffeq=(abstol=1e-14, reltol=1e-14))

res = periodic_orbit(ds, alg, ig)

#%%
function nlls!(du, u, p)
    du[1] = 2u[1] - 2
    du[2] = u[1] - 4u[2]
    du[3] = 0
    du[4] = 0
    du[5] = 0
end
u0 = [0.0, 0.0]
prob = NonlinearLeastSquaresProblem(
    NonlinearFunction(nlls!, resid_prototype = zeros(5)), u0)
@time res = solve(prob, NonlinearSolve.LevenbergMarquardt(), reltol = 1e-12, abstol = 1e-12)





#%%
using OrdinaryDiffEq
f(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
