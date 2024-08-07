using PeriodicOrbits
using LinearAlgebra: norm
using BenchmarkTools
using NonlinearSolve
using CairoMakie
import ForwardDiff

function plot_result(res, T, ds; azimuth = 1.3 * pi, elevation = 0.3 * pi)
    traj, t = trajectory(ds, T, res; Dt = 0.001)
    fig = Figure()
    ax = Axis3(fig[1,1], azimuth = azimuth, elevation=elevation)
    lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)
    scatter!(ax, res)
    return fig
end

@inbounds function lorenz_rule(u, p, t)
    du1 = p[1] * (u[2] - u[1])
    du2 = u[1] * (p[2] - u[3]) - u[2]
    du3 = u[1] * u[2] - p[3] * u[3]
    return SVector{3}(du1, du2, du3)
end
function roessler(u0=[1.0, -2.0, 0.1]; a = 0.2, b = 0.2, c = 5.7)
    return CoupledODEs(roessler_rule, u0, [a, b, c])
end
@inbounds function roessler_rule(u, p, t)
    du1 = p[4] * (-u[2]-u[3])
    du2 = p[4] * (u[1] + p[1]*u[2])
    du3 = p[4] * (p[2] + u[3]*(u[1] - p[3]))
    return [du1, du2, du3]
end
@inbounds function roessler_rule2(u, p, t)
    du1 = p[4] * (-u[2]-u[3])
    du2 = p[4] * (u[1] + p[1]*u[2])
    du3 = p[4] * (p[2] + u[3]*(u[1] - p[3]))
    return SVector{3}(du1, du2, du3)
end
@inbounds function roessler_rule3(du, u, p, t)
    du[1] = -u[2]-u[3]
    du[2] = u[1] + p[1]*u[2]
    du[3] = p[2] + u[3]*(u[1] - p[3])
    return nothing
end
@inbounds function roessler_rule4(u, p, t)
    du1 = -u[2]-u[3]
    du2 = u[1] + p[1]*u[2]
    du3 = p[2] + u[3]*(u[1] - p[3])
    return SVector{3}(du1, du2, du3)
end
@inbounds function roessler_rule5(du, u, p, t)
    du[1] = u[4] * (-u[2]-u[3])
    du[2] = u[4] * (u[1] + p[1]*u[2])
    du[3] = u[4] * (p[2] + u[3]*(u[1] - p[3]))
    du[4] = 0
    return 0
    # return SVector{4}(du1, du2, du3, du4)
end
@inbounds function roessler_rule6(u, p, t)
    du1 = u[4] * (-u[2]-u[3])
    du2 = u[4] * (u[1] + p[1]*u[2])
    du3 = u[4] * (p[2] + u[3]*(u[1] - p[3]))
    du4 = 0
    return SVector{4}(du1, du2, du3, du4)
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
using NonlinearSolve
f(u, p, t) = 1.01 * u
u0 = 1 / 2
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)



#%%

function lorenz!(du, u, p, t)
    du[1] = (p[1] * (u[2] - u[1]))
    du[2] = (u[1] * (p[2] - u[3]) - u[2])
    du[3] = (u[1] * u[2] - p[3] * u[3])
end


u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1.0)
p = [10.0, 28.0, 8 / 3]
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8, saveat=[0.0, 0.5, 0.7])



function func(u0::AbstractArray{L}, alg) where L
    f = (err, v, p) -> begin
            u0 = v[1:3]
            T = v[end]

            tspan = (0.0, 1.0 + alg.n*alg.Δt)
            p = [0.25, 0.2, 3.5, T]
            prob = ODEProblem(roessler_rule, u0, tspan, p)
            sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8, saveat=[0:alg.Δt:((alg.n-1)*alg.Δt)..., 1.0:alg.Δt: (1.0 + (alg.n-1)*alg.Δt)...])
            for i in 1:alg.n
                err[3*i-2:3*i] = sol.u[i] - sol.u[end-alg.n+i]
            end
        end

    p = []
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*3)), u0, p)

    solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-6, abstol = 1e-6, maxiters=1500)
end
function func2(u0::AbstractArray{L}, p0, alg, ds) where L
    f = (err, v, p) -> begin
            u0 =  SVector(v[1:3]...)
            T = v[end]

            bounds = zeros(typeof(T), alg.n*2)
            for i in 0:alg.n-1
                bounds[i+1] = i*alg.Δt
                bounds[i+alg.n+1] = T + i*alg.Δt
            end
            tspan = (0.0, T + alg.n*alg.Δt)
            p0 = typeof(T).(p)
            p0[end] = T
            sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, p=p0, tspan=tspan), Tsit5(), reltol = 1e-8, abstol = 1e-8, saveat=bounds)
            dim = dimension(ds)
            for i in 1:alg.n
                err[dim*i-(dim-1):dim*i] = sol.u[i] - sol.u[i+alg.n]
            end
        end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), u0, p0)

    solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-8, abstol = 1e-8, maxiters=1000)
end
#%%
u0 = [3.8, 2.0, 4.0, 6.0]
p0 = [0.21, 0.3, 3.6]
alg = OptimizedShooting(Δt=1e-4, n=3)
ds2 = CoupledODEs(roessler_rule4, zeros(3), p0)
@time res = func2(u0, p0, alg, ds2)

u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(roessler_rule4, u, [p0[1:end-1]..., T], diffeq=(abstol=1e-10, reltol=1e-10))
plot_result(u, T, ds; azimuth = 1.3pi, elevation=0.1pi)


#%%
function func3(u0::AbstractArray{L}, p0, alg, ds) where L
    bounds = zeros(alg.n*2)
    for i in 0:alg.n-1
        bounds[i+1] = i*alg.Δt
        bounds[i+alg.n+1] = 1.0 + i*alg.Δt
    end
    tspan = (0.0, 1.0 + alg.n*alg.Δt)
    f = (err, v, p) -> begin
            u0 =  SVector(v[1:3]...)
            T = v[end]
            p0 = typeof(T).(p)
            p0[end] = T
            sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, p=p0, tspan=tspan), Tsit5(), reltol = 1e-8, abstol = 1e-8, saveat=bounds)
            dim = dimension(ds)
            for i in 1:alg.n
                err[dim*i-(dim-1):dim*i] = sol.u[i] - sol.u[i+alg.n]
            end
        end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), u0, p0)

    solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-8, abstol = 1e-8, maxiters=1000)
end
#%%
u0 = [0.1, 2.0, 3.0, 11.0]
p0 = [0.25, 0.45, 3.5, 0.0]
alg = OptimizedShooting(Δt=1e-4, n=3)
ds3 = CoupledODEs(roessler_rule2, zeros(3), p0)
@btime res = func3(u0, p0, alg, ds3)

u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(roessler_rule2, u, [p0[1:end-1]..., T], diffeq=(abstol=1e-10, reltol=1e-10))
plot_result(u, 1.0, ds; azimuth = 1.3pi, elevation=0.1pi)


#%%
u0 = [0.1, 2.0, 3.0, 30.0]
p0 = [0.25, 0.35, 3.0, 0.0]
alg = OptimizedShooting(Δt=1e-4, n=3)
ds3 = CoupledODEs(roessler_rule2, zeros(3), p0)
@time res = func3(u0, p0, alg, ds3)

u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(roessler_rule2, u, [p0[1:end-1]..., T], diffeq=(abstol=1e-10, reltol=1e-10))
plot_result(u, 1.0, ds; azimuth = 1.3pi, elevation=0.1pi)



#%%
function func4(u0::AbstractArray{L}, p0, alg, ds) where L
    bounds = zeros(alg.n*2)
    for i in 0:alg.n-1
        bounds[i+1] = i*alg.Δt
        bounds[i+alg.n+1] = 1.0 + i*alg.Δt
    end
    tspan = (0.0, 1.0 + alg.n*alg.Δt)
    f = (err, u0, p) -> begin
            sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, p=p, tspan=tspan), Tsit5(), reltol = 1e-8, abstol = 1e-8, saveat=bounds)
            dim = dimension(ds)
            for i in 1:alg.n
                err[dim*i-(dim-1):dim*i] = sol.u[i] - sol.u[i+alg.n]
            end
        end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), u0, p0)

    solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-8, abstol = 1e-8, maxiters=1000)
end
#%%
u0 = [0.1, 2.0, 3.0, 11.0]
p0 = [0.25, 0.45, 3.5]
alg = OptimizedShooting(Δt=1e-4, n=2)
ds5 = CoupledODEs(roessler_rule5, zeros(4), p0)
@btime res = func4(u0, p0, alg, ds5)
println(res.retcode)

u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(roessler_rule4, u, p0, diffeq=(abstol=1e-10, reltol=1e-10))
plot_result(u, T, ds; azimuth = 1.3pi, elevation=0.1pi)