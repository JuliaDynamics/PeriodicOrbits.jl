using PeriodicOrbits
using LinearAlgebra: norm
using BenchmarkTools
using NonlinearSolve
using CairoMakie
using OrdinaryDiffEq

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

@inbounds function roessler_rule(u, p, t)
    du1 = -u[2]-u[3]
    du2 = u[1] + p[1]*u[2]
    du3 = p[2] + u[3]*(u[1] - p[3])
    return SVector{3}(du1, du2, du3)
end

function transform_rule(rule)
    return (u, p, t) -> begin
        new_p = @view p[1:end-1]
        T = p[end]
        return T * rule(u, new_p, T*t)
    end
end


#%%
function detect(u0, p0, alg, ds)
    D = dimension(ds)
    f = (err, v, p) -> begin
            # period = eltype(v) <: ForwardDiff.Dual ? v[end].value : v[end]
            # @show period
            T = v[end]
            bounds = zeros(eltype(v), alg.n*2)
            for i in 0:alg.n-1
                bounds[i+1] = i*alg.Δt
                bounds[i+alg.n+1] = T + i*alg.Δt
            end
            tspan = (0.0, T + alg.n*alg.Δt)
            if isinplace(ds) 
                u0 = @view v[1:D]
            else
                u0 =  SVector{D}(v[1:D])
            end

            # sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, 
            # tspan=tspan), Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat=bounds)
            sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, 
            tspan=tspan); DynamicalSystemsBase.DEFAULT_DIFFEQ..., ds.diffeq..., saveat=bounds)
            dim = dimension(ds)
            for i in 1:alg.n
                err[dim*i-(dim-1):dim*i] = sol.u[i] - sol.u[i+alg.n]
            end
        end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), u0, p0)

    solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-6, abstol = 1e-6, maxiters=1000)
end
# function detect(u0, p0, alg, ds)
#     rule = dynamic_rule(ds)
#     rule = transform_rule(rule)
#     ds = CoupledODEs(rule, zeros(dimension(ds)), p0)
#     bounds = zeros(alg.n*2)
#     for i in 0:alg.n-1
#         bounds[i+1] = i*alg.Δt
#         bounds[i+alg.n+1] = 1.0 + i*alg.Δt
#     end
#     tspan = (0.0, 1.0 + alg.n*alg.Δt)
#     D = dimension(ds)
#     f = (err, v, p) -> begin
#             u0 =  SVector{D, eltype(u0)}(@view v[1:3])
#             T = v[end]
#             p0 = typeof(T).(p) # needed for ForwardDiff
#             p0[end] = T
#             sol = solve(SciMLBase.remake(ds.integ.sol.prob; u0=u0, p=p0, 
#             tspan=tspan), Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat=bounds)
#             dim = dimension(ds)
#             for i in 1:alg.n
#                 err[dim*i-(dim-1):dim*i] = sol.u[i] - sol.u[i+alg.n]
#             end
#         end

#     prob = NonlinearLeastSquaresProblem(
#         NonlinearFunction(f, resid_prototype = zeros(alg.n*dimension(ds))), u0, p0)

#     solve(prob, NonlinearSolve.LevenbergMarquardt(); reltol = 1e-6, abstol = 1e-6, maxiters=1000)
# end

#%%
u0 = [0.1, -5.0, 1.0, 2.1]
p0 = [10.0, 28.0, 8/3]
alg = OptimizedShooting(Δt=1e-4, n=5)
ds = CoupledODEs(lorenz_rule, zeros(3), p0[1:3])
@time res = detect(u0, p0, alg, ds)


u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(lorenz_rule, u, p0, diffeq=(abstol=1e-14, reltol=1e-14))
plot_result(u, 1*T, ds; azimuth = 1.8pi, elevation=0.1pi)


#%%
u0 = [2.1, 10.0, 3.0, 5.0]
p0 = [10.0, 28.0, 8/3, 0.0]
alg = OptimizedShooting(Δt=1e-4, n=3)
ds = CoupledODEs(lorenz_rule, zeros(3), p0[1:3])
@time res = detect(u0, p0, alg, ds)


u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(lorenz_rule, u, p0[1:end-1], diffeq=(abstol=1e-14, reltol=1e-14))
plot_result(u, 1*T, ds; azimuth = 1.8pi, elevation=0.1pi)

#%%
u0 = [3.1, 4.0, 3.0, 11.0]
p0 = [0.2, 0.2, 3.6, 0.0]
alg = OptimizedShooting(Δt=1e-4, n=3)
ds = CoupledODEs(roessler_rule, zeros(3), p0[1:3])
@time res = detect(u0, p0, alg, ds)

u = SVector{3}(res.u[1:3])
T = res.u[end]
ds = CoupledODEs(roessler_rule, u, p0[1:end-1], diffeq=(abstol=1e-14, reltol=1e-14))
plot_result(u, 1*T, ds; azimuth = 1.3pi, elevation=0.1pi)

#%%
using PredefinedDynamicalSystems

ds = PredefinedDynamicalSystems.lorenz96(5)
u0 = [5*rand(5)..., 15.0]
p0 = [16.0]
alg = OptimizedShooting(Δt=1e-4, n=5)
@time res = detect(u0, p0, alg, ds)


u = SVector{3}(res.u[1:3])
T = res.u[end]
plot_result(u, 1.0*T, ds; azimuth = 1.8pi, elevation=0.1pi)