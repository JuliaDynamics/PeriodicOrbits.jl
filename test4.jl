using BoundaryValueDiffEq
using LinearAlgebra
using CairoMakie

#%%
const g = 9.81
L = 1.0
tspan = (0.0, pi / 2)
function simplependulum!(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g / L) * sin(θ)
end
function bc1!(residual, u, p, t)
    residual[1] = u[end ÷ 2][1] + pi / 2 # the solution at the middle of the time span should be -pi/2
    residual[2] = u[end][1] - pi / 2 # the solution at the end of the time span should be pi/2
end
bvp1 = BVProblem(simplependulum!, bc1!, [pi / 2, pi / 2], tspan)
@time sol1 = solve(bvp1, MIRK4(), dt = 0.05)


#%%
using BoundaryValueDiffEq, OrdinaryDiffEq

@inbounds function roessler_rule(du, u, p, t)
    du[1] = p[4] * (-u[2]-u[3])
    du[2] = p[4] * (u[1] + p[1]*u[2])
    du[3] = p[4] * (p[2] + u[3]*(u[1] - p[3]))
    return nothing
end

# Initial guess for the solution (including guess for T)
initial_guess = [3.0, 4.0, 3.0] # Initial conditions for x, v, and guess for T

# Define the boundary condition function
function bc(residual, u, p, t)
    residual[1:3] = u[1] .- u[end]
    # residual[1] = u[1][1] - u[end][1]  # x(0) - x(T)
    # residual[2] = u[1][2] - u[end][2]  # v(0) - v(T)
    # residual[3] = u[1][3] - u[end][3]  # v(0) - v(T)
    # residual[3] = u[end][3] - u[1][3]  # T should be consistent (T(T) = T(0))
end

# Time span (now from 0 to 1 since T is included as a variable)
time_span = (0.0, 1.0)
p = [0.25, 0.2, 3.5, 6.2]
bvp_prob = BVProblem(roessler_rule, bc, initial_guess, time_span, p)

# Solve the BVP
@time sol = solve(bvp_prob, MIRK4(), dt = 0.01)

#%%


fig = Figure()
ax = Axis3(fig[1,1])
lines!(ax, [x[1] for x in sol.u], [x[2] for x in sol.u], [x[3] for x in sol.u])
# lines!(ax, [x[1] for x in sol.u], [x[2] for x in sol.u])
display(fig)