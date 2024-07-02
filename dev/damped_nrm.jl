using PeriodicOrbits
using DynamicalSystems
using CairoMakie


@inbounds function lorenz_jacob(u, p, t)
    return SMatrix{3,3}(-p[1], p[2] - u[3], u[2], p[1], -1.0, u[1], 0.0, -u[1], -p[3])
end

begin
    ds = Systems.lorenz()
    alg = DampedNewtonRaphsonMees(δ=2^(-6), J=lorenz_jacob, maxiter=10000, disttol=1e-4)
    traj, t = trajectory(ds, 10; Dt=0.5)
    igs = InitialGuess[InitialGuess(x, 7*rand()) for x in traj]
    pos = periodic_orbits(ds, alg, igs)
end


begin
    ds = Systems.lorenz(ρ=350)
    alg = DampedNewtonRaphsonMees(δ=2^(-1), J=lorenz_jacob, maxiter=5000, disttol=1e-6)
    traj, t = trajectory(ds, 10; Dt=0.1)
    # igs = InitialGuess[InitialGuess(x, 5*rand()) for x in traj]
    igs = InitialGuess[InitialGuess(traj[end], 5*rand())]
    @time pos = periodic_orbits(ds, alg, igs)
end

# Example of stable periodic orbit in lorenz attractor
begin
    u = traj[end]
    T = 0.39
    traj3, t = trajectory(ds, T, u; Dt=0.001)

    fig = Figure()
    axis = Axis3(fig[1, 1], aspect = (1, 1, 1), azimuth=-0.15pi, elevation=0.05pi)

    lines!(axis, traj3[:, 1], traj3[:, 2], traj3[:, 3], color = :blue, linewidth = 2.00)
    scatter!(axis, u[1], u[2], u[3], color = :red, markersize = 15.0)
    display(fig)
end

begin
    u = traj3[end]
    T = 0.39
    alg = DampedNewtonRaphsonMees(δ=2^(-1), J=lorenz_jacob, maxiter=5000, disttol=1e-6)
    traj, t = trajectory(ds, 10; Dt=0.1)
    igs = InitialGuess[
        InitialGuess(u + SVector{3}(1.1*rand(3)), 0.49)
    ]
    @time pos = periodic_orbits(ds, alg, igs)
    display(pos[1].points)

    # u = [5.973265115317724, 43.63696712944919, 286.1587689550906]
    # T = 2.716031518506218
    if length(pos[1].points) > 0 
        u = pos.points[1].u
        T = pos.points[1].T

        traj2, t = trajectory(ds, T, u; Dt=0.001)

        fig = Figure()
        axis = Axis3(fig[1, 1], aspect = (1, 1, 1), azimuth=-0.15pi, elevation=0.05pi)

        lines!(axis, traj2[:, 1], traj2[:, 2], traj2[:, 3], color = :blue, linewidth = 2.00)
        scatter!(axis, u[1], u[2], u[3], color = :red, markersize = 15.0)
        display(fig)
    end
end

begin
    # for i in LinRange(0.0, 8.031791028996272, 100)
        u, T = ([1.3888000396978526, 2.9480907150257925, 1.947618665281444], 11.82803972636592)
        traj3, t = trajectory(ds, T, u; Dt=0.001)

        fig = Figure()
        axis = Axis3(fig[1, 1], aspect = (1, 1, 1), azimuth=-0.15pi, elevation=0.05pi, title="Another UPO!")

        lines!(axis, traj3[:, 1], traj3[:, 2], traj3[:, 3], color = :blue, linewidth = 2.00)
        scatter!(axis, u[1], u[2], u[3], color = :red, markersize = 15.0)
        display(fig)
        save("anotherupo.png", fig)
    # end
end


# %%
u = [-28.285420783580218, -13.243360330826398, 91.26460634801842]
T = 0.9
ds = Systems.lorenz()
rule = dynamic_rule(ds)
reverse_f(u, p, t) = -rule(u, p, t)
reverse_J(u, p, t) = -lorenz_jacob(u, p, t)
reverse_ds = CoupledODEs(reverse_f, current_state(ds), current_parameters(ds))
reverse_tands = TangentDynamicalSystem(reverse_ds; J = reverse_J, k=dimension(ds))
reinit!(reverse_tands, u)
step!(reverse_tands, NaN)
current_deviations(reverse_tands)

reinit!(reverse_tands, u)
for t in 0:0.1:1.5
    step!(reverse_tands, 0.1)
    println(current_deviations(reverse_tands))
end



begin
    u, T = ([0.5345029350672663, 0.026172617856423462, 30.960602519089996], 0.19176616542026598)
    reinit!(ds, u)
    step!(ds, T)
    DynamicalSystemsBase.norm(current_state(ds) - u)
end