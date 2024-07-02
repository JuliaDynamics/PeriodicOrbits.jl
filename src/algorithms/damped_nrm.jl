export periodic_orbits, DampedNewtonRaphsonMees

using LinearAlgebra: I, norm, SingularException

@kwdef struct DampedNewtonRaphsonMees <: PeriodicOrbitFinder
    δ = 2^(-3)
    J = nothing
    maxiter = 10000
    disttol = 1e-4
    maxperiod = 50
    inftol = 1e7
    checkperiod = 1.0
end

function periodic_orbits(ds::CoupledODEs, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess})
    tands = TangentDynamicalSystem(ds; J = alg.J, k=dimension(ds))
    pos = PeriodicOrbit{typeof(current_state(ds)), typeof(current_time(ds))}[]

    reverse_f(u, p, t) = -dynamic_rule(ds)(u, p, t)
    reverse_J(u, p, t) = -alg.J(u, p, t)
    reverse_ds = CoupledODEs(reverse_f, current_state(ds), current_parameters(ds))

    reverse_tands = TangentDynamicalSystem(reverse_ds; J = reverse_J, k=dimension(ds))

    dim = dimension(ds)+1

    j = 0
    for ig in igs
        j+=1
        println("Initial guess $j")
        # try
            prev_step = SVector{length(ig.u0)+1}(vcat(ig.u0, ig.T))

            if prev_step[end] >= 0.0
                ds2 = tands
            else
                ds2 = reverse_tands
            end

            for _ in 1:alg.maxiter
                reinit!(ds2, prev_step[1:end-1])
                step!(ds2, abs(prev_step[end]))

                if norm(current_state(ds2) - prev_step[1:end-1]) < alg.disttol
                    push!(pos, PeriodicOrbit{typeof(current_state(ds)), typeof(current_time(ds))}([prev_step[1:end-1]], prev_step[end]))
                    break
                end

                prev_step, status = next_step(prev_step, ds2, alg, dim)
                status == false && break
            end
        # catch e
        #     @warn e.msg
        # end
    end
    return pos
end

function next_step(prev_step, ds, alg, dim)
    X = prev_step[1:end-1]
    T = prev_step[end]
    reinit!(ds, X)

    for t in 0:alg.checkperiod:T
        if maximum(abs.(current_deviations(ds))) > alg.inftol
            return prev_step, false
        end
        step!(ds, abs(t))
    end

    Phi = current_deviations(ds)
    phi = current_state(ds)

    F = ds.original_f
    p = current_parameters(ds)
    t = current_time(ds)

    A = hcat(vcat(Phi-I, F(X, p, t)'), vcat(F(phi, p, t), 0))
    b = vcat(X-phi, 0)

    try
        Δ = A\b
        return prev_step + alg.δ*Δ, true
    catch e
        if e isa SingularException
            return prev_step, false
        else
            rethrow(e)
        end
    end
end