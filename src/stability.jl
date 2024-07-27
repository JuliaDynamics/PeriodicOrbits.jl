export isstable

using LinearAlgebra: eigvals, mul!

"""
    isstable(ds::CoreDynamicalSystem, u0, T [, jac]) → true/false

Determine the local stability of the point `u0` laying on the periodic orbit with period `T`
using the jacobian rule `jac`. Returns `true` if the periodic orbit is stable, `false` if it is unstable.

For discrete systems, the stability is determined using eigenvalues of the jacobian of `T`-th 
iterate of the dynamical system `ds` at the point `u0`. If the maximum absolute value of the eigenvalues 
is less than `1`, the periodic orbit is marked as stable.

For continuous systems, the stability is determined by the Floquet multipliers of the monodromy matrix.
If the maximum absolute value of the Floquet multipliers is less than `1`, the periodic orbit is marked as stable.

The default value of jacobian rule `jac` is obtained via automatic differentiation.
"""
function isstable(ds::CoreDynamicalSystem, u0::AbstractArray{<:Real}, T::Real, jac=jacobian(ds))
    return _isstable(ds, u0, T, jac)
end

# discrete OOP
function _isstable(ds::DeterministicIteratedMap{false}, u0::AbstractArray{<:Real}, T::Integer, jac)
    T < 1 && throw(ArgumentError("Period must be a positive integer."))
    reinit!(ds, u0)
    J = jac(u0, current_parameters(ds), current_time(ds))

    for _ in 2:T
        J = jac(current_state(ds), current_parameters(ds), current_time(ds)) * J
        step!(ds)
    end

    eigs = eigvals(Array(J))
    return maximum(abs.(eigs)) < 1
end

# discrete IIP
function _isstable(ds::DeterministicIteratedMap{true}, u0::AbstractArray{<:Real}, T::Integer, jac!)
    T < 1 && throw(ArgumentError("Period must be a positive integer."))
    J0 = zeros(dimension(ds), dimension(ds))
    J1 = ones(dimension(ds), dimension(ds))
    dummyJ = copy(J0)
    reinit!(ds, u0)
    for _ in 1:T
        jac!(J0, current_state(ds), current_parameters(ds), current_time(ds))
        mul!(dummyJ, J0, J1)
        J1 .= dummyJ
        step!(ds)
    end

    eigs = eigvals(Array(J1))
    return maximum(abs.(eigs)) < 1
end

function _isstable(ds::CoupledODEs, u0::AbstractArray{<:Real}, T::AbstractFloat, jac)
    tands = TangentDynamicalSystem(ds, u0=u0; J=jac)
    step!(tands, T, true)
    monodromy = current_deviations(tands)
    floq_muls = abs.(eigvals(Array(monodromy)))
    sort!(floq_muls)
    x = floq_muls[end-1] + floq_muls[end] - 1
    return x < 1
end