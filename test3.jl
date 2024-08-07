using ForwardDiff
using DynamicalSystemsBase

function henon(u0=zeros(2); a = 1.4, b = 0.3)
    return DeterministicIteratedMap(henon_rule, u0, [a,b])
end # should give lyapunov exponents [0.4189, -1.6229]
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])

function f(x)
    return x[1]^2 + x[2]^2
end

function test(ds, x)
    reinit!(ds, x)
    step!(ds)
end

#%%
ForwardDiff.gradient(f, [1.0, 2.0])
ds = henon()
ForwardDiff.gradient(x->test(ds, x), [1.0, 2.0])