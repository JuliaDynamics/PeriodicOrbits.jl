Base.summary(po::PeriodicOrbit) = "$(typeof(po))($(po.points[1]), $(po.T), $(po.stable))"

function Base.show(io::IO, po::PeriodicOrbit)
    print(io, summary(po))
end

function Base.show(io::IO, ::MIME"text/plain", po::PeriodicOrbit)
    digits = 5
    descriptors = [
        "u0" => round.(po.points[1], digits=digits),
        "T" => round(po.T, digits=digits),
        "stable" => po.stable,
        "discrete" => isdiscretetime(po),
        "length" => length(po.points),
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 3

    println(io, typeof(po))
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end

function Base.summary(ig::InitialGuess)
    digits = 5
    u0 = round.(ig.u0, digits=digits)
    T = isnothing(ig.T) ? "nothing" : round(ig.T, digits=digits)
    return "$(typeof(ig))($(u0), $(T))"
end

function Base.show(io::IO, ig::InitialGuess)
    print(io, summary(ig))
end

function Base.show(io::IO, ::MIME"text/plain", ig::InitialGuess)
    digits = 5
    descriptors = [
        "u0" => round.(ig.u0, digits=digits),
        "T" => isnothing(ig.T) ? "nothing" : round(ig.T, digits=digits)
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 3

    println(io, typeof(ig))
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end