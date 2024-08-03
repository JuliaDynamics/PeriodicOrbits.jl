Base.summary(po::PeriodicOrbit) = "$(typeof(po))($(po.points[1]), $(po.T), $(po.stable))"

function Base.show(io::IO, po::PeriodicOrbit)
    print(io, summary(po))
end

function Base.show(io::IO, ::MIME"text/plain", po::PeriodicOrbit)
    descriptors = [
        "u0" => po.points[1],
        "T" => po.T,
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