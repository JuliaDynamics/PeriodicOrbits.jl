cd(@__DIR__)

using PeriodicOrbits
using PeriodicOrbits.DynamicalSystemsBase

pages = [
    "index.md",
    "tutorial.md",
    "api.md",
    "algorithms.md",
    "examples.md",
    "developer.md",
    "references.md"
]

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib");
    style=:authoryear
)

build_docs_with_style(pages, PeriodicOrbits, DynamicalSystemsBase; bib)