using NoiseContrastiveEstimation
using Documenter
ENV["GKSwstype"] = "100"
@show joinpath(@__DIR__, "src")

makedocs(;
    modules = [NoiseContrastiveEstimation],
    authors = "Francesco Alemanno <francescoalemanno710@gmail.com> and contributors",
    repo = "https://github.com/francescoalemanno/NoiseContrastiveEstimation.jl/blob/{commit}{path}#L{line}",
    sitename = "NoiseContrastiveEstimation.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://francescoalemanno.github.io/NoiseContrastiveEstimation.jl",
        assets = String[],
    ),
    pages = ["Basic Usage" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
    repo = "github.com/francescoalemanno/NoiseContrastiveEstimation.jl",
    push_preview = true,
)
