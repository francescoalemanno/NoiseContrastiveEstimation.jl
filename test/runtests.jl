using NoiseContrastiveEstimation
using Test
@testset "Basic Functional" begin
    lϕ(x, θ) = θ[1] * x + θ[2] * x * x / 2
    gϕ(x, ϕ) = [x, x * x / 2]
    data = randn(200) .* 0.3 .+ 1.5
    noised = foldl(hcat, x .+ randn(20) .* 0.4 for x in data)
    J = CNCE(lϕ, gϕ, data, noised)
    @test J(Cost(), (15, -10)) < J(Cost(), (7.5, -5)) < J(Cost(), (0, 0))
    x = zeros(2)
    v = zeros(2)
    μ=0.9
    η=10.0
    i=0
    while true
        v = μ.* v .- J(Grad(), x.+η.*μ.*v)
        x = x .+ η.*v
        i=i+1
        sum(abs2, v) < 1e-10 && break
    end
    @test J(Cost(), x) < J(Cost(), (15, -10))
    Jgless = CNCE(lϕ, data, noised)
    @test_throws ErrorException("Gradient is not available") Jgless.gϕ(1,2)
    @test J(Cost(), x) == Jgless(Cost(), x)
end

@testset "RBM training" begin
    function gϕ(σ,ξ)
        sc=ξ*σ
        g=similar(ξ)
        for i in 1:size(ξ,1), j in 1:size(ξ,2)
            g[i,j]=2*σ[j]*sc[i]
        end
        g
    end

    function lϕ(σ,ξ)
        sc=ξ*σ
        sc'sc
    end
    #=
    using ReverseDiff
    gϕ(σ,ξ)=ReverseDiff.gradient(x->lϕ(σ,x),ξ)
    =#
    data = [sign.(randn(10)) for i in 1:2]
    noised = [sign.(randn(10).+1.3).*data[i] for j in 1:70, i in eachindex(data)]
    J = CNCE(lϕ, gϕ, data, noised)
    ξ=sign.(randn(4,10))
    J(Grad(),ξ)

    x = sign.(randn(4,10)).*0.01
    v = x.*0
    μ=0.95
    η=0.01
    i=0
    C=J(Cost(), x)
    while true
        v = μ.* v .- J(Grad(), x.+η.*μ.*v)
        x = x .+ η.*v
        i=i+1
        n=J(Cost(), x)-C
        C+=n
        @show n,C
        abs(n) < 1e-8 && break
    end

    dx=sign.(x)

    M=[maximum(abs.(dx*data[1]/length(data[1]))) for i in 1:length(data)]
    @test sum(M)==2
end
