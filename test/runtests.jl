using NoiseContrastiveEstimation
using Test, Random
using ReverseDiff, LinearAlgebra

@testset "Basic Functional" begin
    R = MersenneTwister(125)
    lϕ(x, θ) = θ[1] * x + θ[2] * x * x / 2
    gϕ(x, ϕ) = [x, x * x / 2]
    data = randn(R, 200) .* 0.3 .+ 1.5
    noised = foldl(hcat, x .+ randn(R, 20) .* 0.4 for x in data)
    J = CNCE(f = lϕ, grad_f = gϕ, data = data, noised = noised)
    @test J((15, -10)).J < J((7.5, -5)).J < J((0, 0)).J
    results = nesterov(J, zeros(2), 0.9, 7.0)
    x = results.sol
    @test J(x).J < J((15, -10)).J
    Jgless = CNCE(f = lϕ, data = data, noised = noised)
    @test Jgless.grad_f(1, 2) == 0
    @test J(x).J == Jgless(x).J
end

@testset "RBM training" begin
    R = MersenneTwister(125)
    function gϕ(σ, ξ)
        sc = ξ * σ
        g = similar(ξ)
        for i = 1:size(ξ, 1), j = 1:size(ξ, 2)
            g[i, j] = 2 * σ[j] * sc[i]
        end
        g
    end

    function lϕ(σ, ξ)
        sc = ξ * σ
        sc'sc
    end

    data = [sign.(randn(R, 10)) for i = 1:2]
    noised = [sign.(rand(R, 10) .- 1 / 3) .* data[i] for j = 1:20, i in eachindex(data)]
    J = CNCE(lϕ, gϕ, data, noised)
    x = sign.(randn(R, 4, 10)) .* 0.01
    results = nesterov(J, x, 0.95, 0.05)
    @show results
    dx = sign.(results.sol)
    M = [maximum(abs.(dx * data[i] / length(data[i]))) for i = 1:length(data)]
    @test sum(M) == 2
end

@testset "Estimate Covariate Model" begin
    R = MersenneTwister(125)
    gϕ(x, θ) = [x[1]*x[1], x[2]*x[2], x[1]*x[2], x[1], x[2]]
    lϕ(x, θ) = θ'gϕ(x, θ)

    data = [[x+1,y+1] for (x,y) in zip(randn(R,200),randn(R,200))]
    noised = [x+0.3*randn(2) for j=1:50, x in data]
    J = CNCE(f = lϕ, grad_f = gϕ, data = data, noised = noised)
    results = nesterov(J, zeros(5), 0.9, 2.5)
    @test results.sol[1]<0
    @test results.sol[2]<0
    @test results.sol[4]>0
    @test results.sol[5]>0
end

@testset "DBM training" begin
    R = MersenneTwister(125)
    
    function lϕ(σ, ξ)
        L = length(σ)
        t = L ÷ 3
        sc1 = view(ξ, :, 1:t) * view(σ, 1:t)
        sc2 = view(ξ, :, (t+1):2t) * view(σ, (t+1):2t)
        sc3 = view(ξ, :, (2t+1):L) * view(σ, (2t+1):L)
        dot(sc1 + sc2, sc1 + sc2) + dot(sc2 + sc3, sc2 + sc3)
    end

    gϕ(σ, ξ) = ReverseDiff.gradient(x -> lϕ(σ, x), ξ)

    Nσ = 20
    Nμ = 4
    Ndata = 2
    Nnoise = 20
    p_flip = 1 / 3

    data = [sign.(randn(R, Nσ)) for i = 1:Ndata]
    noised = [sign.(rand(R, Nσ) .- p_flip) .* data[i] for j = 1:Nnoise, i in eachindex(data)]
    J = CNCE(lϕ, gϕ, data, noised)
    x = sign.(randn(R, Nμ, Nσ)) .* 0.01
    results = nesterov(J, x, 0.9, 0.5)
    dx = sign.(results.sol)
    M = [maximum(abs.(dx * data[i] / length(data[i]))) for i = 1:length(data)]
    @test sum(M) == 2
end

struct DeepBM{Nl,T}
    partitions::NTuple{Nl,T}
end

function (dbm::DeepBM{Nl,T})(σ,ξ) where {Nl,T}
    sc = map(p -> view(ξ, :, p) * view(σ, p), dbm.partitions)
    ll = map(length, dbm.partitions)
    length(σ)*sum(x->dot(x,x), (sc[i] + sc[i+1]) / (ll[i] + ll[i+1]) for i in 1:(Nl-1))
end

@testset "DBM general training" begin
    R = MersenneTwister(125)
    
    Nσ = 20
    Nμ = 4
    Ndata = 2
    Nnoise = 20
    p_flip = 1 / 3

    lϕ=DeepBM((1:7,8:15,16:20))
    gϕ(σ, ξ) = ReverseDiff.gradient(x -> lϕ(σ, x), ξ)

    data = [sign.(randn(R, Nσ)) for i = 1:Ndata]
    noised = [sign.(rand(R, Nσ) .- p_flip) .* data[i] for j = 1:Nnoise, i in eachindex(data)]
    J = CNCE(lϕ, gϕ, data, noised)
    x = sign.(randn(R, Nμ, Nσ)) .* 0.01
    results = nesterov(J, x, 0.9, 0.5)

    dx = sign.(results.sol)
    M = [maximum(abs.(dx * data[i] / length(data[i]))) for i = 1:length(data)]
    @test sum(M) == 2
end

Base.@kwdef struct SimParameters
    Nσ::Int
    Nμ::Int
    Ndata::Int
    Nnoise::Int
    p_flip::Float64
end

function partition(L,::Val{N}) where N
    fr=trunc.(Int,LinRange(1,L,N+1))
    R=[ifelse(i==1,1,fr[i]+1):fr[i+1] for i in 1:N]
    ntuple(i->R[i],N)
end


function sim(params, ::Val{Nl}, reps) where Nl
    Rs = [MersenneTwister(125+3i) for i in 1:Threads.nthreads()]
    
    Nσ = params.Nσ
    Nμ = params.Nμ
    Ndata = params.Ndata
    Nnoise = params.Nnoise
    p_flip = params.p_flip

    lϕ=DeepBM(partition(Nσ,Val(Nl)))
    gϕ(σ, ξ) = ReverseDiff.gradient(x -> lϕ(σ, x), ξ)
    tasks = @sync [Threads.@spawn begin
        R=Rs[Threads.threadid()]
        data = [sign.(randn(R, Nσ)) for i = 1:Ndata]
        tdata = [data[i].*sign.(((1-Nσ÷2):(Nσ-Nσ÷2)) .+ 0.01) for i = 1:Ndata]
        noised = [sign.(rand(R, Nσ) .- p_flip) .* data[i] for j = 1:Nnoise, i in eachindex(data)]
        J = CNCE(lϕ, gϕ, data, noised)
        x = sign.(randn(R, Nμ, Nσ)) .* 0.01
        results = nesterov(J, x, 0.95, 0.5, maxiter=1000)
        dx = sign.(results.sol)
        M = [maximum(abs.(dx * data[i] / length(data[i]))) for i = 1:length(data)]
        Mt = [maximum(abs.(dx * tdata[i] / length(tdata[i]))) for i = 1:length(tdata)]
        Mr = mean([maximum(abs.(dx * sign.(randn(R, Nσ)) / Nσ)) for i = 1:100])
        [M;Mt;Mr]
    end 
    for rep = 1:reps]
    
    filter(x->isfinite(sum(x)),fetch.(tasks))
end
params=SimParameters(    
    Nσ = 50,
    Nμ = 20,
    Ndata = 2,
    Nnoise = 20,
    p_flip = 1 / 3)


simsD=sim(params,Val(5),50)
simsS=sim(params,Val(2),50)
using Statistics

mean(simsD),mean(simsS)

