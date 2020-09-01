module NoiseContrastiveEstimation
using ReverseDiff, Random, LinearAlgebra
abstract type AbstractCNCE end

Base.@kwdef struct CNCE{T,F,dF} <: AbstractCNCE
    f::F
    grad_f::dF = (d, p) -> ReverseDiff.gradient(x -> f(d, x), p)
    data::Vector{T}
    noised::Matrix{T}
end

Base.@kwdef struct StochasticCNCE{F,dF,T,pT,RNG} <: AbstractCNCE
    f::F
    grad_f::dF = (d, p) -> ReverseDiff.gradient(x -> f(d, x), p)
    data::Vector{T}
    perturbator::pT 
    K::Int = 5
    minibatch::Int
    rng::RNG = Random.GLOBAL_RNG
end

tsum(a, b) = (a[1] .+ b[1], a[2] .+ b[2])

function (J::StochasticCNCE)(θ)
    κ = J.K
    N = length(J.data)
    obs=0.0
    step = N÷J.minibatch
    T = foldl(
        tsum,
        begin
            rg = (1+ri*step):min(step+ri*step,N)
            res = (0.0, 0.0)
            if !isempty(rg) 
                i = rand(J.rng,rg)
                bϕ = J.f(J.data[i], θ)
                dbϕ = J.grad_f(J.data[i], θ)
                res = foldl(tsum, begin
                    noised=J.perturbator(J.data[i])
                    G = bϕ - J.f(noised, θ)
                    dG = dbϕ - J.grad_f(noised, θ)
                    eG = 1 + exp(-G)
                    grad = (1 / eG - 1) * dG
                    fval = log(eG)
                    obs += 1
                    (fval, grad)
                end for j = 1:κ)
            end
            res
        end for ri = 0:J.minibatch
    )
    T=T .* (2/obs)
    (J = T[1], dJ = T[2])
end


function (J::CNCE)(θ)
    κ = size(J.noised, 1)
    N = length(J.data)
    T = foldl(
        tsum,
        begin
            bϕ = J.f(J.data[i], θ)
            dbϕ = J.grad_f(J.data[i], θ)
            foldl(tsum, begin
                G = bϕ - J.f(J.noised[j, i], θ)
                dG = dbϕ - J.grad_f(J.noised[j, i], θ)
                eG = 1 + exp(-G)
                grad = 2 / (κ * N) * (1 / eG - 1) * dG
                fval = 2 / (κ * N) * log(eG)
                (fval, grad)
            end for j = 1:κ)
        end for i = 1:N
    )
    (J = T[1], dJ = T[2])
end

function approxtol(a, b, rtol, atol)
    a = abs(a)
    b = abs(b)
    aδ = abs(b - a)
    rδ = aδ / max(a, b)
    rδ <= rtol || aδ <= atol
end

function nesterov(J::AbstractCNCE, x0, μ, η; atol = 0, rtol = 1e-7, maxiter = 1e4, verbose=false)
    C, v = J(x0)
    C0 = C
    i = 1
    x = x0 + zero(η) * v
    while i < maxiter
        f, gf = J(x - μ * v)
        v = μ * v + (η/i) * gf
        x = x - v 
        verbose && println(i," ",f," ",norm(gf)," ",norm(v))
        i > 1 && approxtol(f, C, rtol, atol) && break
        C = f
        i += 1
    end
    (sol = x, cost = C, cost_fractional_reduction = 1 - C / C0, epochs = i)
end


export StochasticCNCE, CNCE, nesterov
end
