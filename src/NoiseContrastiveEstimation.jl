module NoiseContrastiveEstimation

struct CNCE{T,F,dF}
    lϕ::F
    gϕ::dF
    data::Vector{T}
    noised::Matrix{T}
end

function CNCE(lϕ, data, noised)
    CNCE(lϕ, (x, y) -> 0, data, noised)
end

function (J::CNCE)(θ)
    κ = size(J.noised, 1)
    N = length(J.data)
    tsum(a, b) = (a[1] + b[1], a[2] + b[2])
    T = foldl(
        tsum,
        begin
            bϕ = J.lϕ(J.data[i], θ)
            dbϕ = J.gϕ(J.data[i], θ)
            foldl(tsum, begin
                G = bϕ - J.lϕ(J.noised[j, i], θ)
                dG = dbϕ - J.gϕ(J.noised[j, i], θ)
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

function nesterov(J::CNCE, x0, μ, η; atol = 0, rtol = 1e-7, maxiter = 1e4)
    J_dJ = J(x0)
    C, v = J_dJ
    C0 = C
    i = 1
    x = x0 + zero(η)*v
    while i < maxiter
        f, gf = J(x - μ * v)
        v = μ * v + η * gf
        x = x - v
        i > 1 && approxtol(f, C, rtol, atol) && break
        C = f
        i += 1
    end
    (sol = x, cost = C, cost_fractional_reduction = 1 - C / C0, epochs = i)
end


export CNCE, nesterov
end
