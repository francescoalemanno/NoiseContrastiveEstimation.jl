module NoiseContrastiveEstimation

struct CNCE{T,F}
    lϕ::F
    data::Vector{T}
    noised::Matrix{T}
end

function (J::CNCE)(θ)
    κ=size(J.noised,1)
    N=length(J.data)
    jn=0.0
    for i=1:N
        bϕ=J.lϕ(J.data[i],θ)
        for j=1:κ
            G=bϕ-J.lϕ(J.noised[j,i],θ)
            jn+=2/(κ*N)*log(1+exp(-G))
        end
    end
    jn
end

export CNCE
end
