using FFTW, Statistics

function KuramotoSivashinsky(Q, N, tmax, h; M=16, tr=1000)
    #@assert N/λ % 1 == 0 "N/λ % 1 != 0"
    nmax = round(Int,tmax/h)
    x = N*(1:Q) / Q
    #x = N*(-(Q-1)/2:(Q-1)/2) / Q
    u = cos.(π*x/N).*(1 .+sin.(π*x/N))
    v = fft(u)
    #cc= μ* fft(cos.(2π*x/λ))
    #Precompute various ETDRK4 scalar quantities
    #Why all the mean operations?
    k = 2π/N*(0.0:Q-1)
    #k = 2π/N*[(0:(Q/2-1))..., 0, (-Q/2+1:-1)...]
    L = (k.^2 - k.^4) # Fourier Multiples
    E =  exp.(h*L)
    E2 = exp.(h*L/2)
    M = 16 # No. of points for complex means
    r = exp.(im*π*((1:M) .-0.5)/M)
    LR = h*L*ones(M)' + ones(Q)*r'
    QQ = h*real.(mean((exp.(LR/2).-1)./LR, dims=2))[:]
    f1 = h*real(mean((-4 .-LR+exp.(LR).*(4 .-3*LR+ LR.^2))./LR.^3,dims=2))[:]
    f2 = h*real(mean((2 .+LR+exp.(LR).*(-2 .+LR))./LR.^3,dims=2))[:]
    f3 = h*real(mean((-4 .-3*LR-LR.^2+exp.(LR).*(4 .-LR))./LR.^3,dims=2))[:]
    g = -0.5im*k
    #g = 0.5im*k*Q
    uu = [u]
    tt = [0.]

    T = plan_fft(v)
    Ti = plan_ifft(v)
    T! = plan_fft!(v)
    Ti! = plan_ifft!(v)
    a = Complex.(zeros(Q))
    b = Complex.(zeros(Q))
    c = Complex.(zeros(Q))
    Nv = Complex.(zeros(Q))
    Na = Complex.(zeros(Q))
    Nb = Complex.(zeros(Q))
    Nc = Complex.(zeros(Q))
    tmp = Complex.(zeros(Q))

    for n = 1:nmax+tr
                            Nv .= g .* (T*real(Ti*v).^2) #.+ cc
        @.  a  =  E2*v + QQ*Nv
                           Na .= g .* (T!*real(Ti!*a).^2) #.+ cc
        @. b  =  E2*v + QQ*Na
                            Nb .= g.* (T!*real(Ti!*b).^2) #.+ cc
        @. c  =  E2*a + QQ*(2Nb-Nv)
        Nc .= g.* (T!*real(Ti!*c).^2) #.+ cc
        @. v =  E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n > tr
            t = n*h
            u = real(Ti*v)
            push!(uu,u)
            push!(tt,t)
        end
    end
    return uu,tt
end
