using Plots
#using PyPlot
#pyplot()


##This Algorithm is taken from
#  http://www.scholarpedia.org/article/Barkley_model

# Simulation is super fast but plotting/animating sucks....

function barkley(T, Nx=100, Ny=100)
    a = 0.75
    b = 0.02
    ε = 0.02
    V = zeros(Nx, Ny, T)
    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    u[35:end,34] = 0.1
    u[35:end,35] = 0.5
    u[35:end,36] = 5
    v[35:end,34] = 1

    u[1:15,14] = 5
    u[1:15,15] = 0.5
    u[1:15,16] = 0.1
    v[1:15,17] = 1




    #u[50:52, 47:48] = 1


    h = 0.75
    Δt = 0.1
    δ = 0.01
    Σ = zeros(Nx, Ny, 2)
    r = 1
    s = 2
    function F(u, uth)
        if u < uth
            u/(1-(Δt/ε)*(1-u)*(u-uth))
        else
            (u + (Δt/ε)*u*(u-uth))/(1+(Δt/ε)*u*(u-uth))
        end
    end

    for m=1:T
        for i=1:Nx, j=1:Ny
            if u[i,j] < δ
                u[i,j] = Δt/h^2 * Σ[i,j,r]
                v[i,j] = (1 - Δt)* v[i,j]
            else
                uth = (v[i,j] + b)/a
                v[i,j] = v[i,j] + Δt*(u[i,j] - v[i,j])
                u[i,j] = F(u[i,j], uth) + Δt/h^2 *Σ[i,j,r]
                Σ[i,j,s] -= 4u[i,j]
                i > 1  ? Σ[i-1,j,s] += u[i,j] : nothing
                i < Nx ? Σ[i+1,j,s] += u[i,j] : nothing
                j > 1  ? Σ[i,j-1,s] += u[i,j] : nothing
                j < Ny ? Σ[i,j+1,s] += u[i,j] : nothing
            end
            Σ[i,j,r] = 0
        end
        r,s = s,r
        V[:,:,m] .= v
        #plot(p)
    end

    #@gif for i=1:T
    #    plot(@view(V[:,:,i]), st=[:contourf])
    #end every 10
    return V
end

Nx = 50
Ny = 50
Tskip = 20
Ttrain = 50
p = 100
T = Tskip + Ttrain + p

V = barkley(T, Nx, Ny)#[:,:,Tskip+1:T]
Vtrain = V[:,:,Tskip + 1:Tskip + Ttrain]
Vtest  = V[:,:,Tskip + Ttrain :  T]



D = 2
τ = 1
B = 1
k = 1
a = 0
b = 0
boundary = 20


function localmodel_stts(s,D,τ,p,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3))
    #R = myReconstruction(s,D,τ,B,k,boundary)
    R = myReconstruction(s,D,τ,B,k,boundary, a, b)
    #M = dimension(s)
    #L = length(s)
    X,Y,L = size(s)

    #Prepare tree but remove the last reconstructed states first
    R_tree = Dataset(copy(R.data))
    for my=1:Y,mx=1:X
        R_tree.data[(mx+X*(my-1))*(L-(D-1)*τ)] = SVector((Inf*ones(length(R[1])))...)
    end
    tree = KDTree(R_tree)


    #Prepare s_pred with end of STTS so all initial queries can be created
    s_pred = s[:,:,L-D*τ:L]


    #New state that will be predicted, allocate once and reuse
    state = zeros(X,Y)
    q = zeros(length(R[1]))
    for n=1:p
        N = size(s_pred)[end]
        for mx=1:X, my=1:Y
            #create q from previous predictions
            qidx = 1
            for   t=N-D*τ+1:τ:N, j=my-B*k:k:my+B*k, i=mx-B*k:k:mx+B*k
                if 0< i <=X && 0< j <=Y
                    q[qidx] = s_pred[i,j,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b
            q[qidx+1] = a*(-1+2*(my-1)/(Y-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]
            ynn = map(y -> y[1+(2*B+1)*B+B+(D-1)*(2*B+1)^2],R[idxs+1])

            state[mx,my] = method(q,xnn,ynn,dists)[1]
        end
        s_pred = cat(3,s_pred,state)
    end
    #return only the predictions without boundary and w/o old STTS
    return s_pred[:,:,D*τ+1:end]
end

Vpred = localmodel_stts(Vtrain, D, τ, p, B, k, a, b)
error = Vtest-Vpred
ε = sum(error, (1,2))[:]
#ε2 = sum(Vpred-Vpred2, (1,2))[:]

@gif for i=1:size(Vtest)[3]
    #l = @layout [a{0.5w} b]
    p1 = plot(@view(Vtest[:,:,i]), st=[:contourf])
    p2 = plot(@view(Vpred[:,:,i]), st=[:contourf])
    #p3 = plot(@view(error[:,:,i]), st=[:contourf])
    plot(p1,p2, layout=(1,2))#, size=(600,1000))
    plot!(title = "barkley L=Orig, R=pred")
end
