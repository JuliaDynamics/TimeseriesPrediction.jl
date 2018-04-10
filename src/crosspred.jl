function crosspred_stts(strain,ytrain,spred,D,τ,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(1))

    R = myReconstruction(strain,D,τ,B,k,boundary, a, b)
    X,Y,L = size(strain)

    #Prepare tree but remove the last reconstructed states first
    tree = KDTree(R)


    #Prepare s_pred with end of STTS so all initial queries can be created

    q = zeros(length(R[1]))
    N = size(spred)[end]
    ypred = zeros((X,Y,N-(D-1)*τ))
    for n=1:N-(D-1)τ
        for mx=1:X, my=1:Y
            #create q from previous predictions
            qidx = 1
            for   t=n:τ:n+(D-1)*τ, j=my-B*k:k:my+B*k, i=mx-B*k:k:mx+B*k
                if 0< i <=X && 0< j <=Y
                    q[qidx] = spred[i,j,t]
                else
                    q[qidx] = boundary
                end
                qidx += 1
            end
            q[qidx]   = a*(-1+2*(mx-1)/(X-1))^b
            q[qidx+1] = a*(-1+2*(my-1)/(Y-1))^b

            #make prediction & put into state
            idxs,dists = TimeseriesPrediction.neighborhood_and_distances(q,R,tree,ntype)
            xnn = R[idxs]   #not used in method...
            ynn = ytrain[idxs+X*Y*(D-1)]    #Indeces in R are shifted by X*Y rel. to ytrain

            ypred[mx,my,n] = method(q,xnn,ynn,dists)[1]
        end
    end
    #return only the predictions without boundary and w/o old STTS
    return ypred
end

Nx = 50
Ny = 50
Tskip = 100
Ttrain = 100
Ttest = 50
T = Tskip + Ttrain + Ttest
D = 2
τ = 1
B = 0
k = 1
a = 1
b = 1
boundary = 20

U,V = barkley(T, Nx, Ny)
Utrain = U[:,:,Tskip + 1:Tskip + Ttrain]
Vtrain = V[:,:,Tskip + 1:Tskip + Ttrain]
Utest  = U[:,:,Tskip + Ttrain + (D-1)τ:  T]
Vtest  = V[:,:,Tskip + Ttrain :  T]







Upred = crosspred_stts(Vtrain,Utrain,Vtest, D, τ, B, k, a, b)
error = abs.(Utest-Upred)
ε = sum(error, (1,2))[:]


# Animation (takes forever)
@time @gif for i=2:Base.size(Utest)[3]
    l = @layout([a b; c d])
    p1 = plot(@view(Vtest[:,:,i+(D-1)τ]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    plot!(title = "Barkley Model")
    p2 = plot(@view(Utest[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("U component")
    p3 = plot(@view(Upred[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("U Prediction")
    p4 = plot(@view(error[:,:,i]),clims=(0,0.1),aspect_ratio=1,st=[:heatmap])
    title!("Model Error")

    plot(p1,p2,p3,p4, layout=l, size=(600,600))
end





Utest  = U[:,:,Tskip + Ttrain :  T]
Vtest  = V[:,:,Tskip + Ttrain + (D-1)τ:  T]







Vpred = crosspred_stts(Utrain,Vtrain,Utest, D, τ, B, k, a, b)
error = abs.(Vtest-Vpred)
ε = sum(error, (1,2))[:]


# Animation (takes forever)
@time @gif for i=2:Base.size(Utest)[3]
    l = @layout([a b; c d])
    p1 = plot(@view(Vtest[:,:,i+(D-1)τ]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    plot!(title = "Barkley Model")
    p2 = plot(@view(Utest[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("U component")
    p3 = plot(@view(Vpred[:,:,i]), clims=(0,0.75),aspect_ratio=1,st=[:heatmap])
    title!("V Prediction")
    p4 = plot(@view(error[:,:,i]),clims=(0,0.1),aspect_ratio=1,st=[:heatmap])
    title!("Model Error")

    plot(p1,p2,p3,p4, layout=l, size=(600,600))
end
