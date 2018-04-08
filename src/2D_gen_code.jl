using PyPlot
using DynamicalSystemsBase
using StaticArrays
using NearestNeighbors
using TimeseriesPrediction

###########################################################################################
#                       Old naive Reconstruction                                          #
###########################################################################################
function md_STReconstruction(s, D,τ,B=1,k=1,boundary=10, a=1,b=1)
    #2D
    dim = 2
    X,Y,L = size(s)
    sub_s = Array{Float64, 2}(L,(2B+1)*(2B+1))

    R = Dataset{(2B+1)*(2B+1)*D+2, Float64}()
    sizehint!(R.data,(L-D*τ)*X*Y)
    #Actual Loop
    for mx=1:X, my=1:Y

        idxs = [(i,j) for i=mx-B*k:k:mx+B*k , j=my-B*k:k:my+B*k]
        for (n,(i,j)) in enumerate(idxs)
            if i<1 || i>X || j<1 || j>Y
                sub_s[:,n] = ones(L) * boundary
            else
                sub_s[:,n] = s[i,j,:]
            end
        end
        Rnext = Reconstruction(Dataset(sub_s), D, τ)
        mat = hcat(convert(Matrix,Rnext),
                ones(length(Rnext))* a*(-1+2*mx/X)^b,
                ones(length(Rnext))* a*(-1+2*my/Y)^b)
        append!(R,Dataset(mat))
    end
    return R
end


###########################################################################################
#                       Better generated Reconstruction                                   #
###########################################################################################

function my_reconstruct_impl(::Val{D}, ::Val{B}) where {D, B}
    gens = [:(0 < mx + $i<=X && 0 < my + $j <= Y ?
     s[mx + $i, my + $j, t + $d*τ] : boundary)
      for i=-B:B, j=-B:B, d=0:D-1]
    quote
        X,Y,L = size(s)
        L -= $(D-1)*τ
        T = eltype(s)
        data = Vector{SVector{$D*(2*$B + 1)^2+2, T}}(L*X*Y)
        for my ∈ 1:Y, mx ∈ 1:X
            for t ∈ 1:L
                #n = mx + X*(my-1) + X*Y*(t-1)
                wx = a*(-1+2*(mx-1)/(X-1))^b
                wy = a*(-1+2*(my-1)/(Y-1))^b
                n = t + L*(mx-1) + L*X*(my-1)
                #println("mx=$mx, my=$my, t=$t, n=$n")
                data[n] = SVector{$D*(2*$B + 1)^2+2, T}($(gens...),wx, wy)
            end
        end
        data
    end
end

@generated function my_reconstruct(s, ::Val{D}, ::Val{B}, τ, boundary, a, b) where {D, B}
    my_reconstruct_impl(Val{D}(), Val{B}())
end

function myReconstruction(s::AbstractArray{T,3}, D,τ::DT,B=1,k=1,boundary=10, a=1,b=1) where {T, DT}
    Reconstruction{D*(2B+1)^2+2,T,DT}(my_reconstruct(s, Val{D}(), Val{B}(),τ,boundary,a,b), τ)
end





###########################################################################################
#                                     Prediction                                          #
###########################################################################################

function (M::AverageLocalModel)(q,xnn,ynn,dists)
    if length(xnn) > 1 && maximum(dists) > 0
        dmax = maximum(dists)
        y_pred = zeros(typeof(ynn[1]))
        Ω = 0.
        for (y,d) in zip(ynn,dists)
            ω2 = (1-0.9(d/dmax)^M.n)^2M.n
            Ω += ω2
            y_pred += ω2*y
        end
        y_pred /= Ω
        return y_pred
    end
    return ynn[1]
end

function localmodel_stts(s,D,τ,p,B=1,k=1,boundary=20, a=1,b=1;
    method::AbstractLocalModel = AverageLocalModel(2),
    ntype::AbstractNeighborhood = FixedMassNeighborhood(3))
    #R = myReconstruction(s,D,τ,B,k,boundary)
    R = md_STReconstruction(s,D,τ,B,k,boundary)
    #M = dimension(s)
    #L = length(s)
    X,Y,L = size(s)

    #Prepare tree but remove the last reconstructed states first
    R_tree = Dataset(copy(R.data))
    for mx=1:X, my=1:Y
        R_tree.data[(mx+X*(my-1))*(L-(D-1)*τ)] = SVector((Inf*ones(length(R[1])))...)
    end
    tree = KDTree(R_tree)


    #Prepare s_pred with end of STTS so all initial queries can be created
    s_pred = s[:,:,L-D*τ:L]


    #New state that will be predicted, allocate once and reuse
    state = zeros(X,Y)
    for n=1:p
        N = size(s_pred)[end]
        for mx=1:X, my=1:Y
            #create q from previous predictions
            q = Float64[]
            idxs = [(i,j,t) for i=mx-B*k:k:mx+B*k , j=my-B*k:k:my+B*k, t=N-D*τ+1:τ:N]
            for (n,(i,j,t)) in enumerate(idxs)
                if i<1 || i>X || j<1 || j>Y
                    push!(q, boundary)
                else
                    push!(q,s_pred[i,j,t])
                end
            end
            push!(q,a*(-1+2*(mx-1)/(X-1))^b)
            push!(q,a*(-1+2*(mY-1)/(Y-1))^b)
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



function coupled_henon(X=10,Y=10)
    function henon(du, u, p, t)
        du[1,:,1] = du[:,1,1] = du[X, :, 1] = du[:, Y, 1] = 0.5
        du[1,:,2] = du[:,1,2] = du[X, :, 2] = du[:, Y ,2] = 0
        for mx=2:X-1 , my=2:Y-1
            du[mx,my,1] = 1 - 1.45*(.5*u[mx,my,1]+ .125*u[mx-1,my,1] + .125u[mx+1,my,1]+ .125u[mx, my-1,1]+ .125u[mx,my+1,1])^2 + 0.3*u[mx,my,2]
            du[mx,my,2] = u[mx,my,1]
        end
        return nothing
    end
    return DiscreteDynamicalSystem(henon,rand(X,Y,2), nothing; t0=0)
end


#Size
X=10
Y=10
ds = coupled_henon(X,Y)
N_train = 1000
p = 30
data = trajectory(ds,N_train+p)
#Reconstruct this #
s = data[1:N_train,SVector(1:X*Y...)]
mat = convert(Matrix,s)'
mat = reshape(mat, (X,Y,N_train))

begin
    subplot(311)
    #Original
    img = Matrix(data[N_train:N_train+p,SVector(1:X*Y...)])
    pcolormesh(img)
    xticks([])
    colorbar()
    subplot(312)
    #Prediction
    s_pred = localmodel_stts(mat,2,1,p,1,1,10, 1,1)
    pred_mat = reshape(s_pred, (X*Y,p+1))'
    pcolormesh(pred_mat)
    colorbar()
    xticks([])

    subplot(313)
    #Error
    ε = abs.(img-pred_mat)
    pcolormesh(ε)
    colorbar()
end
