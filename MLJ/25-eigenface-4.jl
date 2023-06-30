

import MultivariateStats:fit,PCA,predict,reconstruct
import Images:load
import GLMakie:Axis
using GLMakie,MLJ,Images,ImageView,LinearAlgebra
using CSV,DataFrames


    macro memoize(expr)
            local cache = Dict()
            local res = undef
            local params = expr.args
            #@show params
            local id = hash(params)
            if haskey(cache, id) == true
                res = cache[id]
            else
                local val = esc(expr)
        
                push!(cache, (id => val))
                res = cache[id]
            end
        
            return :($res)
        end

function data_prepare(str)
        fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
        
        df = fetch(str)
        
        return df
end

str="lwface"
df=data_prepare(str)
Xtr=df|>Matrix|>transpose
Xte=df[1:25,:]|>Matrix
    dim=60
    
function trian_construct(dims)
    M = fit(PCA, Xtr; maxoutdim=dim)
    Yte = predict(M, Xte')
    Xr = reconstruct(M, Yte)'
    return Xr
end



function plot_two_faces(origindata,data,rd;dims=40)

    fig=Figure(resolution=(1500,3000))
        
    for i in 0:4
        for j in 1:5
            idx=i*5+j
            
            local img = data[idx, :] |> d ->reshape(d, dims, dims)|>rotr90
            local ax = Axis(fig[i, j])
            image!(ax, img)
            hidespines!(ax)
            hidedecorations!(ax)
        
        end

    end

    for i in 0:4
        for j in 1:5
            idx=i*5+j
            
            local img = origindata[idx, :] |> d ->reshape(d, dims, dims)|>rotr90
            local ax = Axis(fig[i+5, j])
            image!(ax, img)
            hidespines!(ax)
            hidedecorations!(ax)
        
        end

    end
    fig
    save("lw-faces-$(rd)-components.png",fig)
    
end
    
dims=[120,200,300,500,700,1000,1200]

for  rd in  dims
   local Xr=trian_construct(rd)
   plot_two_faces(Xte,Xr,rd)
end

















#================backup===========================#
# function plot_face(data;dims=40)
   
#     fig=Figure()
    
#     for i in 0:4
#         for j in 1:5
#             idx=i*5+j
#             local img = data[idx, :] |> d ->reshape(d, dims, dims)|>rotr90
#             local ax = Axis(fig[i, j])
#             image!(ax, img)
#             hidespines!(ax)
#             hidedecorations!(ax)
           
#         end

#     end
    
    
#     fig
#     #save("19-mnist-pca-reconstruct-$(dim)-components.png",fig)
# end
#=================================================#