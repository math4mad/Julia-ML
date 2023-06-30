
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

directory_path="/Users/lunarcheung/Downloads/lfw_funneled"
cd(directory_path)
#directory_files = readdir(directory_path);

# directory_images = filter(x -> ismatch(r"\.(jpg|png|gif){1}$"i, x),
#    directory_files);

arr=[i==10 ? "pairs_10.txt" : "pairs_0$(i).txt" for i in 1:10]

function image_data()
    imgarr=[]
    patharr=[]
    open(arr[1], "r") do file
        for line in eachline(file)
            
            if isfile(line)
                image_path = joinpath(directory_path, line);
                image=imresize(load(image_path), (40, 40))
                img_gray = Gray.(image)
                res=img_gray|>channelview|>Matrix.|>(Float64)|>d->reshape(d,1600,1)
                push!(imgarr,res)
                push!(patharr,image_path)
                @info "success!"
            else
                @info("ERROR: Image not found!")
            end
        end
    end

    return imgarr,patharr
end


#@time  imgarr,patharr= @memoize image_data()
#imshow(imgarr[1])

function plot_image()
    fig=Figure()

    for i in 0:4
        for j in 1:5
            num=i*5+j;
            img=load(patharr[num])|>rotl90
            local ax = Axis(fig[i, j],yreversed=true)
            image!(ax, img)
            #hidespines!(ax)
            hidedecorations!(ax)
        end
    end
    fig 
end


#@time @memoize plot_image()


##chanv = channelview(imgarr[1])
##Gray.(imgarr[1])|>channelview|>Matrix.|>Float64|>Base.Flatten

#imgarr[1]|>channelview|>Matrix.|>(Float64)|>d->reshape(d,1600,1)

Xtrain=reduce(hcat,imgarr)

df = DataFrame(Xtrain',:auto)
CSV.write("lwface.csv", df)
# Xte=Xtrain[:,1:10]

# dim=60
# M = fit(PCA, Xtrain; maxoutdim=dim)
#Yte = predict(M, Xte)

#Xr = reconstruct(M, Yte)'