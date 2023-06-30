import MultivariateStats:fit,PCA,predict
import Images:load
import GLMakie:Axis
using GLMakie,MLJ,Images,ImageView


directory_path="/Users/lunarcheung/Downloads/lfw_funneled"
cd(directory_path)
#directory_files = readdir(directory_path);

# directory_images = filter(x -> ismatch(r"\.(jpg|png|gif){1}$"i, x),
#    directory_files);

arr=[i==10 ? "pairs_10.txt" : "pairs_0$(i).txt" for i in 1:10]
imgarr=[]
patharr=[]
open(arr[1], "r") do file
    for line in eachline(file)
        
        if isfile(line)
            image_path = joinpath(directory_path, line);
            image=imresize(load(image_path), (50, 50))
            push!(imgarr,image)
            push!(patharr,image_path)
            @info "success!"
         else
            @info("ERROR: Image not found!")
        end
    end
end

preview_img=imgarr[100]

imshow(RGB.(preview_img))






