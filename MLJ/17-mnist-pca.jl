


import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie


function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:label=>Multiclass)
    df = fetch(str)|>to_ScienceType
    ytrain, Xtrain=  unpack(df, ==(:label), rng=123);
    cat=ytrain|>levels|>unique
    return ytrain, Xtrain,cat
end

str="mnist_train"
ytrain, Xtrain,cat=data_prepare(str)

#histogram(ytrain,bins=:fd,xticks = 0:1:10,label="mnist-digits-frequency")
#savefig("17-mnist-pca-histogram.png")


PCA = @load PCA pkg=MultivariateStats

maxdim=50
model=PCA(maxoutdim=maxdim)
mach = machine(model, Xtrain) |> fit!
#MLJ.save("mnist-pca-50-components-model.jls", mach)
#Ytr =transform(mach, Xtrain)



function plot_digits_cloud()
    
    fig=Figure(resolution=(1800,1800))
    ax= maxdim==3 ? Axis3(fig[1,1]) : Axis(fig[1,1])
    colors=[:red, :yellow,:purple,:lightblue,:black,:orange,:pink,:blue,:tomato,:lightgreen,]
    markers=['0','1','2','3','4','5','6','7','8','9']

    for (c,color,m) in zip(cat,colors,markers)
        data=Ytr[ytrain.==c,:]
        if maxdim==3
            scatter!(ax,data[:,1], data[:,2],data[:,3],color=(color,0.8),markersize=14,marker=m)
        elseif maxdim==2
            scatter!(ax,data[:,1], data[:,2],color=(color,0.8),markersize=14,marker=m)
        else
            return nothing
        end
        
        
    end
    fig
end

#plot_digits_cloud()







