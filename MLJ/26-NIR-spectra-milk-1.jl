"""
https://nirpyresearch.com/classification-nir-spectra-principal-component-analysis-python/
"""


import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie


function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:labels=>Multiclass)
    df = fetch(str)|>to_ScienceType
    ytrain, Xtrain=  unpack(df, ==(:labels),!=(:Column1), rng=123);
    cat=ytrain|>levels|>unique
    return ytrain, Xtrain,cat
end

str="NIR-spectra-milk"
ytrain, Xtrain,cat=data_prepare(str)

rows,cols=size(Xtrain)

PCA = @load PCA pkg=MultivariateStats

maxdim=2
model=PCA(maxoutdim=maxdim)
mach = machine(model, Xtrain) |> fit!

Ytr =transform(mach, Xtrain)

function plot_data()
    
    fig=Figure(resolution=(800,800))
    ax= maxdim==3 ? Axis3(fig[1,1]) : Axis(fig[1,1])
    colors=[:red, :yellow,:purple,:lightblue,:black,:orange,:pink,:blue,:tomato]
    

    for (c,color) in zip(cat,colors)
        data=Ytr[ytrain.==c,:]
        if maxdim==3
            scatter!(ax,data[:,1], data[:,2],data[:,3],color=(color,0.8),markersize=14)
        elseif maxdim==2
            scatter!(ax,data[:,1], data[:,2],color=(color,0.8),markersize=14)
        else
            return nothing
        end
        
        
    end
    fig
    save("26-NIR-spectra-milk-$(maxdim)-components-pca.png",fig)
end

plot_data()