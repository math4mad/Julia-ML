"""
https://nirpyresearch.com/classification-nir-spectra-principal-component-analysis-python/

pca-svm
"""


import MultivariateStats:fit,predict,reconstruct,PCA
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
Xtrain=Xtrain|>Matrix|>transpose

M = fit(PCA, Xtrain; maxoutdim=2)

Ytr = predict(M, Xtrain)

colors=[:red, :yellow,:purple,:lightblue,:black,:orange,:pink,:blue,:tomato]
labels=cat
fig=Figure()

ax=Axis(fig[1,1],xlabel="PC1",ylabel="PC2")
    for  (label, color) in zip(labels,colors)
        data=Ytr[:,ytrain.==label]
        scatter!(ax, data[1,:],data[2,:];color=(color,0.6),markersize=12,label=label)
    end


fig

#save("26-NIR-spectra-milk-4.png",fig)