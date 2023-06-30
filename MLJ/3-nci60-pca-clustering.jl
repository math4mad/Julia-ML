"""
An Introduction to Statistical Learning.pdf page 18
"""

import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:Column1=> Multiclass,Count=>Continuous)
    df = fetch(str)
    return df
end

str="NCI60"
df=data_prepare(str)


#data 
rows,cols=size(df)
Xtr = df[:,2:end]
Xtr_labels = Vector(df[:,1])

# # split other half to testing set
 Xte=df[1:3:end,2:end]
 Xte_labels = Vector(df[1:3:end,1])

 PCA = @load PCA pkg=MultivariateStats
 KMeans = @load KMeans pkg=Clustering
 

 model=PCA(maxoutdim=2)

 model2 = KMeans(k=3)

 mach = machine(model, Xtr) |> fit!

 Xproj =transform(mach, Xtr)

 mach2= machine(model2, Xproj) |> fit!

 yhat = predict(mach2, Xproj)|>Array

 cat=levels(yhat)

#scatter(Xproj.x1,Xproj.x2,group=yhat,frame=:box)

fig=Figure()
ax=Axis(fig[1,1])

colors=[:red,:orange,:blue]

for (i,c) in enumerate(yhat)
     data=Xproj[i,:]
     scatter!(ax, data.x1,data.x2,color=(colors[c],0.6),markersize=20)
end

fig

#save("3-nci60-pca-.png",fig)


 