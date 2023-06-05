"""
An Introduction to Statistical Learning.pdf page 18
"""

using DataFrames,MLJ,CSV,MLJModelInterface,Plots

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
Xtr = Matrix(df[:,2:end])'
Xtr_labels = Vector(df[:,1])

# # split other half to testing set
 Xte = Matrix(df[1:3:end,2:end])'
 Xte_labels = Vector(df[1:3:end,1])

 PCA = @load PCA pkg=MultivariateStats


 model=PCA(maxoutdim=2)

 mach = machine(model, Xtr) |> fit!

 Xproj =MLJ.transform(mach, Xtr)

 x1,x2=Xproj

 scatter(x1,x2)

