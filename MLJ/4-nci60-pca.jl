"""
An Introduction to Statistical Learning.pdf page 18
直接使用 MultivariateStats 包
"""

import  MultivariateStats:PCA,fit,predict
using DataFrames,CSV,Plots,MultivariateStats,KernelFunctions,LinearAlgebra

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    #to_ScienceType(d)=coerce(d,:Column1=> Multiclass,Count=>Continuous)
    df = fetch(str)
    return df
end

str="NCI60"
df=data_prepare(str)


#data 
rows,cols=size(df)
Xtr = Matrix(df[1:end,2:end])'
Xtr_labels = Vector(df[1:end,1])

## split other half to testing set
 Xte = Matrix(df[1:end,2:end])'
 Xte_labels = Vector(df[1:end,1])

 
 ## workflow
 k = SqExponentialKernel()
 kmatrix=kernelmatrix(k, Xtr)

 #M =fit(PCA, Xtr; maxoutdim=2)
 #M=fit(KernelPCA,Xtr;maxoutdim=2,kernel=(x,y)->exp(-0.2*norm(x-y)^2.0))
 M=fit(KernelPCA,Xtr;maxoutdim=2)

 Yte = predict(M, Xte)
 xs,ys=Yte[1,:],Yte[2,:]
 p1=scatter(xs,ys;ms=2,alpha=0.5,label=false)
 p2=scatter(xs,ys;group=Xte_labels,ms=2,alpha=0.5,label=false)
 plot(p1,p2;layout=(1,2),frame=:box)
 #savefig("nci60-gene-expression-pca.png")

