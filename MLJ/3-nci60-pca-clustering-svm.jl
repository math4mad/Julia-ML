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
 SVC = @load SVC pkg=LIBSVM

 model=PCA(maxoutdim=2)

 model2 = KMeans(k=3)
 
 model3 = SVC()

 mach = machine(model, Xtr) |> fit!

 Xproj =transform(mach, Xtr)

function boundary_data(df,;n=200)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x1])
    ylow,yhigh=extrema(df[:,:x2])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    xtest=MLJ.table(x_test')
    return tx,ty, xtest
end

 tx,ty, xtest=boundary_data(Xproj)

 mach2= machine(model2, Xproj) |> fit!
 
 yhat = predict(mach2, Xproj)
 cat=yhat|>Array|>levels

 mach3 = machine(model3, Xproj, yhat)|>fit!

 ypred=predict(mach3, xtest)|>Array|>d->reshape(d,200,200)


#scatter(Xproj.x1,Xproj.x2,group=yhat,frame=:box)

function plot_model()
    fig = Figure()
    ax = Axis(fig[1, 1],title="NCI60 Machine Learning",subtitle="pca->clustering->svm")

    colors = [:red, :orange, :blue]
    contourf!(ax, tx,ty,ypred)
    for (i, c) in enumerate(Array(yhat))
        data = Xproj[i, :]
        
        scatter!(ax, data.x1, data.x2, color=(colors[c], 0.6), markersize=20)
        text!(ax,data.x1, data.x2;text="v$(i)")
    end

    fig
    #save("NCI60 Machine Learning:pca->clustering->svm-with-tag.png",fig)
end

plot_model()


 