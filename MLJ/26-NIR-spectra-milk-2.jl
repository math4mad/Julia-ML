"""
https://nirpyresearch.com/classification-nir-spectra-principal-component-analysis-python/

pca-svm
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
n=200
Xarr=[]
for i in 1:601
    col=Vector(range(extrema(Xtrain[:,i])...,n))
    push!(Xarr,col)
end

Xtest=reduce(hcat,Xarr)

nums=200
function boundary_data(d,;n=nums)
    n1=n2=n
    xlow,xhigh=extrema(d[:x1])
    ylow,yhigh=extrema(d[:x2])
    tx = LinRange(xlow,xhigh,n1)
    ty = LinRange(ylow,yhigh,n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    x_test=MLJ.table(x_test')
    return tx,ty,x_test
end

PCA = @load PCA pkg=MultivariateStats
SVC = @load SVC pkg=LIBSVM 

maxdim=2
function milk_pca_svm(;dim=2)
    model1=PCA(maxoutdim=dim)
    model2 = SVC()
    mach1 = machine(model1, Xtrain) |> fit!
    Ytr =transform(mach1, Xtrain)
    mach2 = machine(model2, Ytr, ytrain)|>fit!
    Yte=transform(mach1, Xtest)
    tx,ty,x_test=boundary_data(Yte)
    #@show [extrema(x_test[:x1]),extrema(x_test[:x2])]
    yhat = predict(mach2, x_test)|>Array|>d->reshape(d,nums,nums)
    return Ytr,yhat,tx,ty
end



function plot_data()
    Ytr,yhat,tx,ty=milk_pca_svm(;dim=2)
    fig=Figure(resolution=(800,800))
    ax= maxdim==3 ? Axis3(fig[1,1]) : Axis(fig[1,1])
    colors=[:red, :yellow,:purple,:lightblue,:black,:orange,:pink,:blue,:tomato]
    contourf!(ax,tx,ty,yhat,levels=length(cat),colormap=:redsblues)
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
    #save("26-NIR-spectra-milk-$(maxdim)-components-pca-svm.png",fig)
end
@info cat
plot_data()









