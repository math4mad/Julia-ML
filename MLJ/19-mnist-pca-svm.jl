"""
Classification and analysis of the MNIST dataset using PCA and SVM algorithms
https://scindeks-clanci.ceon.rs/data/ipdf/0042-8469/2023/0042-84692302221A.pdf#:~:text=The%20PCA%2C%20an%20unsupervised%20machine%20learning%20technique%2C%20was,classify%20the%20MNIST%20dataset%20into%20classes%20%28Suthaharan%2C%202016%29.

svm 要用 yhat = predict(mach, Xnew) 注意
"""


import MLJ:transform,predict,predict_mode
import LIBSVM
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
y, X,cat=data_prepare(str)
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,shuffle=true)


#plot_mnist(Xtrain)

PCA = @load PCA pkg=MultivariateStats
SVC = @load SVC pkg=LIBSVM   

function different_pca_components()
        dims=[3,10,15,30,50,100,150]
        acc_arr=[]

        for dim in dims
            model1=PCA(maxoutdim=dim)
            model2 = SVC()
            mach1 = machine(model1, Xtrain) |> fit!
            Ytr =transform(mach1, Xtrain)
            mach2 = machine(model2, Ytr, ytrain)|>fit!
            Yte=transform(mach1, Xtest)
            yhat = predict(mach2, Yte)
            res=accuracy(yhat,ytest)
            push!(acc_arr,res)
        end

        return acc_arr
        
end
#acc_arr=different_pca_components()


dims=[3,10,15,30,50,100,150]
dims2=["3","10","15","30","50","100","150"]




acc=[0.5238333333333334,0.9329444444444445,0.9631666666666666,0.9800555555555556,0.9811111111111112,0.9817222222222223,0.9815555555555555].|>(d->round(d,digits=3))|>Vector


function plot_accuracy(acc_arr)
    len=length(acc_arr)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel="pcs",ylabel="accuracy",title="SVM accruacy with n primary components",
      xticks=(1:7, dims2)
    )
    xs=1:len
    acc_arr=round.(acc_arr,digits=3)
    scatterlines!(ax,acc_arr,markercolor = (:red,0.5))
    fig
    save("19-mnist-PCA-SVM-accuracy.png",fig)
end

plot_accuracy(acc)