import MLJ:transform,predict,predict_mode
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


LDA = @load LDA pkg=MultivariateStats
PCA = @load PCA pkg=MultivariateStats


function different_pca_components(;n=10)
    acc_arr=[]
    for i in 1:n
        model1=PCA(maxoutdim=i)
        model2 = LDA()
        mach1 = machine(model1, Xtrain) |> fit!
        Ytr =transform(mach1, Xtrain)
        mach2 = machine(model2, Ytr, ytrain)|>fit!
        Yte=transform(mach1, Xtrain)
        yhat = predict_mode(mach2, Yte)
        res=accuracy(yhat,ytrain)
        push!(acc_arr,res)
    end
    return acc_arr
end



#acc_arr=different_pca_components(;n=15)

 function plot_accuracy(acc_arr)
    len=length(acc_arr)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel="pcs",ylabel="accuracy",title="mnist LDA accuracy with n primary components",yticks = 0.5:0.1:1.0)
    xs=1:len
    acc_arr=round.(acc_arr,digits=3)
    scatterlines!(ax,xs,acc_arr,markercolor = (:red,0.5))
    fig
    save("17-mnist-PCA-LDA.png",fig)
 end



#acc_arr=different_pca_components(;n=201)
#plot_accuracy(acc_arr)

str="mnist_train"
ytrain, Xtrain,cat=data_prepare(str)

size(Xtrain)
