"""
Classification and analysis of the MNIST dataset using PCA and SVM algorithms
https://scindeks-clanci.ceon.rs/data/ipdf/0042-8469/2023/0042-84692302221A.pdf#:~:text=The%20PCA%2C%20an%20unsupervised%20machine%20learning%20technique%2C%20was,classify%20the%20MNIST%20dataset%20into%20classes%20%28Suthaharan%2C%202016%29.

svm 要用 yhat = predict(mach, Xnew) 注意
construct  img   参见 mnist 文件夹图片
"""

import MultivariateStats:fit,PCA,predict
using DataFrames,CSV,GLMakie,MultivariateStats,MLJ




function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv")|>DataFrame
    df = fetch(str)
    return df
end

str="mnist_train"
df=data_prepare(str)

Xtr = Matrix(df[1:2:end,2:end])'
Xtr_labels = Vector(df[1:2:end,:label])

Xte = Matrix(df[2:2:end,2:end])'
Xte_labels = Vector(df[2:2:end,:label])

dim=700

M = fit(PCA, Xtr; maxoutdim=dim)


Yte = predict(M, Xte)

Xr = reconstruct(M, Yte)'




function plot_mnist(data;dims=28)
   
    fig=Figure()
    
    for i in 0:4
        for j in 1:5
            idx=i*5+j
            local img = data[idx, :] |> d ->reshape(d, dims, dims)
            local ax = Axis(fig[i, j],yreversed=true)
            image!(ax, img)
            hidespines!(ax)
            hidedecorations!(ax)
           
        end

    end
    
    
    fig
    save("19-mnist-pca-reconstruct-$(dim)-components.png",fig)
end

plot_mnist(Xr)

