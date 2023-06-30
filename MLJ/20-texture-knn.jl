"""
https://online.stat.psu.edu/stat857/node/235/
"""

import MLJ:transform,predict,predict_mode
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie,Images,ImageView

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:A41=> Multiclass)
    df = fetch(str)|>to_ScienceType
    return df
end

str="Texture"
df=data_prepare(str)
#first(df,10)
y, X =  unpack(df, ==(:A41), rng=123,shuffle=true);

(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123,shuffle=true)
#shuffle 以后准确率会大大提高
#===============================================================================#


    KMeans = @load KMeans pkg=Clustering
    
    model = KNNClassifier(weights=NearestNeighborModels.Inverse())

    mach = machine(model, Xtrain, ytrain) |> fit!

    yhat = predict_mode(mach, Xtest)
    accuracy(yhat, ytest)



# function plot_img(i)
    
#     ax=Axis(fig[1,i])
#     img=Xtrain[i,:]|>Vector|>d->reshape(d,5,8)
#     image!(ax,img)
    
    
# end

# fig=Figure()

# for  i in 1:10
#     plot_img(i)
# end

# fig

# #imshow(Xtrain[1,:]|>Vector|>d->reshape(d,5,8))

# res=Xtrain[1,:]|>Vector|>d->reshape(d,4,10)

# imshow(res)