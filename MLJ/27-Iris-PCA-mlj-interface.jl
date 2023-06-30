import MLJ:transform
using MLJ, GLMakie,RDatasets

# load iris dataset
iris = dataset("datasets", "iris")

# split half to training set
Xtr = Matrix(iris[1:2:end,1:4])'   # pca 需要转置, 属性作为行, 观测值为列
Xtr_labels = Vector(iris[1:2:end,5])

labels=["setosa","versicolor","virginica"]
colors=[:red,:orange,:blue]

# split other half to testing set
Xte = Matrix(iris[2:2:end,1:4])'
Xte_labels = Vector(iris[2:2:end,5])

#X, y = @load_iris ## a table and a vector
#PCA = @load PCA pkg=MultivariateStats

#model = PCA(maxoutdim=3)

#mach = machine(model, X ) |> fit!

#report(mach)
#Xproj = transform(mach, X)


X, y = make_regression(100, 4)
# function plot_pca()
#     fig=Figure()
#         ax=Axis3(fig[1,1],xlabel="PC1",ylabel="PC2",zlabel="PC3")
#         for  (label, color) in zip(labels,colors)
#             data=Yte[:,Xte_labels.==label]
#             scatter!(ax, data[1,:],data[2,:],data[3,:];color=(color,0.6),markersize=16,label=label)
#         end
#         axislegend(ax)
#     fig
# end

#save("27-Iris-PCA.png",fig)