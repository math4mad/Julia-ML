
import MultivariateStats:fit,predict,reconstruct
using MultivariateStats, GLMakie,RDatasets

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


M = fit(PCA, Xtr; maxoutdim=3)


Yte = predict(M, Xte)


#Xr = reconstruct(M, Yte)


fig=Figure()
azimuths = [0, 0.2pi, 0.4pi,-0.45pi]

# for (i,angle) in enumerate(azimuths)
#     local ax=Axis3(fig[1,i],azimuth=angle,elevation =0.2pi)
#     for  (label, color) in zip(labels,colors)
#         data=Yte[:,Xte_labels.==label]
#         scatter!(ax, data[1,:],data[2,:],data[3,:];color=(color,0.6),markersize=16,label=label)
#     end
#     axislegend(ax)
# end

    ax=Axis3(fig[1,1],azimuth=azimuths[4],xlabel="PC1",ylabel="PC2")
    for  (label, color) in zip(labels,colors)
        data=Yte[:,Xte_labels.==label]
        scatter!(ax, data[1,:],data[2,:],data[3,:];color=(color,0.6),markersize=16,label=label)
    end
    axislegend(ax)




fig

save("27-Iris-PCA.png",fig)