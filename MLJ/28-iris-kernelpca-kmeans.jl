
import MLJ:predict,transform
using MLJ,Plots,LinearAlgebra

KernelPCA = @load KernelPCA pkg=MultivariateStats
KMeans = @load KMeans pkg=Clustering

function rbf_kernel(length_scale)
    return (x,y) -> norm(x-y)^2 / ((2 * length_scale)^2)
end

X, y = @load_iris ## a table and a vector
cat=levels(y)|>unique
colors=[:red,:orange,:tomato]

function tune(i)

    model1 = KernelPCA(maxoutdim=2, kernel=rbf_kernel(i))
    mach1 = machine(model1, X) |> fit!
    model2 = KMeans(k=3)
    Xproj = MLJ.transform(mach1, X)
    mach2 = machine(model2, Xproj) |> fit!
    yhat = MLJ.predict(mach2, Xproj)
    return scatter(Xproj[:x1],Xproj[:x2],group=yhat,title="rbf($(i))",frame=:box,showaxis=false,ms=2)
    #savefig("iris-kernelpca-rbf($i)-kmeans.png")
    #label=["setosa" "versicolor" "virginica"]
end

plot_arr=[]
for i in 1:9
    push!(plot_arr,tune(i))
end

plot(plot_arr...,layout=(3,3),)

#savefig("iris-kpca-rbf-kmeans.png")





 
