using MLJ,Plots

PCA = @load PCA pkg=MultivariateStats
KMeans = @load KMeans pkg=Clustering

X, y = @load_iris ## a table and a vector
cat=levels(y)|>unique
colors=[:red,:orange,:tomato]

model1 = PCA(maxoutdim=2)
mach1 = machine(model1, X) |> fit!
model2 = KMeans(k=3)
Xproj = MLJ.transform(mach1, X)
mach2 = machine(model2, Xproj) |> fit!

yhat = MLJ.predict(mach2, Xproj)


scatter(Xproj[:x1],Xproj[:x2],group=yhat,label=["setosa" "versicolor" "virginica"])
#savefig("iris-pca-kmeans.png")
 
