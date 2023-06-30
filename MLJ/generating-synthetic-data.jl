using MLJ, DataFrames,GLMakie,Random
Random.seed!(1222)


#X, y = make_blobs(200, 2; centers=2, cluster_std=[1.0, 3.0])
X, y = make_circles(200; noise=0.1, factor=0.3)

df = DataFrame(X)
df.y = y
cat=df.y|>levels
colors=[:red,:blue]

fig=Figure()
ax=Axis(fig[1,1])
for (i,c) in enumerate(cat)
    data=df[y.==c,:]
    scatter!(data[:,1],data[:,2],color=colors[i])
end
fig


