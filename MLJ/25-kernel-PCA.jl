"""
referece  scikit-learning  2.5.2. Kernel Principal Component Analysis (kPCA)

"""

import MLJ:transform
using  MLJ,GLMakie,DataFrames,Random,LinearAlgebra,KernelFunctions
Random.seed!(1222)
colors=[:red,:blue]
"""
circle_data()
return df
"""
function circle_data()
    X, y = make_circles(400; noise=0.1, factor=0.3)
    df = DataFrame(X)
    df.y = y
    return df
end

function moons_data()
    X, y = make_moons(400; noise=0.1)
    df = DataFrame(X)
    df.y = y
    return df
end

df=circle_data()

df2=moons_data()

X,y=df[:,1:2],df[:,3]
X2,y2=df2[:,1:2],df2[:,3]
cat=levels(y)|>unique
cat2=levels(y2)|>unique

KernelPCA = @load KernelPCA pkg=MultivariateStats
function rbf_kernel(length_scale)
    return (x,y) -> norm(x-y)^2 / ((2 * length_scale)^2)
end

poly_kernel=(d,c=1)->(x,y)->(x'y+c)^d
exp_kernal=(x,y)->exp(x'y)
model=KernelPCA(maxoutdim=2,kernel=rbf_kernel(1))
mach = machine(model, X2)|>fit!

Xproj = transform(mach, X2)

fig=Figure()
ax=Axis(fig[1,1])
ax2=Axis(fig[1,2])
for i in eachindex(cat2)
    local data=Xproj[y.==cat2[i],:]
    scatter!(ax, data[:,1],data[:,2], color=(colors[i],0.8))
    
end

for i in eachindex(cat)
    local data=X2[y.==cat2[i],:]
    scatter!(ax2, data[:,1],data[:,2], color=(colors[i],0.8))
    
end
fig

