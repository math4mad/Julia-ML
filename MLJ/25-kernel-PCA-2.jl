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
X,y=df2[:,1:2],df2[:,3]
schema(df2)

cat=levels(y)|>unique





KernelPCA = @load KernelPCA pkg=MultivariateStats
function rbf_kernel(length_scale)
    return (x,y) -> norm(x-y)^2 / ((2 * length_scale)^2)
end

rbf=(γ=2)->(x,y)->exp(-norm(x-y)^γ)
rbf2=(x,y)->exp(-norm(x-y)^4/2)

poly_kernel=(d,c=1)->(x,y)->(x'y+c)^d
exp_kernal=(x,y)->exp(x'y)
model=KernelPCA(maxoutdim=2,kernel=rbf2)
mach = machine(model, X2)|>fit!
Xproj = transform(mach, X2)


fig=Figure()
ax1=Axis(fig[1,1])
ax2=Axis(fig[1,2])


for i in eachindex(cat)
    local data=df2[y.==cat[i],:]
    local data2=Xproj[y.==cat[i],:]
    scatter!(ax1,data[:,1],data[:,2],color=colors[i])
    scatter!(ax2,data2[:,1],data2[:,2],color=colors[i])
end



fig