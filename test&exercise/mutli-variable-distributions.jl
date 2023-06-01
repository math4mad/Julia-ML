"""
Projection to xz,yz plane
for xz plane,  data is  Point(x,y, pdf(d,y))  x is fixed
for yz plane,  data is  Point(x,y, pdf(d,x))  y is fixed
    

"""

using GLMakie
using Distributions
using Latexify

GLMakie.activate!()

n=100
xs=ys=range(-4,4,n)
μ=[0 , 0]
Σ=[1 0 ; 0 1]
d=MvNormal(μ,Σ)
d2=Normal()
zs=[pdf(d,[x,y]) for x in xs , y in ys]

# 投影到 x-z 平面数据
# div  2  just scale plot
xz=[Point3f(4,y,pdf(d2,y)/2) for y in ys]

# 投影到 y-z 平面数据
# div  2  just scale plot
yz=[Point3f(x,4,pdf(d2,x)/2) for x in ys]


str=latexify([0,0])
fig = Figure(size=(600,600))
ax = Axis3(fig[1, 1])
surface!(xs,ys,zs;space=:data)
lines!(ax, xz;linewidth=3,label="along x Axis")
lines!(ax, yz;linewidth=3,label="along y Axis")
#text!(Point3f(3.0,-0.5,0.18);text="along x Axis")
axislegend(ax)
fig
#save("mutli-normal-dist.png",fig)