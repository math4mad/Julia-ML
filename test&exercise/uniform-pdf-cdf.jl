using GLMakie, Distributions
a=0.2
b=0.6
d=Uniform(a,b)

fig=Figure()
ax1=Axis(fig[1,1])
ax2=Axis(fig[1,2])

tspan=range(0.0,1.0,100)

lines!(ax1,tspan,pdf.(d,tspan))
lines!(ax2,tspan,cdf.(d,tspan))

fig
#save("uniform(0.2,0.6)-pdf-cdf.png",fig)
