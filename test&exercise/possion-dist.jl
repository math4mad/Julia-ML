using GLMakie, Distributions
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="possion dist pdf")
ax2 = Axis(fig[1, 2], xlabel="possion dist cdf")
dist(λ) =Poisson(λ)
ts =1:10
i=1
d=dist(i)
data=rand(d,5000)
pmf=cdf.(d,ts)
stem!(ax1, ts,pdf.(d,ts);label="θ=$(i)",stemwidth =4)
stem!(ax2, ts,cdf.(d,ts);label="θ=$(i)",stemwidth =4)

axislegend(ax1)
axislegend(ax2)
fig