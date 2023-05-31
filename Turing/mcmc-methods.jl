"""
https://storopoli.io/Bayesian-Julia/pages/05_MCMC/
"""


using GLMakie
using Distributions
using Random

Random.seed!(123);

 N = 100_000
 μ = [0, 0]
 Σ = [1 0.8; 0.8 1]
 mvnormal = MvNormal(μ, Σ)

figure=Figure()
ax1=Axis(figure[1,1];xlabel=L"X", ylabel=L"Y")


xs = -3:0.01:3
ys = -3:0.01:3
zs = [pdf(mvnormal, [i, j]) for i in xs, j in ys]


c=contourf!(ax1,xs,ys,zs)
Colorbar(figure[1, 2],c)

figure


