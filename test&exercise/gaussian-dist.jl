using GLMakie, Distributions, Random
Random.seed!(124)

colors = ["#FF410D", "#6EE2FF", "#F7C530", "#95CC5E", "#D0DFE6", "#F79D1E"]
μσpairs = [[-3, 1], [-0.3, 1], [0, 1], [1.2, 1], [4, 1], [5, 1]] #方差相同, 均值变化
μσpairs2 = [[0, 0.8], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]   # 均值相同, 方差变化

ts = range(-10, 10, 200)
fig = Figure(resolution=(1200, 600))
ax1 = Axis(fig[1, 1:2], xlabel="μ change, σ=1")
ax2 = Axis(fig[1, 3], xlabel="μ=0,σ change")

for (p, p2, c) in zip(μσpairs, μσpairs2, colors)
    local pdf1(x) = pdf(Normal(p...), x)
    local pdf2(x) = pdf(Normal(p2...), x)
    lines!(ax1, ts, pdf1.(ts), label="$(p[1]),  $(p[2])", color=c, linewidth=3)
    band!(ax1, ts, fill(0, length(ts)), pdf1.(ts); color=(c, 0.1)) #半透明区域
    lines!(ax2, ts, pdf2.(ts), label="$(p2[1]),  $(p2[2])", color=c, linewidth=3)
    band!(ax2, ts, fill(0, length(ts)), pdf2.(ts); color=(c, 0.1)) #半透明区域
end


axislegend(ax1)
axislegend(ax2)

fig

#ssave("gaussian-dist.png",fig)

