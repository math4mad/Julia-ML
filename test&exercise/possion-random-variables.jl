using GLMakie, Distributions
fig = Figure()
ax1 = Axis(fig[1, 1], limits=(0, 20, 0, 0.4), xlabel="poisson dist pdf")
ax2 = Axis(fig[1, 2], limits=(0, 20, 0, 1), xlabel="poisson dist cdf")
dist(λ) = Poisson(λ)
ts =0:20
arr = [1, 4, 10]
colorarr = [:red, :blue, :orange]
for (i, c) in zip(arr, colorarr)
    local d = dist(i)
    stem!(ax1, ts, pdf.(d, ts); label="θ=$(i)",color=c, stemcolor=c, stemwidth =4,markersize = 20)
    stem!(ax2, ts, cdf.(d, ts); label="θ=$(i)",color=c,stemcolor=c, stemwidth =4)
end
axislegend(ax1)
axislegend(ax2)

fig
save("poisson-random-variables.png",fig)
