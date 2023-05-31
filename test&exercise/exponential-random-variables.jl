using GLMakie, Distributions
fig = Figure()
ax1 = Axis(fig[1, 1], limits=(0, 1, 0, 6), xlabel="exponential dist pdf")
ax2 = Axis(fig[1, 2], limits=(0, 1, 0, 1.2), xlabel="exponential dist cdf")
dist(λ) = Exponential(λ)
ts = range(0.0, 1.0, 100)
arr = [0.2, 0.5, 0.8]
lsarr = [:solid, :dot, :dashdot]
for (i, ls) in zip(arr, lsarr)
    d = dist(i)
    lines!(ax1, ts, pdf.(d, ts); label="θ=$(i)", linestyle=ls, linewidth=4)
    lines!(ax2, ts, cdf.(d, ts); label="θ=$(i)", linestyle=ls, linewidth=4)
end
axislegend(ax1)
axislegend(ax2)

fig
#save("exponential-random-variables.png",fig)
