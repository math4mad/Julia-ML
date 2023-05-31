
"""
in Unifrom  when  b approaching a,  pdf's value  to ∞
"""

using GLMakie, Distributions
a = 0.2
bcollect = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.21, 0.205]
dist(b) = Uniform(0.2, b)
tspan = range(0.0, 1.0, 100)


fig = Figure()

for i in eachindex(bcollect)
    local d = dist(bcollect[i])
    local ax = Axis(fig[1, i], title=L"uni(0.2,%$(bcollect[i]))")
    lines!(ax, tspan, pdf.(d, tspan))
end

fig
#save("different-b-in-uniform(0.2,b).png",fig)