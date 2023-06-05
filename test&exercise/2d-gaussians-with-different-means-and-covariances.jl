"""
Introduction to Probability for Data Science-Michigan Publishing (2021).pdf page308
"""

using GLMakie,Distributions,Random
using FileIO
Random.seed!(12334)

xs=ys=range(-5,5,100)
μ₁=[0 , 2]
Σ₁=[5 0 ; 0 0.5]

μ₂=[1,2]
Σ₂=[1 -0.5;-0.5 1]

μ₃=[0,0]
Σ₃=[2 1.9; 1.9 2]
dist=[MvNormal(μ₁,Σ₁),MvNormal(μ₂,Σ₂),MvNormal(μ₃,Σ₃ )]
colors = [:orange, :lightgreen, :purple]


fig=Figure(resolution=(1800,700))

for i in 1:3
    ax1=Axis(fig[1:4,i])
    ax2=Axis(fig[5,i],height=70)
    img=load("./test&exercise/imgs/cor$(i).png")|>rotr90
    scatter!(ax1, rand(dist[i],200),color=(colors[i],0.8),markersize=14)
    image!(ax2,img)
    hidespines!(ax2)
    hidedecorations!(ax2)

end



fig
#save("2d-gaussians-with-different-means-and-covariances.png",fig)