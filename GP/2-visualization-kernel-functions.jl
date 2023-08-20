"""
https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/examples/gaussian-process-priors/

协方差矩阵的 heatmap 解释了方差的离散程度,对角线方差最小,相关性最大(数据点与数据点自身的相关性)
ConstantKernel() 在 GLMakie 中无法绘制, 批量绘图时去掉

"""

import KernelFunctions:Kernel,kernelmatrix
using GaussianProcesses,CSV,GLMakie,Random,Distributions,KernelFunctions,LinearAlgebra,
      StatsBase
Random.seed!(1223)


num_inputs = 101
xlim = (-5, 5)
X = range(xlim...,101);


num_samples = 5
v = randn(num_inputs, num_samples);


function mvn_sample(K)
    L = cholesky(K + 1e-6 * I)
    f = L.L * v
    return f
end;



function plot_prior(k::Kernel)
    K = kernelmatrix(k, X)
    f = mvn_sample(K)

    fig = Figure()
    ax = Axis(fig[1, 1])

    for i in 1:num_samples
        lines!(ax, X, f[:, i])
    end

    fig
end

function  plot_matrix_heatmap(k::Kernel)
    K = kernelmatrix(k, X)
    fig = Figure()
    ax = Axis(fig[1, 1],yreversed=true)
    hm=heatmap!(ax,X,X,K)
    Colorbar(fig[1, 2], hm)
    fig
end

kernels = [
    Matern12Kernel(),
    Matern32Kernel(),
    Matern52Kernel(),
    SqExponentialKernel(),
    WhiteKernel(),
    ConstantKernel(),
    LinearKernel(),
    compose(PeriodicKernel(), ScaleTransform(0.2)),
    NeuralNetworkKernel(),
    GibbsKernel(; lengthscale=x -> sum(exp ∘ sin, x)),
]
#plot_prior(kernels[4])
#plot_matrix_heatmap(kernels[4])

function plot_series()
    fig=Figure(resolution=(600,2400))

    for i in eachindex(kernels)

        local K = kernelmatrix(kernels[i],X)
        local data= mvn_sample(K)
        kname=string(nameof(typeof(kernels[i])))
        ax1=Axis(fig[i,1];)
       
        for i in 1:num_samples
            lines!(ax1,X,data[:,i];label=i==1 ? "$(kname)" : nothing)
        end
        axislegend(ax1)
    end
    fig
end


function plot_heatmaps()
    fig=Figure(resolution=(300,2400))
    local X=range(-5,5,30)
    for (i,kernel) in enumerate([kernels[1:5]...,kernels[7:10]...])
        local kname=string(nameof(typeof(kernel)))
        local K = kernelmatrix(kernel,X)
        local ax=Axis(fig[i,1],yreversed=true,title="$(kname)")
        heatmap!(ax,X,X,K)
    end
    #save("kernels-heatmap.png",fig)
    fig
end


plot_heatmaps()




