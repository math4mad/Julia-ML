"""
https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/examples/gaussian-process-priors/

协方差矩阵的 heatmap 解释了方差的离散程度,对角线方差最小,相关性最大(数据点与数据点自身的相关性)
"""

import KernelFunctions:Kernel
using GaussianProcesses,CSV,GLMakie,Random,Distributions,KernelFunctions,LinearAlgebra,
      StatsBase
Random.seed!(1223)


num_inputs = 101
xlim = (-5, 5)
X = range(xlim...; length=num_inputs);


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
    ax = Axis(fig[1, 1])
    hm=heatmap!(ax, X,X,K)
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

fig=Figure(resolution=(600,2400))

for i in eachindex(kernels)

    K = kernelmatrix(kernels[i],X)
    data= mvn_sample(K)
    kname=string(nameof(typeof(kernels[i])))
    ax1=Axis(fig[i,1];)
    #ax2=Axis(fig[i,2])
    
    for i in 1:num_samples
        lines!(ax1,X, data[:,i];label=i==1 ? "$(kname)" : nothing)
    end
    axislegend()
    #hm=heatmap!(ax2,X,X,K)
    #Colorbar(fig[i,3], hm)
    
    
end

fig

save("visualization-kernel-functions-1.png",fig)










