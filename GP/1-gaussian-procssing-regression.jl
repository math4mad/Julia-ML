"""
https://juanitorduz.github.io/gaussian_process_reg/
"""

import KernelFunctions:Kernel
using GaussianProcesses,CSV,GLMakie,Random,Distributions,LinearAlgebra,
      StatsBase

Random.seed!(1223)
n=300
f1(x)=sin(4pi*x)
f2(x)=sin(7pi*x)
f(x)=f1(x)+f2(x)
xs=range(0,1,n)

function plot_origin_function()
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, xs, f.(xs); label=L"f(x)=sin(4\pi x)+sin(7\pi x)", linewidth=4)
    lines!(ax, xs, f1.(xs), label=L"f(x)=sin(4\pi x)", linestyle=:dash)
    lines!(ax, xs, f2.(xs), label=L"f(x)=sin(7\pi x)", linestyle=:dash)
    axislegend(ax)
    fig
end
#plot_origin_function()


# generate  training sample observation
noisedata,testdata=(()->begin
    μ=0;δ=0.4
    noise=Normal(μ,δ)|>d->rand(d,n)
    noisedata=data_with_noise=f.(xs)+noise
    test_data=data_with_noise[1:3:end]
    return  noisedata,test_data
end)()


function plot_noise_data()
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, xs, f.(xs); label=L"f(x)=sin(4\pi x)+sin(7\pi x)", linewidth=3,color=(:red,0.7))
    scatter!(ax, xs,data_with_noise,label=L"data",)
    axislegend(ax)
    fig
    #save("gaussian-procssing-regression-2.png",fig)
end
#plot_noise_data()






# GP  workflow

mZero = MeanZero()              
kern = SE(0.0,0.0)
logObsNoise = -1.0   

# prior plot

"""
    参见: https://juliagaussianprocesses.github.io
   /KernelFunctions.jl/stable/examples/gaussian-process-priors/#Random-samples
"""
function plot_prior1(k::Kernel)
    n=300
    xs=range(-5,5,n)
    K = kernelmatrix(k, xs)
    num_samples =5
    v = randn(n, num_samples);
    
    function mvn_sample(K)
        L = cholesky(K + 1e-6 * I)
        f = L.L * v
        return f
    end
    data= mvn_sample(K)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for i in 1:num_samples
        lines!(ax,xs, data[:,i])
    end

    fig
end


function plot_prior2()
    
    n=100
    xs=range(-5,5,n)
    
    covmatrix=(xs*xs')+1e-6 * I
    d=MvNormal(fill(0.0,n),covmatrix)
    fig = Figure(resolution=(1200,400))
    ax = Axis(fig[1, 1])
    for i in 1:10
        lines!(ax,xs,rand(d))
    end

    fig
end



"""
    plot_kernel_matrix(k::Kernel,data,noisedata)
    输入 kernel, 理想数据和有噪音数据
    绘制 heatmap
TBW
"""
function plot_kernel_matrix(k::Kernel,data,noisedata)
    K = kernelmatrix(k, data)
    K2=kernelmatrix(k, noisedata)
    fig=Figure(resolution=(1400,600))
    ax=Axis(fig[1,1],yreversed=true)
    ax2=Axis(fig[1,3],yreversed=true)
    hm1=heatmap!(ax,data,data,K)
    hm2=heatmap!(ax2,noisedata,noisedata,K2)
    Colorbar(fig[1,2],hm1)
    Colorbar(fig[1,4],hm2)
    fig
    #save("kernel-matrix-data-noisedata.png",fig)
end





# posterior process
function main()
    gp = GP(xs,noisedata,mZero,kern,logObsNoise)

    optimize!(gp)

    function plot_gp(xs,samples)
    rows,cols=size(samples)
    fig = Figure()
    ax=Axis(fig[1,1])
    for c  in 1:cols
       lines!(ax, xs,samples[:,c])
    end
    lines!(ax, xs, f.(xs); label=L"f(x)=sin(4\pi x)+sin(7\pi x)", linewidth=4)
    axislegend(ax)
    fig
    #save("gaussian-procssing-regression-3.png",fig)

    end

 sample_n=10;
 sample_n|>n->rand(gp, xs, n)|>d->plot_gp(xs,d)

end

#main()




k=SqExponentialKernel()
data=f.(xs)
plot_kernel_matrix(k,data,noisedata)


