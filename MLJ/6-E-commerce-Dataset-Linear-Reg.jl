"""
model:https://www.kaggle.com/code/mohammedibrahim784/e-commerce-dataset-linear-regression-model
datasets:https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website
aim:  通过上网浏览时间预测年花费
"""


using GLMakie, MLJ,CSV,DataFrames,StatsBase

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,Count=>Continuous)
    df = fetch(str)|>to_ScienceType
    return df
end

str="Ecommerce-Customers"
df=data_prepare(str)
select!(df,4:8)
#schema(df)

#label=["Avg. Session Length","Time on App","Time on Website","Length of Membership"]
axs = []
label=names(df)|>Array
colors = [:orange, :lightgreen, :purple,:lightblue,:red,:green]

fig = Figure(resolution=(1400, 1400))
ax=Axis(fig[1,1])

function plot_diag(i)

    ax = Axis(fig[i, i])
    push!(axs, ax)
    density!(ax, df[:, i]; color=(colors[i], 0.5),
            strokewidth=1.25, strokecolor=colors[i])
end


function plot_cor(i, j)
    ax = Axis(fig[i, j])
    scatter!(ax, df[:, i], df[:, j]; color=colors[j])
end


function plot_pair()
    [(i == j ? plot_diag(i) : plot_cor(i, j)) for i in 1:5, j in 1:5]
end

function add_xy_label()
    for i in 1:5
        Axis(fig[5, i], xlabel=label[i],)
        Axis(fig[i, 1], ylabel=label[i],)
    end
end

function main()

    plot_pair()
    add_xy_label()
    
    return fig
    #save("$(str).png",fig)
end

#main()


df_cov = (cov(df|>Matrix)) .|> d -> round(d, digits=3)
df_cor = (cor(df|>Matrix)) .|> d -> round(d, digits=3)


fig = Figure(resolution=(1200, 600))
ax1 = Axis(fig[1, 1]; xticks=(1:5, label), yticks=(1:5, label), title="ecommerce cov matrix")
ax3 = Axis(fig[1, 3], xticks=(1:5, label), yticks=(1:5, label), title="ecommerce cor matrix")

hm = heatmap!(ax1, df_cov)
Colorbar(fig[1, 2], hm)
[text!(ax1, x, y; text=string(df_cov[x, y]), color=:white, fontsize=18, align=(:center, :center)) for x in 1:5, y in 1:5]

hm2 = heatmap!(ax3, df_cor)
Colorbar(fig[1, 4], hm2)
[text!(ax3, x, y; text=string(df_cor[x, y]), color=:white, fontsize=18, align=(:center, :center)) for x in 1:5, y in 1:5]

fig

#save("6-E-commerce-Dataset-cov-cor-matrix.png",fig)






