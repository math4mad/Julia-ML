using GLMakie

fig=Figure()
ax=Axis(fig[1,1])

lines!(0..10, x -> sin(3x) / (cos(x) + 2),
    label = L"/\begin{bmatrix}1/\\2/\\ 3/\end{bmatrix}")

Legend(fig[1, 2], ax)
fig