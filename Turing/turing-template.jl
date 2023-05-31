using Turing, Distributions

mean(Dirichlet(6, 1))
sum(mean(Dirichlet(6, 1)))