# The usual hard instance for BAI in linear bandits from Soare et al.

using JLD2;
using Distributed;
using Printf;
using IterTools;
using DelimitedFiles;
using NPZ;
using Distributions

include("../thresholds.jl")
include("../peps.jl")
include("../elimination_rules.jl")
include("../stopping_rules.jl")
include("../sampling_rules.jl")
include("../runit.jl")
include("../experiment_helpers.jl")
include("../utils.jl")
include("../envelope.jl")

δ = 0.001
d = 2
K = d + 1
m = 2  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
for k = 1:d
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end
ω = 0.1
v = zeros(d);
v[1] = cos(ω);
v[2] = sin(ω);
push!(arms, v)

θ = zeros(d)
θ[1] = 1.0

μ = [arm'θ for arm in arms]
topm_arms = istar(Topm(arms, m), θ)
println("min abs value of μ: ", minimum(abs.(μ)))
println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

β = LinearThreshold(d, 1, 1, 1)
β = GK16()

pep = BAI(arms);
#pep = Topm(arms, m);
#pep = OSI(arms);

if typeof(pep) != Topm
    w_star = optimal_allocation(pep, θ, false, 10000)
    println("Optimal allocation: ", round.(w_star, digits=3))
end

# LinGame vs LinGapE vs FWS vs LazyTaS vs Oracle
elim_rules = [NoElim()]
stopping_rules = [Force_Stopping(3000, NoStopping())]
sampling_rules = [LinGame(CTracking, NoElimSR, false)]

repeats = 100;
seed = 123;

# One fake run for each algorithm
# This is to have fair comparison of compute times later since Julia compiles stuff at the first calls
@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    ((sampling, stopping, elim),) -> runit(seed, sampling, stopping, elim, θ, pep, β, δ), 
    zip(sampling_rules, stopping_rules, elim_rules)
);

@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    (((sampling, stopping, elim), i),) -> runit(seed + i, sampling, stopping, elim, θ, pep, β, δ), 
    Iterators.product(zip(sampling_rules, stopping_rules, elim_rules), 1:repeats),
);

# uncomment the line below if would like to see stats
# dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

data = collect(data)
writedlm("results_c.txt", data)

# save
#isdir("experiments") || mkdir("experiments")
#@save isempty(ARGS) ? "experiments/hard_lin_$(typeof(pep))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed