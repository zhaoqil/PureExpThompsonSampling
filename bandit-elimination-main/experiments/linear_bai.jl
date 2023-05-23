using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions
using Random;
using JLD;
using JSON;
using HDF5;
using DelimitedFiles


include("../thresholds.jl")
include("../peps.jl")
include("../elimination_rules.jl")
include("../stopping_rules.jl")
include("../sampling_rules.jl")
include("../runit.jl")
include("../experiment_helpers.jl")
include("../utils.jl")
include("../envelope.jl")

δ = 0.01
d = 6
K = 20
m = 5  # for topm

# for reproducibility
rng = MersenneTwister(123)


arms = Vector{Float64}[]
for i in 1:20
    v = randn(rng, d)
    v /= norm(v)
    push!(arms, v)
end

min_norm = 10
for i in 1:length(arms)
    for j in (i+1):length(arms)
        norm_ij = norm(arms[i] - arms[j])
        if norm_ij < min_norm
            global min_pair = [i, j]
            global min_norm = norm_ij
        end
    end
end


θ = arms[min_pair[1]] + 0.01*(arms[min_pair[2]] - arms[min_pair[1]])
println(arms)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/theta.csv", θ)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/arms.csv", arms)
#h5file = h5open("/home/jupyter-zli9/PureExpThompsonSampling/linear_instance.h5", "w")
#h5write(h5file, "theta", θ)
#h5write(h5file, "arms", arms)
#h5close(h5file)
#JLD2.save("/home/jupyter-zli9/PureExpThompsonSampling/linear_instance.h5", Dict("theta" => θ, "arms" => arms))

# # θ = zeros(d)
# # θ[1] = 1
# # θ[2:Int(mid/2)] .= 0.9
# # θ[Int(mid/2)+1:mid] .= 0.8
# # θ[mid+1:end] = rand(rng, d-mid) .- 1   # uniform in [-0.5,0.5]


#μ = [arm'θ for arm in arms]
#topm_arms = istar(Topm(arms, m), θ)
# println("min abs value of μ: ", minimum(abs.(μ)))
# println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
# println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

# # β = LinearThreshold(d, 1, 1, 1)
# # β = GK16()
# β = HeuristicThreshold()
β = 5

pep = BAI(arms);

w_star = optimal_allocation(pep, θ, false, 10000)
println("Optimal allocation: ", round.(w_star, digits=3))

max_samples = 1e6

repeats = 5;
seed = 123;

function run()
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

    # dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);
    
    data = collect(data)
    writedlm("/home/jupyter-zli9/PureExpThompsonSampling/results_sphere_500.txt", data, '\t')

#     # save
#     isdir("experiments/results") || mkdir("experiments/results")
#     @save isempty(ARGS) ? "experiments/results/lin_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

# #################################################
# # LinGapE
# #################################################
# # elim_rules = [NoElim(), CompElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [LinGapE(NoElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

# # run()

#################################################
# LinGame
#################################################

elim_rules = [NoElim()]
stopping_rules = [Force_Stopping(3000, NoStopping())]
sampling_rules = [LinGame(CTracking, NoElimSR, false)]

# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [LinGame(CTracking, NoElimSR, false), LinGame(CTracking, NoElimSR, false), LinGame(CTracking, ElimSR, false)]

run()


# #################################################
# # Oracle
# #################################################
# # elim_rules = [NoElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [FixedWeights(w_star), FixedWeights(w_star)]

# # run()

# #################################################
# # LazyTaS
# #################################################
# # elim_rules = [NoElim(), CompElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [LazyTaS(NoElimSR), LazyTaS(NoElimSR), LazyTaS(ElimSR)]

# # run()

# #################################################
# # FWS
# #################################################
# # elim_rules = [NoElim(), CompElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [FWSampling(NoElimSR), FWSampling(NoElimSR), FWSampling(ElimSR)]

# # run()

# #################################################
# # XY-Adaptive
# #################################################
# # elim_rules = [NoElim()]
# # stopping_rules = [NoStopping()]
# # sampling_rules = [XYAdaptive()]

# # run()

# #################################################
# # RAGE
# #################################################
# # elim_rules = [NoElim()]
# # stopping_rules = [NoStopping()]
# # sampling_rules = [RAGE()]

# # run()

# #################################################
# # LinGIFA
# #################################################
# # elim_rules = [NoElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [LinGIFA(), LinGIFA()]

# # run()