# julia analyze.jl RESULTS_DIRECTORY
using Statistics, Random, Printf
function csvrows(path)
    lines=readlines(path);names=split(first(lines),',')
    [Dict(String.(names).=>String.(split(line,','))) for line in lines[2:end] if !isempty(line)]
end
function interval(logs,alpha,rng)
    n=length(logs);draws=Float64[]
    for sample in 1:50000
        total=0.0;used=0
        while used<n
            start=rand(rng,1:n)
            for j in 0:min(5,n-used)-1;total+=logs[mod1(start+j,n)];end
            used+=min(5,n-used)
        end
        push!(draws,exp(total/n))
    end
    quantile(draws,[alpha/2,1-alpha/2])
end
function analyze(dir)
    screen=csvrows(joinpath(dir,"summary.csv"))
    paired=csvrows(joinpath(dir,"paired.csv"))
    groups=Dict{Tuple,Vector{Dict{String,String}}}()
    for r in paired
        key=(parse(Int,r["size"]),r["comparison"],r["shared_case"],r["global_case"])
        push!(get!(groups,key,Dict{String,String}[]),r)
    end
    comparison=csvrows(joinpath(dir,"comparison.csv"))
    controls=Dict{Tuple,Vector{Dict{String,String}}}()
    for r in comparison
        key=(parse(Int,r["size"]),r["comparison"])
        push!(get!(controls,key,Dict{String,String}[]),r)
    end
    all(length(v)==30 for v in values(groups))||error("Incomplete shared/global trials")
    all(length(v)==30 for v in values(controls))||error("Incomplete old/new trials")
    alpha=0.05/(length(groups)+length(controls));rng=MersenneTwister(171117)
    open(joinpath(dir,"confirmed.csv"),"w") do io
        println(io,"size,comparison,baseline_median_ms,new_median_ms,median_speedup,paired_geometric_speedup,simultaneous_ci_low,simultaneous_ci_high,details")
        for key in sort(collect(keys(groups)))
            data=groups[key];base=parse.(Float64,getindex.(data,"global_ms"));new=parse.(Float64,getindex.(data,"shared_ms"))
            logs=log.(base./new);ci=interval(logs,alpha,rng)
            println(io,join((key[1],key[2],median(base),median(new),median(base)/median(new),exp(mean(logs)),ci...,key[3]*" vs "*key[4]),','))
        end
        for key in sort(collect(keys(controls)))
            data=controls[key];base=parse.(Float64,getindex.(data,"baseline_ms"));new=parse.(Float64,getindex.(data,"new_ms"))
            logs=log.(base./new);ci=interval(logs,alpha,rng)
            println(io,join((key...,median(base),median(new),median(base)/median(new),exp(mean(logs)),ci...,"fresh randomized pairs"),','))
        end
    end
    println("Accepted full-image checks: ",length(screen),"; mismatches: ",sum(parse(Int,r["mismatches"]) for r in screen))
    println("Confirmed groups: ",length(groups)+length(controls),"; 30 pairs/group; family-wise bootstrap intervals saved in confirmed.csv")
end
abspath(PROGRAM_FILE)==(@__FILE__)&&analyze(only(ARGS))
