using Random, Statistics, Printf
function records(path)
    lines=readlines(path);isempty(lines)&&return Dict{String,String}[]
    names=split(first(lines),',')
    [Dict(String.(names).=>String.(split(line,','))) for line in lines[2:end] if !isempty(line)]
end
function interval(logratios,alpha,rng;replicates=50000,block=5)
    n=length(logratios);samples=Vector{Float64}(undef,replicates)
    for trial in 1:replicates
        total=0.0;used=0
        while used<n
            start=rand(rng,1:n)
            for j in 0:min(block,n-used)-1
                total+=logratios[mod1(start+j,n)]
            end
            used+=min(block,n-used)
        end
        samples[trial]=exp(total/n)
    end
    quantile(samples,[alpha/2,1-alpha/2])
end
function analyze_retry(dir)
    rows=records(joinpath(dir,"screen.csv"))
    paired=records(joinpath(dir,"paired_trials.csv"))
    keys=unique((r["size"],r["comparison"],r["shared_case"],r["global_case"]) for r in paired)
    completekeys=filter(key->count(r->(r["size"],r["comparison"],r["shared_case"],r["global_case"])==key,paired)==30,keys)
    isempty(completekeys)&&error("No completed 30-pair comparison yet")
    alpha=0.05/length(completekeys) # Bonferroni family-wise 95% coverage
    rng=MersenneTwister(20261110)
    summary=Dict{String,Any}[]
    open(joinpath(dir,"paired_summary.csv"),"w") do io
        println(io,"size,comparison,shared_case,global_case,global_median_ms,shared_median_ms,median_speedup,paired_geometric_speedup,simultaneous_ci_low,simultaneous_ci_high,supported_win,at_least_1pct")
        for key in sort(completekeys;by=k->(parse(Int,k[1]),k[2],k[3],k[4]))
            data=filter(r->(r["size"],r["comparison"],r["shared_case"],r["global_case"])==key,paired)
            globals=parse.(Float64,getindex.(data,"global_ms"))
            shareds=parse.(Float64,getindex.(data,"shared_ms"))
            logs=log.(globals./shareds);lo,hi=interval(logs,alpha,rng)
            geo=exp(mean(logs));ratio=median(globals)/median(shareds)
            fields=(key...,median(globals),median(shareds),ratio,geo,lo,hi,lo>1,lo>1&&geo>=1.01)
            println(io,join(fields,','))
            push!(summary,Dict("size"=>parse(Int,key[1]),"comparison"=>key[2],"shared_case"=>key[3],"global_case"=>key[4],"global_ms"=>median(globals),"shared_ms"=>median(shareds),"speedup"=>ratio,"geometric_speedup"=>geo,"low"=>lo,"high"=>hi,"supported"=>lo>1,"practical"=>lo>1&&geo>=1.01))
        end
    end
    open(joinpath(dir,"RETRY_RESULTS.md"),"w") do io
        println(io,"# Shared-memory retry\n")
        println(io,isfile(joinpath(dir,"COMPLETE")) ? "All scheduled sizes completed.\n" : "PARTIAL RUN — additional sizes/comparisons may still be running.\n")
        println(io,"$(length(rows)) screened configurations; $(length(completekeys)) independent 30-pair comparisons. Every accepted full-image configuration had zero mismatches.\n")
        println(io,"Fresh paired order was randomized. Intervals use a circular block bootstrap (block length 5, 50,000 replicates) and Bonferroni-adjusted 95% family-wise coverage across the completed comparisons. These are run-specific results, not universal guarantees.\n")
        println(io,"| N | Comparison | Global ms | Shared ms | Speedup | Simultaneous interval | Shared candidate |")
        println(io,"|---|---|---|---|---|---|---|")
        for s in summary
            @printf(io,"| %d | %s | %.4f | %.4f | %.4f× | [%.4f, %.4f] | %s |\n",s["size"],s["comparison"],s["global_ms"],s["shared_ms"],s["speedup"],s["low"],s["high"],s["shared_case"])
        end
        println(io,"\n## Interpretation\n")
        strong=filter(s->s["comparison"]=="best_global"&&s["practical"],summary)
        matched=filter(s->s["comparison"]=="matched"&&s["practical"],summary)
        if !isempty(strong)
            println(io,"Shared memory has at least one independently confirmed ≥1% win over the strongest selected global baseline in this sweep. See the precise sizes/configurations and intervals above.")
        elseif !isempty(matched)
            println(io,"Shared memory has independently confirmed wins over matched implementations, but no confirmed ≥1% win over the strongest global baseline in this sweep. A matched-path win is not an overall fastest-renderer claim.")
        else
            println(io,"No independently confirmed ≥1% shared-memory win was found in the completed comparisons. Do not describe screening noise as a speedup.")
        end
        native=occursin("storage=CuMatrix",read(joinpath(dir,"environment.txt"),String))
        if native
            println(io,"\nThis experiment requires native Julia CuMatrix[y,x] column-major output. Shared-memory tiles transpose horizontally computed pixels into contiguous column-major stores, optionally staging coordinate arrays from global memory. Controls include both horizontal and vertical computation, multiple ILP factors, block sizes and grid sizes. This output-layout requirement differs from the previous flat row-major renderer; a win here does not establish a win over that renderer.")
        else
            println(io,"\nThe strategies are output exchange through warp-local shared memory (packed, padded and XOR-swizzled), and 32×32 shared transpose tiles with optional global-coordinate staging, using the original flat row-major output.")
        end
        println(io,"\nPixel arithmetic, Float32 precision, region and 2,000-iteration limit are unchanged. All timings exclude compilation, allocations, coordinate preparation and output transfer. Original coordinate-only staging results are retained separately. Larger Float32 grids may contain repeated coordinates because of finite precision.")
    end
    println("Analyzed ",length(rows)," configurations and ",length(summary)," completed comparisons")
    for s in summary
        if s["practical"]
            println("SUPPORTED WIN: ",s)
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__)&&analyze_retry(only(ARGS))
