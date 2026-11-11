include("report.jl")
function json_value(v)
    v isa AbstractString && return "\""*replace(v,"\\"=>"\\\\","\""=>"\\\"","\n"=>"\\n")*"\""
    v isa AbstractDict && return "{"*join((json_value(string(k))*":"*json_value(x) for (k,x) in v),",")*"}"
    v isa AbstractVector && return "["*join(json_value.(v),",")*"]"
    v isa Bool && return v ? "true" : "false"
    v===nothing && return "null"
    return string(v)
end
function analyze(dir)
    rows=readrows(joinpath(dir,"summary.csv"))
    big=filter(r->r["size"]=="8192",rows)
    onlyrow(mode,ppt)=first(filter(r->r["experiment"]==mode && r["pixels_per_thread"]==string(ppt),big))
    best(group)=group[argmin([parse(Float64,r["median_ms"]) for r in group])]
    ms(r)=parse(Float64,r["median_ms"])
    confirmed=readrows(joinpath(dir,"confirmation.csv"))
    winner=best(confirmed)
    warp=onlyrow("single_warp",1);ilp=onlyrow("single_warp",2)
    baseline=first(filter(r->r["threads"]=="128" && r["pixels_per_thread"]=="1" && r["assignment"]=="interleaved" && r["layout"]=="linear",confirmed))
    output=Dict{String,Any}("rows"=>length(rows),"nonzero_mismatch_rows"=>count(r->r["mismatches"]!="0",rows),"complete"=>isfile(joinpath(dir,"COMPLETE")),"main_confirmed_best"=>winner,"main_confirmed_baseline"=>baseline,"single_warp"=>warp,"single_warp_ilp"=>ilp,"single_warp_ilp_speedup"=>ms(warp)/ms(ilp),"full_gpu_vs_one_warp"=>ms(warp)/ms(winner),"tuning_speedup"=>ms(baseline)/ms(winner))
    pairs=Any[]
    for mode in ("coordinates_shared","tile_shared")
        for shared in filter(r->r["experiment"]==mode,big)
            direct_mode=mode=="coordinates_shared" ? "coordinates_global" : "tile_global"
            direct=first(filter(r->r["experiment"]==direct_mode && r["threads"]==shared["threads"] && r["layout"]==shared["layout"],big))
            push!(pairs,Dict("shared"=>shared,"direct"=>direct,"shared_speedup"=>ms(direct)/ms(shared)))
        end
    end
    output["shared_comparisons"]=pairs
    if isfile(joinpath(dir,"family_confirmation.csv"))
        families=readrows(joinpath(dir,"family_confirmation.csv"))
        output["family_confirmation"]=families
        output["best_precomputed_global"]=best(filter(r->r["experiment"] in ("coordinates_global","tile_global"),families))
    end
    memory=filter(r->startswith(r["experiment"],"memory_"),rows)
    output["memory_best_direct"]=best(filter(r->r["experiment"]=="memory_direct",memory))
    output["memory_best_shared"]=best(filter(r->r["experiment"]=="memory_shared",memory))
    output["memory_matched_256_threads"]=filter(r->r["threads"]=="256" && r["offset_elements"] in ("0","1","8") && r["stride"] in ("1","32"),memory)
    if isfile(joinpath(dir,"cpu.csv"))
        cpu=readrows(joinpath(dir,"cpu.csv"))
        if !isempty(cpu)
            base=first(filter(r->r["mode"]=="serial" && r["pixels_per_thread"]=="1",cpu))
            output["cpu"]=cpu
            output["cpu_best"]=best(cpu)
            output["cpu_speedups"]=[Dict("mode"=>r["mode"],"pixels_per_thread"=>r["pixels_per_thread"],"speedup"=>ms(base)/ms(r)) for r in cpu]
        end
    end
    write(joinpath(dir,"analysis.json"),json_value(output))
    println("Total configurations: ",length(rows),"; mismatch rows: ",output["nonzero_mismatch_rows"])
    println("Main confirmed best: ",winner)
    println("One-warp ILP speedup: ",output["single_warp_ilp_speedup"],"; full GPU vs one warp: ",output["full_gpu_vs_one_warp"])
    println("Memory direct best: ",output["memory_best_direct"])
    println("Memory shared best: ",output["memory_best_shared"])
end
abspath(PROGRAM_FILE)==(@__FILE__) && analyze(only(ARGS))
