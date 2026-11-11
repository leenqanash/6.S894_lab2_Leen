# Julia standard library only. Run: julia report.jl results/RUN_DIRECTORY
using Statistics, Printf
function readrows(path)
    lines=readlines(path)
    isempty(lines) && return Dict{String,String}[]
    names=split(first(lines),',')
    [Dict(names .=> split(line,',')) for line in lines[2:end] if !isempty(line)]
end
function report(dir)
    rows=readrows(joinpath(dir,"summary.csv"))
    reportpath=joinpath(dir,"report.md")
    open(reportpath,"w") do io
        println(io,"# Julia Mandelbrot experiment results\n")
        println(io,isfile(joinpath(dir,"COMPLETE")) ? "Run completed." : "PARTIAL RUN: do not treat this as a completed sweep.")
        println(io,"\nTimings exclude compilation, allocation and output transfer. Device medians use CUDA events; wall medians are recorded separately. Winners are best observed configurations, not universal optima.\n")
        println(io,"| Size | Baseline median ms | Best median ms | Speedup | Threads/block | Blocks | Pixels/thread | Layout | Assignment |")
        println(io,"|---|---|---|---|---|---|---|---|---|")
        for size in sort(unique(parse(Int,r["size"]) for r in rows if r["experiment"]=="mandelbrot"))
            group=filter(r->r["experiment"]=="mandelbrot" && parse(Int,r["size"])==size,rows)
            valid=filter(r->r["mismatches"]=="0",group)
            isempty(valid) && continue
            best=valid[argmin(parse.(Float64,getindex.(valid,"median_ms")))]
            bs=sort(unique(parse(Int,r["blocks"]) for r in valid))
            candidates=filter(r->r["threads"]=="128" && parse(Int,r["blocks"])==bs[min(3,length(bs))] && r["pixels_per_thread"]=="1" && r["layout"]=="linear" && r["assignment"]=="interleaved",valid)
            isempty(candidates) && continue
            baseline=first(candidates)
            base=parse(Float64,baseline["median_ms"]); ms=parse(Float64,best["median_ms"])
            println(io,"| $size | $base | $ms | $(base/ms)× | $(best["threads"]) | $(best["blocks"]) | $(best["pixels_per_thread"]) | $(best["layout"]) | $(best["assignment"]) |")
        end
        if isfile(joinpath(dir,"confirmation.csv"))
            confirmed=readrows(joinpath(dir,"confirmation.csv"))
            sort!(confirmed;by=r->parse(Float64,r["median_ms"]))
            best=first(confirmed)
            println(io,"\n## Fresh confirmation (15 trials per setting)\n")
            println(io,"Best observed confirmed setting: threads $(best["threads"]), blocks $(best["blocks"]), independent pixels/thread $(best["pixels_per_thread"]), layout $(best["layout"]), assignment $(best["assignment"]); median $(best["median_ms"]) ms.")
        end
        println(io,"\n## Single-warp controls\n")
        for size in sort(unique(parse(Int,r["size"]) for r in rows if r["experiment"]=="single_warp"))
            controls=filter(r->r["experiment"]=="single_warp" && parse(Int,r["size"])==size,rows)
            all(p->any(r->r["pixels_per_thread"]==p,controls),("1","2")) || continue
            base=first(filter(r->r["pixels_per_thread"]=="1",controls))
            ilp=first(filter(r->r["pixels_per_thread"]=="2",controls))
            full=filter(r->r["experiment"]=="mandelbrot" && parse(Int,r["size"])==size,rows)
            best=minimum(parse(Float64,r["median_ms"]) for r in full)
            println(io,"- Size $size: one-warp ILP speedup $(parse(Float64,base["median_ms"])/parse(Float64,ilp["median_ms"]))×; best full-GPU speedup versus one warp $(parse(Float64,base["median_ms"])/best)×.")
        end
        if isfile(joinpath(dir,"cpu.csv"))
            cpu=readrows(joinpath(dir,"cpu.csv"))
            println(io,"\n## CPU ILP and threading\n")
            bases=filter(r->r["mode"]=="serial" && r["pixels_per_thread"]=="1",cpu)
            if !isempty(bases)
                baseline=parse(Float64,first(bases)["median_ms"])
                for r in cpu
                    ms=parse(Float64,r["median_ms"])
                    println(io,"- $(r["mode"]), $(r["pixels_per_thread"]) independent pixels: $ms ms; $(baseline/ms)× versus serial one-pixel CPU control.")
                end
            end
        end
        println(io,"\n## Memory access\n")
        for mode in ("memory_direct","memory_shared")
            group=filter(r->r["experiment"]==mode && r["mismatches"]=="0",rows)
            isempty(group) && continue
            best=group[argmin(parse.(Float64,getindex.(group,"median_ms")))]
            println(io,"- $mode best observed: offset $(best["offset_elements"]) Float32 elements, stride $(best["stride"]), threads $(best["threads"]), median $(best["median_ms"]) ms; useful bandwidth $(best["effective_GB_per_s"]) GB/s.")
        end
        println(io,"\nEffective bandwidth counts useful input + output bytes; it is not measured DRAM transaction bandwidth. Cache state, stride, and repeated address reuse affect these results. Offset zero means an aligned CuArray base; offsets preserve Float32 alignment while changing warp transaction alignment.")
        println(io,"\nMandelbrot creates its coordinates mathematically. Linear staging adds shared stores/loads and barriers without cross-thread reuse; the tile_shared variant instead reuses x across rows and broadcasts y within each row. Compare coordinates_global versus coordinates_shared and tile_global versus tile_shared at identical settings. Nsight Compute is needed to confirm memory sectors, transactions and bank conflicts.")
    end
    println("Saved ",reportpath)
end
abspath(PROGRAM_FILE)==(@__FILE__) && report(only(ARGS))
