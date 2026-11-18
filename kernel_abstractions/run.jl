# Setup: julia setup.jl
# Run:   julia --project=. run.jl
# Defaults: 1024² and 8192², Float32, 2000 iterations, native Output[y,x].
# Override SIZES=1024,2048,4096,8192 and TRIALS=5 through the environment.
using CUDA, KernelAbstractions, Dates, Statistics, Random, SHA, Sockets
include("gpu.jl")

struct Case
    group::Tuple{Int,Int}                # (contiguous rows, columns)
    pixels::Int
    shared::Bool
    early::Bool
    reuse::Bool
end
caseid(c) = "$(c.group[1])x$(c.group[2])_p$(c.pixels)_s$(Int(c.shared))_e$(Int(c.early))_reuse$(Int(c.reuse))"
function cases()
    configs = Case[]
    for group in ((32,4),(32,8),(32,16),(16,16),(8,32)), pixels in (1,2), shared in (false,true), early in (false,true)
        push!(configs,Case(group,pixels,shared,early,false))
    end
    for group in ((32,4),(32,8),(32,16)), pixels in (1,2), shared in (false,true)
        push!(configs,Case(group,pixels,shared,true,true))
    end
    configs
end

# Float32 coordinate identity is exact, not a visual-similarity heuristic.
function unique_coordinates(values)
    unique_values = unique(values)
    lookup = Dict(value=>Int32(i) for (i,value) in enumerate(unique_values))
    unique_values, Int32[lookup[value] for value in values]
end
function prepare(n)
    started = time_ns()
    xs = [coordinate_x(x,n) for x in 1:n]; ys = [coordinate_y(y,n) for y in 1:n]
    ux,xmap = unique_coordinates(xs); uy,ymap = unique_coordinates(ys)
    prep_ms = (time_ns()-started)/1e6
    data = (xs=CuArray(xs),ys=CuArray(ys),ux=CuArray(ux),uy=CuArray(uy),
            xmap=CuArray(xmap),ymap=CuArray(ymap),
            compact=CUDA.zeros(UInt32,length(uy),length(ux)))
    CUDA.synchronize()
    data, prep_ms
end
function action(c,out,data,n,limit)
    if c.reuse
        return ()->begin
            launch!(data.compact,data.ux,data.uy,n,limit;group=c.group,pixels=c.pixels,shared=c.shared,early=c.early)
            expand!(out,data.compact,data.xmap,data.ymap)
        end
    end
    ()->launch!(out,data.xs,data.ys,n,limit;group=c.group,pixels=c.pixels,shared=c.shared,early=c.early)
end
function elapsed_ms(f)
    CUDA.synchronize()
    1000 * CUDA.@elapsed f()
end
function samples(f,trials)
    f(); f(); CUDA.synchronize()            # compilation/warmup excluded
    [elapsed_ms(f) for _ in 1:trials]
end
function release(data)
    for a in values(data); CUDA.unsafe_free!(a); end
end

function run()
    CUDA.functional() || error("An NVIDIA GPU is required; CPU checks are in test.jl")
    CUDA.allowscalar(false)
    sizes = parse.(Int,split(get(ENV,"SIZES","1024,8192"),','))
    trials = parse(Int,get(ENV,"TRIALS","5")); limit = DEFAULT_ITERS
    rng = MersenneTwister(179494); configs = cases()
    dir = joinpath(@__DIR__,"results",Dates.format(now(),"yyyymmdd-HHMMSS")); mkpath(dir)
    println("RESULTS=",dir); flush(stdout)
    open(joinpath(dir,"environment.txt"),"w") do io
        println(io,"$(now()) $(gethostname()); Julia $VERSION; limit=$limit; Float32; CuMatrix[y,x]")
        CUDA.versioninfo(io)
        for name in ("gpu.jl","core.jl","run.jl")
            println(io,name," SHA256 ",bytes2hex(sha256(read(joinpath(@__DIR__,name)))))
        end
    end
    # GPU compilation and exact CPU comparisons for boundary and partial tiles.
    for n in (1,31,65)
        data,_ = prepare(n); out=CUDA.zeros(UInt32,n,n); expected=reference(n,limit)
        for c in configs
            action(c,out,data,n,limit)(); CUDA.synchronize()
            Array(out)==expected || error("CPU mismatch N=$n $(caseid(c))")
        end
        release(data);CUDA.unsafe_free!(out)
    end
    write(joinpath(dir,"correctness.txt"),"All $(length(configs)) configurations passed actual NVIDIA kernels against CPU at N=1,31,65. Every full-size result is also compared exactly with the scalar-state KA reference.\n")
    println("GPU/CPU correctness passed for ",length(configs)," configurations");flush(stdout)
    open(joinpath(dir,"summary.csv"),"w") do summary
      open(joinpath(dir,"trials.csv"),"w") do raw
       open(joinpath(dir,"paired.csv"),"w") do paired
        println(summary,"size,case_id,rows,columns,pixels,shared,early,reuse,unique_x,unique_y,coordinate_host_ms,median_ms,minimum_ms,mismatches")
        println(raw,"size,case_id,trial,device_ms")
        println(paired,"size,comparison,shared_case,global_case,trial,global_first,global_ms,shared_ms")
        for n in sizes
            data,prep_ms=prepare(n);out=CUDA.zeros(UInt32,n,n);ref=similar(out)
            action(Case((32,8),1,false,true,false),ref,data,n,limit)();CUDA.synchronize()
            measured=Dict{String,Float64}()
            for c in shuffle(rng,configs)
                f=action(c,out,data,n,limit); f();CUDA.synchronize()
                mismatches=Int(sum(out .!= ref));mismatches==0||error("Full mismatch N=$n $(caseid(c))")
                times=samples(f,trials);measured[caseid(c)]=median(times)
                println(summary,join((n,caseid(c),c.group...,c.pixels,c.shared,c.early,c.reuse,length(data.ux),length(data.uy),prep_ms,median(times),minimum(times),mismatches),','));flush(summary)
                for (i,ms) in enumerate(times);println(raw,join((n,caseid(c),i,ms),','));end;flush(raw)
            end
            # Keep the plain and coordinate-reuse algorithms as separate comparisons.
            for reuse in (false,true)
                globals=filter(c->!c.shared&&c.reuse==reuse,configs)
                shareds=filter(c->c.shared&&c.reuse==reuse,configs)
                bestg=globals[argmin([measured[caseid(c)] for c in globals])]
                bests=shareds[argmin([measured[caseid(c)] for c in shareds])]
                matched=Case(bests.group,bests.pixels,false,bests.early,reuse)
                for (label,gcase) in (("matched",matched),("selected_global",bestg))
                    g=action(gcase,out,data,n,limit);s=action(bests,out,data,n,limit)
                    samples(g,3);samples(s,3);gt=Float64[];st=Float64[]
                    for trial in 1:30
                        first=rand(rng,Bool)
                        if first;gm=elapsed_ms(g);sm=elapsed_ms(s);else;sm=elapsed_ms(s);gm=elapsed_ms(g);end
                        push!(gt,gm);push!(st,sm)
                        println(paired,join((n,label,caseid(bests),caseid(gcase),trial,first,gm,sm),','));flush(paired)
                    end
                    println("N=$n reuse=$reuse $label global=$(median(gt)) shared=$(median(st)) speedup=$(median(gt)/median(st))");flush(stdout)
                end
            end
            println("N=$n unique image $(length(data.uy))x$(length(data.ux)); lowest screen ",minimum(values(measured))," ms");flush(stdout)
            release(data);CUDA.unsafe_free!(out);CUDA.unsafe_free!(ref);CUDA.reclaim();GC.gc()
        end
       end
      end
    end
    write(joinpath(dir,"COMPLETE"),"Completed $(now())\n")
end
abspath(PROGRAM_FILE)==(@__FILE__) && run()
