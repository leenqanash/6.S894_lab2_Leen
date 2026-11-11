# Confirm leading configurations using fresh, randomized order and more trials.
# Run: julia --project=. confirm.jl results/ACTUAL_RUN_DIRECTORY
isdefined(@__MODULE__, :measure) || include("run.jl")
isdefined(@__MODULE__, :readrows) || include("report.jl")
using Random
function confirm(dir)
    rows=filter(r->r["experiment"]=="mandelbrot" && r["size"]=="8192" && r["mismatches"]=="0",readrows(joinpath(dir,"summary.csv")))
    isempty(rows) && error("No valid 8192² sweep results")
    sort!(rows;by=r->parse(Float64,r["median_ms"]))
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    configs=[(parse(Int,r["threads"]),parse(Int,r["blocks"]),parse(Int,r["pixels_per_thread"]),Symbol(r["layout"]),Symbol(r["assignment"])) for r in first(rows,min(10,length(rows)))]
    baseline=(128,4*sm,1,:linear,:interleaved)
    push!(configs,baseline); unique!(configs)
    shuffle!(MersenneTwister(191),configs)
    n=8192;limit=2000
    out=CUDA.zeros(UInt32,n*n);reference=similar(out)
    @cuda threads=128 blocks=4*sm fastmath=false mandelbrot!(reference,n,limit,Val(1),Val(:linear),Val(:interleaved))
    CUDA.synchronize()
    write(joinpath(dir,"confirmation_trials.csv"),"threads,blocks,pixels_per_thread,layout,assignment,trial,device_ms,wall_ms\n")
    open(joinpath(dir,"confirmation.csv"),"w") do io
        println(io,"threads,blocks,pixels_per_thread,layout,assignment,minimum_ms,median_ms,maximum_ms,mismatches")
        for (threads,blocks,ppt,layout,assignment) in configs
            f=()->(@cuda threads=threads blocks=blocks fastmath=false mandelbrot!(out,n,limit,Val(ppt),Val(layout),Val(assignment)))
            f();CUDA.synchronize();wrong=Int(sum(out .!= reference))
            wrong==0 || error("Confirmation mismatch")
            m=measure(f;trials=15)
            println(io,join((threads,blocks,ppt,layout,assignment,m.minimum_ms,m.median_ms,m.maximum_ms,wrong),','));flush(io)
            open(joinpath(dir,"confirmation_trials.csv"),"a") do raw
                for i in eachindex(m.device)
                    println(raw,join((threads,blocks,ppt,layout,assignment,i,m.device[i],m.wall[i]),','))
                end
            end
            println("Confirmed ",(threads,blocks,ppt,layout,assignment)," median_ms=",m.median_ms)
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && confirm(only(ARGS))
