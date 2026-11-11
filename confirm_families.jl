isdefined(@__MODULE__, :measure) || include("run.jl")
isdefined(@__MODULE__, :readrows) || include("report.jl")
using Random
function confirm_families(dir)
    rows=readrows(joinpath(dir,"summary.csv"))
    selected=Dict{String,String}[]
    # Fresh tests of the fastest member of each matched memory family.
    for mode in ("coordinates_global","coordinates_shared","tile_global","tile_shared")
        group=filter(r->r["experiment"]==mode && r["size"]=="8192",rows)
        sort!(group;by=r->parse(Float64,r["median_ms"]))
        push!(selected,Dict(first(group)))
    end
    shuffle!(MersenneTwister(319),selected)
    n=8192;limit=2000
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    out=CUDA.zeros(UInt32,n*n);reference=similar(out)
    @cuda threads=128 blocks=4*sm fastmath=false mandelbrot!(reference,n,limit,Val(1),Val(:linear),Val(:interleaved))
    CUDA.synchronize()
    xs=CuArray([coordinates(i,n)[1] for i in 0:n-1])
    ys=CuArray([coordinates(i*n,n)[2] for i in 0:n-1])
    open(joinpath(dir,"family_confirmation.csv"),"w") do io
        println(io,"experiment,threads,blocks,layout,minimum_ms,median_ms,maximum_ms,mismatches")
        open(joinpath(dir,"family_confirmation_trials.csv"),"w") do raw
            println(raw,"experiment,threads,blocks,layout,trial,device_ms,wall_ms")
            for r in selected
                threads=parse(Int,r["threads"]);blocks=parse(Int,r["blocks"])
                mode=r["experiment"];shared=endswith(mode,"shared")
                if startswith(mode,"tile_")
                    shape=split(replace(r["layout"],"tile_"=>""),'x')
                    width,height=parse.(Int,shape)
                    f=()->(@cuda threads=threads blocks=blocks fastmath=false coordinates_tiled!(out,xs,ys,n,limit,Val(width),Val(height),Val(shared)))
                else
                    f=()->(@cuda threads=threads blocks=blocks shmem=(shared ? threads*8 : 0) fastmath=false from_global!(out,xs,ys,n,limit,Val(shared)))
                end
                f();CUDA.synchronize();wrong=Int(sum(out .!= reference))
                wrong==0 || error("Family confirmation mismatch")
                m=measure(f;trials=15)
                println(io,join((mode,threads,blocks,r["layout"],m.minimum_ms,m.median_ms,m.maximum_ms,wrong),','));flush(io)
                for i in eachindex(m.device)
                    println(raw,join((mode,threads,blocks,r["layout"],i,m.device[i],m.wall[i]),','))
                end
                flush(raw)
                println("Confirmed family ",mode," ",r["layout"]," median_ms=",m.median_ms)
            end
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && confirm_families(only(ARGS))
