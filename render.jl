# Simple entry point after the full experiment. Default: 8192 × 8192.
# Modes: computed, global, shared, tile_global, tile_shared.
# The launch defaults below are the best observed L40S candidates, not a
# universal optimum for other GPUs. SAVE_IMAGE=1 also writes a binary PGM.
include("run.jl")
function render(mode=isempty(ARGS) ? "global" : first(ARGS))
    mode in ("computed","global","shared","tile_global","tile_shared") || error("Unknown mode: $mode")
    CUDA.functional() || error("Run this entry point on an allocated NVIDIA GPU node.")
    n=parse(Int,get(ENV,"SIZE","8192"));limit=parse(Int,get(ENV,"MAX_ITERS","2000"))
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    out=CUDA.zeros(UInt32,n*n)
    xs=nothing;ys=nothing;coordinate_setup_ms=0.0
    if mode!="computed"
        coordinate_setup_ms=1000*(@elapsed begin
            xs=CuArray([coordinates(i,n)[1] for i in 0:n-1])
            ys=CuArray([coordinates(i*n,n)[2] for i in 0:n-1])
            CUDA.synchronize()
        end)
    end
    if mode=="computed"
        launch=()->(@cuda threads=512 blocks=2*sm fastmath=false mandelbrot!(out,n,limit,Val(2),Val(:linear),Val(:chunked)))
    elseif mode in ("global","shared")
        shared=mode=="shared"
        launch=()->(@cuda threads=512 blocks=4*sm shmem=(shared ? 4096 : 0) fastmath=false from_global!(out,xs,ys,n,limit,Val(shared)))
    else
        shared=mode=="tile_shared"
        launch=()->(@cuda threads=512 blocks=4*sm fastmath=false coordinates_tiled!(out,xs,ys,n,limit,Val(64),Val(8),Val(shared)))
    end
    measured=measure(launch;trials=parse(Int,get(ENV,"TRIALS","5")))
    reference=similar(out)
    @cuda threads=128 blocks=4*sm fastmath=false mandelbrot!(reference,n,limit,Val(1),Val(:linear),Val(:interleaved))
    CUDA.synchronize()
    wrong=Int(sum(out .!= reference));wrong==0 || error("Output mismatch: $wrong pixels")
    copy_ms=1000*(@elapsed host=Array(out))
    checksum=sum(UInt64,host)
    dir=joinpath(@__DIR__,"results");mkpath(dir)
    stamp=Dates.format(now(),"yyyymmdd-HHMMSS")
    path=joinpath(dir,"render-$stamp-$mode.txt")
    open(path,"w") do io
        println(io,"size=$n × $n; max_iters=$limit; mode=$mode; device=$(CUDA.name(CUDA.device()))")
        println(io,"kernel_minimum_ms=$(measured.minimum_ms); kernel_median_ms=$(measured.median_ms); kernel_maximum_ms=$(measured.maximum_ms)")
        println(io,"coordinate_setup_ms=$coordinate_setup_ms; output_copy_with_allocation_ms=$copy_ms; checksum=$checksum; mismatches=$wrong")
        println(io,"Kernel timing excludes compilation, allocations, coordinate setup and host transfer.")
    end
    print(read(path,String))
    if get(ENV,"SAVE_IMAGE","0")=="1"
        gray=Vector{UInt8}(undef,n*n)
        for i in eachindex(host)
            gray[i]=UInt8(clamp(round(Int,255*(1-Float64(host[i])/limit)),0,255))
        end
        imagepath=joinpath(dir,"mandelbrot-$stamp-$mode.pgm")
        open(imagepath,"w") do io
            print(io,"P5\n$n $n\n255\n");write(io,gray)
        end
        println("Image: ",imagepath)
    end
    println("Record: ",path)
end
abspath(PROGRAM_FILE)==(@__FILE__) && render()
