using Dates, Statistics, Printf, SHA, Random
include("gpu.jl")

function measure(f; trials=5)
    f(); CUDA.synchronize() # compilation and warmup excluded
    device=Float64[]; wall=Float64[]
    for _ in 1:trials
        CUDA.synchronize()
        started=time_ns()
        seconds=CUDA.@elapsed f()
        CUDA.synchronize()
        push!(device,seconds*1000)
        push!(wall,(time_ns()-started)/1e6)
    end
    (; minimum_ms=minimum(device), median_ms=median(device),
       maximum_ms=maximum(device), wall_median_ms=median(wall), device, wall)
end

function main()
    CUDA.functional() || error("An NVIDIA GPU allocation is required; no GPU timings were recorded.")
    CUDA.allowscalar(false)
    n=parse(Int,get(ENV,"SIZE","8192"))
    limit=parse(Int,get(ENV,"MAX_ITERS","2000"))
    trials=parse(Int,get(ENV,"TRIALS","5"))
    sizes= n == 8192 ? [1024,2048,4096,8192] : [n]
    stamp=Dates.format(now(),"yyyymmdd-HHMMSS")*"-"*string(getpid())
    dir=joinpath(@__DIR__,"results",stamp); mkpath(dir)
    open(joinpath(dir,"environment.txt"),"w") do io
        println(io,"Started: ",now(),"\nHost: ",gethostname(),"\nJulia: ",VERSION)
        println(io,"CPU: ",Sys.cpu_info()[1].model,"; allocation CPUs: ",get(ENV,"SLURM_CPUS_PER_TASK","unknown"))
        println(io,"Threads: ",Threads.nthreads(),"\nDefault size: ",n,"\nLimit: ",limit,"\nTrials: ",trials)
        CUDA.versioninfo(io)
        for file in filter(f->endswith(f,".jl") || endswith(f,".toml"),readdir(@__DIR__))
            path=joinpath(@__DIR__,file)
            isfile(path) && println(io,file," SHA256 ",bytes2hex(sha256(read(path))))
        end
    end
    println("Results directory: ",dir)
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    summary=open(joinpath(dir,"summary.csv"),"w")
    raw=open(joinpath(dir,"trials.csv"),"w")
    println(summary,"experiment,size,max_iters,threads,blocks,pixels_per_thread,layout,assignment,offset_elements,stride,minimum_ms,median_ms,maximum_ms,wall_median_ms,mpixels_per_s,effective_GB_per_s,mismatches")
    println(raw,"experiment,size,threads,blocks,pixels_per_thread,layout,assignment,offset_elements,stride,trial,device_ms,wall_ms")
    function record(f,experiment,size,threads,blocks,ppt,layout,assignment,offset,stride,mismatches; bytes=0)
        m=measure(f;trials)
        println(summary,join((experiment,size,limit,threads,blocks,ppt,layout,assignment,offset,stride,m.minimum_ms,m.median_ms,m.maximum_ms,m.wall_median_ms,size^2/m.median_ms/1000,bytes/m.median_ms/1e6,mismatches),',')); flush(summary)
        for i in eachindex(m.device)
            println(raw,join((experiment,size,threads,blocks,ppt,layout,assignment,offset,stride,i,m.device[i],m.wall[i]),','))
        end
        flush(raw)
        @printf("%-20s size=%d threads=%d blocks=%d ppt=%d %-10s median=%.4f ms mismatches=%d\n",experiment,size,threads,blocks,ppt,String(layout),m.median_ms,mismatches)
        flush(stdout)
        return m.median_ms
    end
    try
        # Compile/validate every mapping and ILP specialization against CPU.
        # Odd dimensions exercise partially filled blocks and tail pixels.
        for small in (31,65,128)
            expected=Vector{UInt32}(undef,small^2); cpu!(expected,small,limit)
            d=CUDA.zeros(UInt32,small^2)
            for ppt in (1,2,4,8), layout in (:linear,:transpose), assignment in (:interleaved,:chunked)
                CUDA.fill!(d,typemax(UInt32))
                @cuda threads=64 blocks=3 fastmath=false mandelbrot!(d,small,limit,Val(ppt),Val(layout),Val(assignment))
                CUDA.synchronize()
                @assert Array(d)==expected "CPU/GPU validation failed: size=$small ppt=$ppt layout=$layout assignment=$assignment"
            end
            xs=CuArray([coordinates(i,small)[1] for i in 0:small-1])
            ys=CuArray([coordinates(i*small,small)[2] for i in 0:small-1])
            for shared in (false,true)
                @cuda threads=64 blocks=3 shmem=512 fastmath=false from_global!(d,xs,ys,small,limit,Val(shared))
                @assert Array(d)==expected "Coordinate staging validation failed"
            end
            for width in (32,64), height in (2,4,8), shared in (false,true)
                @cuda threads=width*height blocks=3 fastmath=false coordinates_tiled!(d,xs,ys,small,limit,Val(width),Val(height),Val(shared))
                @assert Array(d)==expected "Reusable coordinate tile validation failed"
            end
            CUDA.unsafe_free!(d); CUDA.unsafe_free!(xs); CUDA.unsafe_free!(ys)
        end
        write(joinpath(dir,"correctness.txt"),"PASS: CPU/GPU exact agreement for sizes 31, 65, 128; all mappings, assignments, ILP values, coordinate staging modes.\n")
        for size in sizes
            output=CUDA.zeros(UInt32,size^2)
            reference=similar(output)
            # GPU reference also checked against the CPU above.
            @cuda threads=128 blocks=4*sm fastmath=false mandelbrot!(reference,size,limit,Val(1),Val(:linear),Val(:interleaved))
            CUDA.synchronize()
            best=(Inf,128,4*sm,1,:linear,:interleaved)
            configs=collect(Iterators.product((64,128,256,512),(1,2,4,8),(1,2,4,8),(:linear,:transpose),(:interleaved,:chunked)))
            shuffle!(MersenneTwister(191+size),vec(configs))
            for (threads,ratio,ppt,layout,assignment) in configs
                blocks=sm*ratio
                f=()->(@cuda threads=threads blocks=blocks fastmath=false mandelbrot!(output,size,limit,Val(ppt),Val(layout),Val(assignment)))
                CUDA.fill!(output,typemax(UInt32)); f(); CUDA.synchronize()
                wrong=Int(sum(output .!= reference))
                wrong==0 || error("Full-image mismatch ($wrong pixels)")
                ms=record(f,"mandelbrot",size,threads,blocks,ppt,layout,assignment,0,1,wrong)
                if ms<best[1]; best=(ms,threads,blocks,ppt,layout,assignment); end
            end
            # Single-warp controls isolate ILP and provide a full-GPU baseline.
            # These are intentionally slow at 8192²; retain every trial.
            for ppt in (1,2)
                f=()->(@cuda threads=32 blocks=1 fastmath=false mandelbrot!(output,size,limit,Val(ppt),Val(:linear),Val(:interleaved)))
                f(); CUDA.synchronize(); wrong=Int(sum(output .!= reference))
                wrong==0 || error("Single-warp mismatch")
                record(f,"single_warp",size,32,1,ppt,:linear,:interleaved,0,1,wrong)
            end
            # Hold the block on one SM and vary its available warps.
            for threads in (64,128,256,512,1024)
                f=()->(@cuda threads=threads blocks=1 fastmath=false mandelbrot!(output,size,limit,Val(1),Val(:linear),Val(:interleaved)))
                f();CUDA.synchronize();wrong=Int(sum(output .!= reference))
                wrong==0 || error("Single-SM mismatch")
                record(f,"single_sm",size,threads,1,1,:linear,:interleaved,0,1,wrong)
            end
            # Compare direct coordinate loads with global -> shared staging.
            # Include the coordinate-generation approach above as main baseline.
            xs=CuArray([coordinates(i,size)[1] for i in 0:size-1])
            ys=CuArray([coordinates(i*size,size)[2] for i in 0:size-1])
            for threads in (64,128,256,512), shared in (false,true)
                blocks=4*sm
                f=()->(@cuda threads=threads blocks=blocks shmem=(shared ? threads*8 : 0) fastmath=false from_global!(output,xs,ys,size,limit,Val(shared)))
                f(); CUDA.synchronize(); wrong=Int(sum(output .!= reference))
                wrong==0 || error("Full-image staging mismatch")
                record(f,shared ? "coordinates_shared" : "coordinates_global",size,threads,blocks,1,:linear,:interleaved,0,1,wrong)
            end
            for width in (32,64), height in (2,4,8), shared in (false,true)
                threads=width*height; blocks=4*sm
                f=()->(@cuda threads=threads blocks=blocks fastmath=false coordinates_tiled!(output,xs,ys,size,limit,Val(width),Val(height),Val(shared)))
                f(); CUDA.synchronize(); wrong=Int(sum(output .!= reference))
                wrong==0 || error("Reusable coordinate tile mismatch")
                record(f,shared ? "tile_shared" : "tile_global",size,threads,blocks,1,Symbol("tile_$(width)x$(height)"),:interleaved,0,1,wrong)
            end
            open(joinpath(dir,"best.txt"),"a") do io
                println(io,"size=$size best_observed=(median_ms,threads,blocks,pixels_per_thread,layout,assignment): ",best)
            end
            # Full-size CPU serial and threaded results, excluding allocation.
            if size==n
                host=Vector{UInt32}(undef,size^2)
                write(joinpath(dir,"cpu.csv"),"mode,size,threads,pixels_per_thread,minimum_ms,median_ms,maximum_ms,mismatches\n")
                write(joinpath(dir,"cpu_trials.csv"),"mode,size,threads,pixels_per_thread,trial,wall_ms\n")
                for threaded in (false,true), ppt in (1,2,4,8)
                    cpu_ilp!(host,min(size,128),limit,Val(ppt);threaded)
                    samples=Float64[]
                    for trial in 1:trials
                        ms=1000*(@elapsed cpu_ilp!(host,size,limit,Val(ppt);threaded))
                        push!(samples,ms)
                        open(joinpath(dir,"cpu_trials.csv"),"a") do rawcpu
                            println(rawcpu,join((threaded ? "threaded" : "serial",size,Threads.nthreads(),ppt,trial,ms),','))
                        end
                    end
                    wrong=count(host .!= Array(reference))
                    wrong==0 || error("Full-size CPU/GPU mismatch")
                    open(joinpath(dir,"cpu.csv"),"a") do io
                        println(io,join((threaded ? "threaded" : "serial",size,Threads.nthreads(),ppt,minimum(samples),median(samples),maximum(samples),wrong),','))
                    end
                end
                # Transfer is explicitly separate from compute-only measurements.
                transfer=Float64[]
                for _ in 1:trials
                    push!(transfer,1000*(@elapsed copyto!(host,output)))
                end
                write(joinpath(dir,"transfer.txt"),"Device-to-host preallocated output copy median_ms=$(median(transfer)); output_bytes=$(sizeof(UInt32)*size^2)\n")
            end
            CUDA.unsafe_free!(xs);CUDA.unsafe_free!(ys)
            CUDA.unsafe_free!(output);CUDA.unsafe_free!(reference)
        end
        # 16 MiB useful input footprint; stride and offset change physical reads.
        count_items=4*1024^2
        hostsrc=Float32.(mod.(0:32*count_items+32,1021))
        src=CuArray(hostsrc); dst=CUDA.zeros(Float32,count_items)
        write(joinpath(dir,"memory_alignment.txt"),"Input base modulo 128 bytes: $(UInt(pointer(src))%128)\nElement width: 4 bytes; tested byte offsets are 4 × offset_elements.\n")
        for stride in (1,2,4,8,32), offset in (0,1,2,4,8,16,31), threads in (128,256), shared in (false,true)
            blocks=4*sm
            f=()->(@cuda threads=threads blocks=blocks shmem=(shared ? threads*4 : 0) memory_probe!(dst,src,count_items,offset,stride,Val(shared),Val(1)))
            f(); CUDA.synchronize()
            expected=hostsrc[offset+1:stride:offset+1+(count_items-1)*stride]
            wrong=count(Array(dst) .!= expected)
            wrong==0 || error("Memory probe mismatch")
            record(f,shared ? "memory_shared" : "memory_direct",2048,threads,blocks,1,:linear,:interleaved,offset,stride,wrong;bytes=8*count_items)
        end
        write(joinpath(dir,"SWEEP_COMPLETE"),"Completed $(now())\n")
    catch err
        open(joinpath(dir,"FAILED.txt"),"w") do io
            showerror(io,err,catch_backtrace())
        end
        rethrow()
    finally
        close(summary);close(raw)
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
