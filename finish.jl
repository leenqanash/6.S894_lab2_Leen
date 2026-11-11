# Resume a completed GPU sweep after replacing the allocating CPU ILP path.
# Existing GPU measurements are retained. CPU and memory phases use fresh
# references, exact checks, new raw trial files, and final source hashes.
include("run.jl")
include("report.jl")
include("confirm.jl")
function finish(dir)
    rows=readrows(joinpath(dir,"summary.csv"))
    @assert count(r->r["experiment"]=="mandelbrot" && r["size"]=="8192",rows)==256 "GPU sweep is incomplete"
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    limit=2000;n=8192;trials=5
    open(joinpath(dir,"completion_provenance.txt"),"w") do io
        println(io,"The GPU sweep was retained. Its CPU phase was paused before any completed CPU timing because tuple captures caused allocations. CPU ILP now uses immutable typed helpers.")
        println(io,"Original core is preserved as core_before_cpu_allocation_fix.jl. Coordinates, scalar reference recurrence and GPU kernels are unchanged by this CPU-only fix.")
        for file in filter(f->endswith(f,".jl")||endswith(f,".toml"),readdir(@__DIR__))
            println(io,file," SHA256 ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
        end
    end
    summary=open(joinpath(dir,"summary.csv"),"a");raw=open(joinpath(dir,"trials.csv"),"a")
    try
        if !any(r->startswith(r["experiment"],"memory_"),rows)
            count_items=4*1024^2
            hostsrc=Float32.(mod.(0:32*count_items+32,1021))
            src=CuArray(hostsrc);dst=CUDA.zeros(Float32,count_items)
            write(joinpath(dir,"memory_alignment.txt"),"Input base modulo 128 bytes: $(UInt(pointer(src))%128)\nElement width: 4 bytes.\n")
            for stride in (1,2,4,8,32),offset in (0,1,2,4,8,16,31),threads in (128,256),shared in (false,true)
                blocks=4*sm
                f=()->(@cuda threads=threads blocks=blocks shmem=(shared ? threads*4 : 0) memory_probe!(dst,src,count_items,offset,stride,Val(shared),Val(1)))
                f();CUDA.synchronize()
                expected=hostsrc[offset+1:stride:offset+1+(count_items-1)*stride]
                wrong=count(Array(dst) .!= expected)
                wrong==0 || error("Memory probe mismatch")
                m=measure(f;trials)
                mode=shared ? "memory_shared" : "memory_direct"
                println(summary,join((mode,2048,limit,threads,blocks,1,:linear,:interleaved,offset,stride,m.minimum_ms,m.median_ms,m.maximum_ms,m.wall_median_ms,count_items/m.median_ms/1000,8*count_items/m.median_ms/1e6,wrong),','));flush(summary)
                for i in eachindex(m.device)
                    println(raw,join((mode,2048,threads,blocks,1,:linear,:interleaved,offset,stride,i,m.device[i],m.wall[i]),','))
                end
                flush(raw)
            end
            CUDA.unsafe_free!(src);CUDA.unsafe_free!(dst)
        end
    finally
        close(summary);close(raw)
    end
    println("Memory probe sweep completed");flush(stdout)
    isfile(joinpath(dir,"confirmation.csv")) || confirm(dir)
    report(dir)
    reference=CUDA.zeros(UInt32,n*n)
    @cuda threads=128 blocks=4*sm fastmath=false mandelbrot!(reference,n,limit,Val(1),Val(:linear),Val(:interleaved))
    CUDA.synchronize();expected=Array(reference)
    host=similar(expected)
    cpupath=joinpath(dir,"cpu.csv")
    isfile(cpupath) && cp(cpupath,joinpath(dir,"cpu_before_allocation_fix.csv");force=true)
    open(cpupath,"w") do io
        println(io,"mode,size,threads,pixels_per_thread,minimum_ms,median_ms,maximum_ms,mismatches")
        open(joinpath(dir,"cpu_trials.csv"),"w") do rawcpu
            println(rawcpu,"mode,size,threads,pixels_per_thread,trial,wall_ms")
            for threaded in (false,true),ppt in (1,2,4,8)
                cpu_ilp!(host,128,limit,Val(ppt);threaded)
                # Verify no inner-state allocation for the serial specialization.
                allocated_bytes=@allocated cpu_ilp!(host,128,limit,Val(ppt))
                open(joinpath(dir,"cpu_allocation_checks.txt"),"a") do allocation_log
                    println(allocation_log,"pixels_per_thread=$ppt serial_call_allocated_bytes=$allocated_bytes (128², warmed)")
                end
                # Dynamic Val construction at this orchestration call site can
                # allocate a small object; reject pixel-loop allocation growth.
                allocated_bytes<65536 || error("CPU pixel-loop allocation remains excessive: $allocated_bytes bytes")
                samples=Float64[]
                for trial in 1:trials
                    ms=1000*(@elapsed cpu_ilp!(host,n,limit,Val(ppt);threaded))
                    push!(samples,ms)
                    println(rawcpu,join((threaded ? "threaded" : "serial",n,Threads.nthreads(),ppt,trial,ms),','));flush(rawcpu)
                    println("CPU ",(threaded,ppt,trial)," wall_ms=",ms);flush(stdout)
                end
                wrong=count(host .!= expected)
                wrong==0 || error("Full-size CPU/GPU mismatch")
                println(io,join((threaded ? "threaded" : "serial",n,Threads.nthreads(),ppt,minimum(samples),median(samples),maximum(samples),wrong),','));flush(io)
            end
        end
    end
    transfer=Float64[]
    for _ in 1:trials
        push!(transfer,1000*(@elapsed copyto!(host,reference)))
    end
    write(joinpath(dir,"transfer.txt"),"Preallocated device-to-host copy median_ms=$(median(transfer)); bytes=$(4*n*n)\n")
    write(joinpath(dir,"COMPLETE"),"All phases completed $(now()) by finish.jl\n")
    report(dir)
    println("FINISHED_RESULTS_DIRECTORY=",dir)
end
abspath(PROGRAM_FILE)==(@__FILE__) && finish(only(ARGS))
