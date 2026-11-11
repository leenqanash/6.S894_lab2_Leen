include("../run_retry.jl")
include("kernels.jl")
include("config.jl")
function run_native()
    CUDA.functional()||error("NVIDIA GPU required")
    CUDA.allowscalar(false)
    stamp=Dates.format(now(),"yyyymmdd-HHMMSS")*"-"*string(getpid())
    dir=joinpath(@__DIR__,"results",stamp);mkpath(dir)
    println("RETRY_RESULTS=",dir);flush(stdout)
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    all=native_cases();byid=Dict(c.id=>c for c in all)
    open(joinpath(dir,"environment.txt"),"w") do io
        println(io,"Started $(now()); host $(gethostname()); Julia $VERSION; SM count $sm; max_iters=2000; Float32; storage=CuMatrix[y,x], Julia column-major")
        CUDA.versioninfo(io)
        for path in (@__FILE__,joinpath(@__DIR__,"kernels.jl"),joinpath(@__DIR__,"..","..","gpu.jl"),joinpath(@__DIR__,"..","..","core.jl"))
            println(io,path," SHA256 ",bytes2hex(sha256(read(path))))
        end
    end
    # All specializations must pass odd-size/tail comparisons against CPU.
    valid=Set{String}();skipped=open(joinpath(dir,"skipped.txt"),"w")
    for n in (31,65,128)
        host=zeros(UInt32,n*n);cpu!(host,n,2000);host=vec(permutedims(reshape(host,n,n)))
        out=CUDA.zeros(UInt32,n,n)
        xs=CuArray([coordinates(p,n)[1] for p in 0:n-1])
        ys=CuArray([coordinates(p*n,n)[2] for p in 0:n-1])
        for c in all
            n!=31 && !(c.id in valid) && continue
            try
                CUDA.fill!(out,typemax(UInt32))
                f=native_launcher(c,out,xs,ys,n,2000,sm);f();CUDA.synchronize()
                wrong=count(vec(Array(out)).!=host)
                wrong==0||error("Mismatch: $(c.id), n=$n, wrong=$wrong")
                push!(valid,c.id)
            catch err
                if occursin("Mismatch",sprint(showerror,err));rethrow();end
                delete!(valid,c.id)
                println(skipped,c.id," n=",n," ",sprint(showerror,err));flush(skipped)
            end
        end
        CUDA.unsafe_free!(out);CUDA.unsafe_free!(xs);CUDA.unsafe_free!(ys)
    end
    close(skipped)
    isempty(valid)&&error("No valid kernels")
    println("Correctness: ",length(valid)," variants passed CPU comparisons at 31,65,128");flush(stdout)
    write(joinpath(dir,"correctness.txt"),"$(length(valid)) variants passed CPU comparisons at 31,65,128. Every screened full image is checked against the established GPU reference.\n")
    summary=open(joinpath(dir,"screen.csv"),"w");raw=open(joinpath(dir,"screen_trials.csv"),"w")
    pairs=open(joinpath(dir,"paired_trials.csv"),"w")
    globalcheck=open(joinpath(dir,"global_confirmation.csv"),"w")
    println(globalcheck,"size,case_id,trial,device_ms")
    println(summary,"size,case_id,family,threads,blocks,pixels,precomputed,shared,pad,column,cache,median_ms,minimum_ms,maximum_ms,mismatches")
    println(raw,"size,case_id,trial,device_ms")
    println(pairs,"size,comparison,shared_case,global_case,trial,first_variant,global_ms,shared_ms")
    sizes=parse.(Int,split(get(ENV,"RETRY_SIZES","1024,4096,8192,16384,32768"),','))
    rng=MersenneTwister(6191)
    try
        for n in sizes
            out=CUDA.zeros(UInt32,n,n);ref=similar(out)
            xs=CuArray([coordinates(p,n)[1] for p in 0:n-1])
            ys=CuArray([coordinates(p*n,n)[2] for p in 0:n-1])
            reference=byid["native_dense_t512_b4_p1_pre1_col1"];native_launcher(reference,ref,xs,ys,n,2000,sm)();CUDA.synchronize()
            # Independent CPU spot checks in the required [y,x] matrix layout.
            chosen=unique(vcat([0,n*n-1,n-1,n*(n-1)],rand(rng,0:n*n-1,256)))
            didx=CuArray(chosen.+1)
            got=Array(ref[didx])
            expected=UInt32[pixel(coordinates((q%n)*n+q÷n,n)...,2000) for q in chosen]
            got==expected||error("Native matrix reference failed CPU spot check")
            CUDA.unsafe_free!(didx)
            measured=Dict{String,Float64}()
            order=shuffle(rng,filter(c->c.id in valid,all))
            for c in order
                f=native_launcher(c,out,xs,ys,n,2000,sm)
                f();CUDA.synchronize()
                wrong=Int(sum(out .!= ref));wrong==0||error("Full-image mismatch $(c.id) size=$n")
                f();CUDA.synchronize()
                times=[timed(f) for _ in 1:3]
                med=median(times);measured[c.id]=med
                println(summary,join((n,c.id,c.family,c.threads,c.ratio*sm,c.pixels,c.precomputed,isshared(c),c.pad,c.column,c.cache,med,minimum(times),maximum(times),wrong),','));flush(summary)
                for (trial,ms) in enumerate(times);println(raw,join((n,c.id,trial,ms),','));end;flush(raw)
            end
            globals=filter(c->!isshared(c)&&haskey(measured,c.id),all)
            shareds=filter(c->isshared(c)&&haskey(measured,c.id)&&haskey(measured,native_matched(c)),all)
            sort!(globals;by=c->measured[c.id]);sort!(shareds;by=c->measured[c.id])
            println("SCREEN n=$n best global $(globals[1].id) $(measured[globals[1].id]) ms; best shared $(shareds[1].id) $(measured[shareds[1].id]) ms");flush(stdout)
            # Confirm the top three global candidates before choosing the
            # independent paired control; do not pick an artificially weak one.
            tops=shuffle(rng,globals[1:min(3,length(globals))])
            checkmed=Dict{String,Float64}()
            for gcase in tops
                gf=native_launcher(gcase,out,xs,ys,n,2000,sm)
                gf();CUDA.synchronize()
                samples=[timed(gf) for _ in 1:10]
                checkmed[gcase.id]=median(samples)
                for (trial,ms) in enumerate(samples)
                    println(globalcheck,join((n,gcase.id,trial,ms),','))
                end
                flush(globalcheck)
            end
            strong=tops[argmin([checkmed[c.id] for c in tops])]
            # Fresh data, independently randomized, comparing the leading shared
            # candidates both with their matched path and the strongest global.
            selected=shareds[1:min(2,length(shareds))]
            relative=shareds[argmax([measured[native_matched(c)]/measured[c.id] for c in shareds])]
            any(c->c.id==relative.id,selected)||push!(selected,relative)
            for c in selected
                for (label,base) in (("matched",byid[native_matched(c)]),("best_global",strong))
                    s=native_launcher(c,out,xs,ys,n,2000,sm);g=native_launcher(base,out,xs,ys,n,2000,sm)
                    for _ in 1:5;g();s();end;CUDA.synchronize()
                    gt=Float64[];st=Float64[]
                    for trial in 1:30
                        globalfirst=rand(rng,Bool)
                        if globalfirst;gm=timed(g);ss=timed(s);else;ss=timed(s);gm=timed(g);end
                        push!(gt,gm);push!(st,ss)
                        println(pairs,join((n,label,c.id,base.id,trial,globalfirst ? "global" : "shared",gm,ss),','));flush(pairs)
                    end
                    println("PAIRED n=$n $label shared=$(c.id) global=$(base.id) speedup=$(median(gt)/median(st))");flush(stdout)
                end
            end
            CUDA.unsafe_free!(out);CUDA.unsafe_free!(ref);CUDA.unsafe_free!(xs);CUDA.unsafe_free!(ys)
            CUDA.reclaim();GC.gc()
        end
        write(joinpath(dir,"COMPLETE"),"Completed $(now())\n")
    catch err
        write(joinpath(dir,"FAILED.txt"),sprint(showerror,err,catch_backtrace()));rethrow()
    finally
        close(summary);close(raw);close(pairs);close(globalcheck)
    end
end
abspath(PROGRAM_FILE)==(@__FILE__)&&run_native()
