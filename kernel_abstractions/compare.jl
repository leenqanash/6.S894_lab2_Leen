# Independent randomized comparisons, not reused tuning-screen samples.
# julia --project=. compare.jl RESULTS_DIRECTORY
include("run.jl")
module OriginalCUDA
    include("../shared_retry/native/run_native.jl")
end
@inline function escape_while(cx,cy,limit)
    x2=0f0;y2=0f0;w=0f0;count=UInt32(0)
    while x2+y2<=4f0 && count<limit
        real=x2-y2+cx;imag=w-(x2+y2)+cy
        x2=real*real;y2=imag*imag;z=real+imag;w=z*z;count+=UInt32(1)
    end
    count
end
# Same scalar state, mapping and launch as mandelbrot_single!, isolating syntax.
@kernel unsafe_indices=true function while_control!(out,@Const(xs),@Const(ys),n,limit)
    ly=@index(Local,Cartesian)[1];lx=@index(Local,Cartesian)[2]
    gy=@index(Group,Cartesian)[1];gx=@index(Group,Cartesian)[2]
    @uniform TY=@groupsize()[1]
    @uniform TX=@groupsize()[2]
    y=(gy-1)*TY+ly;x=(gx-1)*TX+lx
    if x<=n&&y<=n;@inbounds out[y,x]=escape_while(xs[x],ys[y],limit);end
end
function paired!(io,n,label,base,new,ref,out,rng)
    for f in (base,new)
        f();CUDA.synchronize()
        Int(sum(out .!= ref))==0||error("Comparison mismatch $label N=$n")
        samples(f,3)
    end
    bt=Float64[];nt=Float64[]
    for trial in 1:30
        first=rand(rng,Bool)
        if first;b=elapsed_ms(base);t=elapsed_ms(new);else;t=elapsed_ms(new);b=elapsed_ms(base);end
        push!(bt,b);push!(nt,t)
        println(io,join((n,label,trial,first,b,t),','));flush(io)
    end
    println("CONFIRM N=$n $label baseline=$(median(bt)) new=$(median(nt)) speedup=$(median(bt)/median(nt))");flush(stdout)
end
function compare(dir)
    CUDA.allowscalar(false);rng=MersenneTwister(981717)
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    legacy=Dict(c.id=>c for c in OriginalCUDA.native_cases())
    open(joinpath(dir,"comparison.csv"),"w") do io
        println(io,"size,comparison,trial,baseline_first,baseline_ms,new_ms")
        for n in (1024,8192)
            data,_=prepare(n);out=CUDA.zeros(UInt32,n,n);ref=similar(out)
            plain=Case((8,32),1,false,true,false)
            reused=Case(n==1024 ? (32,8) : (32,4),1,false,true,true)
            new=action(plain,out,data,n,2000)
            action(plain,ref,data,n,2000)();CUDA.synchronize()
            oldid=n==1024 ? "native_dense_t512_b8_p1_pre1_col1" : "native_tile_t512_b4_p1_col0_s0_c0"
            old=OriginalCUDA.native_launcher(legacy[oldid],out,data.xs,data.ys,n,2000,sm)
            paired!(io,n,"original_global_to_KA",old,new,ref,out,rng)
            paired!(io,n,"plain_to_exact_reuse_including_expand",new,action(reused,out,data,n,2000),ref,out,rng)
            group=plain.group
            wh=()->while_control!(get_backend(out),group)(out,data.xs,data.ys,n,2000;ndrange=(cld(n,group[1])*group[1],cld(n,group[2])*group[2]))
            paired!(io,n,"while_to_bounded_for_same_mapping",wh,new,ref,out,rng)
            two=action(Case(group,2,false,true,false),out,data,n,2000)
            paired!(io,n,"one_to_two_pixels_same_group",new,two,ref,out,rng)
            release(data);CUDA.unsafe_free!(out);CUDA.unsafe_free!(ref);CUDA.reclaim()
        end
    end
    write(joinpath(dir,"COMPARISONS_COMPLETE"),"Completed $(now())\n")
end
abspath(PROGRAM_FILE)==(@__FILE__)&&compare(only(ARGS))
