using Test, Dates, Statistics
include("core.jl")
using .MandelbrotCore
@testset "Mandelbrot arithmetic and coverage" begin
    @test pixel(0f0,0f0,2000)==2000
    @test pixel(3f0,0f0,2000)==1
    @test coordinates(0,8192)==(MandelbrotCore.WX,MandelbrotCore.WY)
    for n in (1,7,31,65,128), ppt in (1,2,4,8), nt in (32,64,128), blocks in (1,3), layout in (:linear,:transpose), assignment in (:interleaved,:chunked)
        seen=zeros(Int,n*n)
        for b in 0:blocks-1
            batch=b*nt*ppt
            while batch<n*n
                for t in 0:nt-1, k in 1:ppt
                    q=assignment==:interleaved ? batch+t+(k-1)*nt : batch+t*ppt+k-1
                    if q<n*n
                        p=MandelbrotCore.mapping(q,n,Val(layout))
                        seen[p+1]+=1
                    end
                end
                batch+=blocks*nt*ppt
            end
        end
        @test all(==(1),seen)
    end
    for n in (31,65,128)
        a=Vector{UInt32}(undef,n*n); b=similar(a)
        cpu!(a,n,2000);cpu!(b,n,2000;threaded=true)
        @test a==b
    end
    for n in (7,31,65,128), p in (1,2,4,8), threaded in (false,true)
        a=Vector{UInt32}(undef,n*n); b=similar(a)
        cpu!(a,n,2000); cpu_ilp!(b,n,2000,Val(p);threaded)
        @test a==b
    end
    for n in (1,7,31,65,128), width in (32,64), height in (2,4,8), blocks in (1,3)
        seen=zeros(Int,n*n)
        tiles_x=cld(n,width); tiles_y=cld(n,height)
        for block in 0:blocks-1
            tile=block
            while tile<tiles_x*tiles_y
                for t in 0:width*height-1
                    col=(tile%tiles_x)*width+t%width
                    row=(tile÷tiles_x)*height+t÷width
                    if col<n && row<n
                        seen[row*n+col+1]+=1
                    end
                end
                tile+=blocks
            end
        end
        @test all(==(1),seen)
    end
    for file in ("core.jl","gpu.jl","run.jl","report.jl")
        parsed=Meta.parseall(read(joinpath(@__DIR__,file),String))
        @test !any(x -> x isa Expr && x.head in (:error,:incomplete),parsed.args)
    end
end
get(ENV,"TESTS_ONLY","0")=="1" && exit()
n=parse(Int,get(ENV,"SIZE","8192")); trials=parse(Int,get(ENV,"TRIALS","3"))
limit=2000
out=Vector{UInt32}(undef,n*n)
cpu!(out,min(n,128),limit);cpu!(out,min(n,128),limit;threaded=true)
path=joinpath(@__DIR__,"results","local_cpu_"*Dates.format(now(),"yyyymmdd-HHMMSS")*".csv")
open(path,"w") do io
    println(io,"host,julia,threads,size,max_iters,mode,trial,wall_ms,checksum")
    for threaded in (false,true)
        for trial in 1:trials
            ms=1000*(@elapsed cpu!(out,n,limit;threaded))
            println(io,join((gethostname(),VERSION,Threads.nthreads(),n,limit,threaded ? "threaded" : "serial",trial,ms,sum(UInt64,out)),','));flush(io)
            println("Local CPU size=$n mode=$(threaded ? "threaded" : "serial") trial=$trial wall_ms=$ms")
        end
    end
end
println("Saved: ",path)
