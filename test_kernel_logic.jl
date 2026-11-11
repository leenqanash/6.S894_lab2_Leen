# Execute the actual non-shared kernel source with CPU index stubs. This checks
# arithmetic and grid-stride assignments; it does NOT validate GPU compilation,
# shared-memory barriers, hardware races, register pressure or GPU performance.
using Test
module KernelLogic
const thread_number=Ref(1)
const block_number=Ref(1)
const threads_per_block=Ref(64)
const block_count=Ref(3)
threadIdx()=(x=thread_number[],)
blockIdx()=(x=block_number[],)
blockDim()=(x=threads_per_block[],)
gridDim()=(x=block_count[],)
CuStaticSharedArray(T,n)=zeros(T,n) # only used by the direct-global tile branch
source=replace(read(joinpath(@__DIR__,"gpu.jl"),String),"using CUDA\n"=>"")
Core.eval(@__MODULE__,Meta.parseall(source))
end
using .KernelLogic.MandelbrotCore
@testset "Actual kernel arithmetic on CPU indices (not GPU validation)" begin
    for n in (7,31,65), ppt in (1,2,4,8), layout in (:linear,:transpose), assignment in (:interleaved,:chunked)
        expected=zeros(UInt32,n*n);cpu!(expected,n,2000)
        out=fill(typemax(UInt32),n*n)
        KernelLogic.threads_per_block[]=64
        for block in 1:3, thread in 1:64
            KernelLogic.block_number[]=block;KernelLogic.thread_number[]=thread
            KernelLogic.mandelbrot!(out,n,2000,Val(ppt),Val(layout),Val(assignment))
        end
        @test out==expected
    end
    for n in (7,31,65), width in (32,64), height in (2,4,8)
        expected=zeros(UInt32,n*n);cpu!(expected,n,2000)
        xs=[coordinates(i,n)[1] for i in 0:n-1]
        ys=[coordinates(i*n,n)[2] for i in 0:n-1]
        out=fill(typemax(UInt32),n*n)
        KernelLogic.threads_per_block[]=width*height
        for block in 1:3, thread in 1:width*height
            KernelLogic.block_number[]=block;KernelLogic.thread_number[]=thread
            KernelLogic.coordinates_tiled!(out,xs,ys,n,2000,Val(width),Val(height),Val(false))
        end
        @test out==expected
    end
end
