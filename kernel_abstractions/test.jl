# Local scalar/coordinate checks: julia --project=. test.jl
# Actual GPU kernels: TEST_BACKEND=gpu julia --project=. test.jl
using Test, KernelAbstractions
include("gpu.jl")
@testset "Scalar recurrence and exact coordinate reuse" begin
    for n in (1,7,31,65), limit in (0,1,23,2000)
        expected = reference(n,limit)
        for y in 1:n, x in 1:n, early in (false,true)
            @test escape_count(coordinate_x(x,n),coordinate_y(y,n),limit,early) == expected[y,x]
        end
    end
    for n in (1024,8192,32768)
        for coordinate in (coordinate_x,coordinate_y)
            values = [coordinate(i,n) for i in 1:n]
            unique_values = unique(values)
            lookup = Dict(v=>Int32(i) for (i,v) in enumerate(unique_values))
            mapping = Int32[lookup[v] for v in values]
            @test unique_values[mapping] == values
        end
    end
end
if get(ENV,"TEST_BACKEND","scalar") == "gpu"
    @eval using CUDA
    CUDA.functional() || error("NVIDIA GPU required for TEST_BACKEND=gpu")
    CUDA.allowscalar(false)
    @testset "Actual KA GPU kernels, limits and partial tiles" begin
        for n in (1,31,65), limit in (0,1,23,2000)
            expected=reference(n,limit)
            xs=CuArray([coordinate_x(x,n) for x in 1:n])
            ys=CuArray([coordinate_y(y,n) for y in 1:n])
            out=CUDA.zeros(UInt32,n,n)
            for group in ((8,4),(32,8)), pixels in (1,2), shared in (false,true), early in (false,true)
                launch!(out,xs,ys,n,limit;group,pixels,shared,early)
                CUDA.synchronize()
                @test Array(out) == expected
            end
            CUDA.unsafe_free!(xs);CUDA.unsafe_free!(ys);CUDA.unsafe_free!(out)
        end
    end
else
    println("Scalar tests passed; actual GPU validation requires TEST_BACKEND=gpu or run.jl.")
end
