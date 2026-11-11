# Run inside an NVIDIA allocation under Nsight Compute, after package setup:
# ncu --profile-from-start off --set full --export profile --force-overwrite \
#     julia --project=. profile.jl
include("gpu.jl")
function profile_selected()
    CUDA.functional() || error("NVIDIA GPU required")
    CUDA.allowscalar(false)
    sm=Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    count_items=4*1024^2
    src=CuArray(Float32.(mod.(0:32*count_items+32,1021)))
    dst=CUDA.zeros(Float32,count_items)
    kernels=Function[]
    for stride in (1,32), offset in (0,1,4,8,16), shared in (false,true)
        push!(kernels,let stride=stride,offset=offset,shared=shared
            ()->(@cuda threads=256 blocks=4*sm shmem=(shared ? 1024 : 0) memory_probe!(dst,src,count_items,offset,stride,Val(shared),Val(1)))
        end)
    end
    n=8192;limit=2000
    image=CUDA.zeros(UInt32,n*n)
    xs=CuArray([coordinates(i,n)[1] for i in 0:n-1])
    ys=CuArray([coordinates(i*n,n)[2] for i in 0:n-1])
    push!(kernels,()->(@cuda threads=512 blocks=2*sm fastmath=false mandelbrot!(image,n,limit,Val(2),Val(:linear),Val(:chunked))))
    push!(kernels,()->(@cuda threads=512 blocks=4*sm fastmath=false from_global!(image,xs,ys,n,limit,Val(false))))
    push!(kernels,()->(@cuda threads=512 blocks=4*sm shmem=4096 fastmath=false from_global!(image,xs,ys,n,limit,Val(true))))
    push!(kernels,()->(@cuda threads=512 blocks=4*sm fastmath=false coordinates_tiled!(image,xs,ys,n,limit,Val(64),Val(8),Val(false))))
    push!(kernels,()->(@cuda threads=512 blocks=4*sm fastmath=false coordinates_tiled!(image,xs,ys,n,limit,Val(64),Val(8),Val(true))))
    open(joinpath(@__DIR__,"profile_launch_order.csv"),"w") do io
        println(io,"launch_index_0based,experiment,stride,offset_elements,shared")
        index=0
        for stride in (1,32), offset in (0,1,4,8,16), shared in (false,true)
            println(io,join((index,"memory_probe",stride,offset,shared),','));index+=1
        end
        for (experiment,shared) in (("main_p2_chunked",false),("coordinates_global",false),("coordinates_shared",true),("tile_global_64x8",false),("tile_shared_64x8",true))
            println(io,join((index,experiment,1,0,shared),','));index+=1
        end
    end
    # Compile each selected specialization before entering the profile region.
    for f in kernels; f(); end
    CUDA.synchronize()
    CUDA.@profile external=true begin
        for f in kernels
            f(); CUDA.synchronize()
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && profile_selected()
