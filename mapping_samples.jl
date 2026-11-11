# Export concrete lane → pixel examples without needing a GPU.
include("core.jl")
using .MandelbrotCore
path=joinpath(@__DIR__,"results","pixel_thread_mapping.csv")
open(path,"w") do io
    println(io,"size,threads_per_block,pixels_per_thread,layout,assignment,block_1based,thread_1based,warp_1based,lane_1based,pixel_slot_1based,array_index_1based,row_0based,column_0based")
    n=8192
    for nt in (64,128,256,512), ppt in (1,2,4,8), layout in (:linear,:transpose), assignment in (:interleaved,:chunked), block in 0:1, thread in 0:31, k in 1:ppt
        tile=block*nt*ppt
        logical=assignment==:interleaved ? tile+thread+(k-1)*nt : tile+thread*ppt+k-1
        p=MandelbrotCore.mapping(logical,n,Val(layout))
        println(io,join((n,nt,ppt,layout,assignment,block+1,thread+1,thread÷32+1,thread%32+1,k,p+1,p÷n,p%n),','))
    end
end
println("Saved ",path)
