# Optional Nsight Compute entry point. Example:
# ncu --profile-from-start off --set full --export ka-profile \
#     julia --project=. profile.jl
# Read registers, local-memory spill traffic, shared-bank conflicts and barriers.
# Private arrays are not assumed to reside entirely in physical registers.
using CUDA
include("gpu.jl")
n=8192;limit=DEFAULT_ITERS
out=CUDA.zeros(UInt32,n,n)
xs=CuArray([coordinate_x(x,n) for x in 1:n])
ys=CuArray([coordinate_y(y,n) for y in 1:n])
configs=[(pixels=p,shared=s) for p in (1,2) for s in (false,true)]
for c in configs
    launch!(out,xs,ys,n,limit;pixels=c.pixels,shared=c.shared)
end
CUDA.synchronize()
CUDA.@profile external=true begin
    for c in configs
        println("PROFILE pixels=$(c.pixels) shared=$(c.shared)")
        launch!(out,xs,ys,n,limit;pixels=c.pixels,shared=c.shared)
        CUDA.synchronize()
    end
end
