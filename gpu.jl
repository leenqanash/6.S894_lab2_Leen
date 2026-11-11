using CUDA
include("core.jl")
using .MandelbrotCore

# Immutable tuple helpers take old states as arguments. Avoid closures over
# variables reassigned by the kernel loop: Julia can box those captures, and
# device code cannot allocate boxed Julia objects.
@inline function group_coordinates(batch,t,nt,n,::Val{P},layout,::Val{ASSIGN}) where {P,ASSIGN}
    logical=ntuple(k -> ASSIGN==:interleaved ? batch+t+(k-1)*nt : batch+t*P+k-1,Val(P))
    ids=ntuple(k -> MandelbrotCore.mapping(logical[k],n,layout),Val(P))
    cx=ntuple(k -> coordinates(ids[k],n)[1],Val(P))
    cy=ntuple(k -> coordinates(ids[k],n)[2],Val(P))
    active=ntuple(k -> logical[k]<n*n,Val(P))
    return ids,cx,cy,active
end
@inline function advance_group(cx::NTuple{P,Float32},cy::NTuple{P,Float32},
    x2::NTuple{P,Float32},y2::NTuple{P,Float32},w::NTuple{P,Float32},
    counts::NTuple{P,UInt32},active::NTuple{P,Bool}) where P
    live=ntuple(k -> active[k] && x2[k]+y2[k]<=4f0,Val(P))
    x=ntuple(k -> x2[k]-y2[k]+cx[k],Val(P))
    y=ntuple(k -> w[k]-(x2[k]+y2[k])+cy[k],Val(P))
    nextx2=ntuple(k -> x[k]*x[k],Val(P))
    nexty2=ntuple(k -> y[k]*y[k],Val(P))
    nextw=ntuple(k -> (x[k]+y[k])*(x[k]+y[k]),Val(P))
    nextcounts=ntuple(k -> counts[k]+UInt32(live[k]),Val(P))
    return nextx2,nexty2,nextw,nextcounts,live
end

# P independent pixel states per CUDA thread, known at compile time.
# Interleaved keeps neighboring lanes on neighboring pixels; chunked spaces
# neighboring lanes by P elements.
function mandelbrot!(out,n,limit,::Val{P},layout,assignment) where P
    t=Int(threadIdx().x)-1
    nt=Int(blockDim().x)
    total=nt*Int(gridDim().x)
    batch=(Int(blockIdx().x)-1)*nt*P
    while batch<n*n
        ids,cx,cy,active=group_coordinates(batch,t,nt,n,Val(P),layout,assignment)
        valid=active
        x2=ntuple(_ -> 0f0,Val(P));y2=x2;w=x2
        counts=ntuple(_ -> UInt32(0),Val(P))
        for step in 1:limit
            x2,y2,w,counts,active=advance_group(cx,cy,x2,y2,w,counts,active)
            alive=false
            for k in 1:P
                alive |= active[k]
            end
            alive || break
        end
        for k in 1:P
            if valid[k]
                @inbounds out[ids[k]+1]=counts[k]
            end
        end
        batch+=total*P
    end
    return
end

# Two matched kernels read precomputed coordinates from global memory.
# Shared staging has no cross-thread reuse here: deliberately measure its cost.
function from_global!(out, xs, ys, n, limit, ::Val{SHARED}) where SHARED
    t = Int(threadIdx().x)
    nt = Int(blockDim().x)
    tile = (Int(blockIdx().x)-1)*nt
    stride = Int(gridDim().x)*nt
    if SHARED
        shared = CuDynamicSharedArray(Float32, 2*nt)
        while tile < n*n
            p = tile+t
            @inbounds shared[t] = p <= n*n ? xs[(p-1)%n+1] : 0f0
            @inbounds shared[nt+t] = p <= n*n ? ys[(p-1)÷n+1] : 0f0
            sync_threads() # Every thread, including tail threads, reaches barrier.
            if p <= n*n
                @inbounds out[p] = pixel(shared[t],shared[nt+t],limit)
            end
            sync_threads() # Prevent next tile overwriting another thread's data.
            tile += stride
        end
    else
        while tile < n*n
            p = tile+t
            if p <= n*n
                @inbounds out[p] = pixel(xs[(p-1)%n+1],ys[(p-1)÷n+1],limit)
            end
            tile += stride
        end
    end
    return
end

# Read-only input microbenchmark: same output stores for every configuration.
# offset is measured in Float32 elements; stride=1 is contiguous lane access.
# Shared and direct paths compute identical local reuse, without fake reuse
# across threads. This separates alignment/stride from fractal divergence.
function memory_probe!(dst, src, count, offset, stride, ::Val{SHARED}, ::Val{REUSE}) where {SHARED,REUSE}
    t = Int(threadIdx().x)
    nt = Int(blockDim().x)
    tile = (Int(blockIdx().x)-1)*nt
    step = Int(gridDim().x)*nt
    if SHARED
        shared = CuDynamicSharedArray(Float32,nt)
        while tile < count
            p=tile+t
            @inbounds shared[t] = p<=count ? src[offset+1+(p-1)*stride] : 0f0
            sync_threads()
            if p<=count
                @inbounds v=shared[t]
                acc=0f0
                for k in 1:REUSE
                    acc = muladd(acc,0.5f0,v)
                end
                @inbounds dst[p]=acc
            end
            sync_threads()
            tile+=step
        end
    else
        while tile<count
            p=tile+t
            if p<=count
                @inbounds v=src[offset+1+(p-1)*stride]
                acc=0f0
                for k in 1:REUSE
                    acc=muladd(acc,0.5f0,v)
                end
                @inbounds dst[p]=acc
            end
            tile+=step
        end
    end
    return
end

# A tile with actual coordinate reuse: W neighboring columns × H rows.
# Adjacent warp lanes follow x; shared x values are reused across H rows,
# and the row coordinate is broadcast to every pixel in that row.
function coordinates_tiled!(out,xs,ys,n,limit,::Val{W},::Val{H},::Val{SHARED}) where {W,H,SHARED}
    t=Int(threadIdx().x)-1
    lx=t%W; ly=t÷W
    tiles_x=cld(n,W); tiles_y=cld(n,H)
    tile=Int(blockIdx().x)-1
    shared=CuStaticSharedArray(Float32,W+H)
    while tile<tiles_x*tiles_y
        first_col=(tile%tiles_x)*W
        first_row=(tile÷tiles_x)*H
        col=first_col+lx; row=first_row+ly
        if SHARED
            if t<W
                @inbounds shared[t+1]=first_col+t<n ? xs[first_col+t+1] : 0f0
            end
            if t<H
                @inbounds shared[W+t+1]=first_row+t<n ? ys[first_row+t+1] : 0f0
            end
            sync_threads()
            if col<n && row<n
                @inbounds out[row*n+col+1]=pixel(shared[lx+1],shared[W+ly+1],limit)
            end
            sync_threads()
        else
            if col<n && row<n
                @inbounds out[row*n+col+1]=pixel(xs[col+1],ys[row+1],limit)
            end
        end
        tile+=Int(gridDim().x)
    end
    return
end
