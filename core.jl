module MandelbrotCore
export pixel, cpu!, cpu_ilp!, coordinates, mapping, DEFAULT_SIZE, DEFAULT_ITERS
const DEFAULT_SIZE = 8192
const DEFAULT_ITERS = 2000
# Match Lab 2's Float32 window, including the original constant rounding.
const ZOOM = Float32(1.0 / 10000.0)
const WX = Float32(-0.743643887 - 0.5 * Float64(ZOOM))
const WY = Float32(0.131825904 - 0.5 * Float64(ZOOM))
@inline function coordinates(p, n)
    col = p % n
    row = p ÷ n
    return (Float32(col) / Float32(n)) * ZOOM + WX,
           (Float32(row) / Float32(n)) * ZOOM + WY
end
@inline function pixel(cx::Float32, cy::Float32, limit)
    x2 = 0f0; y2 = 0f0; w = 0f0; count = UInt32(0)
    while x2 + y2 <= 4f0 && count < limit
        x = x2 - y2 + cx
        y = w - (x2 + y2) + cy
        x2 = x * x; y2 = y * y
        z = x + y; w = z * z
        count += UInt32(1)
    end
    return count
end
# Flat row-major image: p is zero-based, Julia array indices are one-based.
# This avoids confusing Julia's column-major matrices with the original image.
@inline mapping(p, n, ::Val{:linear}) = p
@inline mapping(p, n, ::Val{:transpose}) = (p % n) * n + p ÷ n
function cpu!(out, n, limit; threaded=false)
    if threaded
        Threads.@threads :static for row in 0:n-1
            for col in 0:n-1
                p = row*n + col
                out[p+1] = pixel(coordinates(p,n)..., limit)
            end
        end
    else
        for p in 0:n*n-1
            out[p+1] = pixel(coordinates(p,n)..., limit)
        end
    end
    out
end
# CPU ILP keeps P independent pixels in progress. Julia may additionally
# vectorize tuple arithmetic; SIMD instruction selection is compiler-dependent.
@inline function cpu_group_coordinates(base,n,stop,::Val{P}) where P
    ids=ntuple(k->base+k-1,Val(P))
    cx=ntuple(k->coordinates(ids[k],n)[1],Val(P))
    cy=ntuple(k->coordinates(ids[k],n)[2],Val(P))
    live=ntuple(k->ids[k]<stop,Val(P))
    return ids,cx,cy,live
end
@inline function cpu_group_step(cx::NTuple{P,Float32},cy::NTuple{P,Float32},
    x2::NTuple{P,Float32},y2::NTuple{P,Float32},w::NTuple{P,Float32},
    counts::NTuple{P,UInt32},active::NTuple{P,Bool}) where P
    live=ntuple(k->active[k] && x2[k]+y2[k]<=4f0,Val(P))
    x=ntuple(k->x2[k]-y2[k]+cx[k],Val(P))
    y=ntuple(k->w[k]-(x2[k]+y2[k])+cy[k],Val(P))
    nextx=ntuple(k->x[k]*x[k],Val(P))
    nexty=ntuple(k->y[k]*y[k],Val(P))
    nextw=ntuple(k->(x[k]+y[k])*(x[k]+y[k]),Val(P))
    nextcount=ntuple(k->counts[k]+UInt32(live[k]),Val(P))
    return nextx,nexty,nextw,nextcount,live
end
function cpu_ilp_row!(out,n,limit,row,::Val{P}) where P
    for base in row*n:P:(row+1)*n-1
        ids,cx,cy,active=cpu_group_coordinates(base,n,(row+1)*n,Val(P))
        x2=ntuple(_->0f0,Val(P));y2=x2;w=x2
        counts=ntuple(_->UInt32(0),Val(P))
        for step in 1:limit
            x2,y2,w,counts,active=cpu_group_step(cx,cy,x2,y2,w,counts,active)
            alive=false
            for k in 1:P; alive |= active[k]; end
            alive || break
        end
        for k in 1:P
            ids[k]<(row+1)*n && (out[ids[k]+1]=counts[k])
        end
    end
    return
end
function cpu_ilp!(out,n,limit,ppt::Val;threaded=false)
    if threaded
        Threads.@threads :static for row in 0:n-1
            cpu_ilp_row!(out,n,limit,row,ppt)
        end
    else
        for row in 0:n-1
            cpu_ilp_row!(out,n,limit,row,ppt)
        end
    end
    out
end

end
