# Backend-independent kernels. CUDA is used only by the NVIDIA host runner.
using KernelAbstractions
include("core.jl")
using .MandelbrotCore

# Scalars are private per thread and eligible for registers. A for-loop alone
# does not eliminate divergence: escaped pixels still need masking/early exit.
@inline function escape_count(cx, cy, limit, early)
    x2 = 0f0; y2 = 0f0; w = 0f0; count = UInt32(0)
    for iteration in 1:limit
        live = x2 + y2 <= 4f0
        if live
            total = x2 + y2                # reused in both recurrence terms
            real = x2 - y2 + cx
            imag = w - total + cy
            x2 = real * real; y2 = imag * imag
            z = real + imag; w = z * z
            count += UInt32(1)
        elseif early
            break
        end
    end
    count
end

# One thread/pixel. First workgroup dimension varies y, the contiguous output
# dimension. Both versions use identical arithmetic, mapping and workgroup size.
@kernel unsafe_indices=true function mandelbrot_single!(out, @Const(xs), @Const(ys), nx, ny, limit,
                                                        ::Val{SHARED}, ::Val{EARLY}) where {SHARED,EARLY}
    ly = @index(Local, Cartesian)[1]; lx = @index(Local, Cartesian)[2]
    gy = @index(Group, Cartesian)[1]; gx = @index(Group, Cartesian)[2]
    @uniform TY = @groupsize()[1]
    @uniform TX = @groupsize()[2]
    y = (gy-1)*TY + ly; x = (gx-1)*TX + lx
    if SHARED
        sx = @localmem Float32 (TX,)
        sy = @localmem Float32 (TY,)
        # Each coordinate is fetched once per tile; every thread reaches barrier.
        if ly == 1; @inbounds sx[lx] = x <= nx ? xs[x] : 0f0; end
        if lx == 1; @inbounds sy[ly] = y <= ny ? ys[y] : 0f0; end
        @synchronize
        if y <= ny && x <= nx
            @inbounds out[y,x] = escape_count(sx[lx],sy[ly],limit,EARLY)
        end
    else
        if y <= ny && x <= nx
            @inbounds out[y,x] = escape_count(xs[x],ys[y],limit,EARLY)
        end
    end
end

# Two independent pixel chains, explicitly allocated once in @private arrays.
# Their scalar operations are interleaved within each iteration. Slots differ
# by TY rows, so consecutive lanes still access consecutive y coordinates.
# @private expresses scope, not a promise of physical registers: profile spills.
@kernel unsafe_indices=true function mandelbrot_ilp2!(out, @Const(xs), @Const(ys), nx, ny, limit,
                                                     ::Val{SHARED}, ::Val{EARLY}) where {SHARED,EARLY}
    ly = @index(Local, Cartesian)[1]; lx = @index(Local, Cartesian)[2]
    gy = @index(Group, Cartesian)[1]; gx = @index(Group, Cartesian)[2]
    @uniform TY = @groupsize()[1]
    @uniform TX = @groupsize()[2]
    y = (gy-1)*(2*TY) + ly; x = (gx-1)*TX + lx
    state = @private Float32 (2,5)       # x², y², (x+y)², cx, cy columns
    counts = @private UInt32 (2,)
    active = @private Bool (2,)
    if SHARED
        sx = @localmem Float32 (TX,)
        sy = @localmem Float32 (2*TY,)
        if ly == 1; @inbounds sx[lx] = x <= nx ? xs[x] : 0f0; end
        if lx == 1
            for slot in 1:2
                row = y + (slot-1)*TY
                @inbounds sy[ly+(slot-1)*TY] = row <= ny ? ys[row] : 0f0
            end
        end
        @synchronize
    end
    for slot in 1:2
        row = y + (slot-1)*TY
        valid = row <= ny && x <= nx
        @inbounds begin
            state[slot,1] = 0f0; state[slot,2] = 0f0; state[slot,3] = 0f0
            state[slot,4] = valid ? (SHARED ? sx[lx] : xs[x]) : 0f0
            state[slot,5] = valid ? (SHARED ? sy[ly+(slot-1)*TY] : ys[row]) : 0f0
            counts[slot] = 0; active[slot] = valid
        end
    end
    for iteration in 1:limit
        any_live = false
        for slot in 1:2
            @inbounds begin
                total = state[slot,1] + state[slot,2]
                live = active[slot] && total <= 4f0
                active[slot] = live
                if live
                    real = state[slot,1] - state[slot,2] + state[slot,4]
                    imag = state[slot,3] - total + state[slot,5]
                    state[slot,1] = real*real; state[slot,2] = imag*imag
                    z = real+imag; state[slot,3] = z*z
                    counts[slot] += UInt32(1)
                end
                any_live |= live
            end
        end
        if EARLY && !any_live; break; end
    end
    for slot in 1:2
        row = y + (slot-1)*TY
        if row <= ny && x <= nx; @inbounds out[row,x] = counts[slot]; end
    end
end

# Deliberately small launcher: group=(rows,columns), pixels=1 or 2.
function launch!(out,xs,ys,n,limit; group=(32,8),pixels=1,shared=false,early=true)
    pixels in (1,2) || throw(ArgumentError("pixels must be 1 or 2"))
    all(>(0),group) || throw(ArgumentError("workgroup dimensions must be positive"))
    backend = get_backend(out)
    kernel = pixels == 1 ? mandelbrot_single! : mandelbrot_ilp2!
    ny, nx = size(out)
    groups_y = cld(ny,group[1]*pixels); groups_x = cld(nx,group[2])
    # Round the range explicitly so all lanes participate in shared barriers.
    kernel(backend,group)(out,xs,ys,nx,ny,limit,Val(shared),Val(early);
                         ndrange=(groups_y*group[1],groups_x*group[2]))
    out
end

# Exact reuse only: identical Float32 coordinate pairs have identical recurrence.
# This expands a compact image; no neighboring-pixel interpolation is used.
@kernel function expand_pixels!(out, @Const(compact), @Const(xmap), @Const(ymap))
    index = @index(Global, Cartesian)
    y = index[1]; x = index[2]
    @inbounds out[y,x] = compact[ymap[y],xmap[x]]
end
function expand!(out,compact,xmap,ymap;group=(32,8))
    expand_pixels!(get_backend(out),group)(out,compact,xmap,ymap;ndrange=size(out))
    out
end
