# MODE 0: packed; 1: padded; 2: XOR swizzle. For powers-of-two P ≤ 32,
# the XOR form is a bijection of all 32 banks for both producer and consumer.
@inline function exchange_index(lane,slot,::Val{P},::Val{MODE}) where {P,MODE}
    MODE==2 ? lane*P+(slot ⊻ (lane÷(32÷P))) : lane*(P+MODE)+slot
end

# Shared memory exchanges results between warp lanes so global output stores
# can be contiguous. Both paths perform exactly the same pixel arithmetic.
@inline function retry_inputs(batch,t,n,::Val{P},::Val{PRE},xs,ys) where {P,PRE}
    ids=ntuple(k->batch+t*P+k-1,Val(P))
    active=ntuple(k->ids[k]<n*n,Val(P))
    cx=ntuple(k->active[k] ? (PRE ? (@inbounds xs[ids[k]%n+1]) : coordinates(ids[k],n)[1]) : 0f0,Val(P))
    cy=ntuple(k->active[k] ? (PRE ? (@inbounds ys[ids[k]÷n+1]) : coordinates(ids[k],n)[2]) : 0f0,Val(P))
    return ids,cx,cy,active
end
function output_exchange!(out,xs,ys,n,limit,::Val{P},::Val{SHARED},::Val{PAD},pre) where {P,SHARED,PAD}
    t=Int(threadIdx().x)-1;nt=Int(blockDim().x)
    warp=t÷32;lane=t%32
    batch=(Int(blockIdx().x)-1)*nt*P
    step=Int(gridDim().x)*nt*P
    pitch=PAD==1 ? P+1 : P
    if SHARED
        buffer=CuDynamicSharedArray(UInt32,nt*pitch)
    end
    # Each warp advances independently; all 32 lanes participate in both
    # warp barriers, including the last partially filled group of pixels.
    while batch+warp*32*P<n*n
        ids,cx,cy,active=retry_inputs(batch,t,n,Val(P),pre,xs,ys)
        valid=active
        x2=ntuple(_->0f0,Val(P));y2=x2;w=x2
        counts=ntuple(_->UInt32(0),Val(P))
        for iter in 1:limit
            x2,y2,w,counts,active=advance_group(cx,cy,x2,y2,w,counts,active)
            alive=false
            for k in 1:P;alive|=active[k];end
            alive || break
        end
        if SHARED
            base=warp*32*pitch
            for k in 1:P
                @inbounds buffer[base+exchange_index(lane,k-1,Val(P),Val(PAD))+1]=counts[k]
            end
            sync_warp()
            for k in 0:P-1
                q=lane+32*k
                p=batch+warp*32*P+q
                if p<n*n
                    @inbounds out[p+1]=buffer[base+exchange_index(q÷P,q%P,Val(P),Val(PAD))+1]
                end
            end
            sync_warp()
        else
            for k in 1:P
                if valid[k]
                    @inbounds out[ids[k]+1]=counts[k]
                end
            end
        end
        batch+=step
    end
    return
end

# A standard padded shared transpose: compute along either rows or columns,
# then exchange outputs through a 32×33 buffer for coalesced row-major stores.
# CACHE additionally stages x/y coordinates from global memory cooperatively.
function transpose_tile!(out,xs,ys,n,limit,::Val{SHARED},::Val{COLUMN},::Val{CACHE}) where {SHARED,COLUMN,CACHE}
    t=Int(threadIdx().x)-1;nt=Int(blockDim().x)
    lane=t%32;warp=t÷32;nwarps=nt÷32
    tiles=cld(n,32)
    tile=Int(blockIdx().x)-1
    if SHARED
        buffer=CuStaticSharedArray(UInt32,(33,32))
    end
    if CACHE
        coords=CuStaticSharedArray(Float32,64)
    end
    while tile<tiles*tiles
        ox=(tile%tiles)*32;oy=(tile÷tiles)*32
        if CACHE
            if t<32
                @inbounds coords[t+1]=ox+t<n ? xs[ox+t+1] : 0f0
                @inbounds coords[33+t]=oy+t<n ? ys[oy+t+1] : 0f0
            end
            sync_threads()
        end
        for j in 0:cld(32,nwarps)-1
            axis=warp+j*nwarps
            if axis<32
                lx=COLUMN ? axis : lane
                ly=COLUMN ? lane : axis
                x=ox+lx;y=oy+ly
                value=UInt32(0)
                if x<n && y<n
                    if CACHE
                        @inbounds value=pixel(coords[lx+1],coords[33+ly],limit)
                    else
                        @inbounds value=pixel(xs[x+1],ys[y+1],limit)
                    end
                end
                if SHARED
                    @inbounds buffer[lx+1,ly+1]=value
                elseif x<n && y<n
                    @inbounds out[y*n+x+1]=value
                end
            end
        end
        if SHARED
            sync_threads()
            for j in 0:cld(32,nwarps)-1
                ly=warp+j*nwarps
                x=ox+lane;y=oy+ly
                if ly<32 && x<n && y<n
                    @inbounds out[y*n+x+1]=buffer[lane+1,ly+1]
                end
            end
            sync_threads()
        elseif CACHE
            sync_threads()
        end
        tile+=Int(gridDim().x)
    end
    return
end
