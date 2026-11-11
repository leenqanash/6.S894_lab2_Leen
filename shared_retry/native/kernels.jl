# Native Julia image convention: output[y+1,x+1], column-major storage.
# Horizontal math and contiguous stores point in different directions; the
# shared transpose lets us keep both without changing the required layout.
@inline function native_dense_inputs(batch,t,n,::Val{P},::Val{COLUMN},::Val{PRE},xs,ys) where {P,COLUMN,PRE}
    q=ntuple(k->batch+t*P+k-1,Val(P))
    valid=ntuple(k->q[k]<n*n,Val(P))
    xx=ntuple(k->COLUMN ? q[k]÷n : q[k]%n,Val(P))
    yy=ntuple(k->COLUMN ? q[k]%n : q[k]÷n,Val(P))
    cx=ntuple(k->valid[k] ? (PRE ? (@inbounds xs[xx[k]+1]) : coordinates(yy[k]*n+xx[k],n)[1]) : 0f0,Val(P))
    cy=ntuple(k->valid[k] ? (PRE ? (@inbounds ys[yy[k]+1]) : coordinates(yy[k]*n+xx[k],n)[2]) : 0f0,Val(P))
    output=ntuple(k->xx[k]*n+yy[k],Val(P))
    return output,cx,cy,valid
end
function native_dense!(out,xs,ys,n,limit,::Val{P},column,pre) where P
    t=Int(threadIdx().x)-1;nt=Int(blockDim().x)
    batch=(Int(blockIdx().x)-1)*nt*P;step=Int(gridDim().x)*nt*P
    while batch<n*n
        ids,cx,cy,active=native_dense_inputs(batch,t,n,Val(P),column,pre,xs,ys)
        valid=active;x2=ntuple(_->0f0,Val(P));y2=x2;w=x2
        counts=ntuple(_->UInt32(0),Val(P))
        for iteration in 1:limit
            x2,y2,w,counts,active=advance_group(cx,cy,x2,y2,w,counts,active)
            alive=false
            for k in 1:P;alive|=active[k];end
            alive||break
        end
        for k in 1:P
            if valid[k];@inbounds out[ids[k]+1]=counts[k];end
        end
        batch+=step
    end
    return
end
@inline function native_tile_inputs(ox,oy,lane,warp,nwarps,j,n,::Val{P},::Val{COLUMN},::Val{CACHE},coords,xs,ys) where {P,COLUMN,CACHE}
    lx=ntuple(k->COLUMN ? warp+(j+k-1)*nwarps : lane,Val(P))
    ly=ntuple(k->COLUMN ? lane : warp+(j+k-1)*nwarps,Val(P))
    valid=ntuple(k->lx[k]<32&&ly[k]<32&&ox+lx[k]<n&&oy+ly[k]<n,Val(P))
    cx=ntuple(k->valid[k] ? (CACHE ? (@inbounds coords[lx[k]+1]) : (@inbounds xs[ox+lx[k]+1])) : 0f0,Val(P))
    cy=ntuple(k->valid[k] ? (CACHE ? (@inbounds coords[33+ly[k]]) : (@inbounds ys[oy+ly[k]+1])) : 0f0,Val(P))
    return lx,ly,cx,cy,valid
end
function native_tile!(out,xs,ys,n,limit,::Val{P},column,::Val{SHARED},::Val{CACHE}) where {P,SHARED,CACHE}
    t=Int(threadIdx().x)-1;nt=Int(blockDim().x)
    lane=t%32;warp=t÷32;nwarps=nt÷32
    tiles=cld(n,32);tile=Int(blockIdx().x)-1
    if SHARED;buffer=CuStaticSharedArray(UInt32,(33,32));end
    if CACHE;coords=CuStaticSharedArray(Float32,64);else;coords=nothing;end
    while tile<tiles*tiles
        ox=(tile%tiles)*32;oy=(tile÷tiles)*32
        if CACHE
            if t<32
                @inbounds coords[t+1]=ox+t<n ? xs[ox+t+1] : 0f0
                @inbounds coords[33+t]=oy+t<n ? ys[oy+t+1] : 0f0
            end
            sync_threads()
        end
        for j in 0:P:cld(32,nwarps)-1
            lx,ly,cx,cy,active=native_tile_inputs(ox,oy,lane,warp,nwarps,j,n,Val(P),column,Val(CACHE),coords,xs,ys)
            valid=active;x2=ntuple(_->0f0,Val(P));y2=x2;w=x2
            counts=ntuple(_->UInt32(0),Val(P))
            for iteration in 1:limit
                x2,y2,w,counts,active=advance_group(cx,cy,x2,y2,w,counts,active)
                alive=false
                for k in 1:P;alive|=active[k];end
                alive||break
            end
            for k in 1:P
                if lx[k]<32&&ly[k]<32
                    if SHARED
                        @inbounds buffer[lx[k]+1,ly[k]+1]=counts[k]
                    elseif valid[k]
                        @inbounds out[(ox+lx[k])*n+oy+ly[k]+1]=counts[k]
                    end
                end
            end
        end
        if SHARED
            sync_threads()
            for j in 0:cld(32,nwarps)-1
                # Native column-major stores: adjacent lanes change y.
                lx=warp+j*nwarps;x=ox+lx;y=oy+lane
                if lx<32&&x<n&&y<n
                    @inbounds out[x*n+y+1]=buffer[lx+1,lane+1]
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
