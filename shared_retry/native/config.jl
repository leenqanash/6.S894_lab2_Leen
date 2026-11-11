function native_cases()
    all=Case[]
    for threads in (64,128,256,512),ratio in (4,8),p in (1,2,4,8,16),pre in (false,true),column in (false,true)
        id="native_dense_t$(threads)_b$(ratio)_p$(p)_pre$(Int(pre))_col$(Int(column))"
        push!(all,Case(id,:native_dense,threads,ratio,p,false,0,pre,column,false))
    end
    for threads in (64,128,256,512),ratio in (4,8),p in (1,2,4,8),column in (false,true),(shared,cache) in ((false,false),(true,false),(true,true),(false,true))
        p<=32÷(threads÷32)||continue
        id="native_tile_t$(threads)_b$(ratio)_p$(p)_col$(Int(column))_s$(Int(shared))_c$(Int(cache))"
        push!(all,Case(id,:native_tile,threads,ratio,p,shared,0,true,column,cache))
    end
    all
end
function native_matched(c)
    "native_tile_t$(c.threads)_b$(c.ratio)_p$(c.pixels)_col$(Int(c.column))_s0_c0"
end
function native_launcher(c,out,xs,ys,n,limit,sm)
    threads=c.threads;blocks=sm*c.ratio
    if c.family==:native_dense
        return ()->(@cuda threads=threads blocks=blocks fastmath=false native_dense!(out,xs,ys,n,limit,Val(c.pixels),Val(c.column),Val(c.precomputed)))
    end
    return ()->(@cuda threads=threads blocks=blocks fastmath=false native_tile!(out,xs,ys,n,limit,Val(c.pixels),Val(c.column),Val(c.shared),Val(c.cache)))
end
