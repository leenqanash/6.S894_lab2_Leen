using Test
# Test indexing independently of GPU timing and synchronization.
index(lane,slot,p,mode)=mode==2 ? lane*p+(slot⊻(lane÷(32÷p))) : lane*(p+mode)+slot
@testset "Shared output exchange layouts" begin
    for p in (2,4,8,16),mode in (0,1,2)
        pitch=mode==1 ? p+1 : p
        storage=fill(-1,32*pitch)
        for lane in 0:31,slot in 0:p-1
            storage[index(lane,slot,p,mode)+1]=lane*p+slot
        end
        recovered=[storage[index(q÷p,q%p,p,mode)+1] for q in 0:32*p-1]
        @test recovered==collect(0:32*p-1)
        if mode==2
            for slot in 0:p-1
                @test length(unique(index(lane,slot,p,mode)%32 for lane in 0:31))==32
            end
            for group in 0:p-1
                @test length(unique(index((lane+32*group)÷p,(lane+32*group)%p,p,mode)%32 for lane in 0:31))==32
            end
        end
    end
    for n in (1,7,31,65,128),p in (2,4,8,16),nt in (64,128,256,512),blocks in (1,3)
        counts=zeros(Int,n*n)
        for block in 0:blocks-1,warp in 0:nt÷32-1
            batch=block*nt*p
            while batch+warp*32*p<n*n
                for q in 0:32*p-1
                    pix=batch+warp*32*p+q
                    pix<n*n&&(counts[pix+1]+=1)
                end
                batch+=blocks*nt*p
            end
        end
        @test all(==(1),counts)
    end
end
