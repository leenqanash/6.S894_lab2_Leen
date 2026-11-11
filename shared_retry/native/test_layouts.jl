using Test
@testset "Native matrix mapping and padded transpose" begin
    for n in (1,31,32,33,65),nt in (64,128,256,512),p in (1,2,4,8),col in (false,true)
        nw=nt÷32;p<=32÷nw||continue
        for tile in 0:cld(n,32)^2-1
            ox=(tile%cld(n,32))*32;oy=(tile÷cld(n,32))*32
            produced=Tuple{Int,Int}[]
            for t in 0:nt-1,j in 0:p:cld(32,nw)-1,k in 0:p-1
                lane=t%32;warp=t÷32
                x=col ? warp+(j+k)*nw : lane
                y=col ? lane : warp+(j+k)*nw
                x<32&&y<32&&push!(produced,(x,y))
            end
            @test length(produced)==length(unique(produced))==1024
            loaded=[(t÷32+j*nw,t%32) for t in 0:nt-1 for j in 0:cld(32,nw)-1]
            @test Set(loaded)==Set(produced)
            for (x,y) in loaded
                if ox+x<n&&oy+y<n
                    @test LinearIndices((n,n))[oy+y+1,ox+x+1]==(ox+x)*n+oy+y+1
                end
            end
        end
    end
end
