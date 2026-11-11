# Nsight's raw export is a wide, quoted CSV: header, units row, then launches.
function csv_fields(line)
    fields=String[]; buffer=IOBuffer(); quoted=false
    i=firstindex(line)
    while i<=lastindex(line)
        c=line[i]
        if c=='"'
            next=nextind(line,i)
            if quoted && next<=lastindex(line) && line[next]=='"'
                write(buffer,'"');i=next
            else
                quoted=!quoted
            end
        elseif c==',' && !quoted
            push!(fields,String(take!(buffer)))
        else
            write(buffer,c)
        end
        i=nextind(line,i)
    end
    push!(fields,String(take!(buffer)))
    return fields
end
function analyze_profile(dir)
    lines=readlines(joinpath(dir,"nsight-metrics.csv"))
    header=csv_fields(lines[1]);units=csv_fields(lines[2])
    wanted=["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum","launch__registers_per_thread","launch__shared_mem_per_block_static","launch__shared_mem_per_block_dynamic","sm__warps_active.avg.pct_of_peak_sustained_active","smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum"]
    selected=[m for m in wanted if m in header]
    open(joinpath(dir,"profile_selected_metrics.csv"),"w") do io
        println(io,join(vcat(["ID","case","stride","offset_elements","shared","sectors_per_global_load_request"],selected),','))
        for line in lines[3:end]
            values=csv_fields(line)
            length(values)==length(header) || continue
            row=Dict(header .=> values)
            id=tryparse(Int,row["ID"]);id===nothing && continue
            if id<20
                stride=id<10 ? 1 : 32
                offset=(0,1,4,8,16)[(id%10)÷2+1]
                shared=isodd(id)
                name="memory_probe"
            else
                name=("main_p2_chunked","coordinates_global","coordinates_shared","tile_global_64x8","tile_shared_64x8")[id-19]
                stride=1;offset=0;shared=id in (22,24)
            end
            requests=parse(Float64,replace(row[wanted[1]],","=>""))
            sectors=parse(Float64,replace(row[wanted[2]],","=>""))
            ratio=requests==0 ? 0.0 : sectors/requests
            fields=vcat(string.([id,name,stride,offset,shared,ratio]),[replace(row[m],","=>"") for m in selected])
            println(io,join(fields,','))
            println("Profile ",id," ",name," offset=",offset," stride=",stride," shared=",shared," sectors/request=",ratio)
        end
    end
    open(joinpath(dir,"profile_metric_units.txt"),"w") do io
        for metric in selected
            println(io,metric," = ",units[findfirst(==(metric),header)])
        end
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && analyze_profile(only(ARGS))
