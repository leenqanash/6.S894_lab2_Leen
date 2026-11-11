# Fresh family comparisons, profiler capture, and runnable renderer checks.
include("confirm_families.jl")
function postprocess(dir)
    confirm_families(dir)
    ncu="/orcd/software/core/001/pkg/cuda/13.1.0/bin/ncu"
    if isfile(ncu)
        exportbase=joinpath(abspath(dir),"nsight-profile")
        profilelog=joinpath(dir,"nsight-profile.log")
        try
            open(profilelog,"w") do io
                run(pipeline(`$ncu --profile-from-start off --set full --export $exportbase --force-overwrite $(Base.julia_cmd()) --project=. --threads=2 profile.jl`,stdout=io,stderr=io))
            end
            open(joinpath(dir,"nsight-metrics.csv"),"w") do io
                run(pipeline(`$ncu --import $(exportbase*".ncu-rep") --csv --page raw`,stdout=io,stderr=io))
            end
            write(joinpath(dir,"profiling_status.txt"),"Nsight Compute capture and raw metrics export completed.\n")
        catch err
            write(joinpath(dir,"profiling_status.txt"),"Profiler did not complete; inspect nsight-profile.log.\n"*sprint(showerror,err)*"\n")
        end
    else
        write(joinpath(dir,"profiling_status.txt"),"Nsight Compute executable unavailable.\n")
    end
    for mode in ("global","tile_shared")
        open(joinpath(dir,"renderer-$mode.log"),"w") do io
            run(pipeline(`$(Base.julia_cmd()) --project=. --threads=2 render.jl $mode`,stdout=io,stderr=io))
        end
    end
    write(joinpath(dir,"POSTPROCESS_COMPLETE"),"Family comparisons, profiler attempt and renderer checks completed $(now()).\n")
    println("Postprocessing complete: ",dir)
end
abspath(PROGRAM_FILE)==(@__FILE__) && postprocess(only(ARGS))
